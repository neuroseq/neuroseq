from pathlib import Path

import numpy as np
import pytest
import tifffile

from neuroseq.importers import roi_traces
from neuroseq.importers.importer_tif import TifPipelineParams
from neuroseq.importers.importer_tif import run_roi_average_pipeline
from neuroseq.importers.importer_tif import stimulation_contaminated_frames
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _masks() -> np.ndarray:
    masks = np.zeros((40, 40), dtype=np.int32)
    masks[5:9, 5:9] = 1
    masks[25:30, 20:24] = 2
    return masks


def _movie(n_frames: int = 50, nchannels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, (n_frames * nchannels, 40, 40)).astype(np.uint16)


def _expected(movie: np.ndarray, masks: np.ndarray, channel: int, nch: int):
    frames = movie[channel - 1 :: nch].astype(np.float64)
    return np.array([frames[:, masks == label].mean(axis=1) for label in (1, 2)])


@pytest.mark.parametrize("storage", ["contiguous", "compressed"])
def test_streamed_roi_means_match_whole_file_means(tmp_path: Path, storage) -> None:
    """Chunks smaller than the file, read by memory map or page by page, give the
    same traces as averaging the whole stack."""
    masks = _masks()
    movie = _movie()
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path,
        movie,
        photometric="minisblack",
        compression="zlib" if storage == "compressed" else None,
    )
    result = run_roi_average_pipeline(
        [path],
        masks,
        TifPipelineParams(nchannels=2, functional_chan=2),
        chunk_frames=7,
    )
    np.testing.assert_allclose(result.F, _expected(movie, masks, 2, 2), rtol=1e-5)
    np.testing.assert_allclose(result.mean_image, movie[1::2].mean(axis=0), rtol=1e-5)
    assert not result.Fneu.any() and result.extra["neuropil"] is False


def test_chunks_are_read_lazily(tmp_path: Path) -> None:
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, _movie(30, 1), photometric="minisblack")
    chunks = roi_traces.iter_functional_chunks(path, 1, 1, chunk_frames=8)
    assert [len(chunk) for chunk in chunks] == [8, 8, 8, 6]


def test_neuropil_annulus_avoids_every_cell_and_has_suite2p_size() -> None:
    masks = _masks()
    (first, second) = roi_traces.neuropil_pixels(masks, np.array([1, 2]))
    assert len(first) == len(second) == roi_traces.NEUROPIL_MIN_PIXELS
    near_cells = np.zeros(masks.shape, bool)
    from scipy import ndimage

    near_cells = (
        ndimage.distance_transform_edt(masks == 0)
        <= roi_traces.NEUROPIL_INNER_RADIUS_PX
    )
    assert not near_cells.ravel()[first].any()
    assert not near_cells.ravel()[second].any()
    # The annulus surrounds its own cell: its centre is near the cell's.
    ys, xs = np.unravel_index(first, masks.shape)
    assert abs(ys.mean() - 6.5) < 4 and abs(xs.mean() - 6.5) < 4


def test_pipeline_neuropil_traces_follow_the_surround(tmp_path: Path) -> None:
    masks = _masks()
    movie = np.full((10, 40, 40), 100, dtype=np.uint16)
    movie[:, masks > 0] = 300
    movie[5:] += 50  # the surround (and cells) brighten together
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, movie, photometric="minisblack")
    result = run_roi_average_pipeline(
        [path], masks, TifPipelineParams(nchannels=1), neuropil=True
    )
    np.testing.assert_allclose(result.Fneu[:, :5], 100)
    np.testing.assert_allclose(result.Fneu[:, 5:], 150)
    np.testing.assert_allclose(result.F[:, 0], 300)


def test_imaging_start_offset_moves_the_contaminated_frames() -> None:
    stim = StimArray(
        "laser", StimType.Excitation, np.array([[0], [0], [1], [0]]), "mW", 0.5
    )
    # Pulse at 1.0-1.5 s on the input's axis; frames at 10 Hz.
    np.testing.assert_array_equal(
        stimulation_contaminated_frames(stim, 10, 30), np.arange(10, 15)
    )
    # Imaging started 0.4 s after the input: the pulse is at 0.6-1.1 s of imaging.
    np.testing.assert_array_equal(
        stimulation_contaminated_frames(stim, 10, 30, imaging_start_s=0.4),
        np.arange(6, 11),
    )
    # Imaging started 0.5 s before the input.
    np.testing.assert_array_equal(
        stimulation_contaminated_frames(stim, 10, 30, imaging_start_s=-0.5),
        np.arange(15, 20),
    )
    # A pulse entirely before the first frame lights none.
    assert stimulation_contaminated_frames(stim, 10, 30, imaging_start_s=2.0).size == 0


def test_sliding_baseline_methods() -> None:
    trace = np.r_[np.full(100, 10.0), np.full(20, 30.0), np.full(100, 10.0)][None]
    maximin = roi_traces.sliding_baseline(
        trace, 10.0, win_baseline_s=5.0, sig_baseline_frames=1.0
    )
    np.testing.assert_allclose(maximin, 10.0, atol=1e-6)  # the transient is ignored
    constant = roi_traces.sliding_baseline(trace, 10.0, method="constant")
    np.testing.assert_allclose(constant, 10.0, atol=1e-6)
    prctile = roi_traces.sliding_baseline(
        trace, 10.0, method="constant_prctile", prctile_baseline=50
    )
    np.testing.assert_allclose(prctile, 10.0)
    with pytest.raises(ValueError, match="Unknown baseline"):
        roi_traces.sliding_baseline(trace, 10.0, method="median")


def test_roi_dff_subtracts_neuropil_and_divides_by_the_raw_baseline() -> None:
    F = np.r_[np.full(50, 200.0), np.full(10, 260.0), np.full(50, 200.0)][None]
    Fneu = np.full_like(F, 250.0)  # F - 0.7 Fneu has a baseline of 25
    dff = roi_traces.roi_delta_f_over_f(
        F, Fneu, 10.0, method="constant_prctile", prctile_baseline=20
    )
    # (60 / 200), not 60 / 25.
    assert dff.max() == pytest.approx(0.3)
    np.testing.assert_allclose(dff[0, :50], 0.0, atol=1e-6)
