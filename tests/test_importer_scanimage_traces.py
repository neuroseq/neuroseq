from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_scanimage_traces import DFF_UNITS
from neuroseq.importers.importer_scanimage_traces import baseline_frame_count
from neuroseq.importers.importer_scanimage_traces import delta_f_over_f
from neuroseq.importers.importer_scanimage_traces import extract_recording_dff
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    calcium_rise_fraction,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType

FPS = 30.0


def _cellpose(mapping: dict[int, int] | None = None) -> CellposeImportResult:
    segmentation = np.full((6, 8), -1, dtype=np.int32)
    segmentation[:3, :4] = 0
    segmentation[3:, 4:] = 1
    return CellposeImportResult(
        image_path=Path("fov.tif"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((6, 8, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1]),
        stimulus_to_cellpose_mapping=mapping,
        mapping_strategy="Custom" if mapping else "Linear",
    )


def _block(onset_s: float, seconds: float, dt: float = 0.001) -> StimArray:
    values = np.zeros((int(seconds / dt), 2))
    values[int(onset_s / dt) : int((onset_s + 0.02) / dt), 1] = 1.0
    return StimArray("Block", StimType.External, values, "", dt)


def _synthetic_recording(path: Path, onset_s: float, frames: int) -> None:
    """Two interleaved channels, as ScanImage saves them. ROI 1 (cellpose id 1) has a
    jGCaMP-like transient after ``onset_s``; channel 2 is constant."""
    t = np.arange(frames) / FPS
    response = np.where(t >= onset_s, np.exp(-(t - onset_s) / 0.5), 0.0)
    pages = []
    for frame in range(frames):
        green = np.full((6, 8), 100.0)
        green[3:, 4:] = 100.0 * (1.0 + 0.8 * response[frame])
        pages += [green.astype(np.int16), np.full((6, 8), 7, dtype=np.int16)]
    tifffile.imwrite(path, np.stack(pages))


def test_delta_f_over_f_uses_the_frames_before_the_stimulus() -> None:
    traces = np.array([[100.0, 100.0, 150.0, 200.0]])
    np.testing.assert_allclose(delta_f_over_f(traces, 2), [[0.0, 0.0, 0.5, 1.0]])
    np.testing.assert_allclose(
        delta_f_over_f(np.array([[10.0, 10.0, 10.0, 10.0, 20.0]]), 0)[0, -1], 1.0
    )


def test_delta_f_over_f_of_a_negative_baseline() -> None:
    """Review: only the divisor is made positive (offset-corrected detectors)."""
    np.testing.assert_allclose(
        delta_f_over_f(np.array([[-50.0, -50.0, -25.0]]), 2), [[0.0, 0.0, 0.5]]
    )


def test_baseline_frame_count_stops_at_the_first_stimulus() -> None:
    assert baseline_frame_count(_block(0.5, 1.0), FPS) == 15
    assert baseline_frame_count(None, FPS) == 0


def test_extracts_dff_of_each_mapped_roi_from_a_recording(tmp_path) -> None:
    path = tmp_path / "2026-10-22_00001.tif"
    onset = 0.5
    _synthetic_recording(path, onset, frames=60)
    trace = extract_recording_dff(
        [path],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
    )
    assert trace.units == DFF_UNITS
    assert trace.dt == pytest.approx(1 / FPS)
    assert trace.traces.shape == (2, 60)
    np.testing.assert_array_equal(trace.raw_cellpose_ids, [0, 1])
    np.testing.assert_array_equal(trace.indices, [3, 7])
    # The responding cell's ΔF/F peaks near 0.8; the frame lit by the
    # stimulation laser is interpolated, the other cell stays flat.
    assert np.max(trace.traces[1]) == pytest.approx(0.8, abs=0.1)
    np.testing.assert_allclose(trace.traces[0], 0.0, atol=1e-6)
    # What the rheobase routine measures for neuron 7 after the pulse.
    rise = calcium_rise_fraction(trace, 7, onset, 250.0, 300.0)
    assert rise > 0.5
    assert calcium_rise_fraction(trace, 3, onset, 250.0, 300.0) == pytest.approx(0)


@pytest.mark.parametrize("remove_artefact", [True, False])
def test_stimulation_artefact_removal_can_be_turned_off(
    tmp_path, remove_artefact: bool
) -> None:
    path = tmp_path / "2026-10-22_00002.tif"
    onset = 0.5
    _synthetic_recording(path, onset, frames=60)
    pages = tifffile.imread(path)
    lit = int(onset * FPS)  # the frame recorded while the laser is on
    pages[2 * lit][:3, :4] = 1000  # stimulation light on the flat cell
    tifffile.imwrite(path, pages)
    trace = extract_recording_dff(
        [path],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
        remove_artefact=remove_artefact,
    )
    if remove_artefact:
        np.testing.assert_allclose(trace.traces[0], 0.0, atol=1e-6)
    else:
        assert trace.traces[0, lit] == pytest.approx(9.0)
        # Still measured against the frames before the stimulus.
        np.testing.assert_allclose(trace.traces[0, :lit], 0.0, atol=1e-6)


@pytest.mark.parametrize("remove_artefact", [True, False])
def test_suite2p_extraction_subtracts_neuropil_with_the_same_masks(
    tmp_path, monkeypatch, remove_artefact: bool
) -> None:
    from neuroseq.importers import importer_scanimage_traces as module
    from neuroseq.importers.importer_tif import TifPipelineResult

    onset = 0.5
    lit = int(onset * FPS)
    F = np.full((2, 60), 170.0)
    F[1, lit + 1 :] = 240.0  # ROI 2 responds after the pulse
    F[0, lit] = 1000.0  # stimulation light on ROI 1
    Fneu = np.full((2, 60), 100.0)
    calls: dict[str, Any] = {}

    def fake_suite2p(paths, masks, params, *, progress, register, reference_image):
        calls.update(
            masks=masks,
            fs=params.fs,
            progress=progress,
            register=register,
            reference=reference_image,
        )
        return TifPipelineResult(
            F=F.copy(),
            Fneu=Fneu,
            spks=np.zeros_like(F),
            stat=np.array([]),
            iscell=np.ones((2, 2)),
            ops={},
            mean_image=np.zeros((6, 8)),
            masks=masks,
            extra={"registration": {"reference": "movie", "max_shift_px": 1.5}},
        )

    monkeypatch.setattr(module, "run_suite2p_pipeline", fake_suite2p)
    registered: list[dict] = []
    progress = lambda _message, _fraction: None  # noqa: E731
    trace = extract_recording_dff(
        [tmp_path / "block.tif"],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
        remove_artefact=remove_artefact,
        method="suite2p",
        progress=progress,
        on_registered=registered.append,
    )
    assert calls["fs"] == FPS and calls["progress"] is progress
    # Motion-corrected; fov.tif does not exist, so Suite2p makes its own reference.
    assert calls["register"] is True and calls["reference"] is None
    assert registered == [{"reference": "movie", "max_shift_px": 1.5}]
    assert set(np.unique(calls["masks"])) == {0, 1, 2}  # the Cellpose masks
    np.testing.assert_array_equal(trace.indices, [3, 7])
    # F - 0.7 Fneu: a baseline of 100, rising to 170 (ΔF/F 0.7) for ROI 2.
    assert trace.traces[1, -1] == pytest.approx(0.7)
    if remove_artefact:
        np.testing.assert_allclose(trace.traces[0], 0.0, atol=1e-6)
    else:
        assert trace.traces[0, lit] == pytest.approx(8.3)


def test_fov_reference_image_is_the_segmented_tiff(tmp_path) -> None:
    import dataclasses

    from neuroseq.importers.importer_scanimage_traces import fov_reference_image

    path = tmp_path / "fov_mean.tif"
    tifffile.imwrite(path, np.stack([np.full((6, 8), 10), np.full((6, 8), 20)]))
    image = fov_reference_image(dataclasses.replace(_cellpose(), image_path=path))
    assert image is not None and image.shape == (6, 8)
    np.testing.assert_allclose(image, 15.0)
    png = tmp_path / "fov.png"
    png.write_bytes(b"")
    assert fov_reference_image(dataclasses.replace(_cellpose(), image_path=png)) is None
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff")
    assert (
        fov_reference_image(dataclasses.replace(_cellpose(), image_path=broken)) is None
    )


def test_unknown_extraction_method_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown trace extraction"):
        extract_recording_dff(
            [tmp_path / "block.tif"],
            fps=FPS,
            saved_channels=[1],
            functional_channel=1,
            cellpose_data=_cellpose(),
            method="fiji",  # type: ignore[arg-type]
        )


def test_extraction_needs_masks_and_the_functional_channel(tmp_path) -> None:
    path = tmp_path / "rec.tif"
    _synthetic_recording(path, 0.5, frames=4)
    with pytest.raises(ValueError, match="Cellpose"):
        extract_recording_dff(
            [path],
            fps=FPS,
            saved_channels=[1],
            functional_channel=1,
            cellpose_data=None,
        )
    with pytest.raises(ValueError, match="Channel 3 was not saved"):
        extract_recording_dff(
            [path],
            fps=FPS,
            saved_channels=[1, 2],
            functional_channel=3,
            cellpose_data=_cellpose(),
        )


def _transients(spike_frames: dict[int, list[int]], n_frames: int = 150) -> np.ndarray:
    """ΔF/F of three cells: 0.4 per spike, decaying with jGCaMP8m's 0.5 s, plus noise."""
    t = np.arange(n_frames) / FPS
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.02, (3, n_frames))
    for row, frames in spike_frames.items():
        for frame in frames:
            onset = frame / FPS
            values[row] += np.where(t >= onset, 0.4 * np.exp(-(t - onset) / 0.5), 0.0)
    return values.astype(np.float32)


def test_deconvolution_finds_the_spikes_of_each_cell() -> None:
    pytest.importorskip("suite2p.extraction")

    from neuroseq.data import TraceData
    from neuroseq.importers.importer_scanimage_traces import deconvolve_dff

    trace = TraceData(
        "dF/F",
        _transients({1: [30, 75], 2: [100]}),
        np.array([5, 7, 9]),
        DFF_UNITS,
        1 / FPS,
        np.array([0, 1, 2]),
    )
    spikes = deconvolve_dff(trace)
    np.testing.assert_array_equal(spikes.spike_indices, [7, 7, 9])
    np.testing.assert_allclose(spikes.spike_times, np.array([30, 75, 100]) / FPS)
    np.testing.assert_array_equal(spikes.raw_cellpose_ids, [1, 1, 2])
    # Noise alone (cell 5) never reaches the threshold; a higher one hides all.
    assert deconvolve_dff(trace, threshold=1.0).spike_indices.size == 0


def test_recorded_probe_is_counted_from_deconvolved_spikes(tmp_path) -> None:
    pytest.importorskip("suite2p.extraction")

    from neuroseq.importers.importer_scanimage_traces import deconvolve_dff
    from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
        measure_calcium_spike_count,
    )

    path = tmp_path / "rec.tif"
    onset = 0.5
    _synthetic_recording(path, onset, frames=60)
    trace = extract_recording_dff(
        [path],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
    )
    spikes = deconvolve_dff(trace)
    assert set(spikes.spike_indices) == {7}
    assert measure_calcium_spike_count(spikes, 7, onset, 300.0, 1 / FPS) >= 1
    assert measure_calcium_spike_count(spikes, 3, onset, 300.0, 1 / FPS) == 0


def test_fov_mean_averages_only_the_functional_channel(tmp_path) -> None:
    from neuroseq.importers.importer_scanimage_traces import functional_channel_mean

    path = tmp_path / "fov.tif"
    _synthetic_recording(path, onset_s=10.0, frames=30)  # no transient in range
    mean = functional_channel_mean([path], saved_channels=[1, 2], functional_channel=1)
    assert mean.shape == (6, 8) and mean.dtype == np.float32
    np.testing.assert_allclose(mean, 100.0)  # channel 2 (all 7) is left out
    red = functional_channel_mean([path], saved_channels=[1, 2], functional_channel=2)
    np.testing.assert_allclose(red, 7.0)
    with pytest.raises(ValueError, match="Channel 3 was not saved"):
        functional_channel_mean([path], saved_channels=[1, 2], functional_channel=3)
