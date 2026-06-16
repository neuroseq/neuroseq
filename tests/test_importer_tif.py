"""Unit tests for the raw-TIFF Suite2p pipeline glue code.

The full pipeline calls into Suite2p (heavy + GPU) which we do not exercise
here; we cover the pure-Python pieces and confirm the wrapper passes the right
parameters through.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neuroseq.importer_tif import TifPipelineParams
from neuroseq.importer_tif import _masks_to_stat


def _make_mask_image(shape=(32, 32)) -> np.ndarray:
    """Three labelled blobs on a background of zeros."""
    masks = np.zeros(shape, dtype=np.int32)
    masks[4:10, 4:10] = 1
    masks[15:20, 12:18] = 2
    masks[22:28, 22:30] = 3
    return masks


def _suite2p_available() -> bool:
    """Suite2p 0.x raises ValueError at import on Python 3.11 — be defensive."""
    try:
        import suite2p.detection  # noqa: F401

        return True
    except Exception:
        return False


def test_masks_to_stat_emits_one_entry_per_cell():
    if not _suite2p_available():
        pytest.skip("suite2p v1.x not installed")

    masks = _make_mask_image()
    stat = _masks_to_stat(masks, masks.shape[0], masks.shape[1])

    assert len(stat) == 3
    for entry in stat:
        assert {"ypix", "xpix", "npix", "lam", "med"} <= set(entry.keys())
        assert entry["npix"] == len(entry["xpix"])
        assert entry["lam"].shape == entry["xpix"].shape


def test_tif_pipeline_params_defaults_target_jgcamp8m():
    params = TifPipelineParams()
    assert params.fs == pytest.approx(30.0)
    assert params.tau == pytest.approx(0.5)
    assert params.nchannels == 2
    assert params.functional_chan == 1
    assert params.neuropil_coefficient == pytest.approx(0.7)
    assert params.batch_size == 200
    assert params.baseline == "maximin"


def test_run_suite2p_pipeline_rejects_empty_tif_list():
    # ValueError check fires before suite2p import — no skip needed.
    from neuroseq.importer_tif import run_suite2p_pipeline

    with pytest.raises(ValueError, match="at least one TIFF"):
        run_suite2p_pipeline([], _make_mask_image(), TifPipelineParams())


def test_run_suite2p_pipeline_rejects_non_2d_masks():
    from neuroseq.importer_tif import run_suite2p_pipeline

    bad_masks = np.zeros((4, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="masks must be 2D"):
        run_suite2p_pipeline(
            [Path("/tmp/nonexistent.tif")], bad_masks, TifPipelineParams()
        )


def test_cellpose_result_to_masks_round_trips_background_and_labels():
    from neuroseq.tif_import_dialog import _cellpose_result_to_masks

    seg = np.array(
        [
            [-1, -1, 0, 0],
            [-1, 0, 0, 1],
            [-1, -1, 1, 1],
        ],
        dtype=np.int32,
    )
    result = MagicMock()
    result.segmentation_array = seg

    masks = _cellpose_result_to_masks(result)
    # 0-based -> 1-based, -1 -> 0
    np.testing.assert_array_equal(
        masks,
        np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 2],
                [0, 0, 2, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_load_masks_from_seg_file_unwraps_cellpose_dict(tmp_path: Path):
    from neuroseq.tif_import_dialog import _load_masks_from_seg_file

    masks = _make_mask_image()
    seg_path = tmp_path / "image_seg.npy"
    np.save(seg_path, np.array({"masks": masks, "outlines": None}, dtype=object))

    loaded = _load_masks_from_seg_file(seg_path)
    np.testing.assert_array_equal(loaded, masks)


def test_load_masks_from_seg_file_accepts_plain_array(tmp_path: Path):
    from neuroseq.tif_import_dialog import _load_masks_from_seg_file

    masks = _make_mask_image()
    seg_path = tmp_path / "plain.npy"
    np.save(seg_path, masks)

    loaded = _load_masks_from_seg_file(seg_path)
    np.testing.assert_array_equal(loaded, masks)
