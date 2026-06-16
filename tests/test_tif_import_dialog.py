"""Qt smoke tests for the TIFF import dialog with a mocked Suite2p pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_pipeline_result():
    from neuroseq.importer_tif import TifPipelineResult

    n_neurons, n_frames = 4, 100
    F = np.random.default_rng(0).random((n_neurons, n_frames)).astype(np.float32)
    Fneu = np.zeros_like(F)
    spks = np.zeros_like(F)
    spks[0, 10] = 1.0
    spks[1, 20] = 1.0
    return TifPipelineResult(
        F=F,
        Fneu=Fneu,
        spks=spks,
        stat=np.array([{"med": [0.0, 0.0]} for _ in range(n_neurons)]),
        iscell=np.ones((n_neurons, 2), dtype=np.float32),
        ops={},
        mean_image=np.zeros((16, 16), dtype=np.float32),
        masks=np.zeros((16, 16), dtype=np.int32),
    )


def test_dialog_disables_loaded_radio_when_no_cellpose_data(qtbot):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    assert not dialog._use_loaded_radio.isEnabled()
    assert dialog._use_seg_radio.isChecked()


def test_dialog_requires_tifs(qtbot):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    # No TIFFs added — pressing Run should pop a warning and not start a thread.
    dialog._start_pipeline()
    assert dialog._thread is None


def test_dialog_runs_mocked_pipeline_and_produces_imported_data(
    qtbot, mocker, tmp_path
):
    from neuroseq.tif_import_dialog import TifImportDialog

    # Build a fake cellpose result so the "use loaded" path works.
    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((16, 16), dtype=np.int32) - 1

    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)

    # Stage one TIFF (file does not need to actually exist — pipeline is mocked).
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))

    result = _make_pipeline_result()
    mock_run = mocker.patch(
        "neuroseq.importer_tif.run_suite2p_pipeline",
        return_value=result,
    )
    # Avoid the success dialog blocking the event loop.
    mocker.patch.object(type(dialog).__bases__[0], "accept", lambda self_: None)

    dialog._start_pipeline()
    qtbot.waitUntil(lambda: dialog._pipeline_result is not None, timeout=5000)

    mock_run.assert_called_once()
    assert len(dialog.imported_data) == 2
    traces = dialog.imported_data[0]
    spikes = dialog.imported_data[1]
    assert traces.traces.shape == result.F.shape
    assert spikes.spike_indices.size >= 2


@pytest.mark.parametrize(
    "seg, expected",
    [
        (
            np.array([[-1, 0], [1, -1]], dtype=np.int32),
            np.array([[0, 1], [2, 0]], dtype=np.int32),
        ),
        (
            np.array([[-1, -1], [-1, -1]], dtype=np.int32),
            np.array([[0, 0], [0, 0]], dtype=np.int32),
        ),
    ],
)
def test_cellpose_result_to_masks_round_trip(seg, expected):
    from neuroseq.tif_import_dialog import _cellpose_result_to_masks

    fake_result = MagicMock()
    fake_result.segmentation_array = seg
    np.testing.assert_array_equal(_cellpose_result_to_masks(fake_result), expected)
