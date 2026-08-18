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


@pytest.mark.parametrize(
    ("pipeline", "function_name"),
    [
        ("roi_average", "run_roi_average_pipeline"),
        ("suite2p", "run_suite2p_pipeline"),
    ],
)
def test_pipeline_worker_dispatches_and_forwards_progress(
    qtbot, mocker, pipeline, function_name
):
    from neuroseq.importer_tif import TifPipelineParams
    from neuroseq.tif_import_dialog import _PipelineWorker

    result = _make_pipeline_result()

    def fake_pipeline(*_args, progress, **_kwargs):
        progress("working", 0.5)
        return result

    run = mocker.patch(
        f"neuroseq.importer_tif.{function_name}", side_effect=fake_pipeline
    )
    worker = _PipelineWorker(
        [], np.ones((2, 2), dtype=np.int32), TifPipelineParams(), pipeline
    )
    progress_events = []
    finished_results = []
    worker.progress.connect(
        lambda message, fraction: progress_events.append((message, fraction))
    )
    worker.finished.connect(finished_results.append)

    worker.run()

    run.assert_called_once()
    assert progress_events == [("working", 0.5)]
    assert finished_results == [result]


def test_pipeline_worker_emits_failure(mocker):
    from neuroseq.importer_tif import TifPipelineParams
    from neuroseq.tif_import_dialog import _PipelineWorker

    mocker.patch(
        "neuroseq.importer_tif.run_roi_average_pipeline",
        side_effect=ValueError("bad stack"),
    )
    worker = _PipelineWorker(
        [], np.ones((2, 2), dtype=np.int32), TifPipelineParams(), "roi_average"
    )
    failures = []
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == ["bad stack"]


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


def test_pipeline_selector_hides_suite2p_parameters_for_roi_average(qtbot):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._pipeline_combo.currentText() == "Suite2p"
    dialog._pipeline_combo.setCurrentIndex(1)

    assert dialog._pipeline_combo.currentText() == "Simply average from ROI"
    assert not dialog._tau_spin.isVisibleTo(dialog)
    assert dialog._fs_spin.isVisibleTo(dialog)
    assert dialog._nchannels_spin.isVisibleTo(dialog)
    assert dialog._functional_chan_spin.isVisibleTo(dialog)


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


def test_finalize_roi_average_imports_raw_trace_only(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    result = _make_pipeline_result()
    result.extra["pipeline"] = "roi_average"
    information = mocker.patch(
        "neuroseq.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    accept = mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)

    assert len(dialog.imported_data) == 1
    traces = dialog.imported_data[0]
    assert traces.name == "Simply averaged ROI traces"
    np.testing.assert_array_equal(traces.traces, result.F)
    assert dialog.experiment_duration == pytest.approx(
        result.F.shape[1] / dialog._fs_spin.value()
    )
    information.assert_called_once()
    accept.assert_called_once()


@pytest.mark.parametrize(
    "seg, expected",
    [
        (
            np.array([[-1, 0], [1, -1]], dtype=np.int32),
            np.array([[2, 0], [0, 1]], dtype=np.int32),
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


def test_browse_tifs_dedupes_added_paths(qtbot, mocker, tmp_path):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    a.write_bytes(b"")
    b.write_bytes(b"")

    mocker.patch(
        "neuroseq.tif_import_dialog.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([str(a), str(b), str(a)], ""),
    )
    dialog._browse_tifs()

    assert len(dialog._tif_paths) == 2
    assert dialog._tif_list.count() == 2


def test_remove_selected_tifs_clears_list_and_paths(qtbot, tmp_path):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    a = tmp_path / "a.tif"
    a.write_bytes(b"")
    dialog._tif_paths.append(a)
    dialog._tif_list.addItem(str(a))
    dialog._tif_list.item(0).setSelected(True)

    dialog._remove_selected_tifs()
    assert dialog._tif_paths == []
    assert dialog._tif_list.count() == 0


def test_browse_seg_file_sets_path_and_switches_radio(qtbot, mocker, tmp_path):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=MagicMock())
    qtbot.addWidget(dialog)

    seg = tmp_path / "image_seg.npy"
    seg.write_bytes(b"")
    mocker.patch(
        "neuroseq.tif_import_dialog.getopenfilename", return_value=(str(seg), "")
    )
    dialog._browse_seg_file()
    assert dialog._seg_file_path == seg
    assert dialog._use_seg_radio.isChecked()


def test_resolve_masks_requires_seg_when_seg_radio_selected(qtbot):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    # _use_seg_radio is checked because cellpose_import is None, but no file picked.
    with pytest.raises(ValueError, match="No _seg.npy"):
        dialog._resolve_masks()


def test_resolve_masks_requires_cellpose_when_loaded_radio_selected(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=MagicMock())
    qtbot.addWidget(dialog)
    # Force "use loaded" radio but clear the cellpose import.
    dialog._cellpose_import = None
    dialog._use_loaded_radio.setEnabled(True)
    dialog._use_loaded_radio.setChecked(True)
    with pytest.raises(ValueError, match="No Cellpose data"):
        dialog._resolve_masks()


def test_launch_fiji_runs_qprocess_with_selected_tif(qtbot, mocker, tmp_path):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    fake_fiji = tmp_path / "fiji"
    fake_fiji.write_text("")
    a = tmp_path / "a.tif"
    a.write_bytes(b"")
    dialog._tif_paths.append(a)
    dialog._tif_list.addItem(str(a))
    dialog._tif_list.item(0).setSelected(True)

    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value=str(fake_fiji))
    start_detached = mocker.patch(
        "neuroseq.tif_import_dialog.QtCore.QProcess.startDetached", return_value=True
    )
    dialog._launch_fiji()
    start_detached.assert_called_once_with(str(fake_fiji), [str(a)])


def test_launch_fiji_no_op_when_path_not_resolved(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value="")
    start_detached = mocker.patch(
        "neuroseq.tif_import_dialog.QtCore.QProcess.startDetached"
    )
    dialog._launch_fiji()
    start_detached.assert_not_called()


def test_change_fiji_path_button_forces_prompt(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    resolve = mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value="")

    dialog._change_fiji_button.click()

    resolve.assert_called_once()
    assert resolve.call_args.kwargs.get("force_prompt") is True


def test_launch_cellpose_invokes_subprocess(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch(
        "neuroseq.tif_import_dialog.importlib.util.find_spec",
        return_value=object(),  # cellpose "found"
    )
    popen = mocker.patch("neuroseq.tif_import_dialog.subprocess.Popen")
    dialog._launch_cellpose()
    popen.assert_called_once()


def test_launch_cellpose_warns_when_missing(qtbot, mocker):
    from neuroseq.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch(
        "neuroseq.tif_import_dialog.importlib.util.find_spec", return_value=None
    )
    popen = mocker.patch("neuroseq.tif_import_dialog.subprocess.Popen")
    dialog._launch_cellpose()
    popen.assert_not_called()


def test_pipeline_failed_signal_re_enables_run_button(qtbot, mocker, tmp_path):
    from neuroseq.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((16, 16), dtype=np.int32) - 1
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))

    mocker.patch(
        "neuroseq.importer_tif.run_suite2p_pipeline",
        side_effect=RuntimeError("synthetic failure"),
    )
    dialog._start_pipeline()
    qtbot.waitUntil(lambda: dialog._run_button.isEnabled(), timeout=5000)
    assert dialog._pipeline_result is None
