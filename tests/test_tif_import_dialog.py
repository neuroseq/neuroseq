"""Qt smoke tests for the TIFF import dialog with a mocked Suite2p pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


def _make_pipeline_result():
    from neuroseq.importers.importer_tif import TifPipelineResult

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


def _make_pipeline_result_with_masks():
    result = _make_pipeline_result()
    result.mean_image = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    result.masks = np.array(
        [
            [0, 1, 0],
            [2, 2, 0],
        ],
        dtype=np.int32,
    )
    return result


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
    from neuroseq.importers.importer_tif import TifPipelineParams
    from neuroseq.importers.tif_import_dialog import _PipelineWorker

    result = _make_pipeline_result()

    def fake_pipeline(*_args, progress, **_kwargs):
        progress("working", 0.5)
        return result

    run = mocker.patch(
        f"neuroseq.importers.importer_tif.{function_name}", side_effect=fake_pipeline
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


def test_pipeline_worker_forwards_log_signal(qtbot, mocker):
    from neuroseq.importers.importer_tif import TifPipelineParams
    from neuroseq.importers.tif_import_dialog import _PipelineWorker

    result = _make_pipeline_result()

    def fake_pipeline(*_args, progress, **_kwargs):
        logging.getLogger("suite2p").info("hello")
        progress("working", 0.5)
        return result

    mocker.patch(
        "neuroseq.importers.importer_tif.run_suite2p_pipeline",
        side_effect=fake_pipeline,
    )
    worker = _PipelineWorker(
        [], np.ones((2, 2), dtype=np.int32), TifPipelineParams(), "suite2p"
    )
    log_events = []
    worker.log.connect(log_events.append)

    worker.run()

    assert "INFO: hello" in log_events


def test_pipeline_worker_emits_failure(mocker):
    from neuroseq.importers.importer_tif import TifPipelineParams
    from neuroseq.importers.tif_import_dialog import _PipelineWorker

    mocker.patch(
        "neuroseq.importers.importer_tif.run_roi_average_pipeline",
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
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    assert not dialog._use_loaded_radio.isEnabled()
    assert dialog._use_seg_radio.isChecked()


def test_current_experiment_masks_with_mapping_are_default(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.stimulus_to_cellpose_mapping = {4: 0, 7: 1}
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)

    assert dialog._use_loaded_radio.isChecked()
    assert "with mapping applied" in dialog._use_loaded_radio.text()
    assert dialog._selected_mapping() == {4: 0, 7: 1}


def test_selected_mapping_reconstructs_linear_offset(qtbot):
    """Regression test for #554's Linear-strategy blind spot: a Linear
    import stores only linear_offset, not a materialized
    stimulus_to_cellpose_mapping, so _selected_mapping() used to read a bare
    None there and import under an unintended identity mapping instead of
    the offset the user actually selected."""
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.stimulus_to_cellpose_mapping = None
    cellpose_import.mapping_strategy = "Linear"
    cellpose_import.linear_offset = 2
    cellpose_import.original_cellpose_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)

    assert dialog._use_loaded_radio.isChecked()
    assert dialog._selected_mapping() == {0: 2, 1: 3, 2: 0, 3: 1}


def test_third_radio_selected_resolves_to_none_masks(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    dialog._auto_detect_radio.setChecked(True)

    assert dialog._resolve_masks() is None


def test_auto_detect_radio_forces_and_disables_pipeline_combo(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    dialog._pipeline_combo.setCurrentIndex(1)

    dialog._auto_detect_radio.setChecked(True)

    assert dialog._pipeline_combo.currentData() == "suite2p"
    assert not dialog._pipeline_combo.isEnabled()

    dialog._use_seg_radio.setChecked(True)

    assert dialog._pipeline_combo.isEnabled()


def test_auto_detect_radio_shows_diameter_spinbox_only_in_auto_detect_mode(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog._diameter_spin.isVisibleTo(dialog)

    dialog._auto_detect_radio.setChecked(True)
    assert dialog._diameter_spin.isVisibleTo(dialog)

    dialog._use_seg_radio.setChecked(True)
    assert not dialog._diameter_spin.isVisibleTo(dialog)


def test_dialog_body_is_scrollable_but_run_buttons_stay_fixed(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    assert dialog._scroll_area.widgetResizable()
    assert dialog._scroll_area.widget() is dialog._scroll_content
    assert dialog._scroll_content.isAncestorOf(dialog._pipeline_combo)
    assert not dialog._scroll_content.isAncestorOf(dialog._run_button)


def test_wheel_over_parameter_field_scrolls_without_changing_value(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    dialog.resize(360, 220)
    dialog.show()
    scroll_bar = dialog._scroll_area.verticalScrollBar()
    qtbot.waitUntil(lambda: scroll_bar.maximum() > 0)
    scroll_bar.setValue(0)
    old_value = dialog._fs_spin.value()
    wheel_event = QtGui.QWheelEvent(
        QtCore.QPointF(10.0, 10.0),
        QtCore.QPointF(10.0, 10.0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    handled = dialog.eventFilter(dialog._fs_spin, wheel_event)

    assert handled
    assert dialog._fs_spin.value() == old_value
    assert scroll_bar.value() > 0


def test_dialog_requires_tifs(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    # No TIFFs added — pressing Run should pop a warning and not start a thread.
    dialog._start_pipeline()
    assert dialog._thread is None


def test_pipeline_selector_hides_suite2p_parameters_for_roi_average(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._pipeline_combo.currentText() == "Simply average from ROI"
    assert not dialog._tau_spin.isVisibleTo(dialog)
    assert dialog._fs_spin.isVisibleTo(dialog)
    assert dialog._nchannels_spin.isVisibleTo(dialog)
    assert dialog._functional_chan_spin.isVisibleTo(dialog)
    assert dialog._nchannels_spin.value() == 1
    assert dialog._suppress_stimulation_artefact_checkbox.isVisibleTo(dialog)


def test_stimulation_suppression_selector_uses_project_stim_arrays(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog
    from neuroseq.stim import StimArray
    from neuroseq.stim import StimType

    stim = StimArray("laser", StimType.Excitation, np.ones((2, 1)), "mW", 0.1)
    dialog = TifImportDialog(stim_arrays={"laser": stim})
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._pipeline_combo.setCurrentIndex(1)

    assert dialog._suppress_stimulation_artefact_checkbox.isChecked()
    assert dialog._stim_array_combo.isVisibleTo(dialog)
    assert dialog._stim_array_combo.currentText() == "laser"
    assert dialog._stim_array_combo.currentData() is stim


def test_stimulation_suppression_checkbox_unchecked_with_no_stim_arrays(qtbot):
    """Regression (#580): with no input arrays in the project, the checkbox
    used to stay disabled *and* checked, so the user couldn't untick it even
    though it reads as "suppression is on"."""
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None, stim_arrays={})
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog._suppress_stimulation_artefact_checkbox.isEnabled()
    assert not dialog._suppress_stimulation_artefact_checkbox.isChecked()


def test_start_pipeline_passes_diameter_param(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))
    dialog._auto_detect_radio.setChecked(True)
    dialog._diameter_spin.setValue(24.5)

    result = _make_pipeline_result()
    captured = {}

    def fake_pipeline(tif_paths, masks, params, *, progress):
        captured["tif_paths"] = tif_paths
        captured["masks"] = masks
        captured["params"] = params
        progress("working", 0.5)
        return result

    mocker.patch(
        "neuroseq.importers.importer_tif.run_suite2p_pipeline",
        side_effect=fake_pipeline,
    )
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    accept = mocker.patch.object(dialog, "accept")

    dialog._start_pipeline()
    qtbot.waitUntil(lambda: dialog._pipeline_result is not None, timeout=5000)

    assert captured["tif_paths"] == [tif]
    assert captured["masks"] is None
    assert captured["params"].diameter == pytest.approx(24.5)
    accept.assert_called_once()


def test_auto_detect_pipeline_run_prompts_before_proceeding(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))
    dialog._auto_detect_radio.setChecked(True)
    question = mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    )
    run = mocker.patch("neuroseq.importers.importer_tif.run_suite2p_pipeline")

    dialog._start_pipeline()

    question.assert_called_once()
    run.assert_not_called()
    assert dialog._thread is None

    result = _make_pipeline_result()
    run.side_effect = lambda *_args, progress, **_kwargs: result
    question.reset_mock()
    question.return_value = QtWidgets.QMessageBox.StandardButton.Yes
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    accept = mocker.patch.object(dialog, "accept")

    dialog._start_pipeline()
    qtbot.waitUntil(lambda: dialog._pipeline_result is not None, timeout=5000)

    question.assert_called_once()
    run.assert_called_once()
    accept.assert_called_once()


def test_dialog_runs_mocked_pipeline_and_produces_imported_data(
    qtbot, mocker, tmp_path
):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    # Build a fake cellpose result so the "use loaded" path works.
    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((16, 16), dtype=np.int32) - 1
    cellpose_import.stimulus_to_cellpose_mapping = {4: 0, 7: 1}

    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog._pipeline_combo.setCurrentIndex(0)

    # Stage one TIFF (file does not need to actually exist — pipeline is mocked).
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))

    result = _make_pipeline_result()
    mock_run = mocker.patch(
        "neuroseq.importers.importer_tif.run_suite2p_pipeline",
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
    np.testing.assert_array_equal(
        traces.indices, np.array([4, 7, 2, 3], dtype=np.uint32)
    )
    assert {4, 7}.issubset(set(spikes.spike_indices.tolist()))


def test_finalize_roi_average_imports_raw_trace_only(qtbot, mocker):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    result = _make_pipeline_result()
    result.extra["pipeline"] = "roi_average"
    information = mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    accept = mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)

    assert len(dialog.imported_data) == 1
    traces = dialog.imported_data[0]
    assert traces.name == "Simply averaged ROI traces"
    np.testing.assert_array_equal(traces.traces, result.F)
    # raw_cellpose_ids must be populated (issue #554) - without it,
    # Project.remap_cellpose_indices() has nothing to re-resolve indices from
    # and silently skips this trace forever once the Cellpose mapping changes.
    n_neurons = result.F.shape[0]
    np.testing.assert_array_equal(
        traces.raw_cellpose_ids, np.arange(n_neurons, dtype=np.uint32)
    )
    np.testing.assert_array_equal(traces.indices, np.arange(n_neurons, dtype=np.uint32))
    assert dialog.experiment_duration == pytest.approx(
        result.F.shape[1] / dialog._fs_spin.value()
    )
    information.assert_called_once()
    accept.assert_called_once()


def test_finalize_roi_average_trace_indices_follow_mapping_after_remap(qtbot, mocker):
    """Regression test for #554: changing the Cellpose mapping strategy after
    importing via the (default) "Simply average from ROI" pipeline left the
    already-imported trace's displayed indices stuck on the old mapping,
    because it never got a raw_cellpose_ids array for remap_cellpose_indices()
    to work from."""
    from neuroseq.data import Experiment
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.importers.tif_import_dialog import TifImportDialog
    from neuroseq.project import Project

    cellpose_import = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        stimulus_to_cellpose_mapping={0: 0, 1: 1, 2: 2, 3: 3},
    )
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    result = _make_pipeline_result()
    result.extra["pipeline"] = "roi_average"
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)
    traces = dialog.imported_data[0]
    np.testing.assert_array_equal(
        traces.indices, np.array([0, 1, 2, 3], dtype=np.uint32)
    )

    project = Project()
    project.cellpose_data = cellpose_import
    project.add_experiment(
        Experiment("exp", dialog.experiment_duration, [0.0], [traces])
    )

    # Mapping strategy changes - Cellpose id 0 is now stim 3, id 3 is now stim 0.
    project.cellpose_data = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        stimulus_to_cellpose_mapping={3: 0, 1: 1, 2: 2, 0: 3},
    )
    failed = project.remap_cellpose_indices()

    assert failed == []
    np.testing.assert_array_equal(
        traces.indices, np.array([3, 1, 2, 0], dtype=np.uint32)
    )


def test_seg_file_source_sets_new_cellpose_result_after_finalize(
    qtbot,
    mocker,
    tmp_path,
):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    seg = tmp_path / "fake_seg.npy"
    tif.write_bytes(b"")
    np.save(seg, np.zeros((2, 3), dtype=np.int32))
    dialog._tif_paths.append(tif)
    dialog._seg_file_path = seg
    dialog._new_segmentation_path = seg
    dialog._use_seg_radio.setChecked(True)
    dialog._propagate_new_segmentation = True
    result = _make_pipeline_result_with_masks()
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)

    assert dialog.new_cellpose_result is not None
    assert dialog.new_cellpose_result.segmentation_array.shape == result.masks.shape
    assert dialog.new_cellpose_result.segmentation_path == seg


def test_auto_detect_source_sets_new_cellpose_result_with_no_segmentation_path(
    qtbot,
    mocker,
    tmp_path,
):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._auto_detect_radio.setChecked(True)
    dialog._propagate_new_segmentation = True
    result = _make_pipeline_result_with_masks()
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)

    assert dialog.new_cellpose_result is not None
    assert dialog.new_cellpose_result.segmentation_array.shape == result.masks.shape
    assert dialog.new_cellpose_result.segmentation_path is None


def test_use_loaded_radio_does_not_set_new_cellpose_result(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((2, 3), dtype=np.int32) - 1
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._use_loaded_radio.setChecked(True)
    result = _make_pipeline_result_with_masks()
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QMessageBox.information"
    )
    mocker.patch.object(dialog, "accept")

    dialog._finalize_import(result)

    assert dialog.new_cellpose_result is None


def test_build_cellpose_result_from_pipeline_matches_masks_and_mean_image():
    from neuroseq.importers.tif_import_dialog import (
        _build_cellpose_result_from_pipeline,
    )

    result = _make_pipeline_result_with_masks()

    imported = _build_cellpose_result_from_pipeline(result, Path("stack.tif"))

    assert imported.image_path == Path("stack.tif")
    assert imported.segmentation_path is None
    assert imported.segmentation_array.shape == result.masks.shape
    np.testing.assert_array_equal(
        imported.segmentation_array,
        result.masks.astype(np.int32) - 1,
    )
    assert -1 in np.unique(imported.segmentation_array)
    assert imported.overlay_image.shape == imported.segmentation_array.shape + (3,)


def test_build_cellpose_result_from_pipeline_prefers_matching_background_image():
    from neuroseq.importers.tif_import_dialog import (
        _build_cellpose_result_from_pipeline,
    )

    result = _make_pipeline_result_with_masks()
    # A non-monotonic pattern (unlike mean_image's plain ramp) so a min-max
    # normalized copy can't coincidentally match mean_image's normalized output.
    background = np.array([[60.0, 10.0, 40.0], [20.0, 50.0, 30.0]], dtype=np.float32)

    imported = _build_cellpose_result_from_pipeline(
        result, Path("stack.tif"), background_image=background
    )

    without_override = _build_cellpose_result_from_pipeline(result, Path("stack.tif"))
    assert not np.array_equal(imported.overlay_image, without_override.overlay_image)


def test_build_cellpose_result_from_pipeline_ignores_mismatched_background_image():
    from neuroseq.importers.tif_import_dialog import (
        _build_cellpose_result_from_pipeline,
    )

    result = _make_pipeline_result_with_masks()
    wrong_shape_background = np.zeros((9, 9), dtype=np.float32)

    imported = _build_cellpose_result_from_pipeline(
        result, Path("stack.tif"), background_image=wrong_shape_background
    )

    without_override = _build_cellpose_result_from_pipeline(result, Path("stack.tif"))
    np.testing.assert_array_equal(
        imported.overlay_image, without_override.overlay_image
    )


def test_load_seg_file_background_image_prefers_embedded_img_key(tmp_path: Path):
    from neuroseq.importers.tif_import_dialog import _load_seg_file_background_image

    seg_path = tmp_path / "fake_seg.npy"
    embedded_img = np.arange(6, dtype=np.uint16).reshape(2, 3)
    np.save(
        seg_path,
        np.array(
            {"masks": np.zeros((2, 3), dtype=np.int32), "img": embedded_img},
            dtype=object,
        ),
    )

    loaded = _load_seg_file_background_image(seg_path)

    np.testing.assert_array_equal(loaded, embedded_img)


def test_load_seg_file_background_image_falls_back_to_filename(tmp_path: Path):
    import tifffile

    from neuroseq.importers.tif_import_dialog import _load_seg_file_background_image

    image_path = tmp_path / "AVG_projection.tif"
    projection = np.arange(6, dtype=np.uint16).reshape(2, 3)
    tifffile.imwrite(image_path, projection, photometric="minisblack")

    seg_path = tmp_path / "fake_seg.npy"
    np.save(
        seg_path,
        np.array(
            {"masks": np.zeros((2, 3), dtype=np.int32), "filename": str(image_path)},
            dtype=object,
        ),
    )

    loaded = _load_seg_file_background_image(seg_path)

    np.testing.assert_array_equal(loaded, projection)


def test_load_seg_file_background_image_returns_none_when_unavailable(
    tmp_path: Path,
):
    from neuroseq.importers.tif_import_dialog import _load_seg_file_background_image

    seg_path = tmp_path / "fake_seg.npy"
    np.save(
        seg_path,
        np.array(
            {
                "masks": np.zeros((2, 3), dtype=np.int32),
                "filename": "/does/not/exist.tif",
            },
            dtype=object,
        ),
    )

    assert _load_seg_file_background_image(seg_path) is None


def test_pipeline_progress_dialog_details_toggle(qtbot):
    from neuroseq.importers.tif_import_dialog import _PipelineProgressDialog

    dialog = _PipelineProgressDialog("Starting", "Cancel", 0, 100)
    qtbot.addWidget(dialog)

    assert dialog._log_view.isHidden()

    dialog._details_button.setChecked(True)
    assert not dialog._log_view.isHidden()

    dialog.appendLog("x")
    assert "x" in dialog._log_view.toPlainText()

    dialog._details_button.setChecked(False)
    assert dialog._log_view.isHidden()


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
    from neuroseq.importers.tif_import_dialog import _cellpose_result_to_masks

    fake_result = MagicMock()
    fake_result.segmentation_array = seg
    np.testing.assert_array_equal(_cellpose_result_to_masks(fake_result), expected)


def test_browse_tifs_dedupes_added_paths(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)

    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    a.write_bytes(b"")
    b.write_bytes(b"")

    mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([str(a), str(b), str(a)], ""),
    )
    dialog._browse_tifs()

    assert len(dialog._tif_paths) == 2
    assert dialog._tif_list.count() == 2


def test_remove_selected_tifs_clears_list_and_paths(qtbot, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

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
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=MagicMock())
    qtbot.addWidget(dialog)

    seg = tmp_path / "image_seg.npy"
    seg.write_bytes(b"")
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.getopenfilename",
        return_value=(str(seg), ""),
    )
    dialog._browse_seg_file()
    assert dialog._seg_file_path == seg
    assert dialog._use_seg_radio.isChecked()


def test_resolve_masks_requires_seg_when_seg_radio_selected(qtbot):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    # _use_seg_radio is checked because cellpose_import is None, but no file picked.
    with pytest.raises(ValueError, match="No _seg.npy"):
        dialog._resolve_masks()


def test_resolve_masks_requires_cellpose_when_loaded_radio_selected(qtbot, mocker):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=MagicMock())
    qtbot.addWidget(dialog)
    # Force "use loaded" radio but clear the cellpose import.
    dialog._cellpose_import = None
    dialog._use_loaded_radio.setEnabled(True)
    dialog._use_loaded_radio.setChecked(True)
    with pytest.raises(ValueError, match="No Cellpose data"):
        dialog._resolve_masks()


def test_launch_fiji_runs_qprocess_with_selected_tif(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

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
        "neuroseq.importers.tif_import_dialog.QtCore.QProcess.startDetached",
        return_value=True,
    )
    dialog._launch_fiji()
    start_detached.assert_called_once_with(str(fake_fiji), [str(a)])


def test_launch_fiji_no_op_when_path_not_resolved(qtbot, mocker):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value="")
    start_detached = mocker.patch(
        "neuroseq.importers.tif_import_dialog.QtCore.QProcess.startDetached"
    )
    dialog._launch_fiji()
    start_detached.assert_not_called()


def test_launch_cellpose_invokes_subprocess(qtbot, mocker):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.importlib.util.find_spec",
        return_value=object(),  # cellpose "found"
    )
    # Stub the post-launch GPU warning: it imports torch, whose first-time CUDA
    # init shells out via subprocess.Popen (e.g. `ldconfig -p`). Because
    # `neuroseq.importers.tif_import_dialog.subprocess` is the real subprocess module,
    # the patch below is global and would otherwise count that call too, making
    # this assertion order-dependent on whether torch was already imported.
    mocker.patch("neuroseq.gpu.warn_if_gpu_unavailable")
    popen = mocker.patch("neuroseq.importers.tif_import_dialog.subprocess.Popen")
    dialog._launch_cellpose()
    popen.assert_called_once()


def test_launch_cellpose_warns_when_missing(qtbot, mocker):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    dialog = TifImportDialog(parent=None, cellpose_import=None)
    qtbot.addWidget(dialog)
    mocker.patch(
        "neuroseq.importers.tif_import_dialog.importlib.util.find_spec",
        return_value=None,
    )
    popen = mocker.patch("neuroseq.importers.tif_import_dialog.subprocess.Popen")
    dialog._launch_cellpose()
    popen.assert_not_called()


def test_pipeline_failed_signal_re_enables_run_button(qtbot, mocker, tmp_path):
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((16, 16), dtype=np.int32) - 1
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog._pipeline_combo.setCurrentIndex(0)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))

    mocker.patch(
        "neuroseq.importers.importer_tif.run_suite2p_pipeline",
        side_effect=RuntimeError("synthetic failure"),
    )
    dialog._start_pipeline()
    qtbot.waitUntil(lambda: dialog._run_button.isEnabled(), timeout=5000)
    assert dialog._pipeline_result is None


def test_pipeline_worker_cancel_emits_cancelled_not_failed(mocker):
    from neuroseq.importers.importer_tif import TifPipelineParams
    from neuroseq.importers.tif_import_dialog import _PipelineWorker

    worker = _PipelineWorker(
        [], np.ones((2, 2), dtype=np.int32), TifPipelineParams(), "roi_average"
    )

    def fake_pipeline(*_args, progress, **_kwargs):
        progress("step 1", 0.1)
        worker.cancel()
        progress("step 2", 0.2)
        raise AssertionError("pipeline kept running after cancel")

    mocker.patch(
        "neuroseq.importers.importer_tif.run_roi_average_pipeline",
        side_effect=fake_pipeline,
    )
    events = {"cancelled": 0, "failed": [], "finished": []}
    worker.cancelled.connect(lambda: events.__setitem__("cancelled", 1))
    worker.failed.connect(events["failed"].append)
    worker.finished.connect(events["finished"].append)

    worker.run()

    assert events == {"cancelled": 1, "failed": [], "finished": []}


def test_cancelling_running_pipeline_keeps_dialog_alive_until_thread_exits(
    qtbot, mocker, tmp_path
):
    """Regression test for #592: cancelling must not destroy a running QThread."""
    import threading

    from neuroseq.importers.tif_import_dialog import TifImportDialog

    cellpose_import = MagicMock()
    cellpose_import.segmentation_array = np.zeros((16, 16), dtype=np.int32) - 1
    dialog = TifImportDialog(parent=None, cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog._pipeline_combo.setCurrentIndex(0)
    tif = tmp_path / "fake.tif"
    tif.write_bytes(b"")
    dialog._tif_paths.append(tif)
    dialog._tif_list.addItem(str(tif))

    started = threading.Event()
    release = threading.Event()

    def slow_pipeline(*_args, progress, **_kwargs):
        progress("start", 0.0)
        started.set()
        release.wait(timeout=5)
        progress("next", 0.5)  # raises PipelineCancelled once cancelled
        raise AssertionError("pipeline kept running after cancel")

    mocker.patch(
        "neuroseq.importers.importer_tif.run_suite2p_pipeline",
        side_effect=slow_pipeline,
    )
    rejected = MagicMock()
    mocker.patch.object(
        QtWidgets.QDialog,
        "reject",
        lambda self_: rejected(self_) if self_ is dialog else None,
    )

    dialog._start_pipeline()
    qtbot.waitUntil(started.is_set, timeout=5000)
    progress_dialog = dialog._progress_dialog

    dialog._cancel_pipeline()

    # Still running: dialog must not be rejected and the thread must survive.
    assert dialog._thread is not None and dialog._thread.isRunning()
    rejected.assert_not_called()
    assert not progress_dialog._cancel_button.isEnabled()

    # Closing via Esc/X while the worker runs is also deferred.
    dialog.reject()
    rejected.assert_not_called()

    release.set()
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=5000)
    rejected.assert_called_once_with(dialog)
    assert dialog._run_button.isEnabled()
    assert dialog._pipeline_result is None
