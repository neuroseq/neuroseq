from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.importers.cellpose_import_dialog import CUSTOM_AOI_STRATEGY
from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _make_cellpose_dialog(tmp_path, qtbot: QtBot) -> CellposeImportDialog:
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    image = np.zeros((6, 6, 3), dtype=np.uint8)
    segmentation = np.zeros((6, 6), dtype=np.int32)
    segmentation[1:3, 1:3] = 1
    segmentation[3:5, 3:5] = 4

    image_path = tmp_path / "cellpose_image.tif"
    seg_path = tmp_path / "segmentation.npy"
    Image.fromarray(image).save(image_path)
    np.save(seg_path, segmentation)

    dialog._load_image_file(str(image_path))
    dialog._load_segmentation_file(str(seg_path))

    return dialog


def _select_mapping_strategy(dialog: CellposeImportDialog, strategy: str) -> None:
    for button in dialog.mapping_buttons.buttons():
        if button.text() == strategy:
            button.setChecked(True)
            return
    raise AssertionError(f"Mapping strategy not found: {strategy}")


def test_ctrl_z_and_ctrl_shift_z_shortcuts_drive_aoi_undo_redo(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    """Regression for issue #531: an accidental shape move/resize should be
    undoable with Ctrl+Z (Ctrl+Shift+Z to redo). The undo/redo logic itself is
    covered at the OverlayView level in test_spatial_footprint_widget.py; this
    just checks the dialog's shortcuts are wired to it."""
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    calls = {"undo": 0, "redo": 0}

    def _record_undo() -> bool:
        calls["undo"] += 1
        return True

    def _record_redo() -> bool:
        calls["redo"] += 1
        return True

    monkeypatch.setattr(
        dialog.overlay_widget.overlay_view, "undo_aoi_edit", _record_undo
    )
    monkeypatch.setattr(
        dialog.overlay_widget.overlay_view, "redo_aoi_edit", _record_redo
    )

    dialog._undo_aoi_shortcut.activated.emit()
    dialog._redo_aoi_shortcut.activated.emit()

    assert calls == {"undo": 1, "redo": 1}


def test_custom_areas_of_interest_is_default_mapping_strategy(qtbot: QtBot) -> None:
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    assert dialog._get_mapping_strategy() == CUSTOM_AOI_STRATEGY


def test_cellpose_dialog_opens_at_compact_file_selection_size(qtbot: QtBot):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    with qtbot.waitExposed(dialog, timeout=WAIT_TIMEOUT_MS):
        dialog.showMaximized()
    dialog._set_ui_loaded_state(False)
    qtbot.waitUntil(lambda: dialog.size() == dialog.sizeHint(), timeout=WAIT_TIMEOUT_MS)

    assert not dialog.isMaximized()
    assert dialog.main_content_container is not None
    assert not dialog.main_content_container.isVisible()
    assert dialog.size() == dialog.sizeHint()


def test_external_gui_buttons_are_grouped_at_dialog_bottom(qtbot: QtBot):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    assert dialog.external_gui_group.title() == "External GUIs"
    group_layout = dialog.external_gui_group.layout()
    assert group_layout is not None
    grouped_widgets = [
        group_layout.itemAt(index).widget()
        for index in range(group_layout.count())
        if group_layout.itemAt(index).widget() is not None
    ]
    assert grouped_widgets == [
        dialog.open_fiji_button,
        dialog.run_cellpose_button,
    ]
    assert (
        dialog.layout().indexOf(dialog.external_gui_group)
        == dialog.layout().count() - 1
    )


def test_cellpose_dialog_linear_offset_persisted(
    tmp_path, qtbot: QtBot, qmessagebox_calls: list
):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Linear")
    assert dialog.windowTitle() == "Cell segmentation"
    dialog.linear_offset_spin.setValue(3)

    assert dialog.linear_offset_spin.value() == 3
    assert dialog._get_linear_offset() == 3
    assert len(qmessagebox_calls) == 0


def test_cellpose_dialog_linear_offset_preview_mapping(tmp_path, qtbot: QtBot):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Linear")
    dialog.linear_offset_spin.setValue(3)

    mapping = dialog.overlay_widget.stimulus_to_cellpose_mapping or {}
    original_ids = dialog.importer._original_cellpose_ids
    assert original_ids is not None
    min_original_id = int(np.min(original_ids))
    first_mapping_key = min(mapping) if mapping else -1

    assert mapping != {}
    assert first_mapping_key == 0
    assert mapping[first_mapping_key] == min_original_id + 3
    assert mapping.get(1) == min_original_id


def test_cellpose_dialog_custom_mapping_restores_on_reopen(tmp_path, qtbot: QtBot):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    mapping = {1: 4, 2: 1}
    assert dialog.image_path is not None
    result = dialog.importer.create_result(
        dialog.image_path, mapping, mapping_strategy="Custom"
    )
    reopened = CellposeImportDialog(existing_data=result)
    qtbot.addWidget(reopened)
    assert reopened._get_mapping_strategy() == "Custom"
    assert reopened.custom_mapping_line.text().replace(" ", "") == "1:4,2:1"
    assert reopened.overlay_widget.stimulus_to_cellpose_mapping == mapping

    screen = reopened._get_screen_for_window()
    assert screen is not None
    available = screen.availableGeometry()
    assert not reopened.isMaximized()
    assert reopened.width() <= min(1100, int(available.width() * 0.75))
    assert reopened.height() <= min(760, int(available.height() * 0.75))


def _original_ids(dialog: CellposeImportDialog) -> list[int]:
    ids = dialog.importer._original_cellpose_ids
    assert ids is not None
    return sorted(int(cellpose_id) for cellpose_id in ids)


def test_dialog_cell_selection_is_not_restricted_to_a_single_button(
    tmp_path, qtbot: QtBot
) -> None:
    """Regression: the Spatial Footprints panel splits click/lasso cell
    selection across left (click) and right (lasso) buttons, but that must
    not affect this dialog's own click-to-select (manual mapping
    assignment) - it still responds to any button for either gesture."""
    dialog = _make_cellpose_dialog(tmp_path, qtbot)

    assert dialog.overlay_widget.overlay_view.cell_click_button == 0
    assert dialog.overlay_widget.overlay_view.cell_lasso_button == 0


def test_manual_click_checkbox_only_visible_for_custom_strategy(
    tmp_path, qtbot: QtBot
) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)

    _select_mapping_strategy(dialog, "Linear")
    assert dialog.manual_click_row_widget.isHidden()

    _select_mapping_strategy(dialog, "Custom")
    assert not dialog.manual_click_row_widget.isHidden()

    _select_mapping_strategy(dialog, CUSTOM_AOI_STRATEGY)
    assert dialog.manual_click_row_widget.isHidden()


def test_manual_click_assigns_by_order_and_fills_in_the_rest(
    tmp_path, qtbot: QtBot
) -> None:
    """Regression for issue #532: clicking cells assigns stimulus ids in
    click order, and every cell not clicked still ends up mapped, via the
    remaining ids in ascending cellpose-id order."""
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    original_ids = _original_ids(dialog)
    assert len(original_ids) >= 2

    dialog.manual_click_checkbox.setChecked(True)
    assert dialog.overlay_widget.overlay_view.enable_cell_selection
    assert dialog.custom_mapping_line.isReadOnly()

    # click the highest id first, so it lands on stim 0 - distinct from
    # where ascending-order auto-fill would have put it
    clicked_id = original_ids[-1]
    dialog.overlay_widget.neuron_clicked.emit(clicked_id)

    remaining = [
        cellpose_id for cellpose_id in original_ids if cellpose_id != clicked_id
    ]
    expected = {0: clicked_id}
    expected.update(
        {stim_id + 1: cellpose_id for stim_id, cellpose_id in enumerate(remaining)}
    )
    assert dialog.overlay_widget.stimulus_to_cellpose_mapping == expected


def test_manual_click_again_undoes_it(tmp_path, qtbot: QtBot) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    original_ids = _original_ids(dialog)
    dialog.manual_click_checkbox.setChecked(True)
    clicked_id = original_ids[-1]
    dialog.overlay_widget.neuron_clicked.emit(clicked_id)
    assert dialog._manual_click_order == [clicked_id]

    dialog.overlay_widget.neuron_clicked.emit(clicked_id)  # click again to undo

    assert dialog._manual_click_order == []
    identity_mapping = {
        stim_id: cellpose_id for stim_id, cellpose_id in enumerate(original_ids)
    }
    assert dialog.overlay_widget.stimulus_to_cellpose_mapping == identity_mapping


def test_manual_click_clear_button_resets_order(tmp_path, qtbot: QtBot) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    original_ids = _original_ids(dialog)
    dialog.manual_click_checkbox.setChecked(True)
    dialog.overlay_widget.neuron_clicked.emit(original_ids[-1])
    assert dialog._manual_click_order == [original_ids[-1]]

    dialog.manual_click_clear_button.click()

    assert dialog._manual_click_order == []
    identity_mapping = {
        stim_id: cellpose_id for stim_id, cellpose_id in enumerate(original_ids)
    }
    assert dialog.overlay_widget.stimulus_to_cellpose_mapping == identity_mapping


def test_manual_click_ignored_while_checkbox_unchecked(tmp_path, qtbot: QtBot) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    original_ids = _original_ids(dialog)
    assert not dialog.manual_click_checkbox.isChecked()

    dialog.overlay_widget.neuron_clicked.emit(original_ids[-1])

    assert dialog._manual_click_order == []


def test_manual_click_mode_turns_off_when_switching_strategy(
    tmp_path, qtbot: QtBot
) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    dialog.manual_click_checkbox.setChecked(True)
    assert dialog.overlay_widget.overlay_view.enable_cell_selection

    _select_mapping_strategy(dialog, "Linear")

    assert not dialog.manual_click_checkbox.isChecked()
    assert not dialog.overlay_widget.overlay_view.enable_cell_selection


def test_show_aoi_checkbox_draws_read_only_shapes_under_custom(
    tmp_path, qtbot: QtBot
) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    view = dialog.overlay_widget.overlay_view
    _select_mapping_strategy(dialog, "Custom")

    assert not dialog.show_aoi_in_custom_checkbox.isChecked()
    assert not view._aoi_visible

    dialog.show_aoi_in_custom_checkbox.setChecked(True)
    assert view._aoi_visible
    assert view._aoi_config is not None
    assert not view.aoi_editable

    dialog.show_aoi_in_custom_checkbox.setChecked(False)
    assert not view._aoi_visible


def test_show_aoi_checkbox_does_not_leak_into_other_strategies_or_result(
    tmp_path, qtbot: QtBot
) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    view = dialog.overlay_widget.overlay_view
    _select_mapping_strategy(dialog, "Custom")
    dialog.show_aoi_in_custom_checkbox.setChecked(True)

    # Linear draws its own adjustable shapes, controlled by Show shapes.
    _select_mapping_strategy(dialog, "Linear")
    dialog.show_shapes_checkbox.setChecked(False)
    assert not view._aoi_visible
    dialog.show_shapes_checkbox.setChecked(True)

    _select_mapping_strategy(dialog, CUSTOM_AOI_STRATEGY)
    assert view._aoi_visible
    assert view.aoi_editable


def test_annotate_mean_tiff_populates_file_selection(tmp_path, qtbot: QtBot, mocker):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    image_path = tmp_path / "mean_fluorescence.tif"
    segmentation_path = tmp_path / "mean_fluorescence_seg.npy"
    mask_path = tmp_path / "mean_fluorescence_cp_masks.png"
    Image.fromarray(np.zeros((6, 6), dtype=np.uint8)).save(image_path)
    segmentation = np.zeros((6, 6), dtype=np.int32)
    segmentation[1:3, 1:3] = 1
    np.save(segmentation_path, segmentation)

    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, mask_path),
    )

    dialog._annotate_mean_tiff()

    annotate.assert_called_once_with(
        str(image_path), model_name="cpsam", display_range=None
    )
    assert dialog.image_path == image_path
    assert dialog.segmentation_path == segmentation_path
    assert dialog.image_path_label.text() == image_path.name
    assert dialog.segmentation_path_label.text() == segmentation_path.name


def test_annotate_mean_tiff_uses_selected_cellpose_model(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    """Regression (#579): one-button segmentation always ran cpsam - the
    model picked in the pane's combo box should now be passed through."""
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    image_path = tmp_path / "mean_fluorescence.tif"
    segmentation_path = tmp_path / "mean_fluorescence_seg.npy"
    mask_path = tmp_path / "mean_fluorescence_cp_masks.png"
    Image.fromarray(np.zeros((6, 6), dtype=np.uint8)).save(image_path)
    segmentation = np.zeros((6, 6), dtype=np.int32)
    segmentation[1:3, 1:3] = 1
    np.save(segmentation_path, segmentation)

    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, mask_path),
    )
    dialog.cellpose_model_combo.setCurrentText("cpsam_v2")

    dialog._annotate_mean_tiff()

    annotate.assert_called_once_with(
        str(image_path), model_name="cpsam_v2", display_range=None
    )


def test_annotate_mean_tiff_accepts_a_png_snapshot(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    """Feature (#566): one-button segmentation used to only accept TIFFs -
    a PNG snapshot should go straight through without the multi-frame TIFF
    inspection/averaging path (which can't parse a PNG)."""
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    image_path = tmp_path / "snapshot.png"
    segmentation_path = tmp_path / "snapshot_seg.npy"
    mask_path = tmp_path / "snapshot_cp_masks.png"
    Image.fromarray(np.zeros((6, 6), dtype=np.uint8)).save(image_path)
    segmentation = np.zeros((6, 6), dtype=np.int32)
    segmentation[1:3, 1:3] = 1
    np.save(segmentation_path, segmentation)

    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, mask_path),
    )
    tiff_frame_count = mocker.patch(
        "neuroseq.cellpose_annotation.tiff_frame_count",
    )

    dialog._annotate_mean_tiff()

    tiff_frame_count.assert_not_called()
    annotate.assert_called_once_with(
        str(image_path), model_name="cpsam", display_range=None
    )
    assert dialog.image_path == image_path
    assert dialog.segmentation_path == segmentation_path


def test_annotate_multiframe_tiff_explains_averaging(
    tmp_path, qtbot: QtBot, mocker, qmessagebox_calls
) -> None:
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    image_path = tmp_path / "recording.tif"
    Image.fromarray(np.zeros((6, 6), dtype=np.uint8)).save(image_path)

    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    mocker.patch("neuroseq.cellpose_annotation.tiff_frame_count", return_value=600)
    mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        side_effect=RuntimeError("stop after notice"),
    )

    dialog._annotate_mean_tiff()

    messages = [
        call["text"] for call in qmessagebox_calls if call["name"] == "information"
    ]
    assert any("first 300" in message for message in messages)
    assert any("150 frames total" in message for message in messages)


def test_linear_mapping_has_adjustable_shapes_that_leave_it_unchanged(
    tmp_path, qtbot: QtBot, qmessagebox_calls
) -> None:
    """Issue #606: Linear shows the shapes and lets them be moved, while the
    mapping stays linear; the shapes are kept with the result."""
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    view = dialog.overlay_widget.overlay_view
    _select_mapping_strategy(dialog, "Linear")
    assert dialog.aoi_group.isVisibleTo(dialog)
    assert dialog.show_shapes_checkbox.isVisibleTo(dialog)
    assert not any(radio.isVisibleTo(dialog) for radio in dialog._aoi_role_radios)
    assert view._aoi_visible
    assert view.aoi_editable
    linear = dialog._generate_stimulus_mapping("Linear")

    moved = dialog._current_aoi_config()
    assert moved is not None
    moved.square.x, moved.square.y = 1.0, 2.0
    dialog._on_overlay_aoi_geometry_changed(moved)  # as after a drag
    current = dialog._current_aoi_config()
    assert current is not None
    assert (current.square.x, current.square.y) == pytest.approx((1.0, 2.0), abs=0.01)
    dialog.circle_diameter_spin.setValue(50)  # % of the image's shorter side
    assert view._aoi_config is not None
    assert view._aoi_config.circle.diameter == pytest.approx(3.0, abs=0.01)
    assert dialog._generate_stimulus_mapping("Linear") == linear

    dialog._import_data()
    imported = dialog.get_imported_data()
    assert imported is not None
    assert imported.mapping_strategy == "Linear"
    assert imported.stimulus_to_cellpose_mapping is None
    assert imported.aoi_config is not None
    assert imported.aoi_config.square.x == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("accept", [True, False])
def test_brightness_is_adjusted_before_segmentation(
    tmp_path, qtbot: QtBot, mocker, accept: bool
) -> None:
    """Issue #611: with "Adjust brightness first", the chosen window is
    passed on to the segmentation; cancelling segments nothing."""
    from neuroseq.visualization_support.brightness_contrast import DisplayRange
    from neuroseq.visualization_support.segmentation_brightness_dialog import (
        SegmentationBrightnessDialog,
    )

    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    image_path = tmp_path / "mean_fluorescence.png"
    Image.fromarray(np.arange(36, dtype=np.uint8).reshape(6, 6)).save(image_path)
    segmentation_path = tmp_path / "mean_fluorescence_cpsam_seg.npy"
    np.save(segmentation_path, np.zeros((6, 6), dtype=np.int32))
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, tmp_path / "masks.png"),
    )
    chosen = DisplayRange(40, 200)

    def _exec(brightness: SegmentationBrightnessDialog) -> int:
        assert brightness.controls.histogram_view is not None
        brightness.controls.set_display_range(chosen)
        return (
            QtWidgets.QDialog.DialogCode.Accepted
            if accept
            else QtWidgets.QDialog.DialogCode.Rejected
        )

    mocker.patch.object(SegmentationBrightnessDialog, "exec", _exec)
    dialog.adjust_brightness_checkbox.setChecked(True)
    dialog._annotate_mean_tiff()

    if accept:
        annotate.assert_called_once_with(
            str(image_path), model_name="cpsam", display_range=chosen
        )
    else:
        annotate.assert_not_called()


def test_brightness_dialog_previews_the_window(qtbot: QtBot) -> None:
    from neuroseq.visualization_support.brightness_contrast import DisplayRange
    from neuroseq.visualization_support.segmentation_brightness_dialog import (
        SegmentationBrightnessDialog,
    )

    image = np.linspace(0, 4000, 64, dtype=np.float32).reshape(8, 8)
    dialog = SegmentationBrightnessDialog(image)
    qtbot.addWidget(dialog)
    assert dialog.display_range.is_identity
    before = dialog.preview.pixmap().toImage()
    dialog.controls.set_display_range(DisplayRange(100, 150))
    assert dialog.display_range == DisplayRange(100, 150)
    assert dialog.preview.pixmap().toImage() != before


@pytest.mark.parametrize("also_adjust", [False, True])
def test_auto_brightness_is_applied_before_segmentation(
    tmp_path, qtbot: QtBot, mocker, also_adjust: bool
) -> None:
    """Grab FOV's "Auto brightness": Fiji's Auto window is passed on to the
    segmentation, and "Adjust brightness first" starts from it."""
    from neuroseq.visualization_support.brightness_contrast import (
        enhanced_display_range,
    )
    from neuroseq.visualization_support.brightness_contrast import luminance_histogram
    from neuroseq.visualization_support.segmentation_brightness_dialog import (
        SegmentationBrightnessDialog,
    )

    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    rng = np.random.default_rng(0)
    pixels = np.clip(rng.normal(80, 20, (200, 200)), 0, 255).astype(np.uint8)
    pixels[0, :4] = (0, 0, 255, 255)  # full 0-255 span, so to_8bit is identity
    image_path = tmp_path / "fov_mean.png"
    Image.fromarray(pixels).save(image_path)
    segmentation_path = tmp_path / "fov_mean_cpsam_seg.npy"
    np.save(segmentation_path, np.zeros((200, 200), dtype=np.int32))
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, tmp_path / "masks.png"),
    )
    expected = enhanced_display_range(luminance_histogram(pixels))
    assert not expected.is_identity
    shown = []

    def _exec(brightness: SegmentationBrightnessDialog) -> int:
        shown.append(brightness.display_range)
        return QtWidgets.QDialog.DialogCode.Accepted

    mocker.patch.object(SegmentationBrightnessDialog, "exec", _exec)
    dialog.adjust_brightness_checkbox.setChecked(also_adjust)
    dialog.segment_file(str(image_path), auto_brightness=True)

    annotate.assert_called_once_with(
        str(image_path), model_name="cpsam", display_range=expected
    )
    assert shown == ([expected] if also_adjust else [])


def test_segmenting_a_new_fov_replaces_an_existing_segmentation(
    tmp_path, qtbot: QtBot, mocker, qmessagebox_calls: list
) -> None:
    """Grab FOV in a project that already has a segmentation of another size
    (e.g. an example project's): the new image and its segmentation are loaded
    together, not checked against the old masks or mapping."""
    old = _make_cellpose_dialog(tmp_path, qtbot)
    assert old.image_path is not None
    existing = old.importer.create_result(
        old.image_path, {0: 4, 1: 1}, mapping_strategy="Custom"
    )
    dialog = CellposeImportDialog(existing_data=existing)
    qtbot.addWidget(dialog)
    assert dialog._get_mapping_strategy() == "Custom"

    fov = tmp_path / "2026-11-06_00001_mean.png"
    Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12)).save(fov)
    segmentation = np.zeros((10, 12), dtype=np.int32)
    segmentation[2:5, 2:5] = 1
    segmentation[6:9, 7:10] = 2
    seg_path = tmp_path / "2026-11-06_00001_mean_cpsam_seg.npy"
    np.save(seg_path, segmentation)
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(seg_path, tmp_path / "masks.png"),
    )

    dialog.segment_file(str(fov))

    errors = [call for call in qmessagebox_calls if call["name"] == "critical"]
    assert errors == []
    assert dialog.image_path == fov
    assert dialog.importer.image_data is not None
    assert dialog.importer.image_data.shape[:2] == (10, 12)
    assert dialog.importer.segmentation_data is not None
    assert dialog.importer.segmentation_data.shape == (10, 12)
    assert dialog.import_button.isEnabled()
    # The old custom mapping does not carry over to the new cells.
    assert dialog._get_mapping_strategy() != "Custom"


def _fov_with_segmentation(tmp_path, mocker):
    """A dim 200x200 FOV, its two-cell segmentation, and Cellpose mocked to return it."""
    rng = np.random.default_rng(1)
    pixels = np.clip(rng.normal(60, 6, (200, 200)), 0, 255).astype(np.uint8)
    pixels[0, :2] = (0, 255)  # full range: the dialog shows it as stored
    fov = tmp_path / "fov_mean.png"
    Image.fromarray(pixels).save(fov)
    segmentation = np.zeros((200, 200), dtype=np.int32)
    segmentation[5:10, 5:10] = 1
    segmentation[100:130, 100:130] = 2
    seg_path = tmp_path / "fov_mean_cpsam_seg.npy"
    np.save(seg_path, segmentation)
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(seg_path, tmp_path / "masks.png"),
    )
    return fov, pixels


def test_auto_brightness_is_shown_and_kept_with_the_import(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    """The window Cellpose segmented with is what the preview shows, and it is
    stored with the import for the Spatial Footprints view."""
    from neuroseq.visualization_support.brightness_contrast import DisplayRange
    from neuroseq.visualization_support.brightness_contrast import (
        enhanced_display_range,
    )
    from neuroseq.visualization_support.brightness_contrast import luminance_histogram

    fov, pixels = _fov_with_segmentation(tmp_path, mocker)
    expected = enhanced_display_range(luminance_histogram(pixels))
    assert not expected.is_identity
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    shown = mocker.spy(dialog.overlay_widget, "set_overlay")

    dialog.segment_file(str(fov), auto_brightness=True)

    base = shown.call_args.kwargs["base_image"]
    assert base is not None
    stored = dialog.importer.image_data
    assert stored is not None
    # More contrasted than the stored FOV.
    assert base.std() > stored.std() * 2
    dialog._import_data()
    imported = dialog.get_imported_data()
    assert imported is not None
    assert imported.display_range == (expected.minimum, expected.maximum)

    # Reopened, the window is shown and kept on a re-import.
    reopened = CellposeImportDialog(existing_data=imported)
    qtbot.addWidget(reopened)
    assert reopened._display_range == DisplayRange(expected.minimum, expected.maximum)
    reopened._import_data()
    again = reopened.get_imported_data()
    assert again is not None and again.display_range == imported.display_range


def test_without_auto_brightness_the_fov_is_shown_as_stored(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    fov, _pixels = _fov_with_segmentation(tmp_path, mocker)
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    shown = mocker.spy(dialog.overlay_widget, "set_overlay")
    dialog.segment_file(str(fov))
    np.testing.assert_array_equal(
        shown.call_args.kwargs["base_image"], dialog.importer.image_data
    )
    dialog._import_data()
    imported = dialog.get_imported_data()
    assert imported is not None and imported.display_range is None


def test_segmenting_a_new_fov_keeps_a_linear_mapping_and_its_offset(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    """A Linear mapping names no cells, so the user's choice survives Grab FOV."""
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    dialog._apply_existing_mapping({}, "Linear", 5)
    fov = tmp_path / "2026-11-06_00002_mean.png"
    Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12)).save(fov)
    segmentation = np.zeros((10, 12), dtype=np.int32)
    segmentation[2:5, 2:5] = 1
    seg_path = tmp_path / "2026-11-06_00002_mean_cpsam_seg.npy"
    np.save(seg_path, segmentation)
    mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(seg_path, tmp_path / "masks.png"),
    )

    dialog.segment_file(str(fov))

    assert dialog.image_path == fov
    assert dialog._get_mapping_strategy() == "Linear"
    assert dialog._get_linear_offset() == 5


def test_prefilled_file_is_segmented_with_the_chosen_options(
    tmp_path, qtbot: QtBot, mocker
) -> None:
    """A dropped file waits until "Segment" and then uses the model and
    "Adjust brightness first" chosen in the dialog (issue #658)."""
    from neuroseq.visualization_support.segmentation_brightness_dialog import (
        SegmentationBrightnessDialog,
    )

    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)
    fov, _pixels = _fov_with_segmentation(tmp_path, mocker)
    from neuroseq import cellpose_annotation

    annotate = mocker.spy(cellpose_annotation, "annotate_mean_fluorescence_tiff")
    shown: list[SegmentationBrightnessDialog] = []

    def _exec(brightness: SegmentationBrightnessDialog) -> int:
        shown.append(brightness)
        return QtWidgets.QDialog.DialogCode.Accepted

    mocker.patch.object(SegmentationBrightnessDialog, "exec", _exec)

    assert dialog.segment_prefilled_button.isHidden()
    dialog.prefill_segmentation_file(str(fov))
    assert not dialog.segment_prefilled_button.isHidden()
    assert fov.name in dialog.prefilled_file_label.text()
    annotate.assert_not_called()

    dialog.cellpose_model_combo.setCurrentText("cpsam_v2")
    dialog.adjust_brightness_checkbox.setChecked(True)
    dialog.segment_prefilled_button.click()

    assert len(shown) == 1
    annotate.assert_called_once()
    assert annotate.call_args.kwargs["model_name"] == "cpsam_v2"
    assert dialog.image_path == fov


def test_prefilling_over_existing_data_returns_to_file_selection(
    tmp_path, qtbot: QtBot
) -> None:
    old = _make_cellpose_dialog(tmp_path, qtbot)
    assert old.image_path is not None
    existing = old.importer.create_result(old.image_path, {0: 1})
    dialog = CellposeImportDialog(existing_data=existing)
    qtbot.addWidget(dialog)
    assert dialog.file_selection_container is not None
    assert dialog.file_selection_container.isHidden()

    dialog.prefill_segmentation_file(str(tmp_path / "new_fov.png"))

    assert not dialog.file_selection_container.isHidden()
    assert dialog.image_path is None
    assert dialog.existing_data is not None  # stored data kept until an import
