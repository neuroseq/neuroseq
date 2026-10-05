from __future__ import annotations

import numpy as np
from PIL import Image
from pytestqt.qtbot import QtBot

from neuroseq.cellpose_import_dialog import CUSTOM_AOI_STRATEGY
from neuroseq.cellpose_import_dialog import CellposeImportDialog


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
    dialog.close()


def test_custom_areas_of_interest_is_default_mapping_strategy(qtbot: QtBot) -> None:
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    assert dialog._get_mapping_strategy() == CUSTOM_AOI_STRATEGY


def test_cellpose_dialog_opens_at_compact_file_selection_size(qtbot: QtBot):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    dialog.showMaximized()
    dialog._set_ui_loaded_state(False)
    dialog.show()
    qtbot.waitExposed(dialog)

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
    dialog.close()


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
    dialog.close()


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
    dialog.close()


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
    dialog.close()


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
    dialog.close()


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
    dialog.close()


def test_manual_click_ignored_while_checkbox_unchecked(tmp_path, qtbot: QtBot) -> None:
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    _select_mapping_strategy(dialog, "Custom")
    original_ids = _original_ids(dialog)
    assert not dialog.manual_click_checkbox.isChecked()

    dialog.overlay_widget.neuron_clicked.emit(original_ids[-1])

    assert dialog._manual_click_order == []
    dialog.close()


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
    dialog.close()


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
        "neuroseq.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, mask_path),
    )

    dialog._annotate_mean_tiff()

    annotate.assert_called_once_with(str(image_path), model_name="cpsam")
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
        "neuroseq.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.cellpose_import_dialog.importlib.util.find_spec",
        return_value=object(),
    )
    annotate = mocker.patch(
        "neuroseq.cellpose_annotation.annotate_mean_fluorescence_tiff",
        return_value=(segmentation_path, mask_path),
    )
    dialog.cellpose_model_combo.setCurrentText("cpsam_v2")

    dialog._annotate_mean_tiff()

    annotate.assert_called_once_with(str(image_path), model_name="cpsam_v2")


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
        "neuroseq.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.cellpose_import_dialog.importlib.util.find_spec",
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
    annotate.assert_called_once_with(str(image_path), model_name="cpsam")
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
        "neuroseq.cellpose_import_dialog.getopenfilename",
        return_value=(str(image_path), ""),
    )
    mocker.patch(
        "neuroseq.cellpose_import_dialog.importlib.util.find_spec",
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
