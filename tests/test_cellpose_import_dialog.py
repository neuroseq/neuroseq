from __future__ import annotations

import numpy as np
from PIL import Image
from pytestqt.qtbot import QtBot

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
        dialog.change_fiji_button,
        dialog.run_cellpose_button,
    ]
    assert (
        dialog.layout().indexOf(dialog.external_gui_group)
        == dialog.layout().count() - 1
    )


def test_file_selection_sections_have_descriptive_titles(qtbot: QtBot):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    assert dialog.segmentation_run_group.title() == "Select file and run segementation"
    assert dialog.file_group.title() == "or load cell segmentation from the files"
    assert dialog.annotate_tiff_button.parentWidget() is dialog.segmentation_run_group


def test_cellpose_dialog_linear_offset_persisted(
    tmp_path, qtbot: QtBot, qmessagebox_calls: list
):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    assert dialog.windowTitle() == "Cell segmentation"
    dialog.linear_offset_spin.setValue(3)

    assert dialog.linear_offset_spin.value() == 3
    assert dialog._get_linear_offset() == 3
    assert len(qmessagebox_calls) == 0
    dialog.close()


def test_cellpose_dialog_linear_offset_preview_mapping(tmp_path, qtbot: QtBot):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
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
