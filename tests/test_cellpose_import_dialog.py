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


def test_cellpose_dialog_linear_offset_persisted(
    tmp_path, qtbot: QtBot, qmessagebox_calls: list
):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    dialog.linear_offset_spin.setValue(3)

    assert dialog.linear_offset_spin.value() == 3
    assert dialog._get_linear_offset() == 3
    assert len(qmessagebox_calls) == 0


def test_cellpose_dialog_linear_offset_preview_mapping(tmp_path, qtbot: QtBot):
    dialog = _make_cellpose_dialog(tmp_path, qtbot)
    dialog.linear_offset_spin.setValue(3)

    mapping = dialog.overlay_widget.stimulus_to_cellpose_mapping or {}
    base_id = int(np.min(dialog.importer._original_cellpose_ids))
    first_key = min(mapping) if mapping else -1

    assert mapping != {}
    assert first_key == 0
    assert mapping[first_key] == base_id + 3
