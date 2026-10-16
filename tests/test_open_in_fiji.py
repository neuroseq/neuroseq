"""Tests for the cellpose dialog's 'Open Fiji' button."""

from __future__ import annotations

from pathlib import Path


def test_open_in_fiji_passes_image_path(qtbot, mocker, tmp_path: Path):
    from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog

    dialog = CellposeImportDialog(parent=None)
    qtbot.addWidget(dialog)

    image = tmp_path / "img.tif"
    image.write_bytes(b"")
    dialog.image_path = image

    fake_fiji = tmp_path / "fiji"
    fake_fiji.write_text("")
    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value=str(fake_fiji))
    start_detached = mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.QtCore.QProcess.startDetached",
        return_value=True,
    )

    dialog._open_in_fiji()
    start_detached.assert_called_once_with(str(fake_fiji), [str(image)])


def test_open_in_fiji_without_image_starts_empty_fiji(qtbot, mocker, tmp_path: Path):
    from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog

    dialog = CellposeImportDialog(parent=None)
    qtbot.addWidget(dialog)
    dialog.image_path = None

    fake_fiji = tmp_path / "fiji"
    fake_fiji.write_text("")
    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value=str(fake_fiji))
    start_detached = mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.QtCore.QProcess.startDetached",
        return_value=True,
    )

    dialog._open_in_fiji()
    start_detached.assert_called_once_with(str(fake_fiji), [])


def test_open_in_fiji_no_op_when_fiji_unset(qtbot, mocker):
    from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog

    dialog = CellposeImportDialog(parent=None)
    qtbot.addWidget(dialog)
    mocker.patch("neuroseq.fiji.resolve_fiji_path", return_value="")
    start_detached = mocker.patch(
        "neuroseq.importers.cellpose_import_dialog.QtCore.QProcess.startDetached"
    )
    dialog._open_in_fiji()
    start_detached.assert_not_called()
