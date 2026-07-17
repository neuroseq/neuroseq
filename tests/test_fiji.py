"""Tests for the Fiji path resolver and settings dialog."""

from __future__ import annotations

from pathlib import Path


def test_resolve_fiji_path_returns_existing_path_without_dialog(
    qtbot, mocker, tmp_path: Path
):
    from neuroseq import fiji

    fake_fiji = tmp_path / "ImageJ-linux64"
    fake_fiji.write_text("")

    settings = mocker.MagicMock()
    settings.fiji_executable_path = str(fake_fiji)
    mocker.patch("neuroseq.fiji.GuiSettings", return_value=settings)

    dialog_cls = mocker.patch("neuroseq.fiji.FijiPathDialog")
    result = fiji.resolve_fiji_path(parent=None)

    assert result == str(fake_fiji)
    dialog_cls.assert_not_called()


def test_resolve_fiji_path_force_prompt_shows_dialog_even_when_valid(
    qtbot, mocker, tmp_path: Path
):
    from neuroseq import fiji

    old_fiji = tmp_path / "ImageJ-linux64"
    old_fiji.write_text("")
    new_fiji = tmp_path / "fiji"
    new_fiji.write_text("")

    settings = mocker.MagicMock()
    settings.fiji_executable_path = str(old_fiji)  # already valid
    mocker.patch("neuroseq.fiji.GuiSettings", return_value=settings)

    fake_dialog = mocker.MagicMock()
    fake_dialog.exec.return_value = 1  # Accepted
    fake_dialog.chosen_path = str(new_fiji)
    dialog_cls = mocker.patch("neuroseq.fiji.FijiPathDialog", return_value=fake_dialog)

    result = fiji.resolve_fiji_path(parent=None, force_prompt=True)

    # Dialog is shown despite the existing valid path, pre-filled with it, and
    # the newly chosen path is persisted.
    dialog_cls.assert_called_once()
    assert dialog_cls.call_args.kwargs["initial_path"] == str(old_fiji)
    assert result == str(new_fiji)
    assert settings.fiji_executable_path == str(new_fiji)


def test_resolve_fiji_path_pops_dialog_when_path_missing(qtbot, mocker, tmp_path: Path):
    from neuroseq import fiji

    settings = mocker.MagicMock()
    settings.fiji_executable_path = ""
    mocker.patch("neuroseq.fiji.GuiSettings", return_value=settings)

    new_path = tmp_path / "ImageJ-linux64"
    new_path.write_text("")

    fake_dialog = mocker.MagicMock()
    fake_dialog.exec.return_value = 1  # Accepted
    fake_dialog.chosen_path = str(new_path)
    mocker.patch("neuroseq.fiji.FijiPathDialog", return_value=fake_dialog)

    result = fiji.resolve_fiji_path(parent=None)

    assert result == str(new_path)
    assert settings.fiji_executable_path == str(new_path)


def test_resolve_fiji_path_returns_empty_on_cancel(qtbot, mocker):
    from neuroseq import fiji

    settings = mocker.MagicMock()
    settings.fiji_executable_path = ""
    mocker.patch("neuroseq.fiji.GuiSettings", return_value=settings)

    fake_dialog = mocker.MagicMock()
    fake_dialog.exec.return_value = 0  # Rejected
    mocker.patch("neuroseq.fiji.FijiPathDialog", return_value=fake_dialog)

    assert fiji.resolve_fiji_path(parent=None) == ""


def test_fiji_path_dialog_rejects_nonexistent_path(qtbot, mocker):
    from neuroseq.fiji import FijiPathDialog

    dialog = FijiPathDialog(parent=None, initial_path="")
    qtbot.addWidget(dialog)
    dialog._path_edit.setText("/does/not/exist")

    accept = mocker.patch.object(dialog, "accept")
    dialog._on_save()
    accept.assert_not_called()
    assert "does not exist" in dialog._status_label.text()


def test_fiji_path_dialog_accepts_existing_path(qtbot, tmp_path: Path):
    from neuroseq.fiji import FijiPathDialog

    fake_fiji = tmp_path / "ImageJ-linux64"
    fake_fiji.write_text("")

    dialog = FijiPathDialog(parent=None, initial_path="")
    qtbot.addWidget(dialog)
    dialog._path_edit.setText(str(fake_fiji))

    dialog._on_save()
    assert dialog.chosen_path == str(fake_fiji)
