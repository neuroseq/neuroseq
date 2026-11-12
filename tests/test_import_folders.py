import pathlib

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.gui_settings import GuiSettings
from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
from neuroseq.importers.import_folders import IMPORT_FOLDER
from neuroseq.importers.import_folders import ImportFolders
from neuroseq.importers.import_folders import remember_folder
from neuroseq.importers.import_folders import start_directory
from neuroseq.importers.tif_import_dialog import TifImportDialog
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.save_load_options import SaveLoadOptions


def test_folder_is_per_project_with_global_fallback(tmp_path: pathlib.Path) -> None:
    settings = GuiSettings()
    first, second = Project(), Project()
    first_folders = ImportFolders(lambda: first, settings)
    second_folders = ImportFolders(lambda: second, settings)
    folder_x = tmp_path / "x"
    folder_y = tmp_path / "y"
    folder_x.mkdir()
    folder_y.mkdir()

    assert first_folders.directory() == ""
    first_folders.remember(folder_x / "seg.npy")
    assert first_folders.directory() == str(folder_x)
    # A project without a folder of its own starts where any project left off.
    assert second_folders.directory() == str(folder_x)
    second_folders.remember(folder_y)  # a picked folder is kept as is
    assert second_folders.directory() == str(folder_y)
    assert first_folders.directory() == str(folder_x)
    assert GuiSettings().import_directory == str(folder_y)
    # Browsing alone does not ask to save the project.
    assert not first.dirty

    saved = tmp_path / "project.nseq"
    first.save(saved, SaveLoadOptions(True, True, False, True))
    assert Project(saved).device_folders[IMPORT_FOLDER] == str(folder_x)


def test_helpers_without_folder_memory_keep_qt_default() -> None:
    assert start_directory(None) == ""
    remember_folder(None, "/somewhere/file.tif")  # no error
    folders = ImportFolders(lambda: None)
    folders.remember("")
    folders.remember("/somewhere/file.tif")
    assert folders.directory() == ""


def test_open_project_tabs_do_not_share_the_import_folder(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """#646: the Cell Segmentation folder of one tab is not offered in another."""
    settings = GuiSettings()
    tabs = [ProjectWorkspace(settings=settings) for _ in range(2)]
    for tab in tabs:
        qtbot.addWidget(tab)
    starts: list[str] = []
    picks = iter(
        [
            str(tmp_path / "fov1" / "img_seg.npy"),
            str(tmp_path / "fov2" / "img_seg.npy"),
            "",
            "",
        ]
    )

    def fake_getopenfilename(parent, caption, directory, *args, **kwargs):
        starts.append(directory)
        return next(picks), ""

    monkeypatch.setattr(
        "neuroseq.importers.cellpose_import_dialog.getopenfilename",
        fake_getopenfilename,
    )
    monkeypatch.setattr(
        "neuroseq.importers.tif_import_dialog.getopenfilename", fake_getopenfilename
    )
    monkeypatch.setattr(
        CellposeImportDialog, "_load_segmentation_file", lambda *a: None
    )

    def segmentation_dialog(tab: ProjectWorkspace) -> CellposeImportDialog:
        dialog = CellposeImportDialog(tab, import_folders=tab.import_folders)
        qtbot.addWidget(dialog)
        return dialog

    segmentation_dialog(tabs[0])._browse_segmentation_file()  # picks fov1
    segmentation_dialog(tabs[1])._browse_segmentation_file()  # picks fov2
    first_tif = TifImportDialog(tabs[0], import_folders=tabs[0].import_folders)
    second_tif = TifImportDialog(tabs[1], import_folders=tabs[1].import_folders)
    qtbot.addWidget(first_tif)
    qtbot.addWidget(second_tif)
    first_tif._browse_seg_file()
    second_tif._browse_seg_file()

    assert starts == [
        "",
        str(tmp_path / "fov1"),  # a new project starts from the last folder
        str(tmp_path / "fov1"),
        str(tmp_path / "fov2"),
    ]
    for tab in tabs:
        tab._close_confirmed = True
