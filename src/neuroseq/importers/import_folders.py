"""Start folder of the data import file dialogs, remembered per project (#646).

Each project keeps the folder its import dialogs used last (in
``Project.device_folders``, so it is saved in the ``.nseq`` file). A project
without one starts from the folder used last in any project
(``GuiSettings.import_directory``). Without this, every file dialog started in
Qt's process-wide "last visited" folder, which all open project tabs share.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroseq.gui_settings import GuiSettings
    from neuroseq.project import Project

#: Key of the import folder in ``Project.device_folders``.
IMPORT_FOLDER = "data_import"


class ImportFolders:
    """Start folder for one workspace's import file dialogs.

    ``project`` returns the workspace's current project (it changes when a
    project is loaded into the tab), ``settings`` holds the global fallback.
    """

    def __init__(
        self,
        project: Callable[[], Project | None],
        settings: GuiSettings | None = None,
    ) -> None:
        self._project = project
        self._settings = settings

    def directory(self) -> str:
        """The folder to open the next file dialog in ("" lets Qt choose)."""
        project = self._project()
        folder = project.device_folders.get(IMPORT_FOLDER, "") if project else ""
        if not folder and self._settings is not None:
            folder = self._settings.import_directory
        return folder

    def remember(self, path: str | pathlib.Path | None) -> None:
        """Remember the folder of the picked file (or the picked folder itself)."""
        if not path:
            return
        path = pathlib.Path(path)
        folder = str(path if path.is_dir() else path.parent)
        project = self._project()
        if project is not None:
            project.set_device_folder(IMPORT_FOLDER, folder, mark_dirty=False)
        if self._settings is not None and self._settings.import_directory != folder:
            self._settings.import_directory = folder


def start_directory(folders: ImportFolders | None) -> str:
    """``folders.directory()``, or "" (Qt's default) without a folder memory."""
    return folders.directory() if folders is not None else ""


def remember_folder(
    folders: ImportFolders | None, path: str | pathlib.Path | None
) -> None:
    """``folders.remember(path)`` when there is a folder memory."""
    if folders is not None:
        folders.remember(path)
