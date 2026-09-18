from __future__ import annotations

import pathlib

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.example_projects import getting_started_project
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project


class StartupDialog(QtWidgets.QDialog):
    """Shown once at app launch (when no project file was given on the
    command line) so the user picks how to start: the built-in example
    project, an existing file, a recently opened file, or a truly blank
    project."""

    def __init__(
        self,
        settings: GuiSettings,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to NeuroSeq")
        self.settings = settings
        # The dialog's outcome: either a ready-to-use Project (Example /
        # New), or a file path for the caller to load (Load / Recent).
        # Exactly one of these is set once the dialog is accepted.
        self.chosen_project: Project | None = None
        self.chosen_filename: str | None = None

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("<h2>Welcome to NeuroSeq</h2>")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        self.btn_example = QtWidgets.QPushButton("Open example project")
        self.btn_example.clicked.connect(self._choose_example)
        layout.addWidget(self.btn_example)

        self.btn_load = QtWidgets.QPushButton("Load Project…")
        self.btn_load.clicked.connect(self._choose_load)
        layout.addWidget(self.btn_load)

        self.btn_new = QtWidgets.QPushButton("New Project")
        self.btn_new.clicked.connect(self._choose_new)
        layout.addWidget(self.btn_new)

        # Stale entries (moved/deleted files) are simply dropped here rather
        # than pruned from settings, so a file that reappears later (e.g. a
        # remounted drive) still shows back up without extra bookkeeping.
        self.recent_paths = [
            path for path in settings.recent_projects if pathlib.Path(path).exists()
        ]
        if self.recent_paths:
            layout.addWidget(QtWidgets.QLabel("Recent Projects:"))
            self.recent_list = QtWidgets.QListWidget()
            self.recent_list.setMaximumHeight(120)
            for path in self.recent_paths:
                item = QtWidgets.QListWidgetItem(pathlib.Path(path).name)
                item.setToolTip(path)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                self.recent_list.addItem(item)
            self.recent_list.itemDoubleClicked.connect(self._choose_recent)
            layout.addWidget(self.recent_list)

    def _choose_example(self) -> None:
        self.chosen_project = getting_started_project()
        self.accept()

    def _choose_load(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load project",
            "project.nseq",
            "neuroseq project (*.nseq)",
        )
        if filename:
            self.chosen_filename = filename
            self.accept()

    def _choose_new(self) -> None:
        self.chosen_project = Project()
        self.accept()

    def _choose_recent(self, item: QtWidgets.QListWidgetItem) -> None:
        self.chosen_filename = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.accept()
