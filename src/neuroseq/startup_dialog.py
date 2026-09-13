from __future__ import annotations

import pathlib

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project


class StartupDialog(QtWidgets.QDialog):
    """Shown once at app launch (when no project file was given on the
    command line) so the user picks how to start: a new project, an
    existing file, a recently opened file, or a built-in example."""

    def __init__(
        self,
        settings: GuiSettings,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to NeuroSeq")
        self.resize(520, 480)
        self.settings = settings
        # The dialog's outcome: either a ready-to-use Project (New / an
        # example), or a file path for the caller to load (Load / Recent).
        # Exactly one of these is set once the dialog is accepted.
        self.chosen_project: Project | None = None
        self.chosen_filename: str | None = None

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("<h2>Welcome to NeuroSeq</h2>")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_new = QtWidgets.QPushButton("New Project")
        self.btn_new.clicked.connect(self._choose_new)
        button_row.addWidget(self.btn_new)
        self.btn_load = QtWidgets.QPushButton("Load Project…")
        self.btn_load.clicked.connect(self._choose_load)
        button_row.addWidget(self.btn_load)
        layout.addLayout(button_row)

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

        layout.addWidget(QtWidgets.QLabel("Example Projects:"))
        self.example_tree = QtWidgets.QTreeWidget()
        self.example_tree.setHeaderHidden(True)
        for section, section_projects in example_projects.items():
            section_item = QtWidgets.QTreeWidgetItem([section])
            section_item.setFlags(
                section_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            self.example_tree.addTopLevelItem(section_item)
            for name, generator in section_projects.items():
                child = QtWidgets.QTreeWidgetItem([name])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, generator)
                section_item.addChild(child)
        self.example_tree.expandAll()
        self.example_tree.itemDoubleClicked.connect(self._choose_example)
        layout.addWidget(self.example_tree)

    def _choose_new(self) -> None:
        self.chosen_project = default_project()
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

    def _choose_recent(self, item: QtWidgets.QListWidgetItem) -> None:
        self.chosen_filename = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.accept()

    def _choose_example(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        generator = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if generator is None:
            return  # a section header, not a selectable example
        self.chosen_project = generator()
        self.accept()
