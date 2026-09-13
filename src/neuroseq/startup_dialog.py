from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.example_projects import default_project
from neuroseq.project import Project


class StartupDialog(QtWidgets.QDialog):
    """Shown once at app launch (when no project file was given on the
    command line) so the user picks how to start: the built-in example
    project, an existing file, or a truly blank project."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to NeuroSeq")
        # The dialog's outcome: either a ready-to-use Project (Example /
        # New), or a file path for the caller to load (Load). Exactly one
        # of these is set once the dialog is accepted.
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

    def _choose_example(self) -> None:
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

    def _choose_new(self) -> None:
        self.chosen_project = Project()
        self.accept()
