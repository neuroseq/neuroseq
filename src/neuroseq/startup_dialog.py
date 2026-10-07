from __future__ import annotations

import pathlib
import random

import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq import console_art
from neuroseq.example_projects import getting_started_project
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_timing import DEFAULT_STIM_DT_SECONDS
from neuroseq.stim_timing import seconds_to_samples_ceil

BLANK_STIM_DURATION_SECONDS = 5.0
# Matches the neuron count "Create input array" (StimEditor) starts from.
BLANK_STIM_NEURONS = 10


def new_blank_project() -> Project:
    """An empty project holding a blank (all-zero) 5 s input array, so
    there is something to draw on right away instead of an empty plot that
    first needs "Design static/dynamic input" followed by "Clear" (#584)."""
    project = Project()
    project.add_stim_array(
        StimArray(
            "stimulus",
            StimType.Excitation,
            np.zeros(
                (
                    seconds_to_samples_ceil(
                        BLANK_STIM_DURATION_SECONDS, DEFAULT_STIM_DT_SECONDS
                    ),
                    BLANK_STIM_NEURONS,
                )
            ),
            "nS",
            DEFAULT_STIM_DT_SECONDS,
        )
    )
    # add_stim_array marks dirty as a side effect; a fresh project that
    # hasn't been touched yet shouldn't prompt to save on close.
    project.dirty = False
    return project


_BRAIN_ICON_HEIGHT = 48


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

        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(self._brain_icon(mirrored=True))
        title = QtWidgets.QLabel("<h2>Welcome to NeuroSeq</h2>")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(title, stretch=1)
        title_row.addWidget(self._brain_icon(mirrored=False))
        layout.addLayout(title_row)

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

    def _brain_icon(self, mirrored: bool) -> QtWidgets.QLabel:
        """A small randomly-retinted brain (see console_widget.py for the
        console's own), flanking the title - mirrored on one side so the
        pair faces inward, toward the title, instead of both facing the
        same way."""
        image = QtGui.QImage()
        image.loadFromData(
            console_art.tinted_brain_image_bytes(
                random.Random(), console_art.startup_brain_image_path()
            )
        )
        if mirrored:
            image = image.mirrored(True, False)
        pixmap = QtGui.QPixmap.fromImage(image).scaledToHeight(
            _BRAIN_ICON_HEIGHT, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        label = QtWidgets.QLabel(self)
        label.setPixmap(pixmap)
        return label

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
        self.chosen_project = new_blank_project()
        self.accept()

    def _choose_recent(self, item: QtWidgets.QListWidgetItem) -> None:
        self.chosen_filename = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.accept()
