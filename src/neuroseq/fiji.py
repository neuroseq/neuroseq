"""Helpers for locating and launching Fiji/ImageJ from neuroseq dialogs."""

from __future__ import annotations

import logging
from pathlib import Path

from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.gui_settings import GuiSettings
from neuroseq.gui_settings import _autodetect_fiji

logger = logging.getLogger(__name__)


class FijiPathDialog(QtWidgets.QDialog):
    """Small modal dialog to set/edit the Fiji executable path."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, *, initial_path: str = ""
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Locate Fiji/ImageJ executable")
        self.setModal(True)
        self.resize(540, 180)
        self._chosen_path: str = initial_path

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Browse to the Fiji launcher binary (e.g. "
            "<code>ImageJ-linux64</code> inside <code>Fiji.app</code>) or paste "
            "the full path below; use <b>Auto-detect</b> to search common "
            "locations. Install Fiji from "
            "<a href='https://fiji.sc/'>https://fiji.sc/</a>."
        )
        info.setWordWrap(True)
        info.setOpenExternalLinks(True)
        layout.addWidget(info)

        path_row = QtWidgets.QHBoxLayout()
        self._path_edit = QtWidgets.QLineEdit(initial_path)
        self._path_edit.setPlaceholderText("/opt/Fiji.app/ImageJ-linux64")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        autodetect_btn = QtWidgets.QPushButton("Auto-detect")
        autodetect_btn.clicked.connect(self._autodetect)
        path_row.addWidget(self._path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        path_row.addWidget(autodetect_btn)
        layout.addLayout(path_row)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self._status_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Fiji executable",
            self._path_edit.text() or "",
            "All files (*)",
        )
        if filename:
            self._path_edit.setText(filename)

    def _autodetect(self) -> None:
        detected = _autodetect_fiji()
        if detected:
            self._path_edit.setText(detected)
            self._status_label.setText(f"Detected: {detected}")
            self._status_label.setStyleSheet("QLabel { color: green; }")
        else:
            self._status_label.setText("Could not auto-detect Fiji on this system.")
            self._status_label.setStyleSheet("QLabel { color: #b35900; }")

    def _on_save(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            self._status_label.setText("Please provide a path or cancel.")
            self._status_label.setStyleSheet("QLabel { color: red; }")
            return
        if not Path(path_str).exists():
            self._status_label.setText(f"Path does not exist: {path_str}")
            self._status_label.setStyleSheet("QLabel { color: red; }")
            return
        self._chosen_path = path_str
        self.accept()

    @property
    def chosen_path(self) -> str:
        return self._chosen_path


def resolve_fiji_path(
    parent: QtWidgets.QWidget | None, *, force_prompt: bool = False
) -> str:
    """Return a valid Fiji path, prompting the user via dialog if needed.

    With ``force_prompt=True`` the chooser is always shown (pre-filled with the
    saved path) even when a valid path already exists, so a wrong-but-valid path
    can be corrected from the GUI.

    Returns an empty string if the user cancels the dialog.
    """
    settings = GuiSettings()
    path = settings.fiji_executable_path
    if not force_prompt and path and Path(path).exists():
        return path

    dialog = FijiPathDialog(parent, initial_path=path)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return ""

    chosen = dialog.chosen_path
    if chosen:
        settings.fiji_executable_path = chosen
        logger.info("Saved Fiji executable path: %s", chosen)
    return chosen
