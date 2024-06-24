from __future__ import annotations

import brian2 as b2
import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.stim_console_widget import StimConsoleWidget
from formozov_project.stim_widget import StimWidget


class StimDialog(QtWidgets.QDialog):
    """A dialog to edit 2-d numpy arrays of neuron stimuli interactively using a Python console.

    The input stimuli are not modified, and the edited stimuli are available as self.stim.
    """

    def __init__(
        self,
        stims: np.ndarray | None = None,
        dt: b2.Quantity = 1.0 * b2.ms,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Stimuli")
        self.name = "stim"
        self.stims = stims.copy() if stims is not None else np.zeros((1, 1))
        self.units = b2.nS
        self.dt = dt / b2.second
        layout = QtWidgets.QVBoxLayout()
        # horizontal row of widgets for name, units, dt
        top_horizontal_layout = QtWidgets.QHBoxLayout()
        self.text_name = QtWidgets.QLineEdit()
        self.text_name.setText(self.name)
        top_horizontal_layout.addWidget(QtWidgets.QLabel("Name:"))
        top_horizontal_layout.addWidget(self.text_name)
        top_horizontal_layout.addWidget(QtWidgets.QLabel(f"Units: {self.units}"))
        top_horizontal_layout.addWidget(QtWidgets.QLabel(f"dt: {self.dt} s"))
        layout.addLayout(top_horizontal_layout)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.stim_widget = StimWidget()
        split_left_right.addWidget(self.stim_widget)
        self.stim_console_widget = StimConsoleWidget(stims=self.stims)
        split_left_right.addWidget(self.stim_console_widget)
        layout.addWidget(split_left_right)
        self.setLayout(layout)
        layout.addWidget(self.button_box)
        self.stim_console_widget.stims_updated.connect(self._update_stims)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._update_stims(self.stims, valid=True)

    def text_name_changed(self, text: str) -> None:
        self.name = text

    def _update_stims(self, stims: np.ndarray, valid: bool) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            valid
        )
        if valid:
            self.stim_widget.set_data(stims, dt=self.dt)
            self.stims = stims
