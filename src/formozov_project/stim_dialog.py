from __future__ import annotations

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
        dt: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Stimuli")
        self.stims = stims.copy() if stims is not None else np.zeros((1, 1))
        self.dt = dt
        layout = QtWidgets.QVBoxLayout()
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

        layout.addWidget(self.buttonBox)
        self.stim_console_widget.stims_updated.connect(self._update_stim)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self._update_stim(self.stims, valid=True)

    def _update_stim(self, stim: np.ndarray, valid: bool) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            valid
        )
        if valid:
            self.stim_widget.set_data(stim, dt=self.dt)
            self.stims = stim
