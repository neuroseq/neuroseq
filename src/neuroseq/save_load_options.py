from __future__ import annotations

from dataclasses import dataclass

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QDialog
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QVBoxLayout


@dataclass
class SaveLoadOptions:
    include_stim_arrays: bool
    include_model: bool
    include_simulation_data: bool
    include_experimental_data: bool


class SaveLoadOptionsDialog(QDialog):
    """A dialog that provides checkboxes to choose which parts of the model should be saved or loaded."""

    def __init__(self, action="Save", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.options = SaveLoadOptions(True, True, True, True)

        self.setWindowTitle(f"{action} Project")
        self.setModal(True)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Please select which data you want to include:"))

        self.checkbox_stim_arrays = QCheckBox("Input Arrays")
        self.checkbox_stim_arrays.setChecked(True)
        self.checkbox_stim_arrays.checkStateChanged.connect(
            self.checkbox_stim_arrays_changed
        )
        self.layout.addWidget(self.checkbox_stim_arrays)

        self.checkbox_model = QCheckBox("Model")
        self.checkbox_model.setChecked(True)
        self.checkbox_model.checkStateChanged.connect(self.checkbox_model_changed)
        self.layout.addWidget(self.checkbox_model)

        self.checkbox_simulation_data = QCheckBox("Simulation data")
        self.checkbox_simulation_data.setChecked(True)
        self.checkbox_simulation_data.checkStateChanged.connect(
            self.checkbox_simulation_data_changed
        )
        self.layout.addWidget(self.checkbox_simulation_data)

        self.checkbox_experimental_data = QCheckBox("Experimental Data")
        self.checkbox_experimental_data.setChecked(True)
        self.checkbox_experimental_data.checkStateChanged.connect(
            self.checkbox_experimental_data_changed
        )
        self.layout.addWidget(self.checkbox_experimental_data)

        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton(f"{action}")
        self.cancel_button = QPushButton("Cancel")
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def checkbox_stim_arrays_changed(self, state: int) -> None:
        self.options.include_stim_arrays = state == Qt.CheckState.Checked

    def checkbox_model_changed(self, state: int) -> None:
        self.options.include_model = state == Qt.CheckState.Checked
        self.checkbox_simulation_data.setEnabled(self.options.include_model)
        if not self.options.include_model:
            # if model is not included, simulation data cannot be included either
            self.checkbox_simulation_data.setChecked(False)
            self.options.include_simulation_data = False

    def checkbox_simulation_data_changed(self, state: int) -> None:
        self.options.include_simulation_data = state == Qt.CheckState.Checked

    def checkbox_experimental_data_changed(self, state: int) -> None:
        self.options.include_experimental_data = state == Qt.CheckState.Checked
