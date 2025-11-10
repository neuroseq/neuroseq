from __future__ import annotations

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QDialog
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QVBoxLayout


class SaveOptionsDialog(QDialog):
    """A dialog that provides checkboxes to choose which parts of the model should be saved."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.include_simulation_data = True
        self.include_experimental_data = True

        self.setWindowTitle("Save Model")
        self.setModal(True)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Please select which data you want to include:"))

        checkbox_simulation_data = QCheckBox("Simulation data")
        checkbox_simulation_data.setChecked(True)
        checkbox_simulation_data.checkStateChanged.connect(
            self.checkbox_simulation_data_changed
        )
        self.layout.addWidget(checkbox_simulation_data)

        checkbox_experimental_data = QCheckBox("Experimental Data")
        checkbox_experimental_data.setChecked(True)
        checkbox_experimental_data.checkStateChanged.connect(
            self.checkbox_experimental_data_changed
        )
        self.layout.addWidget(checkbox_experimental_data)

        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def checkbox_simulation_data_changed(self, state: int) -> None:
        self.include_simulation_data = state == Qt.CheckState.Checked

    def checkbox_experimental_data_changed(self, state: int) -> None:
        self.include_experimental_data = state == Qt.CheckState.Checked
