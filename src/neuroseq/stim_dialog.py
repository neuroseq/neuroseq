from __future__ import annotations

from qtpy import QtWidgets
from qtpy.QtWidgets import QDialog
from qtpy.QtWidgets import QGridLayout
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QPushButton

from neuroseq.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_timing import validate_stim_dt_seconds


class EditStimDialog(QDialog):
    """A dialog that allows a user to edit the properties of a StimArray."""

    def __init__(
        self,
        stim_array: StimArray,
        existing_names: list[str],
        parent: QtWidgets.QWidget | None = None,
        title: str = "Edit input array",
    ) -> None:
        super().__init__(parent)
        self.existing_names = existing_names

        self.name = stim_array.name
        self.stim_type = stim_array.stim_type
        self.n_neurons = stim_array.values.shape[1]
        self.units = stim_array.units
        self.dt = stim_array.dt
        self._name_error = ""
        self._dt_error = ""

        self.setWindowTitle(title)
        self.setModal(True)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Name:"), 0, 0)
        self.txt_name = QtWidgets.QLineEdit(self)
        self.txt_name.setText(self.name)
        self.txt_name.textChanged.connect(self.txt_name_text_changed)
        self.layout.addWidget(self.txt_name, 0, 1)

        self.layout.addWidget(QLabel("Type:"), 1, 0)
        self.cmb_type = QtWidgets.QComboBox(self)
        self.cmb_type.addItem("Excitatory", StimType.Excitation)
        self.cmb_type.addItem("Inhibitory", StimType.Inhibition)
        self.cmb_type.addItem("External", StimType.External)
        self.cmb_type.setCurrentIndex(
            {StimType.Excitation: 0, StimType.Inhibition: 1, StimType.External: 2}[
                self.stim_type
            ]
        )
        self.cmb_type.currentIndexChanged.connect(self.cmb_type_index_changed)
        self.layout.addWidget(self.cmb_type, 1, 1)

        self.layout.addWidget(QLabel("Number of neurons:"), 2, 0)
        self.spin_neurons = QtWidgets.QSpinBox(self)
        self.spin_neurons.setMinimum(1)
        self.spin_neurons.setMaximum(1000000)
        self.spin_neurons.setValue(self.n_neurons)
        self.spin_neurons.valueChanged.connect(self.spin_neurons_value_changed)
        self.layout.addWidget(self.spin_neurons, 2, 1)

        self.layout.addWidget(QLabel("Units:"), 3, 0)
        self.txt_units = QtWidgets.QLineEdit(self)
        self.txt_units.setText(self.units)
        self.txt_units.textChanged.connect(self.txt_units_text_changed)
        self.layout.addWidget(self.txt_units, 3, 1)

        self.layout.addWidget(QLabel("Timestep (s):"), 4, 0)
        self.txt_dt = QtWidgets.QLineEdit(self)
        self.txt_dt.setText(f"{self.dt}")
        self.txt_dt.textChanged.connect(self.txt_dt_text_changed)
        timestep_row = QHBoxLayout()
        timestep_row.addWidget(self.txt_dt)
        self.cmb_dt_presets = QtWidgets.QComboBox(self)
        self.cmb_dt_presets.addItem("Custom timestep", None)
        for label, seconds in (
            ("SysCon2 — 520.8125 µs", _SYSCON2_TTL_PULSE_QUANTUM_S),
            ("1 ms", 0.001),
            ("0.1 ms", 0.0001),
            ("50 µs", 0.00005),
            ("30 Hz imaging", 1 / 30),
        ):
            self.cmb_dt_presets.addItem(label, seconds)
        self.cmb_dt_presets.activated.connect(self._apply_dt_preset)
        timestep_row.addWidget(self.cmb_dt_presets)
        self.layout.addLayout(timestep_row, 4, 1)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: red")
        self.layout.addWidget(self.lbl_error, 5, 0, 1, 2)

        self.button_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_cancel = QPushButton("Cancel")
        self.button_layout.addWidget(self.btn_save)
        self.button_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(self.button_layout, 6, 0, 1, 2)

        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.txt_name_text_changed(self.name)
        self.txt_dt_text_changed(self.txt_dt.text())

    def _apply_dt_preset(self, index: int) -> None:
        seconds = self.cmb_dt_presets.itemData(index)
        if seconds is not None:
            self.txt_dt.setText(str(seconds))

    def _update_validation_state(self) -> None:
        error = self._name_error or self._dt_error
        self.btn_save.setEnabled(error == "")
        self.txt_name.setStyleSheet("color: red" if self._name_error else "")
        self.txt_dt.setStyleSheet("color: red" if self._dt_error else "")
        self.lbl_error.setText(error)

    def txt_name_text_changed(self, text: str) -> None:
        self.name = text
        if text in self.existing_names:
            self._name_error = f"An input array with name '{text}' already exists."
        else:
            self._name_error = ""
        self._update_validation_state()

    def cmb_type_index_changed(self, index: int) -> None:
        self.stim_type = {
            0: StimType.Excitation,
            1: StimType.Inhibition,
            2: StimType.External,
        }[index]

    def spin_neurons_value_changed(self, value: int) -> None:
        self.n_neurons = value

    def txt_units_text_changed(self, text: str) -> None:
        self.units = text

    def txt_dt_text_changed(self, text: str) -> None:
        try:
            self.dt = validate_stim_dt_seconds(float(text))
            self._dt_error = ""
        except (TypeError, ValueError):
            self._dt_error = "Input timestep must be a positive finite number."
        preset = self.cmb_dt_presets.findData(self.dt) if not self._dt_error else -1
        self.cmb_dt_presets.setCurrentIndex(max(0, preset))
        self._update_validation_state()
