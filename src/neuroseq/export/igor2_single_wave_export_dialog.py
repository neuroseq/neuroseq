from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.utils import detect_period
from neuroseq.utils import parse_number


class Igor2SingleWaveExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        stim_name: str,
        stim_array: StimArray,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stim_name = stim_name
        self.stim_array = stim_array
        self._auto_name = self._suggested_experiment_name(0)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Export single wave to Igor2")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        stim_name_label = QtWidgets.QLabel(self.stim_name)
        stim_name_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Input array:", stim_name_label)
        if self.stim_array.stim_type == StimType.Inhibition:
            form.addRow(
                "",
                QtWidgets.QLabel("Inhibitory array: values are negated (x -1)."),
            )

        self.neuron_combo = QtWidgets.QComboBox()
        for neuron_index in range(int(self.stim_array.values.shape[1])):
            self.neuron_combo.addItem(f"Neuron {neuron_index}", neuron_index)
        self.neuron_combo.currentIndexChanged.connect(
            self._update_suggested_experiment_name
        )
        form.addRow("Neuron to export:", self.neuron_combo)

        self.experiment_name_edit = QtWidgets.QLineEdit(self._auto_name)
        form.addRow("Experiment name:", self.experiment_name_edit)

        self._unit_edit = QtWidgets.QLineEdit()
        self._unit_edit.setPlaceholderText(
            f"(leave blank to keep '{self.stim_array.units}')"
        )
        form.addRow("Unit of measurement (optional):", self._unit_edit)

        self._scale_edit = QtWidgets.QLineEdit()
        self._scale_edit.setPlaceholderText(
            "(blank = 1.0; accepts e.g. 1e-12, 10^-12, 2.5*10^(-12))"
        )
        form.addRow("Scaling factor (optional):", self._scale_edit)

        self._as_sweeps_check = QtWidgets.QCheckBox(
            "Export as sweeps (auto-detect period)", self
        )
        self._as_sweeps_check.setToolTip(
            "If the selected column is a repetition of a smaller block, "
            "export it as a multi-sweep .ibw (one column per repeat). "
            "Ignored when saving as .csv."
        )
        self._as_sweeps_check.toggled.connect(self._update_sweep_status)
        form.addRow("", self._as_sweeps_check)

        self._sweep_status = QtWidgets.QLabel("", self)
        self._sweep_status.setWordWrap(True)
        self._sweep_status.setObjectName("sweep_detection_status")
        form.addRow("", self._sweep_status)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_sweep_status()

    def _suggested_experiment_name(self, neuron_index: int) -> str:
        return f"{self.stim_name}_neuron_{neuron_index}"

    def _update_suggested_experiment_name(self) -> None:
        previous_auto_name = self._auto_name
        self._auto_name = self._suggested_experiment_name(self.selected_neuron_index)
        current_text = self.experiment_name_edit.text().strip()
        if not current_text or current_text == previous_auto_name:
            self.experiment_name_edit.setText(self._auto_name)
        self._update_sweep_status()

    def _update_sweep_status(self) -> None:
        if not self._as_sweeps_check.isChecked():
            self._sweep_status.setText("")
            return
        column = np.asarray(
            self.stim_array.values[:, self.selected_neuron_index],
            dtype=np.float32,
        )
        period = detect_period(column)
        if period is None or period <= 0 or column.size % period != 0:
            self._sweep_status.setText(
                "No periodic block detected — will export as a single sweep."
            )
            return
        n_sweeps = column.size // period
        if n_sweeps < 2:
            self._sweep_status.setText(
                "Only one repetition — will export as a single sweep."
            )
            return
        self._sweep_status.setText(
            f"Period detected: {period} samples × {n_sweeps} sweeps "
            f"({column.size} total)."
        )

    @property
    def selected_neuron_index(self) -> int:
        current_data = self.neuron_combo.currentData()
        if current_data is None:
            return int(self.neuron_combo.currentIndex())
        return int(current_data)

    @property
    def experiment_name(self) -> str:
        return self.experiment_name_edit.text().strip()

    @property
    def unit_override(self) -> str | None:
        text = self._unit_edit.text().strip()
        return text if text else None

    @property
    def scaling_factor(self) -> float:
        text = self._scale_edit.text().strip()
        if not text:
            return 1.0
        return parse_number(text)

    @property
    def as_sweeps(self) -> bool:
        return self._as_sweeps_check.isChecked()

    def accept(self) -> None:
        if not self.experiment_name:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                "Experiment name cannot be empty.",
            )
            return
        scale_text = self._scale_edit.text().strip()
        if scale_text:
            try:
                value = parse_number(scale_text)
            except ValueError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid scaling factor",
                    str(exc),
                )
                return
            import math

            if not math.isfinite(value):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid scaling factor",
                    "Scaling factor must be a finite number.",
                )
                return
        super().accept()
