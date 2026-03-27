from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.stim import StimArray


class Igor2PatchClampExportDialog(QtWidgets.QDialog):
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
        self.setWindowTitle("Export Stimulus for Igor2 Patch Clamp")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        stim_name_label = QtWidgets.QLabel(self.stim_name)
        stim_name_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Stimulus array:", stim_name_label)

        self.neuron_combo = QtWidgets.QComboBox()
        for neuron_index in range(int(self.stim_array.values.shape[1])):
            self.neuron_combo.addItem(f"Neuron {neuron_index}", neuron_index)
        self.neuron_combo.currentIndexChanged.connect(
            self._update_suggested_experiment_name
        )
        form.addRow("Neuron to export:", self.neuron_combo)

        self.experiment_name_edit = QtWidgets.QLineEdit(self._auto_name)
        form.addRow("Experiment name:", self.experiment_name_edit)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _suggested_experiment_name(self, neuron_index: int) -> str:
        return f"{self.stim_name}_neuron_{neuron_index}"

    def _update_suggested_experiment_name(self) -> None:
        previous_auto_name = self._auto_name
        self._auto_name = self._suggested_experiment_name(self.selected_neuron_index)
        current_text = self.experiment_name_edit.text().strip()
        if not current_text or current_text == previous_auto_name:
            self.experiment_name_edit.setText(self._auto_name)

    @property
    def selected_neuron_index(self) -> int:
        current_data = self.neuron_combo.currentData()
        if current_data is None:
            return int(self.neuron_combo.currentIndex())
        return int(current_data)

    @property
    def experiment_name(self) -> str:
        return self.experiment_name_edit.text().strip()

    def accept(self) -> None:
        if not self.experiment_name:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                "Experiment name cannot be empty.",
            )
            return
        super().accept()
