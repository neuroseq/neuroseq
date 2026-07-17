"""Configuration dialog for the ELiSe train/validation/replay protocol.

Collects the settings passed to
:meth:`neuroseq.example_models.elise.Elise.run_protocol` — the in-GUI
reproduction of Kriener et al. (2024), Fig. 2.
"""

from __future__ import annotations

from qtpy import QtWidgets


class EliseProtocolDialog(QtWidgets.QDialog):
    """Ask the user for ELiSe protocol epochs/cycles and the teacher mode."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run ELiSe protocol")

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.teacher_mode = QtWidgets.QComboBox()
        # userData carries the value passed to run_protocol.
        self.teacher_mode.addItem("Direct (paper-faithful)", "direct")
        self.teacher_mode.addItem("Opsin kinetics (realistic)", "opsin")
        self.teacher_mode.setToolTip(
            "Direct: nudge the soma straight to the crisp target voltage "
            "(matches the paper and the metric target — reproduces Fig. 2).\n"
            "Opsin: derive the teacher from the 3-state opsin kinetics "
            "(biologically realistic, but the kinetic smear means it does not "
            "reach the paper's low replay error)."
        )
        form.addRow("Teacher mode", self.teacher_mode)

        self.training_epochs = self._spin(1, 1_000_000, 30)
        form.addRow("Training epochs", self.training_epochs)
        self.training_cycles = self._spin(1, 1_000_000, 25)
        form.addRow("Training cycles / epoch", self.training_cycles)
        self.validation_cycles = self._spin(2, 1000, 2)
        form.addRow("Validation cycles / epoch", self.validation_cycles)
        self.replay_epochs = self._spin(1, 1_000_000, 20)
        form.addRow("Replay epochs", self.replay_epochs)
        self.replay_cycles = self._spin(2, 1000, 2)
        form.addRow("Replay cycles / epoch", self.replay_cycles)

        self.replay_learning = QtWidgets.QCheckBox("Keep plasticity on during replay")
        self.replay_learning.setChecked(True)
        form.addRow("", self.replay_learning)

        self._estimate = QtWidgets.QLabel()
        self._estimate.setWordWrap(True)
        layout.addWidget(self._estimate)
        for spin in (self.training_epochs, self.training_cycles):
            spin.valueChanged.connect(self._update_estimate)
        self._update_estimate()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(lo: int, hi: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def _update_estimate(self) -> None:
        total = self.training_epochs.value() * self.training_cycles.value()
        self._estimate.setText(
            f"≈ {total} training cycles. Paper-scale is ~10,000 "
            "(e.g. 100 epochs × 100 cycles); direct mode already converges by "
            "~500 cycles. Larger runs take longer but can be cancelled."
        )

    def get_values(self) -> dict:
        """Return the run_protocol keyword arguments chosen by the user."""
        return {
            "teacher_mode": self.teacher_mode.currentData(),
            "training_epochs": self.training_epochs.value(),
            "training_cycles": self.training_cycles.value(),
            "validation_cycles": self.validation_cycles.value(),
            "replay_epochs": self.replay_epochs.value(),
            "replay_cycles": self.replay_cycles.value(),
            "replay_learning": self.replay_learning.isChecked(),
        }
