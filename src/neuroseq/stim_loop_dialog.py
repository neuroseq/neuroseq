from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.project import Project


class StimLoopDialog(QtWidgets.QDialog):
    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Loop stim array")
        self.project = project

        self.status_label = QtWidgets.QLabel(self)
        self.spinbox = QtWidgets.QSpinBox(self)
        self.spinbox.setRange(1, 1000)
        self.spinbox.setValue(2)
        self.spinbox.setSingleStep(1)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.addWidget(QtWidgets.QLabel("Number of repetitions:", self))
        spin_row.addWidget(self.spinbox)
        spin_row.addStretch(1)
        layout.addLayout(spin_row)

        layout.addWidget(self.button_box)

        self.spinbox.valueChanged.connect(lambda _value: self._refresh_status_label())
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._refresh_status_label()

    @property
    def loop_count(self) -> int:
        return int(self.spinbox.value())

    def _reference_stim(self):
        if not self.project.stim_arrays:
            return None
        return next(iter(self.project.stim_arrays.values()))

    def _refresh_status_label(self) -> None:
        reference_stim = self._reference_stim()
        if reference_stim is None:
            self.status_label.setText("(no stim arrays)")
            return
        n_timestamps = int(reference_stim.values.shape[0])
        current_ms = n_timestamps * reference_stim.dt * 1000.0
        looped_ms = current_ms * self.loop_count
        self.status_label.setText(
            f"Current stim length: {current_ms:.1f} ms ({n_timestamps} samples) "
            f"-> after x {self.loop_count} = {looped_ms:.1f} ms"
        )
