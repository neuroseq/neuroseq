from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.project import Project
from neuroseq.utils import detect_period


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
        self.detect_button = QtWidgets.QPushButton("Detect period and trim", self)
        self.detect_button.setToolTip(
            "Find the smallest repeating block in the current stim and trim "
            "the array to one period. Loop then composes that with the "
            "multiplier."
        )
        self.detect_caption = QtWidgets.QLabel("", self)
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

        detect_row = QtWidgets.QHBoxLayout()
        detect_row.addWidget(self.detect_button)
        detect_row.addWidget(self.detect_caption)
        detect_row.addStretch(1)
        layout.addLayout(detect_row)

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.addWidget(QtWidgets.QLabel("Number of repetitions:", self))
        spin_row.addWidget(self.spinbox)
        spin_row.addStretch(1)
        layout.addLayout(spin_row)

        layout.addWidget(self.button_box)

        self.detect_button.clicked.connect(self._on_detect_clicked)
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

    def _on_detect_clicked(self) -> None:
        reference_stim = self._reference_stim()
        if reference_stim is None:
            self.detect_caption.setText("(no stim arrays)")
            return

        period = detect_period(reference_stim.values)
        if period is None or period <= 0 or period >= reference_stim.values.shape[0]:
            self.detect_caption.setText("No periodic block detected; array unchanged.")
            return

        new_length_secs = period * reference_stim.dt
        self.project.trim_stim_arrays(total_time_seconds=new_length_secs)
        self.detect_caption.setText(
            f"Trimmed to {new_length_secs * 1000.0:.1f} ms "
            f"({period} samples) - 1 period"
        )
        self._refresh_status_label()
