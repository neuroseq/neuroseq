from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.stimulation.stim_timing import format_milliseconds


class StimPadDialog(QtWidgets.QDialog):
    """Ask how much time to add to the input arrays (#647).

    Either a number of seconds to add, or the time point the arrays should
    end at, padding the difference; :func:`padded_length_seconds` turns the
    entry into the new total length.
    """

    def __init__(
        self,
        stim_name: str,
        current_seconds: float,
        add_seconds: float,
        *,
        to_end_time: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pad input arrays")
        self.status_label = QtWidgets.QLabel(
            f"Current duration of '{stim_name}': {current_seconds:g} s "
            f"({format_milliseconds(current_seconds * 1000)} ms)",
            self,
        )
        self.radio_add = QtWidgets.QRadioButton("Seconds to add:", self)
        self.edit_add = QtWidgets.QLineEdit(f"{add_seconds}", self)
        self.radio_end_time = QtWidgets.QRadioButton("Pad to end time (s):", self)
        self.edit_end_time = QtWidgets.QLineEdit(
            f"{current_seconds + add_seconds:g}", self
        )
        self.edit_end_time.setToolTip(
            "The time point the input arrays should end at; the difference to "
            "the current end is added."
        )
        (self.radio_end_time if to_end_time else self.radio_add).setChecked(True)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        form = QtWidgets.QGridLayout()
        form.addWidget(self.radio_add, 0, 0)
        form.addWidget(self.edit_add, 0, 1)
        form.addWidget(self.radio_end_time, 1, 0)
        form.addWidget(self.edit_end_time, 1, 1)
        layout.addLayout(form)
        layout.addWidget(self.button_box)

        self.radio_add.toggled.connect(self._mode_changed)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._mode_changed()

    @property
    def to_end_time(self) -> bool:
        return self.radio_end_time.isChecked()

    @property
    def value_text(self) -> str:
        edit = self.edit_end_time if self.to_end_time else self.edit_add
        return edit.text().strip()

    def _mode_changed(self) -> None:
        self.edit_add.setEnabled(not self.to_end_time)
        self.edit_end_time.setEnabled(self.to_end_time)
