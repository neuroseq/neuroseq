"""Options for saving the imaging data shown in the Imaging tab (issue #609)."""

from __future__ import annotations

from qtpy import QtWidgets


class SaveImagingDialog(QtWidgets.QDialog):
    """Choose the frames to save and whether to keep the original pixels.

    Frames are counted from 1, as in the time slider tooltip and in Fiji.
    """

    def __init__(
        self,
        frame_count: int,
        current_frame: int,
        *,
        original: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save imaging data")
        self._current_frame = current_frame

        self.first_spin = QtWidgets.QSpinBox()
        self.last_spin = QtWidgets.QSpinBox()
        for spin in (self.first_spin, self.last_spin):
            spin.setRange(1, max(1, frame_count))
        self.first_spin.setValue(1)
        self.last_spin.setValue(max(1, frame_count))
        self.first_spin.valueChanged.connect(self._keep_order_after_first)
        self.last_spin.valueChanged.connect(self._keep_order_after_last)
        first_current = QtWidgets.QPushButton("Current frame")
        first_current.setAutoDefault(False)
        first_current.clicked.connect(
            lambda: self.first_spin.setValue(self._current_frame)
        )
        last_current = QtWidgets.QPushButton("Current frame")
        last_current.setAutoDefault(False)
        last_current.clicked.connect(
            lambda: self.last_spin.setValue(self._current_frame)
        )
        all_frames = QtWidgets.QPushButton("All frames")
        all_frames.setAutoDefault(False)
        all_frames.clicked.connect(self._select_all)

        form = QtWidgets.QFormLayout()
        first_row = QtWidgets.QHBoxLayout()
        first_row.addWidget(self.first_spin, 1)
        first_row.addWidget(first_current)
        form.addRow("First frame:", first_row)
        last_row = QtWidgets.QHBoxLayout()
        last_row.addWidget(self.last_spin, 1)
        last_row.addWidget(last_current)
        form.addRow("Last frame:", last_row)
        form.addRow("", all_frames)
        self.count_label = QtWidgets.QLabel()
        form.addRow("", self.count_label)

        self.original_radio = QtWidgets.QRadioButton("Original frames (TIFF substack)")
        self.original_radio.setToolTip(
            "The recorded pixel values of the shown channel, as a stack that "
            "Fiji opens directly."
        )
        self.marked_radio = QtWidgets.QRadioButton(
            "With stimulation marks (TIFF or GIF)"
        )
        self.marked_radio.setToolTip(
            "The frames as displayed, with the stimulated cells marked in red."
        )
        (self.original_radio if original else self.marked_radio).setChecked(True)
        content = QtWidgets.QVBoxLayout()
        content.addWidget(self.original_radio)
        content.addWidget(self.marked_radio)
        form.addRow("Save:", content)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update_count()

    @property
    def frame_range(self) -> tuple[int, int]:
        """First and last frame to save, counted from 0, both included."""
        return self.first_spin.value() - 1, self.last_spin.value() - 1

    @property
    def original(self) -> bool:
        return self.original_radio.isChecked()

    def _select_all(self) -> None:
        self.first_spin.setValue(1)
        self.last_spin.setValue(self.last_spin.maximum())

    def _keep_order_after_first(self, value: int) -> None:
        if self.last_spin.value() < value:
            self.last_spin.setValue(value)
        self._update_count()

    def _keep_order_after_last(self, value: int) -> None:
        if self.first_spin.value() > value:
            self.first_spin.setValue(value)
        self._update_count()

    def _update_count(self) -> None:
        first, last = self.frame_range
        count = last - first + 1
        self.count_label.setText(f"{count} frame{'s' if count != 1 else ''}")
