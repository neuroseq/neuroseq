import math

from qtpy import QtWidgets

from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim_timing import format_milliseconds
from neuroseq.stim_timing import samples_to_milliseconds
from neuroseq.stim_timing import snap_milliseconds_to_dt
from neuroseq.stim_widget import StimType


class StimIntensityDialog(QtWidgets.QDialog):
    def __init__(
        self,
        draw_stim: DrawStim,
        stim_units: str,
        stim_type: StimType,
        draw_stim_length: int,
        dt_seconds: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Design stimulus")
        self.draw_stim = draw_stim
        self.stim_units = stim_units
        self.stim_type = stim_type
        self.dt_seconds = dt_seconds
        self.draw_stim_length = draw_stim_length
        # Restored designs may previously have stored Logarithmic / Random;
        # those modes are no longer offered here. Fall back to Linear (Ramp)
        # so the dialog always opens on a supported radio.
        if self.draw_stim.draw_stim_type not in (
            DrawStimType.Constant,
            DrawStimType.Linear,
        ):
            self.draw_stim.draw_stim_type = DrawStimType.Linear
        # Tracked independently of the QLineEdit contents so the OK button can
        # require every field to be valid at once, rather than the last-edited
        # field alone (value2 is excluded while it's hidden - see intensity_type_changed).
        self._value1_valid = True
        self._value2_valid = True
        self._length_valid = True

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label_intensity_type = QtWidgets.QLabel("Select intensity type:", self)
        self.layout.addWidget(self.label_intensity_type)
        self.rad_group_intensity = QtWidgets.QButtonGroup(self)
        # DrawStimType.Constant
        self.rad_intensity_constant = QtWidgets.QRadioButton("Constant", self)
        self.rad_intensity_constant.setChecked(
            self.draw_stim.draw_stim_type == DrawStimType.Constant
        )
        self.layout.addWidget(self.rad_intensity_constant)
        self.rad_group_intensity.addButton(self.rad_intensity_constant)
        self.rad_intensity_constant.clicked.connect(self.intensity_type_changed)
        # DrawStimType.Linear — surfaced as "Ramp" in the UI.
        self.rad_intensity_linear = QtWidgets.QRadioButton("Ramp", self)
        self.rad_intensity_linear.setChecked(
            self.draw_stim.draw_stim_type == DrawStimType.Linear
        )
        self.layout.addWidget(self.rad_intensity_linear)
        self.rad_group_intensity.addButton(self.rad_intensity_linear)
        self.rad_intensity_linear.clicked.connect(self.intensity_type_changed)

        self.label_value1 = QtWidgets.QLabel("", self)
        self.layout.addWidget(self.label_value1)
        self.txt_value1 = QtWidgets.QLineEdit(str(self.draw_stim.value1), self)
        self.layout.addWidget(self.txt_value1)
        self.txt_value1.textChanged.connect(self.start_intensity_changed)

        self.label_value2 = QtWidgets.QLabel("", self)
        self.layout.addWidget(self.label_value2)
        self.txt_value2 = QtWidgets.QLineEdit(str(self.draw_stim.value2), self)
        self.layout.addWidget(self.txt_value2)
        self.txt_value2.textChanged.connect(self.end_intensity_changed)

        dt_ms = format_milliseconds(self.dt_seconds * 1000.0)
        self.label_length = QtWidgets.QLabel(
            "Length in milliseconds (rounded to nearest multiple of selected "
            f"stim dt = {dt_ms} ms):",
            self,
        )
        self.layout.addWidget(self.label_length)
        self.txt_length = QtWidgets.QLineEdit(
            format_milliseconds(
                samples_to_milliseconds(self.draw_stim_length, self.dt_seconds)
            ),
            self,
        )
        self.layout.addWidget(self.txt_length)
        self.txt_length.textChanged.connect(self.length_changed)

        self.layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("Ok", self)
        self.btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_ok)
        self.btn_cancel = QtWidgets.QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(button_layout)
        self.intensity_type_changed()

    def intensity_type_changed(self):
        if self.rad_intensity_constant.isChecked():
            self.draw_stim.draw_stim_type = DrawStimType.Constant
            self.label_value1.setText(f"Intensity in {self.stim_units}:")
            self.label_value2.setEnabled(False)
            self.label_value2.setVisible(False)
            self.txt_value2.setEnabled(False)
            self.txt_value2.setVisible(False)
        else:
            self.draw_stim.draw_stim_type = DrawStimType.Linear
            self.label_value1.setText(f"Initial intensity in {self.stim_units}:")
            self.label_value2.setText(f"Final intensity in {self.stim_units}:")
            self.label_value2.setEnabled(True)
            self.label_value2.setVisible(True)
            self.txt_value2.setEnabled(True)
            self.txt_value2.setVisible(True)
            self.end_intensity_changed(self.txt_value2.text())
        self._update_ok_enabled()

    def _parse_intensity(self, text, input_widget: QtWidgets.QLineEdit) -> float | None:
        try:
            intensity = float(text)
            if self.stim_type != StimType.External and intensity < 0:
                raise ValueError
            input_widget.setStyleSheet("")
            return intensity
        except ValueError:
            input_widget.setStyleSheet("color: red;")
            return None

    def _update_ok_enabled(self) -> None:
        value2_ok = self._value2_valid or not self.txt_value2.isVisible()
        self.btn_ok.setEnabled(self._value1_valid and value2_ok and self._length_valid)

    def start_intensity_changed(self, text) -> None:
        intensity = self._parse_intensity(text, self.txt_value1)
        self._value1_valid = intensity is not None
        if intensity is not None:
            self.draw_stim.value1 = intensity
        self._update_ok_enabled()

    def end_intensity_changed(self, text) -> None:
        intensity = self._parse_intensity(text, self.txt_value2)
        self._value2_valid = intensity is not None
        if intensity is not None:
            self.draw_stim.value2 = intensity
        self._update_ok_enabled()

    def length_changed(self, text: str) -> None:
        try:
            duration_ms = float(text)
            if not math.isfinite(duration_ms) or duration_ms <= 0.0:
                raise ValueError
            self.draw_stim_length, _ = snap_milliseconds_to_dt(
                duration_ms, self.dt_seconds
            )
            self.txt_length.setStyleSheet("")
            self._length_valid = True
        except (ValueError, OverflowError):
            self.txt_length.setStyleSheet("color: red;")
            self._length_valid = False
        self._update_ok_enabled()
