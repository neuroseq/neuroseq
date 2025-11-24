from qtpy import QtWidgets

from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim_widget import StimType


class StimIntensityDialog(QtWidgets.QDialog):
    def __init__(
        self,
        draw_stim: DrawStim,
        stim_units: str,
        stim_type: StimType,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stimulus intensity")
        self.draw_stim = draw_stim
        self.stim_units = stim_units
        self.stim_type = stim_type

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
        # DrawStimType.Linear
        self.rad_intensity_linear = QtWidgets.QRadioButton("Linear gradient", self)
        self.rad_intensity_linear.setChecked(
            self.draw_stim.draw_stim_type == DrawStimType.Linear
        )
        self.layout.addWidget(self.rad_intensity_linear)
        self.rad_group_intensity.addButton(self.rad_intensity_linear)
        self.rad_intensity_linear.clicked.connect(self.intensity_type_changed)
        # DrawStimType.Logarithmic
        self.rad_intensity_log = QtWidgets.QRadioButton("Logarithmic gradient", self)
        self.rad_intensity_log.setChecked(
            self.draw_stim.draw_stim_type == DrawStimType.Logarithmic
        )
        self.layout.addWidget(self.rad_intensity_log)
        self.rad_group_intensity.addButton(self.rad_intensity_log)
        self.rad_intensity_log.clicked.connect(self.intensity_type_changed)
        # DrawStimType.Random
        self.rad_intensity_random = QtWidgets.QRadioButton("Random", self)
        self.rad_intensity_random.setChecked(
            self.draw_stim.draw_stim_type == DrawStimType.Random
        )
        self.layout.addWidget(self.rad_intensity_random)
        self.rad_group_intensity.addButton(self.rad_intensity_random)
        self.rad_intensity_random.clicked.connect(self.intensity_type_changed)

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
            self.label_value1.setText(f"Initial intensity in {self.stim_units}:")
            self.label_value2.setText(f"Final intensity in {self.stim_units}:")
            self.label_value2.setEnabled(True)
            self.label_value2.setVisible(True)
            self.txt_value2.setEnabled(True)
            self.txt_value2.setVisible(True)
            if self.rad_intensity_linear.isChecked():
                self.draw_stim.draw_stim_type = DrawStimType.Linear
            elif self.rad_intensity_log.isChecked():
                self.draw_stim.draw_stim_type = DrawStimType.Logarithmic
            elif self.rad_intensity_random.isChecked():
                self.draw_stim.draw_stim_type = DrawStimType.Random
                self.label_value1.setText(f"Minimum intensity in {self.stim_units}:")
                self.label_value2.setText(f"Maximum intensity in {self.stim_units}:")
            self.end_intensity_changed(self.txt_value2.text())

    def _parse_intensity(self, text, input_widget: QtWidgets.QLineEdit) -> float | None:
        try:
            intensity = float(text)
            if (
                self.stim_type != StimType.External
                and intensity < 0
                or self.draw_stim.draw_stim_type == DrawStimType.Logarithmic
                and intensity <= 0
            ):
                raise ValueError
            input_widget.setStyleSheet("")
            return intensity
        except ValueError:
            input_widget.setStyleSheet("color: red;")
            self.btn_ok.setEnabled(False)
            return None

    def start_intensity_changed(self, text) -> None:
        intensity = self._parse_intensity(text, self.txt_value1)
        if intensity is not None:
            self.draw_stim.value1 = intensity
            self.btn_ok.setEnabled(True)

    def end_intensity_changed(self, text) -> None:
        intensity = self._parse_intensity(text, self.txt_value2)
        if intensity is not None:
            self.draw_stim.value2 = intensity
            self.btn_ok.setEnabled(True)
