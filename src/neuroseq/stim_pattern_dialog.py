from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from neuroseq.stdp_synaptic_tuning_dialog import _SelfRaisingDialog
from neuroseq.stim import DrawStimType
from neuroseq.stim_patterns import generate_fuer_elise
from neuroseq.stim_patterns import generate_per_neuron
from neuroseq.stim_patterns import generate_ramp
from neuroseq.stim_patterns import generate_sequential
from neuroseq.stim_patterns import generate_simultaneous
from neuroseq.stim_patterns import is_finite_number
from neuroseq.stim_patterns import parse_neuron_indices

if TYPE_CHECKING:
    from neuroseq.stim_editor import StimEditor


_DEFAULT_STATE: dict[str, float | int | str | bool] = {
    "scope_specific": False,
    "neuron_indices": "",
    "pulse_ms": 100.0,
    "ipi_ms": 18.0,
    "magnitude": 1.0,
    "value1": 1.0,
    "value2": 2.0,
    "repeats": 1,
    "pulse_repetitions": 1,
    "start_time_ms": 0.0,
}


@dataclass(slots=True)
class StimPatternContext:
    stim_name: str
    units: str
    dt_seconds: float
    n_neurons: int


class StimPatternDialog(_SelfRaisingDialog):
    """Non-modal dialog, so no Qt parent (see _SelfRaisingDialog's
    docstring): a parent sets WM_TRANSIENT_FOR, and on X11/KWin a transient
    dialog can end up sharing its owner's "raise group" so clicking it
    never brings it above another sibling popup."""

    def __init__(self, editor: StimEditor) -> None:
        super().__init__()
        self.editor = editor
        self._context: StimPatternContext | None = None
        self.setModal(False)
        self.setWindowTitle("Design static input")
        self.resize(420, 480)

        layout = QtWidgets.QVBoxLayout(self)

        target_form = QtWidgets.QFormLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_units = QtWidgets.QLabel("-", self)
        self.lbl_dt_ms = QtWidgets.QLabel("-", self)
        target_form.addRow("Input array:", self.lbl_stim_name)
        target_form.addRow("Units:", self.lbl_units)
        target_form.addRow("dt:", self.lbl_dt_ms)
        layout.addLayout(target_form)

        scope_group = QtWidgets.QGroupBox("Scope", self)
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        self.radio_scope_all = QtWidgets.QRadioButton("All neurons", self)
        self.radio_scope_specific = QtWidgets.QRadioButton("Specific neurons", self)
        self.radio_scope_all.setChecked(True)
        scope_layout.addWidget(self.radio_scope_all)
        specific_row = QtWidgets.QHBoxLayout()
        specific_row.addWidget(self.radio_scope_specific)
        self.edit_neuron_indices = QtWidgets.QLineEdit(self)
        self.edit_neuron_indices.setPlaceholderText("e.g. 0, 2, 5-8")
        specific_row.addWidget(self.edit_neuron_indices, 1)
        scope_layout.addLayout(specific_row)
        self.lbl_neuron_hint = QtWidgets.QLabel(
            "Comma-separated indices and ranges (e.g. 0, 2, 5-8).", self
        )
        self.lbl_neuron_hint.setWordWrap(True)
        scope_layout.addWidget(self.lbl_neuron_hint)
        layout.addWidget(scope_group)

        pattern_group = QtWidgets.QGroupBox("Pattern type", self)
        pattern_layout = QtWidgets.QVBoxLayout(pattern_group)
        self.radio_sequential = QtWidgets.QRadioButton("Sequential", self)
        self.radio_simultaneous = QtWidgets.QRadioButton("Simultaneous", self)
        self.radio_fuer_elise = QtWidgets.QRadioButton("Für Elise", self)
        self.radio_constant = QtWidgets.QRadioButton("Constant", self)
        self.radio_linear = QtWidgets.QRadioButton("Linear gradient", self)
        self.radio_ramp = QtWidgets.QRadioButton("Ramp input", self)
        for widget in [
            self.radio_sequential,
            self.radio_simultaneous,
            self.radio_fuer_elise,
            self.radio_constant,
            self.radio_linear,
            self.radio_ramp,
        ]:
            pattern_layout.addWidget(widget)
        layout.addWidget(pattern_group)

        self.parameters_form = QtWidgets.QFormLayout()
        self.spin_pulse_ms = self._make_double_spin(100.0)
        self.spin_ipi_ms = self._make_double_spin(18.0)
        self.spin_magnitude = self._make_double_spin(1.0)
        self.spin_value1 = self._make_double_spin(1.0)
        self.spin_value2 = self._make_double_spin(2.0)
        self.spin_repeats = QtWidgets.QSpinBox(self)
        self.spin_repeats.setRange(0, 10_000)
        self.spin_repeats.setValue(1)
        self.spin_pulse_repetitions = QtWidgets.QSpinBox(self)
        self.spin_pulse_repetitions.setRange(0, 10_000)
        self.spin_pulse_repetitions.setValue(1)
        self.spin_start_time_ms = self._make_double_spin(0.0, minimum=0.0)

        self.lbl_magnitude = QtWidgets.QLabel("Magnitude:", self)
        self.lbl_value1 = QtWidgets.QLabel("Initial magnitude:", self)
        self.lbl_value2 = QtWidgets.QLabel("Final magnitude:", self)
        self.lbl_repeats = QtWidgets.QLabel("Repeats:", self)
        self.lbl_pulse_repetitions = QtWidgets.QLabel("Pulse repetition:", self)

        self.parameters_form.addRow("Pulse duration (ms):", self.spin_pulse_ms)
        self.parameters_form.addRow(
            "Inter-pulse interval (ms):",
            self.spin_ipi_ms,
        )
        self.parameters_form.addRow(self.lbl_magnitude, self.spin_magnitude)
        self.parameters_form.addRow(self.lbl_value1, self.spin_value1)
        self.parameters_form.addRow(self.lbl_value2, self.spin_value2)
        self.parameters_form.addRow(self.lbl_repeats, self.spin_repeats)
        self.parameters_form.addRow(
            self.lbl_pulse_repetitions, self.spin_pulse_repetitions
        )
        self.parameters_form.addRow("Start time (ms):", self.spin_start_time_ms)
        layout.addLayout(self.parameters_form)
        layout.addStretch(1)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply", self)
        self.btn_reset_start = QtWidgets.QPushButton("Reset start time", self)
        self.btn_clear = QtWidgets.QPushButton("Clear input array", self)
        self.btn_close = QtWidgets.QPushButton("Close", self)
        button_layout.addWidget(self.btn_apply)
        button_layout.addWidget(self.btn_reset_start)
        button_layout.addWidget(self.btn_clear)
        button_layout.addWidget(self.btn_close)
        layout.addLayout(button_layout)

        self.radio_scope_all.toggled.connect(self._scope_changed)
        self.radio_scope_specific.toggled.connect(self._scope_changed)
        for button in [
            self.radio_sequential,
            self.radio_simultaneous,
            self.radio_fuer_elise,
            self.radio_constant,
            self.radio_linear,
            self.radio_ramp,
        ]:
            button.toggled.connect(self._pattern_changed)
        self.btn_apply.clicked.connect(self.apply_pattern)
        self.btn_reset_start.clicked.connect(self.reset_start_time)
        self.btn_clear.clicked.connect(self.editor.clear_stim_array)
        self.btn_close.clicked.connect(self.hide)

        self._scope_changed()

    def _make_double_spin(
        self,
        value: float,
        *,
        minimum: float = -1_000_000_000.0,
        maximum: float = 1_000_000_000.0,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setDecimals(6)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def set_context(self, context: StimPatternContext) -> None:
        self._context = context
        self.lbl_stim_name.setText(context.stim_name)
        self.lbl_units.setText(context.units)
        self.lbl_dt_ms.setText(f"{context.dt_seconds * 1000.0:g} ms")
        self.lbl_neuron_hint.setText(
            "Comma-separated indices and ranges (e.g. 0, 2, 5-8); "
            f"valid range 0–{max(0, context.n_neurons - 1)}."
        )
        if self.radio_scope_specific.isChecked():
            self.radio_constant.setChecked(True)
        else:
            self.radio_sequential.setChecked(True)
        self._scope_changed()

    def capture_state(self) -> dict[str, float | int | str | bool]:
        """The user-entered values, for stashing on the active project (see
        Project.static_input_state). The pattern radio isn't included:
        set_context() re-picks it for the scope on every stim selection."""
        return {
            "scope_specific": self.radio_scope_specific.isChecked(),
            "neuron_indices": self.edit_neuron_indices.text(),
            "pulse_ms": float(self.spin_pulse_ms.value()),
            "ipi_ms": float(self.spin_ipi_ms.value()),
            "magnitude": float(self.spin_magnitude.value()),
            "value1": float(self.spin_value1.value()),
            "value2": float(self.spin_value2.value()),
            "repeats": int(self.spin_repeats.value()),
            "pulse_repetitions": int(self.spin_pulse_repetitions.value()),
            "start_time_ms": float(self.spin_start_time_ms.value()),
        }

    def restore_state(self, state: dict[str, float | int | str | bool] | None) -> None:
        """Load values saved by capture_state(); None restores the defaults
        a freshly opened dialog has (a project that never used it)."""
        values = {**_DEFAULT_STATE, **(state or {})}
        if values["scope_specific"]:
            self.radio_scope_specific.setChecked(True)
        else:
            self.radio_scope_all.setChecked(True)
        self.edit_neuron_indices.setText(str(values["neuron_indices"]))
        self.spin_pulse_ms.setValue(float(values["pulse_ms"]))
        self.spin_ipi_ms.setValue(float(values["ipi_ms"]))
        self.spin_magnitude.setValue(float(values["magnitude"]))
        self.spin_value1.setValue(float(values["value1"]))
        self.spin_value2.setValue(float(values["value2"]))
        self.spin_repeats.setValue(int(values["repeats"]))
        self.spin_pulse_repetitions.setValue(int(values["pulse_repetitions"]))
        self.spin_start_time_ms.setValue(float(values["start_time_ms"]))

    def reset_start_time(self) -> None:
        self.spin_start_time_ms.setValue(0.0)

    def _scope_changed(self) -> None:
        is_all = self.radio_scope_all.isChecked()
        self.edit_neuron_indices.setVisible(not is_all)
        self.lbl_neuron_hint.setVisible(not is_all)
        for button in [
            self.radio_simultaneous,
            self.radio_fuer_elise,
        ]:
            button.setVisible(is_all)
        for button in [
            self.radio_constant,
            self.radio_linear,
        ]:
            button.setVisible(not is_all)
        # Sequential and Ramp both live in both scopes (#582).
        self.radio_sequential.setVisible(True)
        self.radio_ramp.setVisible(True)
        all_options = [
            self.radio_sequential,
            self.radio_simultaneous,
            self.radio_fuer_elise,
            self.radio_ramp,
        ]
        specific_options = [
            self.radio_sequential,
            self.radio_constant,
            self.radio_linear,
            self.radio_ramp,
        ]
        if is_all and not any(b.isChecked() for b in all_options):
            self.radio_sequential.setChecked(True)
        if not is_all and not any(b.isChecked() for b in specific_options):
            self.radio_constant.setChecked(True)
        self._pattern_changed()

    def _set_row_visible(
        self,
        label_widget: QtWidgets.QWidget,
        field_widget: QtWidgets.QWidget,
        visible: bool,
    ) -> None:
        label_widget.setVisible(visible)
        field_widget.setVisible(visible)

    def _pattern_changed(self) -> None:
        is_all = self.radio_scope_all.isChecked()
        is_specific = not is_all
        is_ramp = self.radio_ramp.isChecked()
        # Every pattern uses either a single magnitude or a start/end value
        # pair, never both/neither - so "single magnitude" is just "not a
        # value pair" (Ramp, and Linear gradient in specific scope).
        uses_value_pair = is_ramp or (is_specific and self.radio_linear.isChecked())
        uses_single_magnitude = not uses_value_pair
        # generate_sequential() has no repeats concept in either scope.
        uses_repeats = not self.radio_sequential.isChecked()
        uses_pulse_repetitions = is_all and self.radio_fuer_elise.isChecked()

        self._set_row_visible(
            self.lbl_magnitude, self.spin_magnitude, uses_single_magnitude
        )
        self._set_row_visible(self.lbl_value1, self.spin_value1, uses_value_pair)
        self._set_row_visible(self.lbl_value2, self.spin_value2, uses_value_pair)
        self._set_row_visible(self.lbl_repeats, self.spin_repeats, uses_repeats)
        self._set_row_visible(
            self.lbl_pulse_repetitions,
            self.spin_pulse_repetitions,
            uses_pulse_repetitions,
        )

        if uses_value_pair:
            self.lbl_value1.setText("Initial magnitude:")
            self.lbl_value2.setText("Final magnitude:")

    def _validate_inputs(self) -> bool:
        pulse_ms = float(self.spin_pulse_ms.value())
        ipi_ms = float(self.spin_ipi_ms.value())
        repeats = int(self.spin_repeats.value())
        pulse_repetitions = int(self.spin_pulse_repetitions.value())
        magnitude_values = [
            float(self.spin_magnitude.value()),
            float(self.spin_value1.value()),
            float(self.spin_value2.value()),
            float(self.spin_start_time_ms.value()),
        ]
        if pulse_ms <= 0.0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid pattern",
                "Pulse duration must be positive.",
            )
            return False
        if ipi_ms < 0.0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid pattern",
                "Inter-pulse interval must be non-negative.",
            )
            return False
        if self.lbl_repeats.isVisible() and repeats < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid pattern",
                "Repeats must be at least 1.",
            )
            return False
        if self.lbl_pulse_repetitions.isVisible() and pulse_repetitions < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid pattern",
                "Pulse repetition must be at least 1.",
            )
            return False
        if not all(is_finite_number(value) for value in magnitude_values):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid pattern",
                "All numeric inputs must be finite.",
            )
            return False
        return True

    def _current_profile(self) -> DrawStimType:
        if self.radio_constant.isChecked():
            return DrawStimType.Constant
        return DrawStimType.Linear

    def apply_pattern(self) -> None:
        if self._context is None or not self._validate_inputs():
            return
        pulse_ms = float(self.spin_pulse_ms.value())
        ipi_ms = float(self.spin_ipi_ms.value())
        repeats = int(self.spin_repeats.value())
        dt_seconds = self._context.dt_seconds
        n_neurons = self._context.n_neurons

        neuron_indices: list[int] | None = None
        if self.radio_scope_specific.isChecked():
            try:
                neuron_indices = parse_neuron_indices(
                    self.edit_neuron_indices.text(), n_neurons
                )
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid pattern", str(exc))
                return

        try:
            if self.radio_ramp.isChecked():
                pattern = generate_ramp(
                    n_neurons=n_neurons,
                    ramp_ms=pulse_ms,
                    ipi_ms=ipi_ms,
                    start_magnitude=float(self.spin_value1.value()),
                    end_magnitude=float(self.spin_value2.value()),
                    repeats=repeats,
                    dt_seconds=dt_seconds,
                    neuron_index=neuron_indices,
                )
            elif self.radio_sequential.isChecked():
                pattern = generate_sequential(
                    n_neurons=n_neurons,
                    pulse_ms=pulse_ms,
                    ipi_ms=ipi_ms,
                    magnitude=float(self.spin_magnitude.value()),
                    dt_seconds=dt_seconds,
                    neuron_index=neuron_indices,
                )
            elif self.radio_scope_all.isChecked():
                if self.radio_simultaneous.isChecked():
                    pattern = generate_simultaneous(
                        n_neurons=n_neurons,
                        pulse_ms=pulse_ms,
                        ipi_ms=ipi_ms,
                        magnitude=float(self.spin_magnitude.value()),
                        repeats=repeats,
                        dt_seconds=dt_seconds,
                    )
                else:
                    pattern = generate_fuer_elise(
                        n_neurons=n_neurons,
                        pulse_ms=pulse_ms,
                        ipi_ms=ipi_ms,
                        magnitude=float(self.spin_magnitude.value()),
                        repeats=repeats,
                        dt_seconds=dt_seconds,
                        pulse_repetitions=int(self.spin_pulse_repetitions.value()),
                    )
            else:
                assert neuron_indices is not None
                pattern = generate_per_neuron(
                    neuron_index=neuron_indices,
                    n_neurons=n_neurons,
                    pulse_ms=pulse_ms,
                    ipi_ms=ipi_ms,
                    profile=self._current_profile(),
                    value1=float(self.spin_value1.value())
                    if self.radio_linear.isChecked()
                    else float(self.spin_magnitude.value()),
                    value2=float(self.spin_value2.value())
                    if self.radio_linear.isChecked()
                    else 0.0,
                    repeats=repeats,
                    dt_seconds=dt_seconds,
                )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid pattern", str(exc))
            return

        start_samples = int(
            round(self.spin_start_time_ms.value() / 1000.0 / dt_seconds)
        )
        end_samples = start_samples + pattern.shape[0]
        # generate_ramp/generate_per_neuron fill unselected columns with zeros,
        # so writing the full neuron range would erase other neurons' existing
        # stimuli in this window. Restrict the write to the selected neurons.
        if neuron_indices is not None:
            neuron_target: slice | list[int] = neuron_indices
            values_to_apply = pattern[:, neuron_indices]
        else:
            neuron_target = slice(0, n_neurons)
            values_to_apply = pattern
        self.editor.apply_pattern(
            neuron_target,
            slice(start_samples, end_samples),
            values_to_apply,
        )
        self.spin_start_time_ms.setValue(end_samples * dt_seconds * 1000.0)
