from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
from matplotlib.ticker import MultipleLocator
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import TraceData
from neuroseq.input_designer_dialog import _set_native_minimized_position
from neuroseq.input_designer_dialog import _WheelToScrollFilter
from neuroseq.resources.analysis_plugins.epsp_amplitudes import _to_mv
from neuroseq.rheobase import bisection_converged
from neuroseq.rheobase import count_spikes
from neuroseq.rheobase import next_bracket
from neuroseq.stim_timing import samples_to_milliseconds
from neuroseq.stim_timing import snap_milliseconds_to_dt

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray
    from neuroseq.stim_editor import StimEditor


class RheobaseDialog(QtWidgets.QDialog):
    """Closed-loop search for the smallest direct stimulus that evokes a spike."""

    simulation_requested = QtCore.Signal(float)
    AMPLITUDE_AND_DURATION_STRATEGY = "Amplitude and duration"
    FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY = "Fixed amplitude, varying duration"
    FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY = "Fixed duration, varying amplitude"

    def __init__(
        self,
        editor: StimEditor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        self.editor = editor
        self.project = editor.project
        self._running = False
        self._awaiting_simulation = False
        self._phase = "idle"
        self._iteration = 0
        self._view_block_samples: int | None = None
        self._low = 0.0
        self._high = 0.0
        self._pending_amplitude = 0.0
        self._pulse_onset_seconds = 0.0
        self._duration_values_ms: list[float] = []
        self._duration_index = 0
        self._current_duration_ms = 0.0
        self._duration_low_ms = 0.0
        self._duration_high_ms = 0.0
        self._duration_stage = "idle"
        self._samples: list[tuple[float, float, int]] = []
        self._thresholds: list[tuple[float, float]] = []
        self._rheobase: float | None = None
        self._chronaxie_ms: float | None = None
        self._fixed_spiking_duration_ms: float | None = None
        self._theme = theme_module.LIGHT
        self._expanded_height: int | None = None
        self._compact_height: int | None = None
        self._logged_recovery_note = False
        self._resolution_warning_logged = False
        self.setWindowTitle("Minimal stimulus for spiking (rheobase)")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        info_form = QtWidgets.QFormLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_stim_units = QtWidgets.QLabel("-", self)
        self.lbl_stim_dt = QtWidgets.QLabel("-", self)
        self.lbl_stim_neurons = QtWidgets.QLabel("-", self)
        info_form.addRow("Stim array:", self.lbl_stim_name)
        info_form.addRow("Units:", self.lbl_stim_units)
        info_form.addRow("dt:", self.lbl_stim_dt)
        info_form.addRow("Neurons:", self.lbl_stim_neurons)
        layout.addLayout(info_form)

        self.btn_toggle_parameters = QtWidgets.QToolButton(self)
        self.btn_toggle_parameters.setText("Hide parameters")
        self.btn_toggle_parameters.setCheckable(True)
        self.btn_toggle_parameters.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_toggle_parameters.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle_parameters.setToolTip(
            "Hide the protocol parameters to keep this window compact."
        )
        self.btn_toggle_parameters.toggled.connect(self._set_parameters_hidden)
        layout.addWidget(self.btn_toggle_parameters)

        self.parameters_panel = QtWidgets.QWidget(self)
        parameters_layout = QtWidgets.QVBoxLayout(self.parameters_panel)
        parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_layout.setSpacing(6)
        form = QtWidgets.QFormLayout()
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.cmb_sweeping_strategy = QtWidgets.QComboBox(self)
        self.cmb_sweeping_strategy.addItems(
            [
                self.AMPLITUDE_AND_DURATION_STRATEGY,
                self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY,
                self.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY,
            ]
        )
        self.spn_neuron = QtWidgets.QSpinBox(self)
        self.spn_neuron.setMinimum(0)
        self.spn_neuron.setValue(0)
        self.spn_min_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_min_amplitude.setDecimals(3)
        self.spn_min_amplitude.setRange(-1e12, 1e12)
        self.spn_min_amplitude.setValue(1.0)
        self.spn_max_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_max_amplitude.setDecimals(3)
        self.spn_max_amplitude.setRange(-1e12, 1e12)
        self.spn_max_amplitude.setValue(100.0)
        self.spn_fixed_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_fixed_amplitude.setDecimals(3)
        self.spn_fixed_amplitude.setRange(-1e12, 1e12)
        self.spn_fixed_amplitude.setValue(10.0)
        self.spn_tolerance = QtWidgets.QDoubleSpinBox(self)
        self.spn_tolerance.setDecimals(3)
        self.spn_tolerance.setRange(0.000_001, 1e12)
        self.spn_tolerance.setValue(0.05)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_min_amplitude)
        amplitude_row.addWidget(self.spn_max_amplitude)
        amplitude_row.addWidget(self.spn_tolerance)
        amplitude_row.addWidget(self.lbl_units)
        fixed_amplitude_row = QtWidgets.QHBoxLayout()
        fixed_amplitude_row.addWidget(self.spn_fixed_amplitude)
        self.lbl_fixed_units = QtWidgets.QLabel(self)
        fixed_amplitude_row.addWidget(self.lbl_fixed_units)
        self.spn_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pulse_ms.setDecimals(3)
        self.spn_pulse_ms.setRange(0.001, 100_000.0)
        self.spn_pulse_ms.setValue(10.0)
        self.spn_pulse_ms.setSuffix(" ms")
        self.spn_max_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_max_pulse_ms.setDecimals(3)
        self.spn_max_pulse_ms.setRange(0.001, 100_000.0)
        self.spn_max_pulse_ms.setValue(50.0)
        self.spn_max_pulse_ms.setSuffix(" ms")
        self.spn_duration_tolerance_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_duration_tolerance_ms.setDecimals(3)
        self.spn_duration_tolerance_ms.setRange(0.001, 100_000.0)
        self.spn_duration_tolerance_ms.setValue(1.0)
        self.spn_duration_tolerance_ms.setSuffix(" ms")
        self.spn_recovery_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_recovery_ms.setDecimals(3)
        self.spn_recovery_ms.setRange(0.001, 100_000.0)
        self.spn_recovery_ms.setValue(200.0)
        self.spn_recovery_ms.setSuffix(" ms")
        self.spn_response_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_response_ms.setDecimals(3)
        self.spn_response_ms.setRange(0.001, 100_000.0)
        self.spn_response_ms.setValue(20.0)
        self.spn_response_ms.setSuffix(" ms")
        self.spn_response_ms.setToolTip(
            "Extra window kept after the pulse ends in which spikes still count. "
            "The analysed window is the pulse duration plus this margin, so it "
            "scales with the pulse rather than fixing a long tail for short pulses."
        )
        self.spn_threshold_mv = QtWidgets.QDoubleSpinBox(self)
        self.spn_threshold_mv.setDecimals(3)
        self.spn_threshold_mv.setRange(-100.0, 100.0)
        self.spn_threshold_mv.setValue(-20.0)
        self.spn_threshold_mv.setSuffix(" mV")
        self.spn_max_iterations = QtWidgets.QSpinBox(self)
        self.spn_max_iterations.setRange(1, 10_000)
        self.spn_max_iterations.setValue(25)
        self.chk_follow_input = QtWidgets.QCheckBox(
            "Follow new input on each iteration", self
        )
        self.chk_follow_input.setChecked(True)
        form.addRow("Trace to analyze:", self.cmb_trace)
        form.addRow("Sweeping strategy:", self.cmb_sweeping_strategy)
        form.addRow("Neuron ID:", self.spn_neuron)
        self.lbl_amplitude_range = QtWidgets.QLabel(
            "Pulse amplitude range/tolerance:", self
        )
        self.lbl_fixed_amplitude = QtWidgets.QLabel("Fixed pulse amplitude:", self)
        form.addRow(self.lbl_amplitude_range, amplitude_row)
        form.addRow(self.lbl_fixed_amplitude, fixed_amplitude_row)
        duration_row = QtWidgets.QHBoxLayout()
        duration_row.addWidget(self.spn_pulse_ms)
        duration_row.addWidget(self.spn_max_pulse_ms)
        duration_row.addWidget(self.spn_duration_tolerance_ms)
        self.lbl_duration_range = QtWidgets.QLabel(
            "Pulse duration range/tolerance (ms):", self
        )
        form.addRow(self.lbl_duration_range, duration_row)
        form.addRow("Recovery time (ms):", self.spn_recovery_ms)
        form.addRow("Post-pulse window (ms):", self.spn_response_ms)
        form.addRow("Spike threshold (mV):", self.spn_threshold_mv)
        form.addRow("Maximum iterations:", self.spn_max_iterations)
        form.addRow("Live view:", self.chk_follow_input)
        parameters_layout.addLayout(form)

        self.parameters_scroll = QtWidgets.QScrollArea(self)
        self.parameters_scroll.setWidget(self.parameters_panel)
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameters_scroll.setMinimumHeight(80)
        layout.addWidget(self.parameters_scroll, 1)
        self._wheel_filter = _WheelToScrollFilter(self.parameters_scroll)
        value_widgets: list[QtWidgets.QWidget] = list(
            self.parameters_panel.findChildren(QtWidgets.QAbstractSpinBox)
        )
        value_widgets += self.parameters_panel.findChildren(QtWidgets.QComboBox)
        for value_widget in value_widgets:
            value_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            value_widget.installEventFilter(self._wheel_filter)

        for spin in (
            self.spn_pulse_ms,
            self.spn_max_pulse_ms,
            self.spn_duration_tolerance_ms,
            self.spn_min_amplitude,
            self.spn_max_amplitude,
            self.spn_fixed_amplitude,
            self.spn_tolerance,
        ):
            spin.valueChanged.connect(self._on_parameter_changed)
        self.spn_max_iterations.valueChanged.connect(
            lambda _value: self._update_estimate()
        )
        self.cmb_sweeping_strategy.currentIndexChanged.connect(
            self._update_sweeping_strategy_controls
        )

        self.plot_window = QtWidgets.QDialog(self)
        self.plot_window.setWindowTitle(
            "Minimal stimulus for spiking (rheobase) — plots"
        )
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        self._plot_window_positioned = False
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.rheobase_plot = MatplotlibWidget(self.plot_window)
        self.rheobase_plot.setMinimumHeight(500)
        figure = self.rheobase_plot.getFigure()
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.18)
        self.rheobase_axes = figure.add_subplot(grid[0, 0])
        self.search_axes = figure.add_subplot(grid[1, 0])
        plot_layout.addWidget(self.rheobase_plot)
        self.plot_window.resize(720, 560)

        self.lbl_status = QtWidgets.QLabel("Rheobase: waiting for first probe", self)
        layout.addWidget(self.lbl_status)
        self.lbl_estimate = QtWidgets.QLabel("", self)
        layout.addWidget(self.lbl_estimate)
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setMaximumHeight(130)
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.btn_start = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Start", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.btn_stop = cast(
            QtWidgets.QPushButton,
            buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole),
        )
        self.btn_clear = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Clear stim array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self.editor.clear_stim_array)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.apply_theme(self._theme)
        self._update_sweeping_strategy_controls()
        self.refresh()
        self._update_estimate()

    def _set_parameters_hidden(self, hidden: bool) -> None:
        if hidden:
            self._expanded_height = self.height()
            panel_height = self.parameters_scroll.height()
            dialog_layout = self.layout()
            spacing = dialog_layout.spacing() if dialog_layout is not None else 0
            self._compact_height = max(1, self.height() - panel_height - spacing)
        self.parameters_scroll.setVisible(not hidden)
        self.btn_toggle_parameters.setText(
            "Show parameters" if hidden else "Hide parameters"
        )
        self.btn_toggle_parameters.setArrowType(
            QtCore.Qt.ArrowType.RightArrow if hidden else QtCore.Qt.ArrowType.DownArrow
        )
        self.btn_toggle_parameters.setToolTip(
            "Show the protocol parameters."
            if hidden
            else "Hide the protocol parameters to keep this window compact."
        )
        QtCore.QTimer.singleShot(0, self._resize_for_parameter_visibility)

    def _resize_for_parameter_visibility(self) -> None:
        height = (
            self._compact_height
            if self.btn_toggle_parameters.isChecked()
            else self._expanded_height
        )
        if height is not None:
            self.resize(self.width(), max(height, self.minimumSizeHint().height()))

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self.plot_window.show()
        if not self._plot_window_positioned:
            self._plot_window_positioned = True
            self.plot_window.move(self.x() + self.width() + 20, self.y())
        self.plot_window.raise_()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if self._running:
            self._finish("Routine stopped (dialog closed).")
        self.plot_window.close()
        super().closeEvent(event)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.WindowStateChange
            and self.isMinimized()
        ):
            QtCore.QTimer.singleShot(0, self._position_minimized_window)

    def _position_minimized_window(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        margin = 12
        available = screen.availableGeometry()
        minimized_height = 32
        _set_native_minimized_position(
            int(self.winId()),
            available.left() + margin,
            available.bottom() - minimized_height - margin + 1,
        )

    def _update_sweeping_strategy_controls(self) -> None:
        strategy = self.cmb_sweeping_strategy.currentText()
        fixed_amplitude = strategy == self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
        fixed_duration = strategy == self.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
        self.lbl_amplitude_range.setVisible(not fixed_amplitude)
        self.spn_min_amplitude.setVisible(not fixed_amplitude)
        self.spn_max_amplitude.setVisible(not fixed_amplitude)
        self.spn_tolerance.setVisible(not fixed_amplitude)
        self.lbl_units.setVisible(not fixed_amplitude)
        self.lbl_fixed_amplitude.setVisible(fixed_amplitude)
        self.spn_fixed_amplitude.setVisible(fixed_amplitude)
        self.lbl_fixed_units.setVisible(fixed_amplitude)
        self.spn_max_pulse_ms.setVisible(not fixed_duration)
        self.spn_duration_tolerance_ms.setVisible(not fixed_duration)
        self.lbl_duration_range.setText(
            "Pulse duration (ms):"
            if fixed_duration
            else "Pulse duration range/tolerance (ms):"
        )
        self._update_plot()
        self._update_estimate()

    def refresh(self) -> None:
        self.project = self.editor.project
        selected = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        if self.project is not None:
            self.cmb_trace.addItems(self.project.get_data_names(TraceData))
            self.cmb_trace.setCurrentText(selected)
        modifier = self.editor.current_stim_modifier
        if modifier is not None:
            stim = modifier.stim_array
            n_neurons = stim.values.shape[1]
            self.spn_neuron.setMaximum(max(0, n_neurons - 1))
            self.lbl_units.setText(stim.units)
            self.lbl_fixed_units.setText(stim.units)
            self.lbl_stim_name.setText(stim.name)
            self.lbl_stim_units.setText(stim.units or "-")
            self.lbl_stim_dt.setText(f"{stim.dt * 1000.0:g} ms")
            self.lbl_stim_neurons.setText(str(n_neurons))
        else:
            self.lbl_units.setText("")
            self.lbl_fixed_units.setText("")
            self.lbl_stim_name.setText("-")
            self.lbl_stim_units.setText("-")
            self.lbl_stim_dt.setText("-")
            self.lbl_stim_neurons.setText("-")
        self._update_plot()

    def start(self) -> None:
        self.refresh()
        if self.project is None or self.editor.current_stim_modifier is None:
            self._append_log("No stimulation array is selected.")
            return
        if not self._confirm_start_over_existing_input():
            self._append_log("Routine cancelled; existing stimulation input kept.")
            return
        if not self.cmb_trace.currentText():
            duration = 0.001
            self._running = True
            self._phase = "initialize_trace"
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._append_log(
                f"Running an initial {duration:.3f} s simulation to create "
                "trace monitors."
            )
            self._request_simulation(duration)
            return
        self._begin_search(clear_log=True)

    def _begin_search(self, *, clear_log: bool) -> None:
        strategy = self.cmb_sweeping_strategy.currentText()
        min_amplitude = self.spn_min_amplitude.value()
        max_amplitude = self.spn_max_amplitude.value()
        fixed_amplitude = strategy == self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
        fixed_duration = strategy == self.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
        if not fixed_amplitude and max_amplitude <= min_amplitude:
            self._finish("Maximum amplitude must be greater than minimum amplitude.")
            return
        if (
            not fixed_duration
            and self.spn_max_pulse_ms.value() < self.spn_pulse_ms.value()
        ):
            self._finish("Maximum pulse duration must be at least the minimum.")
            return
        self._running = True
        self._awaiting_simulation = False
        self._phase = "idle"
        self._iteration = 0
        self._view_block_samples = None
        self._low = min_amplitude
        self._high = max_amplitude
        self._pending_amplitude = max_amplitude
        self._pulse_onset_seconds = 0.0
        self._duration_values_ms = self._duration_sweep_values()
        self._duration_index = 0
        self._current_duration_ms = self.spn_pulse_ms.value()
        self._duration_low_ms = self.spn_pulse_ms.value()
        self._duration_high_ms = self.spn_max_pulse_ms.value()
        self._duration_stage = "idle"
        self._samples = []
        self._thresholds = []
        self._rheobase = None
        self._chronaxie_ms = None
        self._fixed_spiking_duration_ms = None
        self._logged_recovery_note = False
        self._resolution_warning_logged = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if clear_log:
            self.log.clear()
        self._maybe_log_duration_resolution_warning()
        self._append_log(
            f"Estimated simulations for the current settings: "
            f"~{self._estimated_probe_count()}."
        )
        self._update_plot()
        if fixed_amplitude:
            self._begin_fixed_amplitude_duration_search()
        elif fixed_duration:
            self._begin_duration_search(self.spn_pulse_ms.value(), "fixed duration")
        else:
            self._begin_strength_duration_search()

    def _duration_sweep_values(self) -> list[float]:
        return [float(self.spn_pulse_ms.value()), float(self.spn_max_pulse_ms.value())]

    def _begin_strength_duration_search(self) -> None:
        self._duration_stage = "rheobase"
        self._begin_duration_search(
            self.spn_max_pulse_ms.value(), "rheobase estimate at maximum duration"
        )

    def _begin_duration_search(self, duration_ms: float, description: str) -> None:
        min_amplitude = self.spn_min_amplitude.value()
        max_amplitude = self.spn_max_amplitude.value()
        self._current_duration_ms = self._effective_duration_ms(duration_ms)
        self._low = min_amplitude
        self._high = max_amplitude
        self._iteration = 0
        self._phase = "probe_max"
        self._append_log(
            f"Searching duration {self._current_duration_ms:.6g} ms ({description})."
        )
        self._probe(max_amplitude)

    def _begin_fixed_amplitude_duration_search(self) -> None:
        self._duration_stage = "duration_probe_max"
        self._duration_low_ms = self.spn_pulse_ms.value()
        self._duration_high_ms = self.spn_max_pulse_ms.value()
        self._probe_fixed_amplitude_duration(self._duration_high_ms, "maximum duration")

    def _probe_fixed_amplitude_duration(
        self, duration_ms: float, description: str
    ) -> None:
        self._current_duration_ms = self._effective_duration_ms(duration_ms)
        self._phase = "duration_probe"
        self._append_log(
            f"Testing fixed amplitude {self.spn_fixed_amplitude.value():.6g} "
            f"{self._unit_label()} at duration {self._current_duration_ms:.6g} ms "
            f"({description})."
        )
        self._probe(self.spn_fixed_amplitude.value())

    def stop(self) -> None:
        self._finish("Routine stopped.")

    def _require_stim(self) -> StimArray:
        modifier = self.editor.current_stim_modifier
        if modifier is None:
            raise RuntimeError("No stimulation array is selected.")
        return modifier.stim_array

    def _require_project(self) -> Project:
        if self.project is None:
            raise RuntimeError("No project is available.")
        return self.project

    def _one_sample_ms(self) -> float:
        return samples_to_milliseconds(1, self._require_stim().dt)

    def _duration_tolerance_ms(self) -> float:
        return max(self.spn_duration_tolerance_ms.value(), self._one_sample_ms())

    def _effective_duration_ms(self, duration_ms: float) -> float:
        _samples, effective_ms = snap_milliseconds_to_dt(
            duration_ms, self._require_stim().dt
        )
        return effective_ms

    def _duration_threshold_exists(self, duration_ms: float) -> bool:
        return any(
            abs(existing_duration - duration_ms) <= 1e-9
            for existing_duration, _threshold in self._thresholds
        )

    def _duration_already_sampled(self, duration_ms: float) -> bool:
        return any(
            abs(existing_duration - duration_ms) <= 1e-9
            for existing_duration, _amplitude, _spike_count in self._samples
        )

    def _maybe_log_duration_resolution_warning(self) -> None:
        if self._resolution_warning_logged:
            return
        one_sample_ms = self._one_sample_ms()
        if self.spn_duration_tolerance_ms.value() >= one_sample_ms:
            return
        self._resolution_warning_logged = True
        self._append_log(
            f"Stim dt is {one_sample_ms:.6g} ms; durations are snapped to that "
            "grid, so finer tolerances cannot be resolved."
        )

    def _probe(self, amplitude: float) -> None:
        stim = self._require_stim()
        neuron_id = self.spn_neuron.value()
        if neuron_id >= stim.values.shape[1]:
            self._finish(f"Neuron {neuron_id} is outside the stimulation array.")
            return
        recovery_n, _recovery_ms = snap_milliseconds_to_dt(
            self.spn_recovery_ms.value(), stim.dt
        )
        pulse_n, effective_ms = snap_milliseconds_to_dt(
            self._current_duration_ms, stim.dt
        )
        self._current_duration_ms = effective_ms
        tail_n, _tail_ms = snap_milliseconds_to_dt(
            self.spn_response_ms.value(), stim.dt
        )
        samples = recovery_n + pulse_n + tail_n
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        block[recovery_n : recovery_n + pulse_n, neuron_id] = amplitude
        block_start = self._append_block(block)
        self._pending_amplitude = amplitude
        self._pulse_onset_seconds = (block_start + recovery_n) * stim.dt
        if not self._logged_recovery_note:
            self._append_log(
                "Each probe is appended after the existing input with a recovery "
                "lead before the pulse so the membrane returns to rest, mirroring "
                "the STDP routine's append-and-run-forward model."
            )
            self._logged_recovery_note = True
        self._append_log(
            f"Probe {len(self._samples) + 1}: testing amplitude "
            f"{amplitude:.6g} {self._unit_label()} at "
            f"{self._current_duration_ms:.6g} ms."
        )
        self._run_to_sample(block_start + samples)

    def _append_block(self, block: np.ndarray) -> int:
        stim = self._require_stim()
        project = self._require_project()
        current_sample = int(np.ceil(project.b2_model.get_sim_time_secs() / stim.dt))
        start = max(stim.values.shape[0], current_sample)
        self.editor.apply_pattern(
            slice(0, stim.values.shape[1]), slice(start, start + block.shape[0]), block
        )
        if self.chk_follow_input.isChecked():
            if self._view_block_samples is None:
                self._view_block_samples = block.shape[0]
            block_end = start + block.shape[0]
            history_start = max(0, block_end - self._view_block_samples)
            history_stop = history_start + self._view_block_samples
            self.editor.focus_time_range(
                history_start * stim.dt, history_stop * stim.dt
            )
        return start

    def _run_to_sample(self, stop_sample: int) -> None:
        stim = self._require_stim()
        current_time = self._require_project().b2_model.get_sim_time_secs()
        run_seconds = max(stim.dt, stop_sample * stim.dt - current_time)
        self._request_simulation(run_seconds)

    def _request_simulation(self, seconds: float) -> None:
        self._awaiting_simulation = True
        self.simulation_requested.emit(seconds)

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if not self._awaiting_simulation:
            return
        self._awaiting_simulation = False
        if self._phase == "initialize_trace":
            self.refresh()
            if not self.cmb_trace.currentText():
                self._finish(
                    "The initial simulation completed, but no simulation "
                    "trace monitor became available. Add a StateMonitor to "
                    "the model and try again."
                )
                return
            self._append_log("Trace monitor created; starting the rheobase search.")
            self._begin_search(clear_log=False)
            return
        trace = self.project.get_data(self.cmb_trace.currentText(), TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return
        try:
            spike_count = self._measure_spikes(trace)
        except ValueError as exc:
            self._finish(f"Spike measurement failed: {exc}")
            return
        amplitude = self._pending_amplitude
        self._samples.append((self._current_duration_ms, amplitude, spike_count))
        self._append_log(
            f"Probe {len(self._samples)}: amplitude {amplitude:.6g} "
            f"{self._unit_label()} at {self._current_duration_ms:.6g} ms -> "
            f"{self._format_spike_count(spike_count)}."
        )
        self._update_plot()
        if self._phase == "duration_probe":
            self._handle_fixed_amplitude_duration_result(spike_count)
            return
        if self._phase == "probe_max":
            if spike_count == 0:
                self._finish(
                    "No spike at the maximum amplitude "
                    f"{self.spn_max_amplitude.value():.6g} {self._unit_label()}; "
                    f"increase the range for {self._current_duration_ms:.6g} ms."
                )
                return
            self._high = self.spn_max_amplitude.value()
            self._phase = "probe_min"
            self._probe(self.spn_min_amplitude.value())
            return
        if self._phase == "probe_min":
            if spike_count > 0:
                self._complete_duration_threshold(
                    self.spn_min_amplitude.value(),
                    "Neuron spikes already at the minimum amplitude "
                    f"{self.spn_min_amplitude.value():.6g} {self._unit_label()}; "
                    "threshold <= minimum.",
                )
                return
            self._low = self.spn_min_amplitude.value()
            self._phase = "bisect"
            self._bisect_step()
            return
        if self._phase == "bisect":
            self._low, self._high = next_bracket(
                self._low, self._high, amplitude, spike_count > 0
            )
            self._bisect_step()

    def _handle_fixed_amplitude_duration_result(self, spike_count: int) -> None:
        if self._duration_stage == "duration_probe_max":
            if spike_count == 0:
                self._update_plot()
                self._finish(self._final_fixed_amplitude_summary())
                return
            self._duration_high_ms = self._current_duration_ms
            self._duration_stage = "duration_probe_min"
            self._probe_fixed_amplitude_duration(
                self.spn_pulse_ms.value(), "minimum duration"
            )
            return
        if self._duration_stage == "duration_probe_min":
            if spike_count > 0:
                self._fixed_spiking_duration_ms = self._current_duration_ms
                self._update_plot()
                self._finish(self._final_fixed_amplitude_summary())
                return
            self._duration_low_ms = self._current_duration_ms
            self._duration_stage = "duration_bisect"
            self._bisect_fixed_amplitude_duration()
            return
        if self._duration_stage == "duration_bisect":
            if spike_count > 0:
                self._duration_high_ms = self._current_duration_ms
            else:
                self._duration_low_ms = self._current_duration_ms
            self._bisect_fixed_amplitude_duration()

    def _bisect_fixed_amplitude_duration(self) -> None:
        if (
            self._duration_high_ms - self._duration_low_ms
            <= self._duration_tolerance_ms()
        ):
            self._fixed_spiking_duration_ms = self._duration_high_ms
            self._update_plot()
            self._finish(self._final_fixed_amplitude_summary())
            return
        mid = 0.5 * (self._duration_low_ms + self._duration_high_ms)
        snapped_mid = self._effective_duration_ms(mid)
        if self._duration_already_sampled(snapped_mid):
            self._fixed_spiking_duration_ms = self._duration_high_ms
            self._update_plot()
            self._finish(self._final_fixed_amplitude_summary())
            return
        self._probe_fixed_amplitude_duration(mid, "duration bisection")

    def _bisect_step(self) -> None:
        if bisection_converged(self._low, self._high, self.spn_tolerance.value()):
            self._complete_duration_threshold(
                self._high,
                f"Threshold at {self._current_duration_ms:.6g} ms ~= "
                f"{self._high:.6g} {self._unit_label()} "
                f"(bracket [{self._low:.6g}, {self._high:.6g}]).",
            )
            return
        if self._iteration >= self.spn_max_iterations.value():
            self._complete_duration_threshold(
                self._high,
                "Maximum iterations reached; "
                f"threshold at {self._current_duration_ms:.6g} ms ~= "
                f"{self._high:.6g} {self._unit_label()}.",
            )
            return
        self._iteration += 1
        self._probe(0.5 * (self._low + self._high))

    def _complete_duration_threshold(self, threshold: float, message: str) -> None:
        self._thresholds.append((self._current_duration_ms, threshold))
        self._update_strength_duration_metrics()
        self._update_plot()
        self._append_log(message)
        strategy = self.cmb_sweeping_strategy.currentText()
        if strategy == self.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY:
            self._finish(self._final_summary())
            return
        self._handle_strength_duration_threshold()

    def _handle_strength_duration_threshold(self) -> None:
        if self._duration_stage == "rheobase":
            if self.spn_max_pulse_ms.value() == self.spn_pulse_ms.value():
                self._finish(self._final_summary())
                return
            self._duration_stage = "duration_min"
            self._begin_duration_search(
                self.spn_pulse_ms.value(), "minimum duration bound"
            )
            return
        if self._duration_stage == "duration_min":
            target = 2.0 * self._rheobase if self._rheobase is not None else None
            threshold = self._thresholds[-1][1]
            if target is None or threshold <= target:
                self._finish(self._final_summary())
                return
            self._duration_low_ms = self._current_duration_ms
            self._duration_high_ms = self.spn_max_pulse_ms.value()
            self._duration_stage = "duration_chronaxie_bisect"
            self._bisect_strength_duration()
            return
        if self._duration_stage == "duration_chronaxie_bisect":
            target = 2.0 * self._rheobase if self._rheobase is not None else None
            threshold = self._thresholds[-1][1]
            if target is not None and threshold > target:
                self._duration_low_ms = self._current_duration_ms
            else:
                self._duration_high_ms = self._current_duration_ms
            self._bisect_strength_duration()

    def _bisect_strength_duration(self) -> None:
        if (
            self._duration_high_ms - self._duration_low_ms
            <= self._duration_tolerance_ms()
        ):
            self._update_strength_duration_metrics()
            self._update_plot()
            self._finish(self._final_summary())
            return
        mid = 0.5 * (self._duration_low_ms + self._duration_high_ms)
        snapped_mid = self._effective_duration_ms(mid)
        if self._duration_threshold_exists(snapped_mid):
            self._update_strength_duration_metrics()
            self._update_plot()
            self._finish(self._final_summary())
            return
        self._begin_duration_search(mid, "duration bisection")

    def _update_strength_duration_metrics(self) -> None:
        if not self._thresholds:
            self._rheobase = None
            self._chronaxie_ms = None
            return
        sorted_thresholds = sorted(self._thresholds, key=lambda item: item[0])
        self._rheobase = min(threshold for _duration, threshold in sorted_thresholds)
        target = 2.0 * self._rheobase
        self._chronaxie_ms = None
        if len(sorted_thresholds) < 2:
            return
        for (d0, a0), (d1, a1) in zip(
            sorted_thresholds, sorted_thresholds[1:], strict=False
        ):
            if (a0 - target) == 0.0:
                self._chronaxie_ms = d0
                return
            if (a0 - target) * (a1 - target) <= 0.0 and a0 != a1:
                fraction = (target - a0) / (a1 - a0)
                self._chronaxie_ms = d0 + fraction * (d1 - d0)
                return
        first_duration, first_threshold = sorted_thresholds[0]
        if first_threshold <= target:
            self._chronaxie_ms = first_duration

    def _final_summary(self) -> str:
        if self._rheobase is None:
            return "Duration sweep finished without a rheobase estimate."
        summary = f"Duration sweep finished; rheobase ~= {self._rheobase:.6g} {self._unit_label()}"
        if self._chronaxie_ms is not None:
            summary += f", chronaxie ~= {self._chronaxie_ms:.6g} ms"
        else:
            summary += "; chronaxie is outside the swept duration range"
        return summary + "."

    def _final_fixed_amplitude_summary(self) -> str:
        amplitude = self.spn_fixed_amplitude.value()
        units = self._unit_label()
        if self._fixed_spiking_duration_ms is None:
            return (
                f"Fixed-amplitude duration sweep finished; no spike at "
                f"{amplitude:.6g} {units} in the swept duration range."
            )
        return (
            "Fixed-amplitude duration sweep finished; first spiking duration "
            f"at {amplitude:.6g} {units} is "
            f"{self._fixed_spiking_duration_ms:.6g} ms."
        )

    def _measure_spikes(self, trace: TraceData) -> int:
        neuron_id = self.spn_neuron.value()
        matches = np.flatnonzero(trace.indices == neuron_id)
        if matches.size == 0:
            raise ValueError(f"Neuron {neuron_id} is not recorded in the trace.")
        trace_mv = _to_mv(trace.traces[int(matches[0])], trace.units)
        dt_ms = 1000.0 * trace.dt
        onset_ms = 1000.0 * self._pulse_onset_seconds
        # Count spikes over the pulse plus a fixed post-pulse margin, so the
        # window tracks the pulse instead of trailing a long fixed tail after a
        # short pulse (which could catch unrelated rebound activity).
        window_ms = self._current_duration_ms + self.spn_response_ms.value()
        start = max(0, int(np.floor(onset_ms / dt_ms)))
        stop = min(
            trace_mv.size,
            int(np.ceil((onset_ms + window_ms) / dt_ms)) + 1,
        )
        return count_spikes(trace_mv[start:stop], self.spn_threshold_mv.value())

    def _update_plot(self) -> None:
        self.rheobase_axes.clear()
        self.search_axes.clear()
        if self._thresholds:
            threshold_pairs = sorted(self._thresholds, key=lambda item: item[0])
            threshold_durations = np.asarray(
                [duration for duration, _threshold in threshold_pairs]
            )
            threshold_values = np.asarray(
                [threshold for _duration, threshold in threshold_pairs]
            )
            self.rheobase_axes.plot(
                threshold_durations,
                threshold_values,
                color="#000000",
                linewidth=2.5,
                marker="o",
                label="Threshold",
            )
        if self._samples:
            durations = np.asarray([sample[0] for sample in self._samples])
            amplitudes = np.asarray([sample[1] for sample in self._samples])
            spike_counts = np.asarray([sample[2] for sample in self._samples])
            spiking = spike_counts > 0
            if np.any(~spiking):
                self.rheobase_axes.scatter(
                    durations[~spiking],
                    amplitudes[~spiking],
                    facecolors="none",
                    edgecolors="#1f77b4",
                    linewidths=1.5,
                    label="No spike",
                )
            if np.any(spiking):
                self.rheobase_axes.scatter(
                    durations[spiking],
                    amplitudes[spiking],
                    color="#1f77b4",
                    label="Spike",
                )
            probe_x = np.arange(1, len(self._samples) + 1)
            probe_y = amplitudes
            self.search_axes.plot(
                probe_x, probe_y, color="#e28743", linewidth=2, marker="o"
            )
        if (
            self.cmb_sweeping_strategy.currentText()
            == self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
            and self._samples
        ):
            fixed_amplitude = self.spn_fixed_amplitude.value()
            self.rheobase_axes.axhline(
                fixed_amplitude, color="#000000", linewidth=2, linestyle="--"
            )
            if self._fixed_spiking_duration_ms is not None:
                self.rheobase_axes.axvline(
                    self._fixed_spiking_duration_ms,
                    color="#bd2c2c",
                    linewidth=2,
                    linestyle="--",
                )
                self.rheobase_axes.annotate(
                    "first spike",
                    xy=(self._fixed_spiking_duration_ms, 0.02),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    color="#bd2c2c",
                )
        if self._rheobase is not None:
            self.rheobase_axes.axhline(
                self._rheobase, color="#3c9300", linewidth=2, linestyle="--"
            )
            self.rheobase_axes.axhline(
                2.0 * self._rheobase,
                color="#bd2c2c",
                linewidth=2,
                linestyle="--",
            )
            if self._chronaxie_ms is not None:
                self.rheobase_axes.axvline(
                    self._chronaxie_ms, color="#bd2c2c", linewidth=2, linestyle="--"
                )
            self.rheobase_axes.annotate(
                "rheobase",
                xy=(0.02, self._rheobase),
                xycoords=("axes fraction", "data"),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="center",
                color="#3c9300",
            )
            self.rheobase_axes.annotate(
                "2x rheobase",
                xy=(0.02, 2.0 * self._rheobase),
                xycoords=("axes fraction", "data"),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="center",
                color="#bd2c2c",
            )
            if self._chronaxie_ms is not None:
                self.rheobase_axes.annotate(
                    "chronaxie",
                    xy=(self._chronaxie_ms, 0.02),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    color="#bd2c2c",
                )
        if self._running and self._phase == "bisect":
            self.search_axes.axhspan(self._low, self._high, color="#2ca02c", alpha=0.12)
        self._update_status_label()
        self._style_plot_axes()
        self.rheobase_plot.getFigure().subplots_adjust(
            left=0.12, right=0.98, bottom=0.11, top=0.95, hspace=0.28
        )
        self.rheobase_plot.draw()

    def _update_status_label(self) -> None:
        units = self._unit_label()
        if (
            self.cmb_sweeping_strategy.currentText()
            == self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
        ):
            amplitude = self.spn_fixed_amplitude.value()
            if self._fixed_spiking_duration_ms is not None:
                self.lbl_status.setText(
                    f"First spiking duration: {self._fixed_spiking_duration_ms:.6g} ms "
                    f"at {amplitude:.6g} {units}"
                )
            elif self._samples:
                self.lbl_status.setText(
                    f"Fixed-amplitude sweep: no spike yet at {amplitude:.6g} {units}"
                )
            else:
                self.lbl_status.setText(
                    "Fixed-amplitude sweep: waiting for first probe"
                )
            return
        if self._rheobase is not None:
            text = f"Rheobase ~= {self._rheobase:.6g} {units}"
            if self._chronaxie_ms is not None:
                text += f"; chronaxie ~= {self._chronaxie_ms:.6g} ms"
            elif len(self._thresholds) > 1:
                text += "; chronaxie outside swept range"
            text += f" (current bracket width {self._high - self._low:.6g})"
            self.lbl_status.setText(text)
        elif self._samples:
            self.lbl_status.setText(
                f"Rheobase: bracket [{self._low:.6g}, {self._high:.6g}] {units}"
            )
        else:
            self.lbl_status.setText("Rheobase: waiting for first probe")

    def _style_plot_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        units = self._unit_label()
        for axes, ylabel in (
            (self.rheobase_axes, f"Stimulus strength\n({units})"),
            (self.search_axes, f"Amplitude ({units})"),
        ):
            axes.set_facecolor(face)
            axes.set_ylabel(ylabel, color=text)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        self.rheobase_axes.set_title("")
        self.rheobase_axes.set_xlabel("Stimulus duration (ms)", color=text)
        self.search_axes.set_title("")
        self.search_axes.set_xlabel("Probe", color=text)
        self.search_axes.xaxis.set_major_locator(MultipleLocator(1))
        self.rheobase_plot.getFigure().set_facecolor(face)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme_module.normalize_theme(theme)
        self._style_plot_axes()
        self.rheobase_plot.draw()

    def simulation_failed(self, message: str) -> None:
        if self._running and self._awaiting_simulation:
            self._finish(f"Simulation failed: {message}")

    def _finish(self, message: str) -> None:
        self._running = False
        self._awaiting_simulation = False
        self._phase = "idle"
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(message)

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _unit_label(self) -> str:
        return self.lbl_units.text() or "a.u."

    @staticmethod
    def _format_spike_count(spike_count: int) -> str:
        if spike_count == 0:
            return "no spike"
        if spike_count == 1:
            return "1 spike"
        return f"{spike_count} spikes"

    def _on_parameter_changed(self, _value: float) -> None:
        self._update_plot()
        self._update_estimate()

    @staticmethod
    def _bisection_steps(span: float, tolerance: float) -> int:
        """Number of halvings to bring ``span`` down to ``tolerance``."""
        if span <= 0.0 or tolerance <= 0.0:
            return 0
        steps = 0
        width = span
        while width > tolerance and steps < 10_000:
            width *= 0.5
            steps += 1
        return steps

    def _amplitude_search_estimate(self) -> int:
        span = self.spn_max_amplitude.value() - self.spn_min_amplitude.value()
        bisections = min(
            self.spn_max_iterations.value(),
            self._bisection_steps(span, self.spn_tolerance.value()),
        )
        # Two bracket probes (max, then min) plus the bisection probes.
        return 2 + bisections

    def _estimated_probe_count(self) -> int:
        """Rough number of simulations the current settings will run."""
        strategy = self.cmb_sweeping_strategy.currentText()
        if strategy == self.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY:
            return self._amplitude_search_estimate()
        duration_span = self.spn_max_pulse_ms.value() - self.spn_pulse_ms.value()
        duration_bisections = self._bisection_steps(
            duration_span, self.spn_duration_tolerance_ms.value()
        )
        if strategy == self.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY:
            # Two bracket probes plus the duration bisection probes.
            return 2 + duration_bisections
        # "Amplitude and duration" runs a full amplitude search at each duration
        # stage (rheobase at the max duration, the min-duration bound, then the
        # chronaxie bisection), so multiply the two searches out.
        duration_stages = 2 + duration_bisections
        return duration_stages * self._amplitude_search_estimate()

    def _update_estimate(self) -> None:
        self.lbl_estimate.setText(
            "Estimated simulations for the current settings: "
            f"~{self._estimated_probe_count()}"
        )

    def _nonempty_stim_array_names(self) -> list[str]:
        project = self.project
        if project is None:
            return []
        return [
            name
            for name, stim in project.stim_arrays.items()
            if bool(np.any(stim.values))
        ]

    def _prompt_existing_input_choice(self, names: list[str]) -> str:
        """Ask what to do about non-empty stim arrays; return the chosen action."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Existing stimulation input")
        listed = "\n".join(f"  • {name}" for name in names)
        box.setText(
            "The routine appends its probe pulses after any existing input, which "
            "lengthens every simulation. These stimulation arrays already contain "
            f"input:\n\n{listed}"
        )
        box.setInformativeText(
            "Clear all stimulation arrays first, append after the existing input, "
            "or cancel."
        )
        clear_button = box.addButton(
            "Clear all stim arrays", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton("Append anyway", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_button or clicked is None:
            return "cancel"
        if clicked is clear_button:
            return "clear"
        return "append"

    def _confirm_start_over_existing_input(self) -> bool:
        """Warn about non-empty stim arrays; return whether to proceed."""
        names = self._nonempty_stim_array_names()
        if not names:
            return True
        choice = self._prompt_existing_input_choice(names)
        if choice == "cancel":
            return False
        if choice == "clear":
            self.editor.clear_all_stim_arrays()
            self.refresh()
            self._append_log("Cleared all stimulation arrays before starting.")
        return True
