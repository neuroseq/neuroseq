from __future__ import annotations

from dataclasses import dataclass
from math import copysign
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
from neuroseq.qt_helpers import SelfRaisingDialog
from neuroseq.qt_helpers import WheelToScrollFilter
from neuroseq.qt_helpers import set_native_minimized_position
from neuroseq.qt_helpers import wrap_in_scroll_area
from neuroseq.stimulation.dynamic_input.dialog_helpers import confirm_start_routine
from neuroseq.stimulation.dynamic_input.dynamic_input_blocks import RoutineArrayCursor
from neuroseq.stimulation.dynamic_input.dynamic_input_blocks import run_to_sample
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import EPSPMeasurement
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import measure_epsp
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import RoutineBackend
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import SimulationHost
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import backend_property
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import snap_milliseconds_to_dt

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim_editor import StimEditor


class StdpSynapticTuningBackend(RoutineBackend):
    """Lifecycle backend for the synaptic tuning dialog.

    Owns the Qt-free EPSP-targeting decision graph; the dialog supplies
    rendering, trace measurement, and append-block callbacks.
    """

    def __init__(self, host: SimulationHost) -> None:
        super().__init__(host)
        self.baseline_epsp: float | None = None
        self.epsp_percentages: list[float] = []
        self.pairing_times_ms: list[float] = []
        self.adaptive_pairing_magnitude_ms = 0.0
        self.probe_pulse_times: list[float] = []
        self.iteration = 0
        self.resolution_warning_logged = False
        self.config: StdpSynapticTuningRunConfig | None = None

    def begin_protocol(self, config: StdpSynapticTuningRunConfig) -> None:
        self.config = config
        self.begin()
        self.phase = "idle"
        self.iteration = 0
        self.baseline_epsp = None
        self.epsp_percentages = []
        self.pairing_times_ms = []
        self.adaptive_pairing_magnitude_ms = config.initial_pairing_time_ms
        self.resolution_warning_logged = False
        self._measurement_changed()
        self.probe()

    def _cfg(self) -> StdpSynapticTuningRunConfig:
        if self.config is None:
            raise RuntimeError("STDP tuning has not been configured")
        return self.config

    def probe(self) -> None:
        c = self._cfg()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(c.pulse_ms, c.stim_dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(
            StdpSynapticTuningDialog.LEAD_MS, c.stim_dt
        )
        period_n, _period_ms = snap_milliseconds_to_dt(
            StdpSynapticTuningDialog.PROBE_PERIOD_MS,
            c.stim_dt,
            minimum_samples=pulse_n + 1,
        )
        tail_n, _tail_ms = snap_milliseconds_to_dt(
            StdpSynapticTuningDialog.RESPONSE_TAIL_MS, c.stim_dt
        )
        samples = (
            lead_n
            + (StdpSynapticTuningDialog.PROBE_PULSES - 1) * period_n
            + pulse_n
            + tail_n
        )
        block = np.zeros((samples, c.n_neurons), dtype=np.float32)
        pulse_offsets = [
            lead_n + index * period_n
            for index in range(StdpSynapticTuningDialog.PROBE_PULSES)
        ]
        for offset in pulse_offsets:
            block[offset : offset + pulse_n, c.presynaptic_id] = c.amplitude
        block_start = self._host.append_block(block)
        self.probe_pulse_times = [
            (block_start + offset) * c.stim_dt for offset in pulse_offsets
        ]
        self.phase = "baseline_probe" if self.baseline_epsp is None else "post_probe"
        self._log("Appended five presynaptic probe pulses.")
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def pairing(self, potentiate: bool) -> None:
        c = self._cfg()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(c.pulse_ms, c.stim_dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(
            StdpSynapticTuningDialog.LEAD_MS, c.stim_dt
        )
        adaptive = (
            c.pairing_strategy == StdpSynapticTuningDialog.ADAPTIVE_PAIRING_STRATEGY
        )
        self._maybe_log_pairing_resolution_warning(c.stim_dt)
        if adaptive:
            if len(self.epsp_percentages) > 1:
                epsp_new = self.epsp_percentages[-1]
                epsp_old = self.epsp_percentages[-2]
                target_error = abs(epsp_new - c.target)
                epsp_change = abs(epsp_new - epsp_old)
                increment = c.pairing_increment_ms
                if target_error > epsp_change:
                    self.adaptive_pairing_magnitude_ms -= increment
                else:
                    self.adaptive_pairing_magnitude_ms += increment
            # Keep the magnitude bounded: never below one simulation sample (so
            # pre and post never collapse to the same time) and never above the
            # configured spinbox maximum (so it cannot drift away unboundedly).
            min_magnitude_ms = 1000.0 * c.stim_dt
            max_magnitude_ms = c.max_pairing_magnitude_ms
            self.adaptive_pairing_magnitude_ms = min(
                max_magnitude_ms,
                max(min_magnitude_ms, self.adaptive_pairing_magnitude_ms),
            )
            pairing_time_ms = (
                self.adaptive_pairing_magnitude_ms
                if potentiate
                else -self.adaptive_pairing_magnitude_ms
            )
        else:
            magnitude_ms = c.pairing_interval_ms
            pairing_time_ms = magnitude_ms if potentiate else -magnitude_ms
        gap_n, gap_ms = snap_milliseconds_to_dt(abs(pairing_time_ms), c.stim_dt)
        pairing_time_ms = copysign(gap_ms, pairing_time_ms)
        tail_n, _tail_ms = snap_milliseconds_to_dt(
            StdpSynapticTuningDialog.RESPONSE_TAIL_MS, c.stim_dt
        )
        samples = lead_n + gap_n + pulse_n + tail_n
        block = np.zeros((samples, c.n_neurons), dtype=np.float32)
        pre_before_post = pairing_time_ms >= 0.0
        first = c.presynaptic_id if pre_before_post else c.postsynaptic_id
        second = c.postsynaptic_id if pre_before_post else c.presynaptic_id
        block[lead_n : lead_n + pulse_n, first] = c.amplitude
        block[lead_n + gap_n : lead_n + gap_n + pulse_n, second] = c.amplitude
        self._host.append_block(block)
        self.phase = "pairing"
        self.pairing_times_ms.append(pairing_time_ms)
        direction = (
            "potentiation (pre → post)"
            if pre_before_post
            else "depression (post → pre)"
        )
        self._log(f"Appended one {direction} pairing at {pairing_time_ms:+.3f} ms.")
        self._measurement_changed()
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def _maybe_log_pairing_resolution_warning(self, dt_seconds: float) -> None:
        c = self._cfg()
        if self.resolution_warning_logged:
            return
        if c.pairing_strategy != StdpSynapticTuningDialog.ADAPTIVE_PAIRING_STRATEGY:
            return
        one_sample_ms = 1000.0 * dt_seconds
        if c.pairing_increment_ms >= one_sample_ms:
            return
        self.resolution_warning_logged = True
        self._log(
            f"Input dt is {one_sample_ms:.6g} ms; pairing times are snapped to "
            "that grid, so finer adaptive increments cannot be resolved."
        )

    def record_measurement(self, amplitude: float, units: str = "") -> None:
        c = self._cfg()
        if self.baseline_epsp is None:
            if amplitude <= 0.0:
                self.finish("Baseline EPSP must be positive before relative targeting.")
                return
            self.baseline_epsp = amplitude
            percentage = 100.0
            self._log(
                f"Baseline EPSP: {amplitude:.6g} {units} (five-pulse average = 100%)."
            )
        else:
            percentage = 100.0 * amplitude / self.baseline_epsp
            self._log(
                f"Current EPSP: {amplitude:.6g} {units} "
                f"({percentage:.3f}% of baseline; change from initial "
                f"{percentage - 100.0:+.3f}%)."
            )
        self.epsp_percentages.append(percentage)
        self._measurement_changed()
        difference = percentage - c.target
        converged = abs(difference) < c.tolerance
        if converged and not c.run_all_iterations:
            self.finish("EPSP is within tolerance; target achieved.")
            return
        if self.iteration >= c.max_iterations:
            self.finish("Maximum plasticity induction iterations reached.")
            return
        if converged:
            self._log(
                "EPSP is within tolerance, but the routine is configured to "
                "run through the last iteration."
            )
        self.iteration += 1
        self.pairing(potentiate=difference < 0.0)

    def _advance(self) -> None:
        if self.on_advance is not None:
            self.on_advance()


@dataclass(frozen=True)
class StdpSynapticTuningRunConfig:
    target: float
    tolerance: float
    max_iterations: int
    run_all_iterations: bool
    pairing_strategy: str
    pairing_interval_ms: float
    initial_pairing_time_ms: float
    pairing_increment_ms: float
    max_pairing_magnitude_ms: float
    presynaptic_id: int
    postsynaptic_id: int
    amplitude: float
    pulse_ms: float
    stim_dt: float
    n_neurons: int


class StdpSynapticTuningDialog(SelfRaisingDialog):
    """Closed-loop EPSP targeting with an STDP induction protocol."""

    simulation_requested = QtCore.Signal(float)
    PROBE_PULSES = 5
    PROBE_PERIOD_MS = 100.0
    PULSE_MS = 5.0
    LEAD_MS = 20.0
    RESPONSE_TAIL_MS = 50.0
    FIXED_PAIRING_STRATEGY = "Fixed pre-post pairing time"
    ADAPTIVE_PAIRING_STRATEGY = "Adaptive (single parameter, dt increment)"

    def __init__(
        self,
        editor: StimEditor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Cached modeless routine windows must outlive transient host windows;
        # detach an optional host parent to avoid Qt deleting the cache first.
        if parent is not None:
            self.setParent(None)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        self.editor = editor
        self.project = editor.project
        self._backend = StdpSynapticTuningBackend(self)
        self._backend.on_advance = self._advance_domain
        self._backend.on_measurement = self._update_plots
        self._backend.on_finished = self._on_backend_finished
        self._running = False
        self._awaiting_simulation = False
        self._paused = False
        self._pending_simulation_seconds: float | None = None
        self._phase = "idle"
        self._theme = theme_module.LIGHT
        self._expanded_height: int | None = None
        self._compact_height: int | None = None
        self._array_cursor: RoutineArrayCursor | None = None
        self._append_start_sample = 0
        self.setWindowTitle("STDP synaptic tuning")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        # Current stim-array info at the top, mirroring the Design static input
        # window.
        info_row = QtWidgets.QHBoxLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_stim_size = QtWidgets.QLabel("-", self)
        self.lbl_stim_units = QtWidgets.QLabel("-", self)
        self.lbl_stim_dt = QtWidgets.QLabel("-", self)
        self.lbl_stim_neurons = QtWidgets.QLabel("-", self)
        for title, value in (
            ("Input array:", self.lbl_stim_name),
            ("Size:", self.lbl_stim_size),
            ("Units:", self.lbl_stim_units),
            ("Neurons:", self.lbl_stim_neurons),
            ("dt:", self.lbl_stim_dt),
        ):
            info_row.addWidget(QtWidgets.QLabel(title, self))
            info_row.addWidget(value)
            info_row.addSpacing(8)
        info_row.addStretch()
        layout.addLayout(info_row)

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
        explanation = QtWidgets.QLabel(
            "Target a postsynaptic EPSP with five-pulse measurements and "
            "single pre/post STDP induction pairings.",
            self.parameters_panel,
        )
        explanation.setWordWrap(True)
        parameters_layout.addWidget(explanation)
        form = QtWidgets.QFormLayout()
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.spn_presynaptic = QtWidgets.QSpinBox(self)
        self.spn_presynaptic.setMinimum(0)
        self.spn_presynaptic.setValue(0)
        self.spn_postsynaptic = QtWidgets.QSpinBox(self)
        self.spn_postsynaptic.setMinimum(0)
        self.spn_postsynaptic.setValue(1)
        self.spn_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_amplitude.setDecimals(3)
        # A positive minimum: a zero-amplitude probe evokes no EPSP and dead-ends
        # the routine at the measurement step.
        self.spn_amplitude.setRange(0.001, 1e12)
        self.spn_amplitude.setValue(100.0)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_amplitude)
        amplitude_row.addWidget(self.lbl_units)
        self.spn_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pulse_ms.setDecimals(3)
        self.spn_pulse_ms.setRange(0.001, 1000.0)
        self.spn_pulse_ms.setValue(self.PULSE_MS)
        self.spn_pulse_ms.setSuffix(" ms")
        self.cmb_pairing_strategy = QtWidgets.QComboBox(self)
        self.cmb_pairing_strategy.addItems(
            [self.FIXED_PAIRING_STRATEGY, self.ADAPTIVE_PAIRING_STRATEGY]
        )
        self.spn_pairing_interval_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pairing_interval_ms.setRange(0.001, 1000.0)
        self.spn_pairing_interval_ms.setValue(30.0)
        self.spn_pairing_interval_ms.setSuffix(" ms")
        self.spn_initial_pairing_time_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_initial_pairing_time_ms.setRange(0.0, 1000.0)
        self.spn_initial_pairing_time_ms.setValue(30.0)
        self.spn_initial_pairing_time_ms.setSuffix(" ms")
        self.spn_pairing_increment_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pairing_increment_ms.setRange(0.001, 1000.0)
        self.spn_pairing_increment_ms.setValue(1.0)
        self.spn_pairing_increment_ms.setSuffix(" ms")
        self.spn_target = QtWidgets.QDoubleSpinBox(self)
        self.spn_target.setRange(0.0, 10_000.0)
        self.spn_target.setValue(120.0)
        self.spn_target.setSuffix(" %")
        self.spn_tolerance = QtWidgets.QDoubleSpinBox(self)
        self.spn_tolerance.setRange(0.001, 1000.0)
        self.spn_tolerance.setValue(1.0)
        self.spn_tolerance.setSuffix(" %")
        self.spn_max_iterations = QtWidgets.QSpinBox(self)
        self.spn_max_iterations.setRange(1, 10_000)
        self.spn_max_iterations.setValue(20)
        self.chk_run_all_iterations = QtWidgets.QCheckBox(
            "Run until the last iteration regardless of convergence to the target",
            self,
        )
        self.chk_run_all_iterations.setChecked(False)
        self.chk_run_all_iterations.setToolTip(
            "Continue plasticity induction after the EPSP enters the target "
            "tolerance, stopping only at the maximum iteration count."
        )
        form.addRow("Trace to analyze:", self.cmb_trace)
        form.addRow("Presynaptic neuron ID:", self.spn_presynaptic)
        form.addRow("Postsynaptic neuron ID:", self.spn_postsynaptic)
        form.addRow("Pulse amplitude:", amplitude_row)
        form.addRow("Pulse duration (ms):", self.spn_pulse_ms)
        form.addRow("Pairing strategy:", self.cmb_pairing_strategy)
        self.lbl_pairing_interval = QtWidgets.QLabel("Pre-post pairing time:", self)
        form.addRow(self.lbl_pairing_interval, self.spn_pairing_interval_ms)
        self.lbl_initial_pairing_time = QtWidgets.QLabel(
            "Initial guess for pre-post pairing time:", self
        )
        form.addRow(self.lbl_initial_pairing_time, self.spn_initial_pairing_time_ms)
        self.lbl_pairing_increment = QtWidgets.QLabel("dt increment:", self)
        form.addRow(self.lbl_pairing_increment, self.spn_pairing_increment_ms)
        form.addRow("Target relative EPSP amplitude:", self.spn_target)
        form.addRow("Tolerance window (deadband):", self.spn_tolerance)
        form.addRow("Maximum plasticity induction iterations:", self.spn_max_iterations)
        form.addRow("Stopping rule:", self.chk_run_all_iterations)
        parameters_layout.addLayout(form)
        # Wrap the parameters in a scroll area so they scroll instead of being
        # squished into unreadable rows on short screens.
        self.parameters_scroll = QtWidgets.QScrollArea(self)
        self.parameters_scroll.setWidget(self.parameters_panel)
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # A small minimum lets the panel scroll rather than shrink its rows when
        # the dialog is short.
        self.parameters_scroll.setMinimumHeight(80)
        layout.addWidget(self.parameters_scroll, 1)
        # A wheel scroll over a value widget must scroll the panel, not silently
        # change the field, so route those events to the scroll area.
        self._wheel_filter = WheelToScrollFilter(self.parameters_scroll)
        value_widgets: list[QtWidgets.QWidget] = list(
            self.parameters_panel.findChildren(QtWidgets.QAbstractSpinBox)
        )
        value_widgets += self.parameters_panel.findChildren(QtWidgets.QComboBox)
        for value_widget in value_widgets:
            value_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            value_widget.installEventFilter(self._wheel_filter)
        self.cmb_pairing_strategy.currentIndexChanged.connect(
            self._update_pairing_strategy_controls
        )
        self._update_pairing_strategy_controls()

        # The plots live in their own window so the controls and the plots each
        # get room, and the controls stay compact on small screens.
        # No Qt parent for the popups (this differs from most of the app's
        # other detached windows): a parent sets WM_TRANSIENT_FOR, and some
        # window managers (observed on KWin/X11) then treat the whole chain
        # of transient windows sharing one owner as a single "raise group" -
        # clicking any one of them just re-asserts the group's foreground
        # status without ever reordering members relative to each other, so
        # a popup can stay stuck behind its sibling no matter which is
        # clicked. Each is still closed (hidden) with this dialog in
        # closeEvent() below.
        self.plot_window = SelfRaisingDialog()
        # No Qt parent, so delete it with this dialog, not whenever Python frees it
        self.destroyed.connect(self.plot_window.deleteLater)
        self.plot_window.setWindowTitle("STDP synaptic tuning — plots")
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        self._plot_window_positioned = False
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.epsp_plot = MatplotlibWidget(self.plot_window)
        self.epsp_plot.setMinimumSize(680, 500)
        figure = self.epsp_plot.getFigure()
        grid = figure.add_gridspec(3, 1, height_ratios=(3, 2, 2), hspace=0.14)
        self.epsp_axes = figure.add_subplot(grid[0, 0])
        self.difference_axes = figure.add_subplot(grid[1, 0], sharex=self.epsp_axes)
        self.pairing_time_axes = figure.add_subplot(grid[2, 0], sharex=self.epsp_axes)
        self.spn_target.valueChanged.connect(lambda _value: self._update_plots())
        self.spn_tolerance.valueChanged.connect(lambda _value: self._update_plots())
        # Compatibility alias: both axes now share one tightly packed canvas.
        self.difference_plot = self.epsp_plot
        plot_layout.addWidget(wrap_in_scroll_area(self.epsp_plot, self.plot_window))
        self.plot_window.resize(720, 640)

        self.lbl_relative_difference = QtWidgets.QLabel(
            "Relative difference from target: waiting for baseline", self
        )
        layout.addWidget(self.lbl_relative_difference)
        self.log = QtWidgets.QPlainTextEdit(self)
        self._backend.on_log = self._append_log
        self.log.setMaximumHeight(130)
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.btn_start = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Start routine", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.btn_stop = cast(
            QtWidgets.QPushButton,
            buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole),
        )
        self.btn_pause = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Pause", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.btn_clear = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Clear input array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_reset = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Reset input array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_reset.setToolTip(
            "Trim the input array back to the simulation head so the routine does "
            "not run a long empty tail."
        )
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear.clicked.connect(self.editor.clear_stim_array)
        self.btn_reset.clicked.connect(self.editor.reset_stim_array)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.apply_theme(self._theme)
        self.refresh()

    def _update_pairing_strategy_controls(self) -> None:
        adaptive = (
            self.cmb_pairing_strategy.currentText() == self.ADAPTIVE_PAIRING_STRATEGY
        )
        self.lbl_pairing_interval.setVisible(not adaptive)
        self.spn_pairing_interval_ms.setVisible(not adaptive)
        self.lbl_initial_pairing_time.setVisible(adaptive)
        self.spn_initial_pairing_time_ms.setVisible(adaptive)
        self.lbl_pairing_increment.setVisible(adaptive)
        self.spn_pairing_increment_ms.setVisible(adaptive)

    def _set_parameters_hidden(self, hidden: bool) -> None:
        """Collapse the parameter panel so the plots can use its space."""
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
        """Resize after Qt has recalculated the layout's size constraints."""
        height = (
            self._compact_height
            if self.btn_toggle_parameters.isChecked()
            else self._expanded_height
        )
        if height is not None:
            self.resize(self.width(), max(height, self.minimumSizeHint().height()))

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        # The plots live in separate windows; bring them up alongside the dialog.
        self.plot_window.show()
        if not self._plot_window_positioned:
            self._plot_window_positioned = True
            self.plot_window.move(self.x() + self.width() + 20, self.y())
        self.plot_window.raise_()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        # The dialog is cached and reused, so closing only hides it. Stop any
        # running routine first, otherwise it would keep consuming simulation
        # callbacks while hidden and resume mid-run when reopened.
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
        """Place Windows' native minimized dialog at the screen bottom-left."""
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        margin = 12
        available = screen.availableGeometry()
        minimized_height = 32
        set_native_minimized_position(
            int(self.winId()),
            available.left() + margin,
            available.bottom() - minimized_height - margin + 1,
        )

    def refresh(self) -> None:
        self.project = self.editor.project
        selected = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        if self.project is None:
            return
        trace_names = self.project.get_data_names(TraceData)
        self.cmb_trace.addItems(trace_names)
        preferred_trace = self._preferred_trace_name(trace_names, selected)
        if preferred_trace:
            self.cmb_trace.setCurrentText(preferred_trace)
        modifier = self.editor.current_stim_modifier
        if modifier is not None:
            stim = modifier.stim_array
            n_neurons = stim.values.shape[1]
            maximum = max(0, n_neurons - 1)
            self.spn_presynaptic.setMaximum(maximum)
            self.spn_postsynaptic.setMaximum(maximum)
            self.lbl_units.setText(stim.units)
            self.lbl_stim_name.setText(stim.name)
            self.lbl_stim_size.setText(f"{stim.values.shape[0]} samples")
            self.lbl_stim_units.setText(stim.units or "-")
            self.lbl_stim_dt.setText(f"{stim.dt * 1000.0:g} ms")
            self.lbl_stim_neurons.setText(str(n_neurons))
        else:
            self.lbl_stim_name.setText("-")
            self.lbl_stim_size.setText("-")
            self.lbl_stim_units.setText("-")
            self.lbl_stim_dt.setText("-")
            self.lbl_stim_neurons.setText("-")

    def _preferred_trace_name(
        self, trace_names: list[str], previously_selected: str
    ) -> str:
        if not trace_names:
            return ""
        if self._is_simulation_voltage_trace(previously_selected):
            return previously_selected
        for trace_name in trace_names:
            if self._is_simulation_voltage_trace(trace_name):
                return trace_name
        if previously_selected in trace_names:
            return previously_selected
        return trace_names[0]

    def _is_simulation_voltage_trace(self, trace_name: str) -> bool:
        if self.project is None or not trace_name:
            return False
        if not trace_name.startswith("Simulation :: ") or not trace_name.endswith(
            " / v"
        ):
            return False
        trace = self.project.get_data(trace_name, TraceData)
        if trace is None:
            return False
        return bool(np.any(trace.indices == self.spn_postsynaptic.value()))

    def start(self) -> None:
        self.refresh()
        if self.project is None or self.editor.current_stim_modifier is None:
            self._append_log("No input array is selected.")
            return
        choice = confirm_start_routine(self, self.editor)
        if choice == "cancel":
            self._append_log("Routine cancelled.")
            return
        stim = self._require_stim()
        project = self._require_project()
        prepare_seconds = 0.0
        if choice == "clear":
            self.editor.zero_current_stim_array()
            # Trim the (now all-zero) tail back to the simulation head, same
            # as reset_stim_array() always did - there is nothing real left
            # to lose, so "start from the beginning" means right after
            # whatever has actually been simulated so far (0 for a fresh
            # project), not literally sample 0 if some has already run.
            self.editor.reset_stim_array()
            target_secs = max(project.b2_model.get_sim_time_secs(), float(stim.dt))
            self._append_start_sample = seconds_to_samples_nearest(
                target_secs, float(stim.dt)
            )
        else:
            # Continue after the last existing stimulus - not the
            # simulation head, which can be well behind it (e.g. an example
            # project's pre-authored stim array that has never been
            # simulated in this session). Using the simulation head here
            # would make reset_stim_array() silently discard that
            # pre-existing content instead of preserving it.
            self._append_start_sample = (
                self.editor.current_stim_array_last_stimulus_sample()
            )
            last_stimulus_secs = self._append_start_sample * float(stim.dt)
            current_sim_secs = project.b2_model.get_sim_time_secs()
            if last_stimulus_secs > current_sim_secs + float(stim.dt) * 0.5:
                # The network has to actually run through that pre-existing
                # content before the routine starts appending and measuring
                # after it.
                prepare_seconds = run_to_sample(
                    project, stim, self._append_start_sample
                )
        # Get this dialog out of the way so the user can watch the stim
        # array and traces fill up as the routine runs. Z-order hints
        # (lower()/raise()) are not reliably honored by every window manager
        # - especially for a window that still has focus from the Start
        # click - so minimizing is used instead, since window state is
        # enforced by the OS/WM rather than merely requested. Restored
        # automatically when the routine finishes (_on_backend_finished).
        # The plot window is left alone: minimizing/restoring it churns its
        # live matplotlib canvas's Qt window state, which on macOS can fire
        # a deferred redraw after the canvas is gone - a real crash seen in
        # CI, not just a maybe.
        self.showMinimized()
        main_window = self.editor.window()
        if main_window is not None:
            main_window.raise_()
            main_window.activateWindow()
        needs_trace_init = not self._is_simulation_voltage_trace(
            self.cmb_trace.currentText()
        )
        if needs_trace_init or prepare_seconds > 0.0:
            duration = max(prepare_seconds, 0.001)
            self._running = True
            self._phase = "initialize_trace"
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_pause.setEnabled(True)
            if prepare_seconds > 0.0:
                self._append_log(
                    f"Simulating {duration:.3f} s to catch up to the "
                    "existing stimuli before the protocol starts."
                )
            else:
                self._append_log(
                    f"Running an initial {duration:.3f} s simulation to "
                    "create trace monitors."
                )
            self._request_simulation(duration)
            return
        self._begin_protocol(clear_log=True)

    def _begin_protocol(self, *, clear_log: bool) -> None:
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._finish("Pre- and postsynaptic neuron IDs must differ.")
            return
        stim = self._require_stim()
        config = StdpSynapticTuningRunConfig(
            target=self.spn_target.value(),
            tolerance=self.spn_tolerance.value(),
            max_iterations=self.spn_max_iterations.value(),
            run_all_iterations=self.chk_run_all_iterations.isChecked(),
            pairing_strategy=self.cmb_pairing_strategy.currentText(),
            pairing_interval_ms=self.spn_pairing_interval_ms.value(),
            initial_pairing_time_ms=self.spn_initial_pairing_time_ms.value(),
            pairing_increment_ms=self.spn_pairing_increment_ms.value(),
            max_pairing_magnitude_ms=self.spn_initial_pairing_time_ms.maximum(),
            presynaptic_id=self.spn_presynaptic.value(),
            postsynaptic_id=self.spn_postsynaptic.value(),
            amplitude=self.spn_amplitude.value(),
            pulse_ms=self.spn_pulse_ms.value(),
            stim_dt=float(stim.dt),
            n_neurons=int(stim.values.shape[1]),
        )
        initial_estimate_seconds = (
            config.pairing_interval_ms + config.pulse_ms
        ) / 1000.0
        self._array_cursor = RoutineArrayCursor(
            self._append_start_sample, initial_estimate_seconds
        )
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        if clear_log:
            self.log.clear()
        self._backend.begin_protocol(config)

    def stop(self) -> None:
        self._finish("Routine stopped.")

    def toggle_pause(self) -> None:
        self._backend.toggle_pause()
        self.btn_pause.setText("Resume" if self._paused else "Pause")

    def _require_stim(self) -> StimArray:
        modifier = self.editor.current_stim_modifier
        if modifier is None:
            raise RuntimeError("No input array is selected.")
        return modifier.stim_array

    def _require_project(self) -> Project:
        if self.project is None:
            raise RuntimeError("No project is available.")
        return self.project

    def append_block(self, block: np.ndarray) -> int:
        return self._append_block(block)

    def _append_block(self, block: np.ndarray) -> int:
        if self._array_cursor is None:
            raise RuntimeError("No routine is running.")
        stim = self._require_stim()
        return self._array_cursor.place_block(self.editor, stim, block)

    def _request_simulation(self, seconds: float) -> None:
        self._backend.request_simulation(seconds)

    def request_simulation(self, seconds: float) -> None:
        self.simulation_requested.emit(seconds)

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if not self._awaiting_simulation:
            # A simulation the designer did not request (e.g. a manual run)
            # completed; do not advance the closed-loop protocol.
            return
        self._backend.simulation_finished()

    def _advance_domain(self) -> None:
        assert self.project is not None
        if self._phase == "initialize_trace":
            self.refresh()
            if not self._is_simulation_voltage_trace(self.cmb_trace.currentText()):
                self._finish(
                    "The initial simulation completed, but no simulation voltage "
                    "trace for the postsynaptic neuron became available. Add a "
                    "StateMonitor for v to the model and try again."
                )
                return
            self._append_log("Trace monitor created; starting the EPSP protocol.")
            self._begin_protocol(clear_log=False)
            return
        if self._phase == "pairing":
            self._backend.probe()
            return
        self.refresh()
        trace_name = self.cmb_trace.currentText()
        if not self._is_simulation_voltage_trace(trace_name):
            self._finish(
                "The routine could not find a simulation voltage trace for the "
                "postsynaptic neuron after simulation."
            )
            return
        trace = self.project.get_data(trace_name, TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return
        try:
            measurement = self._measure_epsps(trace)
        except ValueError as exc:
            self._finish(f"EPSP measurement failed: {exc}")
            return
        self._backend.record_measurement(measurement.amplitude, measurement.units)

    def _measure_epsps(self, trace: TraceData) -> EPSPMeasurement:
        """Associate the shared EPSP plugin's detections with probe pulses."""
        return measure_epsp(
            trace,
            self.spn_postsynaptic.value(),
            self._probe_pulse_times,
            baseline_ms=12.0,
            response_ms=40.0,
        )

    def _update_plots(self) -> None:
        x = np.arange(1, len(self._epsp_percentages) + 1)
        values = np.asarray(self._epsp_percentages)
        target = self.spn_target.value()
        tolerance = self.spn_tolerance.value()
        self.epsp_axes.clear()
        self.epsp_axes.axhspan(
            target - tolerance,
            target + tolerance,
            color="#2ca02c",
            alpha=0.18,
            label="Deadband",
        )
        self.epsp_axes.axhline(target, color="#d62728", linewidth=2, linestyle="--")
        self.epsp_axes.plot(x, values, color="#1f77b4", linewidth=2, marker="o")
        self.difference_axes.clear()
        self.difference_axes.axhspan(
            -tolerance,
            tolerance,
            color="#2ca02c",
            alpha=0.18,
        )
        self.difference_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.difference_axes.plot(
            x, values - target, color="#b245a8", linewidth=2, marker="o"
        )
        self.pairing_time_axes.clear()
        pairing_x = np.arange(1, len(self._pairing_times_ms) + 1)
        self.pairing_time_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.pairing_time_axes.plot(
            pairing_x,
            self._pairing_times_ms,
            color="#e28743",
            linewidth=2,
            marker="o",
        )
        if values.size:
            self.difference_axes.set_xlim(0.5, values.size + 0.5)
        if values.size:
            difference = float(values[-1] - target)
            self.lbl_relative_difference.setText(
                f"Relative difference from target: {difference:+.3f}% "
                f"(absolute {abs(difference):.3f}%, tolerance {self.spn_tolerance.value():.3f}%)"
            )
        else:
            self.lbl_relative_difference.setText(
                "Relative difference from target: waiting for baseline"
            )
        self._style_plot_axes()
        self.epsp_plot.getFigure().subplots_adjust(
            left=0.14, right=0.98, bottom=0.09, top=0.98, hspace=0.14
        )
        self.epsp_plot.draw()

    def _style_plot_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        for axes, ylabel in (
            (self.epsp_axes, "EPSP\namplitude (%)"),
            (self.difference_axes, "Δ target\n(%)"),
            (self.pairing_time_axes, "Pairing time\n(ms)"),
        ):
            axes.set_facecolor(face)
            axes.set_ylabel(ylabel, color=text)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        self.epsp_plot.getFigure().set_facecolor(face)
        self.pairing_time_axes.xaxis.set_major_locator(MultipleLocator(1))
        self.epsp_axes.tick_params(labelbottom=False)
        self.difference_axes.tick_params(labelbottom=False)
        self.epsp_axes.set_xlabel("")
        self.difference_axes.set_xlabel("")
        self.pairing_time_axes.set_xlabel("Measurement / iteration", color=text)

    def apply_theme(self, theme: str) -> None:
        """Recolor every Matplotlib canvas for the application theme."""
        self._theme = theme_module.normalize_theme(theme)
        self._style_plot_axes()
        self.epsp_plot.draw()

    def simulation_failed(self, message: str) -> None:
        if self._running and self._awaiting_simulation:
            self._finish(f"Simulation failed: {message}")

    def _finish(self, message: str) -> None:
        self._backend.finish(message)

    def _on_backend_finished(self, message: str) -> None:
        # The backend can call finish() on itself (e.g. target reached)
        # without going through this dialog's _finish(), so this callback -
        # wired to RoutineBackend.on_finished - is the one hook guaranteed to
        # run on every completion path; it is what actually resets the
        # buttons and trims any padding placed ahead of the routine's last
        # block, so a stopped or completed run never leaves a long empty
        # tail. It is also what restores the dialog after start() minimized
        # it out of the way of the stim array/traces.
        del message
        self.showNormal()
        self.editor.reset_stim_array()
        self._array_cursor = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")

    @property
    def _running(self) -> bool:
        return self._backend.running

    @_running.setter
    def _running(self, value: bool) -> None:
        self._backend.running = value

    @property
    def _awaiting_simulation(self) -> bool:
        return self._backend.awaiting_simulation

    @_awaiting_simulation.setter
    def _awaiting_simulation(self, value: bool) -> None:
        self._backend.awaiting_simulation = value

    @property
    def _paused(self) -> bool:
        return self._backend.paused

    @_paused.setter
    def _paused(self, value: bool) -> None:
        self._backend.paused = value

    @property
    def _pending_simulation_seconds(self) -> float | None:
        return self._backend.pending_simulation_seconds

    @_pending_simulation_seconds.setter
    def _pending_simulation_seconds(self, value: float | None) -> None:
        self._backend.pending_simulation_seconds = value

    @property
    def _phase(self) -> str:
        return self._backend.phase

    @_phase.setter
    def _phase(self, value: str) -> None:
        self._backend.phase = value

    _baseline_epsp = backend_property("baseline_epsp")
    _epsp_percentages = backend_property("epsp_percentages")
    _pairing_times_ms = backend_property("pairing_times_ms")
    _adaptive_pairing_magnitude_ms = backend_property("adaptive_pairing_magnitude_ms")
    _probe_pulse_times = backend_property("probe_pulse_times")
    _iteration = backend_property("iteration")
    _resolution_warning_logged = backend_property("resolution_warning_logged")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)
