from __future__ import annotations

from collections.abc import Callable
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
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.dialog_helpers import _SelfRaisingDialog
from neuroseq.dynamic_input.dialog_helpers import _set_native_minimized_position
from neuroseq.dynamic_input.dialog_helpers import _WheelToScrollFilter
from neuroseq.dynamic_input.dialog_helpers import _wrap_in_scroll_area
from neuroseq.dynamic_input.dialog_helpers import confirm_start_routine
from neuroseq.dynamic_input.dynamic_input_blocks import RoutineArrayCursor
from neuroseq.dynamic_input.dynamic_input_blocks import run_to_sample
from neuroseq.dynamic_input.dynamic_input_measurement import EPSPMeasurement
from neuroseq.dynamic_input.dynamic_input_measurement import NoEpspDetected
from neuroseq.dynamic_input.dynamic_input_measurement import measure_calcium_response
from neuroseq.dynamic_input.dynamic_input_measurement import measure_epsp
from neuroseq.dynamic_input.dynamic_input_routine import RoutineBackend
from neuroseq.dynamic_input.dynamic_input_routine import SimulationHost
from neuroseq.dynamic_input.dynamic_input_routine import backend_property
from neuroseq.dynamic_input.stdp_window import classify_window
from neuroseq.dynamic_input.stdp_window import delta_t_sweep
from neuroseq.dynamic_input.stdp_window import interleave_by_magnitude
from neuroseq.dynamic_input.stdp_window import percent_change
from neuroseq.stimulation.stim_timing import format_milliseconds
from neuroseq.stimulation.stim_timing import samples_to_milliseconds
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import snap_milliseconds_to_dt

# Backwards-compatible import for callers/tests that used the former local
# exception name before measurement was shared between the routines.
_NoEpspDetected = NoEpspDetected

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim_editor import StimEditor


class StdpWindowBackend(RoutineBackend):
    """Lifecycle backend for the STDP timing-window dialog.

    Owns the Qt-free Δt-sweep decision graph; the dialog supplies
    rendering, trace/weight measurement, checkpoint restore, and
    append-block callbacks.
    """

    def __init__(self, host: SimulationHost) -> None:
        super().__init__(host)
        self.delta_ts: list[float] = []
        self.index = 0
        self.pairing_repeat = 0
        self.baseline_epsp: float | None = None
        self.last_epsp: float | None = None
        self.pending_delta_t = 0.0
        self.results: list[tuple[float, float]] = []
        self.epsp_trajectory: list[float] = []
        self.probe_pulse_times: list[float] = []
        self.checkpoint: str | None = None
        self.baseline_sample = 0
        self.point_start = 0
        self.saturation_warned = False
        self.resolution_warning_logged = False
        self.config: StdpWindowRunConfig | None = None
        self.on_restore_baseline: Callable[[], None] | None = None

    def _cfg(self) -> StdpWindowRunConfig:
        if self.config is None:
            raise RuntimeError("STDP window sweep has not been configured")
        return self.config

    def begin_sweep(
        self,
        config: StdpWindowRunConfig,
        delta_ts: list[float],
        *,
        weight_baseline: float | None,
    ) -> None:
        self.config = config
        self.begin()
        self.delta_ts = list(delta_ts)
        self.index = 0
        self.pairing_repeat = 0
        self.results = []
        self.epsp_trajectory = []
        self.probe_pulse_times = []
        self.saturation_warned = False
        self.phase = "idle"
        if weight_baseline is not None:
            self.baseline_epsp = weight_baseline
            self.last_epsp = weight_baseline
            self.epsp_trajectory.append(weight_baseline)
            self._log(f"Baseline synaptic weight: {weight_baseline:.6g}.")
            self._measurement_changed()
            self.begin_point()
        else:
            self.baseline_epsp = None
            self.last_epsp = None
            self.phase = "baseline_probe"
            self.probe()

    def probe(self) -> None:
        c = self._cfg()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(c.pulse_ms, c.stim_dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(c.recovery_ms, c.stim_dt)
        period_n, _period_ms = snap_milliseconds_to_dt(
            StdpWindowDialog.PROBE_PERIOD_MS, c.stim_dt, minimum_samples=pulse_n + 1
        )
        tail_n, _tail_ms = snap_milliseconds_to_dt(c.response_ms, c.stim_dt)
        samples = (
            lead_n + (StdpWindowDialog.PROBE_PULSES - 1) * period_n + pulse_n + tail_n
        )
        block = np.zeros((samples, c.n_neurons), dtype=np.float32)
        pulse_offsets = [
            lead_n + index * period_n for index in range(StdpWindowDialog.PROBE_PULSES)
        ]
        for offset in pulse_offsets:
            block[offset : offset + pulse_n, c.presynaptic_id] = c.amplitude
        block_start = self._host.append_block(block)
        self.probe_pulse_times = [
            (block_start + offset) * c.stim_dt for offset in pulse_offsets
        ]
        self._log("Appended five presynaptic probe pulses.")
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def pairing(self, delta_t_ms: float) -> None:
        c = self._cfg()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(c.pulse_ms, c.stim_dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(c.recovery_ms, c.stim_dt)
        gap_n, gap_ms = snap_milliseconds_to_dt(abs(delta_t_ms), c.stim_dt)
        self.pending_delta_t = copysign(gap_ms, delta_t_ms)
        delta_t_ms = self.pending_delta_t
        tail_n, _tail_ms = snap_milliseconds_to_dt(c.response_ms, c.stim_dt)
        samples = lead_n + gap_n + pulse_n + tail_n
        block = np.zeros((samples, c.n_neurons), dtype=np.float32)
        pre_before_post = delta_t_ms >= 0.0
        first = c.presynaptic_id if pre_before_post else c.postsynaptic_id
        second = c.postsynaptic_id if pre_before_post else c.presynaptic_id
        block[lead_n : lead_n + pulse_n, first] = c.amplitude
        block[lead_n + gap_n : lead_n + gap_n + pulse_n, second] = c.amplitude
        self._host.append_block(block)
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def begin_point(self) -> None:
        c = self._cfg()
        self.pending_delta_t = self.delta_ts[self.index]
        self.pairing_repeat = 0
        if c.sweep_mode == StdpWindowDialog.RESET_MODE and self.on_restore_baseline:
            self.on_restore_baseline()
        self.apply_next_pairing()

    def apply_next_pairing(self) -> None:
        c = self._cfg()
        self.pairing_repeat += 1
        self.phase = "pair"
        self._log(
            f"Δt {self.pending_delta_t:+.3g} ms: pairing "
            f"{self.pairing_repeat}/{c.pairings_per_point}."
        )
        self.pairing(self.pending_delta_t)

    def should_continue_pairing(self) -> bool:
        return self.pairing_repeat < self._cfg().pairings_per_point

    def begin_probe_phase(self) -> None:
        self.phase = "probe"
        self.probe()

    def record_baseline_probe(self, amplitude: float, units: str) -> None:
        self.baseline_epsp = amplitude
        self.last_epsp = amplitude
        self.epsp_trajectory.append(amplitude)
        self._log(f"Baseline EPSP: {amplitude:.6g} {units}.")
        self._measurement_changed()
        self.index = 0
        self.begin_point()

    def handle_probe_measurement(self, measurement: EPSPMeasurement | None) -> None:
        if self.baseline_epsp is None or self.last_epsp is None:
            self.finish("Baseline EPSP is unavailable.")
            return
        delta_t = self.pending_delta_t
        label = classify_window(delta_t)
        if measurement is None:
            # An undetectable EPSP is a skipped point, not a fatal error:
            # record NaN and carry on so one shunted response does not abort
            # the sweep.
            self.epsp_trajectory.append(float("nan"))
            self.results.append((delta_t, float("nan")))
            self._log(f"Δt {delta_t:+.3g} ms → EPSP undetectable; point skipped.")
        else:
            self.epsp_trajectory.append(measurement.amplitude)
            percent = self._percent_for(measurement.amplitude)
            self.results.append((delta_t, percent))
            if self._cfg().sweep_mode != StdpWindowDialog.RESET_MODE:
                self.last_epsp = measurement.amplitude
            self._log(f"Δt {delta_t:+.3g} ms → {percent:+.3f}% EPSP ({label}).")
            self._maybe_warn_saturation(measurement.amplitude)
        self._measurement_changed()
        self.advance_point()

    def record_weight_point(self, weight: float) -> None:
        if self.baseline_epsp is None or self.last_epsp is None:
            self.finish("Baseline synaptic weight is unavailable.")
            return
        delta_t = self.pending_delta_t
        label = classify_window(delta_t)
        self.epsp_trajectory.append(weight)
        percent = self._percent_for(weight)
        self.results.append((delta_t, percent))
        if self._cfg().sweep_mode != StdpWindowDialog.RESET_MODE:
            self.last_epsp = weight
        self._log(f"Δt {delta_t:+.3g} ms → {percent:+.3f}% weight ({label}).")
        self._maybe_warn_saturation(weight)
        self._measurement_changed()
        self.advance_point()

    def _percent_for(self, measure: float) -> float:
        """Percent change: from the fixed baseline (reset) or the running
        value (continuous)."""
        baseline = self.baseline_epsp
        last = self.last_epsp
        assert baseline is not None and last is not None
        reference = (
            baseline if self._cfg().sweep_mode == StdpWindowDialog.RESET_MODE else last
        )
        return percent_change(reference, measure, reference)

    def _maybe_warn_saturation(self, epsp: float) -> None:
        baseline = self.baseline_epsp
        if self.saturation_warned or baseline is None or baseline == 0.0:
            return
        ratio = epsp / baseline
        if ratio < 0.25 or ratio > 4.0:
            self.saturation_warned = True
            self._log(
                "EPSP drift is far from baseline; later Δt points may be "
                "affected by synaptic saturation."
            )

    def advance_point(self) -> None:
        self.index += 1
        if self.index >= len(self.delta_ts):
            measured = sum(1 for _dt, percent in self.results if percent == percent)
            self.finish(
                f"STDP window sweep finished; measured {measured}/"
                f"{len(self.results)} points."
            )
            return
        self.begin_point()

    def _advance(self) -> None:
        if self.on_advance is not None:
            self.on_advance()


@dataclass(frozen=True)
class StdpWindowRunConfig:
    dt_min_ms: float
    dt_max_ms: float
    dt_points: int
    include_zero: bool
    pairings_per_point: int
    readout: str
    sweep_mode: str
    presynaptic_id: int
    postsynaptic_id: int
    amplitude: float
    pulse_ms: float
    recovery_ms: float
    response_ms: float
    stim_dt: float
    n_neurons: int
    max_iterations: int = 0


class StdpWindowDialog(_SelfRaisingDialog):
    """Sweep spike timing to measure the functional STDP timing window."""

    simulation_requested = QtCore.Signal(float)
    PROBE_PULSES = 5
    PROBE_PERIOD_MS = 100.0
    # Readout modes. EPSP is functional (needs a clean postsynaptic EPSP);
    # weight reads the synapse weight directly and is immune to shunting /
    # feedback inhibition, so it works on recurrent networks.
    EPSP_READOUT = "EPSP amplitude change (%)"
    WEIGHT_READOUT = "Synaptic weight change (%)"
    # Sweep modes. Reset restores a baseline checkpoint before each Δt so every
    # point measures one pairing from the identical state (the clean textbook
    # window). Continuous keeps one running synapse (shows cumulative drift) and
    # visits Δt interleaved by magnitude to stay near baseline.
    RESET_MODE = "Reset to baseline each Δt (independent points)"
    CONTINUOUS_MODE = "Continuous sweep (interleaved, no reset)"
    SIMULATION_TARGET = "Brian2 simulation"
    SYSCON2_TARGET = "SysCon2 (calcium imaging)"
    CALCIUM_READOUT = "Calcium response (dF/F) change (%)"
    CALCIUM_BASELINE_MS = 100.0

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
        self._backend = StdpWindowBackend(self)
        self._backend.on_advance = self._advance_domain
        self._backend.on_measurement = self._update_plot
        self._backend.on_restore_baseline = self._restore_baseline
        self._backend.on_finished = self._on_backend_finished
        self._running = False
        self._awaiting_simulation = False
        self._paused = False
        self._pending_simulation_seconds: float | None = None
        self._phase = "idle"
        self._theme = theme_module.LIGHT
        self._expanded_height: int | None = None
        self._compact_height: int | None = None
        self._plot_window_positioned = False
        self._array_cursor: RoutineArrayCursor | None = None
        self._append_start_sample = 0
        self._point_span_end_sample = 0
        # SysCon2 target state (see RheobaseDialog): range of the block in flight and the
        # trace used for the previous probe, to catch a forgotten re-import.
        self._block_range = (0, 0)
        self._last_measured_dataset: str | None = None
        self._data_before_block: dict[str, object] = {}
        self._host_connected = False

        self.setWindowTitle("STDP timing window (LTP/LTD vs Δt)")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

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
        self.btn_syscon_dt = QtWidgets.QPushButton("Use SysCon2 dt", self)
        self.btn_syscon_dt.clicked.connect(self._fix_syscon_dt)
        info_row.addWidget(self.btn_syscon_dt)
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
        note = QtWidgets.QLabel(
            "Sweep pre/post spike timing and measure the synaptic change per "
            "pairing. Use a suprathreshold pulse amplitude (the rheobase routine "
            "can estimate one). Reset mode measures each Δt independently from the "
            "baseline weight — the clean textbook window; continuous mode keeps "
            "one running synapse and shows cumulative drift. The EPSP readout "
            "needs a clean postsynaptic EPSP (isolated pair, e.g. the two-neuron "
            "plasticity example); on recurrent networks use the weight readout.",
            self.parameters_panel,
        )
        note.setWordWrap(True)
        parameters_layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.cmb_readout = QtWidgets.QComboBox(self)
        self.cmb_readout.addItems([self.EPSP_READOUT, self.WEIGHT_READOUT])
        self.cmb_sweep_mode = QtWidgets.QComboBox(self)
        self.cmb_sweep_mode.addItems([self.RESET_MODE, self.CONTINUOUS_MODE])
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.cmb_synapse = QtWidgets.QComboBox(self)
        self.spn_presynaptic = QtWidgets.QSpinBox(self)
        self.spn_presynaptic.setMinimum(0)
        self.spn_presynaptic.setValue(0)
        self.spn_postsynaptic = QtWidgets.QSpinBox(self)
        self.spn_postsynaptic.setMinimum(0)
        self.spn_postsynaptic.setValue(1)
        self.spn_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_amplitude.setDecimals(3)
        self.spn_amplitude.setRange(0.001, 1e12)
        self.spn_amplitude.setValue(100.0)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_amplitude)
        amplitude_row.addWidget(self.lbl_units)
        self.spn_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pulse_ms.setDecimals(3)
        self.spn_pulse_ms.setRange(0.001, 1000.0)
        self.spn_pulse_ms.setValue(5.0)
        self.spn_pulse_ms.setSuffix(" ms")
        self.spn_dt_min_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_dt_min_ms.setDecimals(3)
        self.spn_dt_min_ms.setRange(-1000.0, 1000.0)
        self.spn_dt_min_ms.setValue(-100.0)
        self.spn_dt_min_ms.setSuffix(" ms")
        self.spn_dt_max_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_dt_max_ms.setDecimals(3)
        self.spn_dt_max_ms.setRange(-1000.0, 1000.0)
        self.spn_dt_max_ms.setValue(100.0)
        self.spn_dt_max_ms.setSuffix(" ms")
        self.spn_dt_points = QtWidgets.QSpinBox(self)
        self.spn_dt_points.setRange(2, 500)
        self.spn_dt_points.setValue(21)
        self.chk_include_zero = QtWidgets.QCheckBox(self)
        self.chk_include_zero.setChecked(False)
        self.spn_pairings_per_point = QtWidgets.QSpinBox(self)
        self.spn_pairings_per_point.setRange(1, 100)
        self.spn_pairings_per_point.setValue(1)
        self.spn_recovery_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_recovery_ms.setDecimals(3)
        self.spn_recovery_ms.setRange(0.001, 100_000.0)
        self.spn_recovery_ms.setValue(150.0)
        self.spn_recovery_ms.setSuffix(" ms")
        self.spn_response_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_response_ms.setDecimals(3)
        self.spn_response_ms.setRange(0.001, 100_000.0)
        self.spn_response_ms.setValue(50.0)
        self.spn_response_ms.setSuffix(" ms")

        self.cmb_target = QtWidgets.QComboBox(self)
        self.cmb_target.addItems([self.SIMULATION_TARGET, self.SYSCON2_TARGET])
        self.cmb_target.setToolTip(
            "Brian2: simulate the protocol. SysCon2: deliver each block to the cells "
            "through SysCon2 and measure the calcium response of the postsynaptic cell "
            "in data you import (continuous sweep only - cells cannot be reset)."
        )
        self.cmb_target.currentTextChanged.connect(self._update_target_controls)
        form.addRow("Stimulation target:", self.cmb_target)
        form.addRow("Readout:", self.cmb_readout)
        form.addRow("Sweep mode:", self.cmb_sweep_mode)
        self.lbl_trace = QtWidgets.QLabel("Trace to analyze:", self)
        form.addRow(self.lbl_trace, self.cmb_trace)
        self.lbl_synapse = QtWidgets.QLabel("Synapse weight:", self)
        form.addRow(self.lbl_synapse, self.cmb_synapse)
        form.addRow("Presynaptic neuron ID:", self.spn_presynaptic)
        form.addRow("Postsynaptic neuron ID:", self.spn_postsynaptic)
        form.addRow("Pulse amplitude:", amplitude_row)
        form.addRow("Pulse duration (ms):", self.spn_pulse_ms)
        form.addRow("Δt minimum (ms):", self.spn_dt_min_ms)
        form.addRow("Δt maximum (ms):", self.spn_dt_max_ms)
        form.addRow("Number of Δt points:", self.spn_dt_points)
        form.addRow("Include Δt = 0:", self.chk_include_zero)
        form.addRow("Pairings per point:", self.spn_pairings_per_point)
        form.addRow("Recovery time (ms):", self.spn_recovery_ms)
        self.lbl_response = QtWidgets.QLabel("Post-pulse window (ms):", self)
        form.addRow(self.lbl_response, self.spn_response_ms)
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

        for widget in (
            self.spn_dt_min_ms,
            self.spn_dt_max_ms,
            self.spn_dt_points,
            self.chk_include_zero,
            self.spn_pairings_per_point,
        ):
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(lambda _value: self._update_estimate())
            else:
                widget.valueChanged.connect(lambda _value: self._update_estimate())
        self.cmb_readout.currentIndexChanged.connect(self._on_readout_changed)
        self.cmb_sweep_mode.currentIndexChanged.connect(
            lambda _index: self._update_estimate()
        )

        # No Qt parent for this popup: a parent sets WM_TRANSIENT_FOR, and
        # some window managers (observed on KWin/X11) then treat the whole
        # chain of transient windows sharing one owner as a single "raise
        # group" - clicking one never brings it above a sibling in the same
        # chain.
        self.plot_window = _SelfRaisingDialog()
        self.plot_window.setWindowTitle("STDP timing window (LTP/LTD vs Δt) — plots")
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.window_plot = MatplotlibWidget(self.plot_window)
        self.window_plot.setMinimumSize(680, 500)
        figure = self.window_plot.getFigure()
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.18)
        self.window_axes = figure.add_subplot(grid[0, 0])
        self.trajectory_axes = figure.add_subplot(grid[1, 0])
        plot_layout.addWidget(_wrap_in_scroll_area(self.window_plot, self.plot_window))
        self.plot_window.resize(720, 560)

        self.lbl_status = QtWidgets.QLabel("STDP window: waiting for baseline", self)
        layout.addWidget(self.lbl_status)
        self.lbl_estimate = QtWidgets.QLabel("", self)
        layout.addWidget(self.lbl_estimate)
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
                "Start", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
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
        self.btn_data_imported = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Block done / data imported",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            ),
        )
        self.btn_data_imported.setToolTip(
            "SysCon2 target: click after you ran the sent block. For probe blocks, also "
            "acquire, import the calcium data and select the new trace first."
        )
        self.btn_data_imported.setEnabled(False)
        self.btn_data_imported.clicked.connect(self._on_data_imported_clicked)
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
        self._update_readout_controls()
        self._update_target_controls()
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

    def refresh(self) -> None:
        self.project = self.editor.project
        selected = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        if self.project is not None:
            trace_names = self.project.get_data_names(TraceData)
            self.cmb_trace.addItems(trace_names)
            preferred_trace = self._preferred_trace_name(trace_names, selected)
            if preferred_trace:
                self.cmb_trace.setCurrentText(preferred_trace)
        syscon2_available = self.editor.stimulation_host is not None
        syscon2_item = cast(QtGui.QStandardItemModel, self.cmb_target.model()).item(1)
        if syscon2_item is not None:
            syscon2_item.setEnabled(syscon2_available)
        if not syscon2_available and self._target_is_syscon2():
            self.cmb_target.setCurrentIndex(0)
        selected_synapse = self.cmb_synapse.currentText()
        self.cmb_synapse.clear()
        if self.project is not None:
            synapse_names = self.project.get_data_names(SynapseData)
            self.cmb_synapse.addItems(synapse_names)
            if selected_synapse in synapse_names:
                self.cmb_synapse.setCurrentText(selected_synapse)
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
            self.lbl_stim_dt.setText(f"{format_milliseconds(stim.dt * 1000.0)} ms")
            self.lbl_stim_neurons.setText(str(n_neurons))
        else:
            self.lbl_units.setText("")
            self.lbl_stim_name.setText("-")
            self.lbl_stim_size.setText("-")
            self.lbl_stim_units.setText("-")
            self.lbl_stim_dt.setText("-")
            self.lbl_stim_neurons.setText("-")
        self._update_estimate()
        self._update_plot()

    def _preferred_trace_name(
        self, trace_names: list[str], previously_selected: str
    ) -> str:
        if not trace_names:
            return ""
        if self._target_is_syscon2():
            # Imported calcium traces: keep the user's pick, else offer the newest.
            return (
                previously_selected
                if previously_selected in trace_names
                else trace_names[-1]
            )
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

    def _readout(self) -> str:
        return self.cmb_readout.currentText()

    def _target_is_syscon2(self) -> bool:
        return self.cmb_target.currentText() == self.SYSCON2_TARGET

    def _weight_mode(self) -> bool:
        return not self._target_is_syscon2() and self._readout() == self.WEIGHT_READOUT

    def _reset_mode(self) -> bool:
        # Cells in a dish cannot be rewound to a checkpoint: SysCon2 is always continuous.
        return (
            not self._target_is_syscon2()
            and self.cmb_sweep_mode.currentText() == self.RESET_MODE
        )

    def _update_target_controls(self, _text: str = "") -> None:
        syscon2 = self._target_is_syscon2()
        self.btn_syscon_dt.setVisible(syscon2)
        self.btn_syscon_dt.setEnabled(not self._running)
        self.cmb_readout.setEnabled(not syscon2)
        self.cmb_sweep_mode.setEnabled(not syscon2)
        self.btn_data_imported.setVisible(syscon2)
        if syscon2:
            self._connect_host()
        self._update_readout_controls()

    def _connect_host(self) -> None:
        host = self.editor.stimulation_host
        if host is None or self._host_connected:
            return
        host.block_sent.connect(self._on_block_sent)
        host.block_failed.connect(self._on_block_failed)
        host.block_completed.connect(self._on_block_completed)
        self._host_connected = True

    def _on_readout_changed(self, _index: int) -> None:
        self._update_readout_controls()

    def _update_readout_controls(self) -> None:
        weight = self._weight_mode()
        self.lbl_trace.setVisible(not weight)
        self.cmb_trace.setVisible(not weight)
        self.lbl_response.setVisible(not weight)
        self.spn_response_ms.setVisible(not weight)
        self.lbl_synapse.setVisible(weight)
        self.cmb_synapse.setVisible(weight)
        self._update_estimate()
        self._update_plot()

    def _read_synapse_weight(self) -> float | None:
        """Return the current weight of the pre→post synapse, or None."""
        name = self.cmb_synapse.currentText()
        if not name or self.project is None:
            return None
        data = self.project.get_data(name, SynapseData)
        if data is None or data.values.size == 0:
            return None
        pre = self.spn_presynaptic.value()
        post = self.spn_postsynaptic.value()
        values = data.values
        if pre >= values.shape[0] or post >= values.shape[1]:
            return None
        series = values[pre, post]
        if series.size == 0:
            return None
        return float(series[-1])

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
        if self._target_is_syscon2():
            self._start_syscon2(choice, stim)
            return
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
            # pre-existing content instead of preserving it. (Only matters
            # for continuous mode - reset mode always anchors to the
            # simulation head itself, by design.)
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
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._finish("Pre- and postsynaptic neuron IDs must differ.")
            return
        if self.spn_dt_max_ms.value() <= self.spn_dt_min_ms.value():
            self._finish("Δt maximum must be greater than Δt minimum.")
            return
        if not self._current_delta_t_values():
            self._finish("No Δt points to sweep.")
            return
        if self._weight_mode():
            needs_init = self._read_synapse_weight() is None
        else:
            needs_init = not self._is_simulation_voltage_trace(
                self.cmb_trace.currentText()
            )
        if needs_init or prepare_seconds > 0.0:
            duration = max(prepare_seconds, 0.001)
            self._running = True
            self._phase = (
                "initialize_weight" if self._weight_mode() else "initialize_trace"
            )
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_pause.setEnabled(True)
            if prepare_seconds > 0.0:
                self._append_log(
                    f"Simulating {duration:.3f} s to catch up to the "
                    "existing stimuli before the sweep starts."
                )
            else:
                self._append_log(
                    f"Running an initial {duration:.3f} s simulation to "
                    "create monitors."
                )
            self._request_simulation(duration)
            return
        self._begin_sweep(clear_log=True)

    def _fix_syscon_dt(self) -> bool:
        if self._running:
            return False
        if not self.editor.prepare_syscon_timestep(self):
            return False
        self.refresh()
        return True

    def _start_syscon2(self, choice: str, stim: StimArray) -> None:
        """Begin a sweep whose blocks are delivered through SysCon2."""
        host = self.editor.stimulation_host
        if not self._fix_syscon_dt():
            return
        if host is None or host.phase != "idle":
            self._append_log("SysCon2 is not ready (a block is still in flight).")
            return
        if self.spn_amplitude.value() > 1.0:
            self._append_log(
                "For SysCon2 the pulse amplitude must lie within 0-1 "
                "(fraction of the export target power)."
            )
            return
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._append_log("Pre- and postsynaptic neuron IDs must differ.")
            return
        if self.spn_dt_max_ms.value() <= self.spn_dt_min_ms.value():
            self._append_log("Δt maximum must be greater than Δt minimum.")
            return
        if not self.cmb_trace.currentText():
            self._append_log("Import a calcium trace and select it first.")
            return
        if choice == "clear":
            self.editor.zero_current_stim_array()
            self.editor.reset_stim_array()
            self._append_start_sample = 0
        else:
            self._append_start_sample = (
                self.editor.current_stim_array_last_stimulus_sample()
            )
        self._last_measured_dataset = None
        self._append_log(
            "SysCon2 target: continuous sweep, amplitudes are fractions (0-1) of the "
            "export target power. Every block is sent separately; run it, and for the "
            "baseline/probe blocks acquire and import the calcium data whose time axis "
            "starts with that block. Make the post-pulse window span a few imaging frames."
        )
        self._begin_sweep(clear_log=True)

    def _begin_sweep(self, *, clear_log: bool) -> None:
        delta_ts = self._current_delta_t_values()
        if not delta_ts:
            self._finish("No Δt points to sweep.")
            return
        stim = self._require_stim()
        config = StdpWindowRunConfig(
            dt_min_ms=self.spn_dt_min_ms.value(),
            dt_max_ms=self.spn_dt_max_ms.value(),
            dt_points=self.spn_dt_points.value(),
            include_zero=self.chk_include_zero.isChecked(),
            pairings_per_point=self.spn_pairings_per_point.value(),
            readout=(
                self.CALCIUM_READOUT
                if self._target_is_syscon2()
                else self.cmb_readout.currentText()
            ),
            sweep_mode=(
                self.CONTINUOUS_MODE
                if self._target_is_syscon2()
                else self.cmb_sweep_mode.currentText()
            ),
            presynaptic_id=self.spn_presynaptic.value(),
            postsynaptic_id=self.spn_postsynaptic.value(),
            amplitude=self.spn_amplitude.value(),
            pulse_ms=self.spn_pulse_ms.value(),
            recovery_ms=self.spn_recovery_ms.value(),
            response_ms=self.spn_response_ms.value(),
            stim_dt=float(stim.dt),
            n_neurons=int(stim.values.shape[1]),
        )
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        if clear_log:
            self.log.clear()
        self._append_log(
            f"Sweeping {len(delta_ts)} Δt points; estimated simulations "
            f"~{self._estimated_probe_count()}."
        )
        self._maybe_log_delta_t_resolution_warning()
        self._append_log(
            "Pulse amplitude must evoke pre/post spikes during pairings; use "
            "rheobase first if needed."
        )
        self._update_plot()
        # Reset-mode checkpoint bookkeeping touches the live Brian2 model
        # directly (project.b2_model.store), so it stays a distinct dialog-side
        # step rather than moving into the backend's begin_sweep().
        self._checkpoint = None
        self._baseline_sample = 0
        self._point_start = 0
        self._array_cursor = None
        if self._reset_mode():
            project = self._require_project()
            self._baseline_sample = int(
                np.ceil(project.b2_model.get_sim_time_secs() / stim.dt)
            )
            self._point_start = self._baseline_sample
            self._checkpoint = f"STDP window baseline {id(self)}"
            project.b2_model.store(self._checkpoint)
            # Reset mode reuses the same array region for every Δt point (each
            # rewinds to this checkpoint), so - unlike the append-forward
            # routines - the padded window is sized exactly once, to fit the
            # worst-case point, and never needs to grow or jump afterwards.
            worst_case_gap_ms = max(abs(config.dt_min_ms), abs(config.dt_max_ms))
            point_span_seconds = (
                config.pairings_per_point
                * (
                    config.recovery_ms
                    + worst_case_gap_ms
                    + config.pulse_ms
                    + config.response_ms
                )
                / 1000.0
            )
            baseline_seconds = self._baseline_sample * float(stim.dt)
            self.editor.pad_stim_array_to(baseline_seconds + point_span_seconds)
            self._point_span_end_sample = stim.values.shape[0]
            self.editor.focus_time_range(
                baseline_seconds, baseline_seconds + point_span_seconds
            )
        else:
            probe_estimate_ms = (
                config.recovery_ms
                + (StdpWindowDialog.PROBE_PULSES - 1) * StdpWindowDialog.PROBE_PERIOD_MS
                + config.pulse_ms
                + config.response_ms
            )
            self._array_cursor = RoutineArrayCursor(
                self._append_start_sample, probe_estimate_ms / 1000.0
            )
        weight_baseline: float | None = None
        if self._weight_mode():
            weight_baseline = self._read_synapse_weight()
            if weight_baseline is None:
                self._finish("Could not read the baseline synaptic weight.")
                return
        self._backend.begin_sweep(config, delta_ts, weight_baseline=weight_baseline)

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
        start = self._place_block(block)
        self._block_range = (start, start + block.shape[0])
        return start

    def _restore_baseline(self) -> None:
        """Reset mode: rewind the network to the baseline checkpoint and
        clear the point's stim region.

        Every point reuses the same (padded-once) region, so a point with
        fewer or shorter pairings than the previous one must not inherit
        that previous point's leftover pulses past its own end - clearing
        the whole region before writing keeps it honest without resizing
        the array (and therefore without moving the camera) between points.
        """
        if self._checkpoint is not None:
            self._require_project().b2_model.restore(self._checkpoint)
        self._point_start = self._baseline_sample
        stim = self._require_stim()
        if self._point_span_end_sample > self._baseline_sample:
            zeros = np.zeros(
                (
                    self._point_span_end_sample - self._baseline_sample,
                    stim.values.shape[1],
                ),
                dtype=np.float32,
            )
            self.editor.apply_pattern(
                slice(0, stim.values.shape[1]),
                slice(self._baseline_sample, self._point_span_end_sample),
                zeros,
            )

    def _place_block(self, block: np.ndarray) -> int:
        stim = self._require_stim()
        if self._reset_mode():
            # A reset point deliberately reuses the baseline region, already
            # cleared and padded once in _restore_baseline() /
            # _begin_sweep() - blocks are just written in place at the
            # point's own cursor.
            start = self._point_start
            self.editor.apply_pattern(
                slice(0, stim.values.shape[1]),
                slice(start, start + block.shape[0]),
                block,
            )
            self._point_start = start + block.shape[0]
            return start
        if self._array_cursor is None:
            raise RuntimeError("No routine is running.")
        return self._array_cursor.place_block(self.editor, stim, block)

    def _request_simulation(self, seconds: float) -> None:
        self._backend.request_simulation(seconds)

    def request_simulation(self, seconds: float) -> None:
        if not self._target_is_syscon2():
            self.simulation_requested.emit(seconds)
            return
        host = self.editor.stimulation_host
        if host is None or self.project is None:
            self._finish("SysCon2 is not available.")
            return
        self._data_before_block = {
            name: self.project.get_data(name, TraceData)
            for name in self.project.get_data_names(TraceData)
        }
        start, stop = self._block_range
        self._append_log("Exporting block and sending it to SysCon2...")
        # A failure is reported through host.block_failed -> _on_block_failed.
        host.send_block(
            self.project,
            self._require_stim(),
            start,
            stop,
            "stdp_window",
            preserve_amplitude=True,
        )

    def _on_block_sent(self, _result: object) -> None:
        if not (self._running and self._target_is_syscon2()):
            return
        self.refresh()
        if self._phase == "pair":
            self._append_log(
                "Pairing block sent to SysCon2. Run it, then click "
                "'Block done / data imported' (no data needed for pairings)."
            )
        else:
            self._append_log(
                "Probe block sent to SysCon2. Run it and acquire, import the calcium "
                "data, select the new trace, then click 'Block done / data imported'."
            )
        self.btn_data_imported.setEnabled(True)

    def _on_block_failed(self, message: str) -> None:
        if self._running and self._target_is_syscon2():
            self._finish(f"SysCon2 failed: {message}")

    def _on_block_completed(self) -> None:
        if self._running and self._target_is_syscon2():
            self._backend.simulation_finished()

    def _on_data_imported_clicked(self) -> None:
        host = self.editor.stimulation_host
        if host is None:
            return
        self.btn_data_imported.setEnabled(False)
        host.acquisition_imported()

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if not self._awaiting_simulation:
            return
        self._backend.simulation_finished()

    def _advance_domain(self) -> None:
        assert self.project is not None
        if self._phase in ("initialize_trace", "initialize_weight"):
            self.refresh()
            if self._weight_mode():
                if not self.cmb_synapse.currentText():
                    self._finish(
                        "The initial simulation completed, but no synapse weight "
                        "monitor became available. Add a StateMonitor on the "
                        "synapse weight to the model and try again."
                    )
                    return
                self._append_log(
                    "Synapse monitor created; starting the STDP window sweep."
                )
            else:
                if not self._is_simulation_voltage_trace(self.cmb_trace.currentText()):
                    self._finish(
                        "The initial simulation completed, but no simulation "
                        "voltage trace for the postsynaptic neuron became "
                        "available. Add a StateMonitor for v to the model and "
                        "try again."
                    )
                    return
                self._append_log(
                    "Trace monitor created; starting the STDP window sweep."
                )
            self._begin_sweep(clear_log=False)
            return
        if self._phase == "pair":
            if self._backend.should_continue_pairing():
                self._backend.apply_next_pairing()
                return
            if self._weight_mode():
                weight = self._read_synapse_weight()
                if weight is None:
                    self._finish(
                        "Could not read the synaptic weight after the pairing."
                    )
                    return
                self._backend.record_weight_point(weight)
                return
            self._backend.begin_probe_phase()
            return

        measured = (
            self._measure_calcium()
            if self._target_is_syscon2()
            else self._measure_simulated()
        )
        if measured is None:
            return
        measurement = measured[0]

        if self._phase == "baseline_probe":
            if measurement is None or measurement.amplitude <= 0.0:
                self._finish(
                    "No baseline EPSP could be measured; the postsynaptic "
                    "response may be shunted by network activity. Try the "
                    "'Synaptic weight' readout or an isolated-synapse project."
                )
                return
            self._backend.record_baseline_probe(
                measurement.amplitude, measurement.units
            )
            return

        if self._phase == "probe":
            self._backend.handle_probe_measurement(measurement)

    def _measure_simulated(self) -> tuple[EPSPMeasurement | None] | None:
        """EPSP measurement from the simulated voltage trace; ``None`` once finished."""
        assert self.project is not None
        self.refresh()
        trace_name = self.cmb_trace.currentText()
        if not self._is_simulation_voltage_trace(trace_name):
            self._finish(
                "The routine could not find a simulation voltage trace for the "
                "postsynaptic neuron after simulation."
            )
            return None
        trace = self.project.get_data(trace_name, TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return None
        try:
            return (self._measure_epsps(trace),)
        except NoEpspDetected:
            return (None,)
        except ValueError as exc:
            self._finish(f"EPSP measurement failed: {exc}")
            return None

    def _measure_calcium(self) -> tuple[EPSPMeasurement | None] | None:
        """Calcium response from the imported trace; ``None`` to wait/finish."""
        assert self.project is not None
        self.refresh()
        self._select_newest_if_stale(self.cmb_trace)
        name = self.cmb_trace.currentText()
        if not name or not self._is_fresh_dataset(name):
            self._append_log(
                "Import the new recording, select its trace, and click again."
                if name
                else "Select the imported calcium trace first."
            )
            self._reawait_acquisition()
            return None
        trace = self.project.get_data(name, TraceData)
        if trace is None:
            self._finish(f"Trace '{name}' is unavailable.")
            return None
        block_start_seconds = self._block_range[0] * float(self._require_stim().dt)
        relative_pulse_times = [
            t - block_start_seconds for t in self._probe_pulse_times
        ]
        try:
            measurement = measure_calcium_response(
                trace,
                self.spn_postsynaptic.value(),
                relative_pulse_times,
                baseline_ms=self.CALCIUM_BASELINE_MS,
                response_ms=self.spn_response_ms.value(),
            )
        except ValueError as exc:
            self._finish(f"Calcium measurement failed: {exc}")
            return None
        self._last_measured_dataset = name
        return (measurement,)

    def _is_fresh_dataset(self, name: str) -> bool:
        if self.project is None or not name or name.startswith("Simulation :: "):
            return False
        data = self.project.get_data(name, TraceData)
        return data is not None and data is not self._data_before_block.get(name)

    def _select_newest_if_stale(self, combo: QtWidgets.QComboBox) -> None:
        """Offer a recording imported after this block, including same-name replacements."""
        if self._is_fresh_dataset(combo.currentText()):
            return
        for index in reversed(range(combo.count())):
            if self._is_fresh_dataset(combo.itemText(index)):
                combo.setCurrentIndex(index)
                return

    def _reawait_acquisition(self) -> None:
        """Stay on the current probe: ask the user to fix the import and click again."""
        host = self.editor.stimulation_host
        if host is not None:
            host.phase = "awaiting_acquisition"
        self._backend.awaiting_simulation = True
        self.btn_data_imported.setEnabled(True)

    def _measure_epsps(self, trace: TraceData) -> EPSPMeasurement:
        return measure_epsp(
            trace,
            self.spn_postsynaptic.value(),
            self._probe_pulse_times,
            baseline_ms=12.0,
            response_ms=self.spn_response_ms.value(),
        )

    def _current_delta_t_values(self) -> list[float]:
        values = self._snapped_delta_ts(
            delta_t_sweep(
                self.spn_dt_min_ms.value(),
                self.spn_dt_max_ms.value(),
                self.spn_dt_points.value(),
                self.chk_include_zero.isChecked(),
            )
        )
        if not self._reset_mode():
            # Continuous sweep drifts, so visit offsets interleaved by magnitude
            # to keep the running weight near baseline.
            values = interleave_by_magnitude(values)
        return values

    def _raw_delta_t_values(self) -> list[float]:
        return delta_t_sweep(
            self.spn_dt_min_ms.value(),
            self.spn_dt_max_ms.value(),
            self.spn_dt_points.value(),
            self.chk_include_zero.isChecked(),
        )

    def _snapped_delta_ts(self, values: list[float]) -> list[float]:
        if self.editor.current_stim_modifier is None:
            return values
        dt_seconds = self.editor.current_stim_modifier.stim_array.dt
        snapped: list[float] = []
        seen: set[int] = set()
        for value in values:
            samples, effective_ms = snap_milliseconds_to_dt(abs(value), dt_seconds)
            signed_samples = samples if value >= 0.0 else -samples
            if signed_samples in seen:
                continue
            seen.add(signed_samples)
            snapped.append(copysign(effective_ms, value))
        return snapped

    def _maybe_log_delta_t_resolution_warning(self) -> None:
        if self._resolution_warning_logged:
            return
        raw_values = self._raw_delta_t_values()
        if len(raw_values) < 2:
            return
        spacing = min(
            abs(b - a) for a, b in zip(raw_values, raw_values[1:], strict=False)
        )
        one_sample_ms = samples_to_milliseconds(1, self._require_stim().dt)
        if spacing >= one_sample_ms:
            return
        self._resolution_warning_logged = True
        self._append_log(
            f"Input dt is {one_sample_ms:.6g} ms; Δt values are snapped to that "
            "grid, so finer spacing cannot be resolved."
        )

    def _estimated_probe_count(self) -> int:
        n_points = len(self._current_delta_t_values())
        pairings = self.spn_pairings_per_point.value()
        if self._weight_mode():
            # One init run + the pairings; the weight is read directly, no probes.
            return 1 + n_points * pairings
        return 1 + n_points * (pairings + 1)

    def _update_estimate(self) -> None:
        self.lbl_estimate.setText(
            "Estimated simulations for the current settings: "
            f"~{self._estimated_probe_count()}"
        )

    def _update_plot(self) -> None:
        self.window_axes.clear()
        self.trajectory_axes.clear()
        finite_results = [
            (delta_t, percent)
            for delta_t, percent in self._results
            if percent == percent  # drop NaN (undetectable) points
        ]
        if finite_results:
            sorted_results = sorted(finite_results, key=lambda item: item[0])
            x = np.asarray([delta_t for delta_t, _percent in sorted_results])
            y = np.asarray([percent for _delta_t, percent in sorted_results])
            self.window_axes.plot(x, y, color="#2f2f2f", linewidth=1.5, alpha=0.7)
            positive = x > 0.0
            negative = x < 0.0
            zero = x == 0.0
            self.window_axes.scatter(
                x[positive], y[positive], color="#2ca02c", zorder=3
            )
            self.window_axes.scatter(
                x[negative], y[negative], color="#d62728", zorder=3
            )
            if np.any(zero):
                self.window_axes.scatter(x[zero], y[zero], color="#7f7f7f", zorder=3)
        self.window_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.window_axes.axvline(0.0, color="#7f7f7f", linewidth=1)
        self.window_axes.text(
            0.98,
            0.92,
            "LTP",
            color="#2ca02c",
            ha="right",
            transform=self.window_axes.transAxes,
        )
        self.window_axes.text(
            0.02,
            0.92,
            "LTD",
            color="#d62728",
            ha="left",
            transform=self.window_axes.transAxes,
        )
        if self._epsp_trajectory:
            probe_x = np.arange(len(self._epsp_trajectory))
            self.trajectory_axes.plot(
                probe_x,
                self._epsp_trajectory,
                color="#1f77b4",
                linewidth=2,
                marker="o",
            )
            self.trajectory_axes.axhline(
                self._epsp_trajectory[0],
                color="#7f7f7f",
                linewidth=1,
                linestyle="--",
            )
        self._update_status()
        self._style_plot_axes()
        self.window_plot.getFigure().subplots_adjust(
            left=0.13, right=0.98, bottom=0.09, top=0.98, hspace=0.18
        )
        self.window_plot.draw()

    def _update_status(self) -> None:
        if not self._results:
            self.lbl_status.setText("STDP window: waiting for baseline")
            return
        delta_t, percent = self._results[-1]
        if percent != percent:  # NaN → undetectable point
            detail = "undetectable"
        else:
            detail = f"{percent:+.3f}% ({classify_window(delta_t)})"
        self.lbl_status.setText(
            f"{len(self._results)}/{len(self._delta_ts)} points · last Δt "
            f"{delta_t:+.3g} ms → {detail}"
        )

    def _style_plot_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        for axes in (self.window_axes, self.trajectory_axes):
            axes.set_facecolor(face)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        weight = self._weight_mode()
        self.window_axes.set_xlabel("Δt = t_post − t_pre (ms)", color=text)
        self.window_axes.set_ylabel(
            "Weight change (%)" if weight else "EPSP change (%)", color=text
        )
        self.trajectory_axes.set_xlabel("Point" if weight else "Probe", color=text)
        self.trajectory_axes.set_ylabel(
            "Synaptic weight" if weight else "EPSP (mV)", color=text
        )
        self.trajectory_axes.xaxis.set_major_locator(MultipleLocator(1))
        self.window_plot.getFigure().set_facecolor(face)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme_module.normalize_theme(theme)
        self._style_plot_axes()
        self.window_plot.draw()

    def simulation_failed(self, message: str) -> None:
        if self._running and self._awaiting_simulation:
            self._finish(f"Simulation failed: {message}")

    def _finish(self, message: str) -> None:
        host = self.editor.stimulation_host
        if host is not None:
            host.cancel()
        self.btn_data_imported.setEnabled(False)
        self._backend.finish(message)
        self._append_log(message)

    def _on_backend_finished(self, message: str) -> None:
        # The backend can call finish() on itself (e.g. sweep complete)
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
        self.btn_data_imported.setEnabled(False)
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

    _delta_ts = backend_property("delta_ts")
    _index = backend_property("index")
    _pairing_repeat = backend_property("pairing_repeat")
    _baseline_epsp = backend_property("baseline_epsp")
    _last_epsp = backend_property("last_epsp")
    _pending_delta_t = backend_property("pending_delta_t")
    _results = backend_property("results")
    _epsp_trajectory = backend_property("epsp_trajectory")
    _probe_pulse_times = backend_property("probe_pulse_times")
    _checkpoint = backend_property("checkpoint")
    _baseline_sample = backend_property("baseline_sample")
    _point_start = backend_property("point_start")
    _saturation_warned = backend_property("saturation_warned")
    _resolution_warning_logged = backend_property("resolution_warning_logged")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)
