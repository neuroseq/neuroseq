from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from neuroseq.dynamic_input_blocks import RoutineArrayCursor
from neuroseq.dynamic_input_blocks import run_to_sample
from neuroseq.dynamic_input_measurement import measure_spike_count
from neuroseq.dynamic_input_routine import RoutineBackend
from neuroseq.dynamic_input_routine import SimulationHost
from neuroseq.dynamic_input_routine import backend_property
from neuroseq.rheobase import bisection_converged
from neuroseq.rheobase import next_bracket
from neuroseq.stdp_synaptic_tuning_dialog import _SelfRaisingDialog
from neuroseq.stdp_synaptic_tuning_dialog import _set_native_minimized_position
from neuroseq.stdp_synaptic_tuning_dialog import _WheelToScrollFilter
from neuroseq.stdp_synaptic_tuning_dialog import _wrap_in_scroll_area
from neuroseq.stdp_synaptic_tuning_dialog import confirm_start_routine
from neuroseq.stim_timing import seconds_to_samples_nearest
from neuroseq.stim_timing import snap_milliseconds_to_dt

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray
    from neuroseq.stim_editor import StimEditor


class RheobaseBackend(RoutineBackend):
    """Lifecycle backend for the rheobase dialog.

    Owns the Qt-free rheobase decision graph; the dialog supplies rendering,
    trace measurement, and append-block callbacks.
    """

    def __init__(self, host: SimulationHost) -> None:
        super().__init__(host)
        self.config: RheobaseRunConfig | None = None
        self.on_summarize: Callable[[], str] | None = None
        self.on_summarize_fixed_amplitude: Callable[[], str] | None = None
        self.low = 0.0
        self.high = 0.0
        self.pending_amplitude = 0.0
        self.pulse_onset_seconds = 0.0
        self.duration_values_ms: list[float] = []
        self.duration_index = 0
        self.current_duration_ms = 0.0
        self.duration_low_ms = 0.0
        self.duration_high_ms = 0.0
        self.duration_stage = "idle"
        self.rheobase: float | None = None
        self.chronaxie_ms: float | None = None
        self.fixed_spiking_duration_ms: float | None = None
        self.iteration = 0
        self.logged_recovery_note = False
        self.resolution_warning_logged = False
        self.samples: list[tuple[float, float, int]] = []
        self.thresholds: list[tuple[float, float]] = []

    def _advance(self) -> None:
        if self.on_advance is not None:
            self.on_advance()

    def begin_search(self, config: RheobaseRunConfig) -> None:
        self.config = config
        self.begin()
        self.phase = "idle"
        self.iteration = 0
        self.low = config.min_amplitude
        self.high = config.max_amplitude
        self.pending_amplitude = config.max_amplitude
        self.pulse_onset_seconds = 0.0
        self.duration_values_ms = [config.pulse_ms, config.max_pulse_ms]
        self.duration_index = 0
        self.current_duration_ms = config.pulse_ms
        self.duration_low_ms = config.pulse_ms
        self.duration_high_ms = config.max_pulse_ms
        self.duration_stage = "idle"
        self.samples = []
        self.thresholds = []
        self.rheobase = None
        self.chronaxie_ms = None
        self.fixed_spiking_duration_ms = None
        self.logged_recovery_note = False
        self.resolution_warning_logged = False
        self._maybe_log_duration_resolution_warning()
        if config.strategy == RheobaseDialog.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY:
            self._begin_fixed_amplitude_duration_search()
        elif (
            config.strategy == RheobaseDialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
        ):
            self._begin_duration_search(config.pulse_ms, "fixed duration")
        else:
            self._begin_strength_duration_search()

    def _cfg(self) -> RheobaseRunConfig:
        if self.config is None:
            raise RuntimeError("Rheobase search has not been configured")
        return self.config

    def _duration_sweep_values(self) -> list[float]:
        c = self._cfg()
        return [c.pulse_ms, c.max_pulse_ms]

    def _begin_strength_duration_search(self) -> None:
        self.duration_stage = "rheobase"
        self._begin_duration_search(
            self._cfg().max_pulse_ms, "rheobase estimate at maximum duration"
        )

    def _begin_duration_search(self, duration_ms: float, description: str) -> None:
        c = self._cfg()
        self.current_duration_ms = self._effective_duration_ms(duration_ms)
        self.low = c.min_amplitude
        self.high = c.max_amplitude
        self.iteration = 0
        self.phase = "probe_max"
        self._log(
            f"Searching duration {self.current_duration_ms:.6g} ms ({description})."
        )
        self.probe(c.max_amplitude)

    def _begin_fixed_amplitude_duration_search(self) -> None:
        c = self._cfg()
        self.duration_stage = "duration_probe_max"
        self.duration_low_ms = c.pulse_ms
        self.duration_high_ms = c.max_pulse_ms
        self._probe_fixed_amplitude_duration(c.max_pulse_ms, "maximum duration")

    def _probe_fixed_amplitude_duration(
        self, duration_ms: float, description: str
    ) -> None:
        c = self._cfg()
        self.current_duration_ms = self._effective_duration_ms(duration_ms)
        self.phase = "duration_probe"
        self._log(
            f"Testing fixed amplitude {c.fixed_amplitude:.6g} at duration {self.current_duration_ms:.6g} ms ({description})."
        )
        self.probe(c.fixed_amplitude)

    def _one_sample_ms(self) -> float:
        return 1000.0 * self._cfg().stim_dt

    def _duration_tolerance_ms(self) -> float:
        return max(self._cfg().duration_tolerance_ms, self._one_sample_ms())

    def _effective_duration_ms(self, duration_ms: float) -> float:
        _samples, effective_ms = snap_milliseconds_to_dt(
            duration_ms, self._cfg().stim_dt
        )
        return effective_ms

    def _duration_threshold_exists(self, duration_ms: float) -> bool:
        return any(abs(d - duration_ms) <= 1e-9 for d, _ in self.thresholds)

    def _duration_already_sampled(self, duration_ms: float) -> bool:
        return any(abs(d - duration_ms) <= 1e-9 for d, _, _ in self.samples)

    def _maybe_log_duration_resolution_warning(self) -> None:
        c = self._cfg()
        if (
            not self.resolution_warning_logged
            and c.duration_tolerance_ms < self._one_sample_ms()
        ):
            self.resolution_warning_logged = True
            self._log(
                f"Input dt is {self._one_sample_ms():.6g} ms; durations are snapped to that grid, so finer tolerances cannot be resolved."
            )

    def probe(self, amplitude: float) -> None:
        c = self._cfg()
        recovery_n, _ = snap_milliseconds_to_dt(c.recovery_ms, c.stim_dt)
        pulse_n, effective_ms = snap_milliseconds_to_dt(
            self.current_duration_ms, c.stim_dt
        )
        self.current_duration_ms = effective_ms
        tail_n, _ = snap_milliseconds_to_dt(c.response_ms, c.stim_dt)
        samples = recovery_n + pulse_n + tail_n
        block = np.zeros((samples, c.n_neurons), dtype=np.float32)
        block[recovery_n : recovery_n + pulse_n, c.neuron_id] = amplitude
        block_start = self._host.append_block(block)
        self.pending_amplitude = amplitude
        self.pulse_onset_seconds = (block_start + recovery_n) * c.stim_dt
        if not self.logged_recovery_note:
            self._log(
                "Each probe is appended after the existing input with a recovery lead before the pulse so the membrane returns to rest."
            )
            self.logged_recovery_note = True
        self._log(
            f"Probe {len(self.samples) + 1}: testing amplitude {amplitude:.6g} at {self.current_duration_ms:.6g} ms."
        )
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def record_probe_result(self, spike_count: int) -> None:
        c = self._cfg()
        amplitude = self.pending_amplitude
        self.samples.append((self.current_duration_ms, amplitude, spike_count))
        self._log(
            f"Probe {len(self.samples)}: amplitude {amplitude:.6g} at {self.current_duration_ms:.6g} ms -> {'no spike' if spike_count == 0 else f'{spike_count} spikes'}."
        )
        self._measurement_changed()
        if self.phase == "duration_probe":
            self._handle_fixed_amplitude_duration_result(spike_count)
            return
        if self.phase == "probe_max":
            if spike_count == 0:
                self.finish(
                    f"No spike at the maximum amplitude {c.max_amplitude:.6g}; increase the range."
                )
                return
            self.high = c.max_amplitude
            self.phase = "probe_min"
            self.probe(c.min_amplitude)
            return
        if self.phase == "probe_min":
            if spike_count > 0:
                self._complete_duration_threshold(
                    c.min_amplitude,
                    f"Neuron spikes already at the minimum amplitude {c.min_amplitude:.6g}; threshold <= minimum.",
                )
                return
            self.low = c.min_amplitude
            self.phase = "bisect"
            self._bisect_step()
            return
        if self.phase == "bisect":
            self.low, self.high = next_bracket(
                self.low, self.high, amplitude, spike_count > 0
            )
            self._bisect_step()

    def _handle_fixed_amplitude_duration_result(self, spike_count: int) -> None:
        if self.duration_stage == "duration_probe_max":
            if spike_count == 0:
                self._measurement_changed()
                self.finish(
                    self.on_summarize_fixed_amplitude()
                    if self.on_summarize_fixed_amplitude
                    else "Duration sweep finished."
                )
                return
            self.duration_high_ms = self.current_duration_ms
            self.duration_stage = "duration_probe_min"
            self._probe_fixed_amplitude_duration(
                self._cfg().pulse_ms, "minimum duration"
            )
            return
        if self.duration_stage == "duration_probe_min":
            if spike_count > 0:
                self.fixed_spiking_duration_ms = self.current_duration_ms
                self._measurement_changed()
                self.finish(
                    self.on_summarize_fixed_amplitude()
                    if self.on_summarize_fixed_amplitude
                    else "Duration sweep finished."
                )
                return
            self.duration_low_ms = self.current_duration_ms
            self.duration_stage = "duration_bisect"
            self._bisect_fixed_amplitude_duration()
            return
        if self.duration_stage == "duration_bisect":
            if spike_count > 0:
                self.duration_high_ms = self.current_duration_ms
            else:
                self.duration_low_ms = self.current_duration_ms
            self._bisect_fixed_amplitude_duration()

    def _bisect_fixed_amplitude_duration(self) -> None:
        if (
            self.duration_high_ms - self.duration_low_ms
            <= self._duration_tolerance_ms()
            or self._duration_already_sampled(
                self._effective_duration_ms(
                    0.5 * (self.duration_low_ms + self.duration_high_ms)
                )
            )
        ):
            self.fixed_spiking_duration_ms = self.duration_high_ms
            self._measurement_changed()
            self.finish(
                self.on_summarize_fixed_amplitude()
                if self.on_summarize_fixed_amplitude
                else "Duration sweep finished."
            )
            return
        self._probe_fixed_amplitude_duration(
            0.5 * (self.duration_low_ms + self.duration_high_ms), "duration bisection"
        )

    def _bisect_step(self) -> None:
        c = self._cfg()
        if (
            bisection_converged(self.low, self.high, c.tolerance)
            or self.iteration >= c.max_iterations
        ):
            self._complete_duration_threshold(
                self.high,
                f"Threshold at {self.current_duration_ms:.6g} ms ~= {self.high:.6g}.",
            )
            return
        self.iteration += 1
        self.probe(0.5 * (self.low + self.high))

    def _complete_duration_threshold(self, threshold: float, message: str) -> None:
        self.thresholds.append((self.current_duration_ms, threshold))
        self._update_strength_duration_metrics()
        self._measurement_changed()
        self._log(message)
        if (
            self._cfg().strategy
            == RheobaseDialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
        ):
            self.finish(self.on_summarize() if self.on_summarize else message)
            return
        self._handle_strength_duration_threshold()

    def _handle_strength_duration_threshold(self) -> None:
        c = self._cfg()
        threshold = self.thresholds[-1][1]
        if self.duration_stage == "rheobase":
            if c.max_pulse_ms == c.pulse_ms:
                self.finish(
                    self.on_summarize()
                    if self.on_summarize
                    else "Duration sweep finished."
                )
                return
            self.duration_stage = "duration_min"
            self._begin_duration_search(c.pulse_ms, "minimum duration bound")
            return
        if self.duration_stage == "duration_min":
            if threshold <= 2.0 * (min(t for _, t in self.thresholds)):
                self.finish(
                    self.on_summarize()
                    if self.on_summarize
                    else "Duration sweep finished."
                )
                return
            self.duration_low_ms = self.current_duration_ms
            self.duration_high_ms = c.max_pulse_ms
            self.duration_stage = "duration_chronaxie_bisect"
            self._bisect_strength_duration()
            return
        if self.duration_stage == "duration_chronaxie_bisect":
            if threshold > 2.0 * (min(t for _, t in self.thresholds)):
                self.duration_low_ms = self.current_duration_ms
            else:
                self.duration_high_ms = self.current_duration_ms
            self._bisect_strength_duration()

    def _bisect_strength_duration(self) -> None:
        if (
            self.duration_high_ms - self.duration_low_ms
            <= self._duration_tolerance_ms()
            or self._duration_threshold_exists(
                self._effective_duration_ms(
                    0.5 * (self.duration_low_ms + self.duration_high_ms)
                )
            )
        ):
            self.finish(
                self.on_summarize() if self.on_summarize else "Duration sweep finished."
            )
            return
        self._begin_duration_search(
            0.5 * (self.duration_low_ms + self.duration_high_ms), "duration bisection"
        )

    def _update_strength_duration_metrics(self) -> None:
        if not self.thresholds:
            self.rheobase = self.chronaxie_ms = None
            return
        pairs = sorted(self.thresholds)
        self.rheobase = min(value for _, value in pairs)
        target = 2.0 * self.rheobase
        self.chronaxie_ms = None
        for (d0, a0), (d1, a1) in zip(pairs, pairs[1:], strict=False):
            if a0 == target:
                self.chronaxie_ms = d0
                return
            if (a0 - target) * (a1 - target) <= 0.0 and a0 != a1:
                self.chronaxie_ms = d0 + (target - a0) * (d1 - d0) / (a1 - a0)
                return
        if pairs[0][1] <= target:
            self.chronaxie_ms = pairs[0][0]


@dataclass(frozen=True)
class RheobaseRunConfig:
    strategy: str
    min_amplitude: float
    max_amplitude: float
    fixed_amplitude: float
    pulse_ms: float
    max_pulse_ms: float
    duration_tolerance_ms: float
    recovery_ms: float
    response_ms: float
    neuron_id: int
    threshold_mv: float
    tolerance: float
    max_iterations: int
    stim_dt: float
    n_neurons: int


class RheobaseDialog(_SelfRaisingDialog):
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
        # Cached modeless routine windows must outlive transient host windows;
        # detach an optional host parent to avoid Qt deleting the cache first.
        if parent is not None:
            self.setParent(None)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        self.editor = editor
        self.project = editor.project
        self._backend = RheobaseBackend(self)
        self._backend.on_advance = self._advance_domain
        self._backend.on_summarize = self._final_summary
        self._backend.on_summarize_fixed_amplitude = self._final_fixed_amplitude_summary
        self._backend.on_log = self._append_log
        self._backend.on_measurement = self._on_measurement_changed
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
        self.setWindowTitle("Minimal input for spiking (rheobase)")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        info_form = QtWidgets.QFormLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_stim_units = QtWidgets.QLabel("-", self)
        self.lbl_stim_dt = QtWidgets.QLabel("-", self)
        self.lbl_stim_neurons = QtWidgets.QLabel("-", self)
        info_form.addRow("Input array:", self.lbl_stim_name)
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

        # No Qt parent for this popup: a parent sets WM_TRANSIENT_FOR, and
        # some window managers (observed on KWin/X11) then treat the whole
        # chain of transient windows sharing one owner as a single "raise
        # group" - clicking one never brings it above a sibling in the same
        # chain.
        self.plot_window = _SelfRaisingDialog()
        self.plot_window.setWindowTitle("Minimal input for spiking (rheobase) — plots")
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        self._plot_window_positioned = False
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.rheobase_plot = MatplotlibWidget(self.plot_window)
        self.rheobase_plot.setMinimumSize(680, 500)
        figure = self.rheobase_plot.getFigure()
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.18)
        self.rheobase_axes = figure.add_subplot(grid[0, 0])
        self.search_axes = figure.add_subplot(grid[1, 0])
        plot_layout.addWidget(
            _wrap_in_scroll_area(self.rheobase_plot, self.plot_window)
        )
        self.plot_window.resize(720, 560)

        self.lbl_status = QtWidgets.QLabel("Rheobase: waiting for first probe", self)
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
        # Get this dialog and its plots out of the way so the user can watch
        # the stim array and traces fill up as the routine runs. Z-order
        # hints (lower()/raise()) are not reliably honored by every window
        # manager - especially for a window that still has focus from the
        # Start click - so minimizing is used instead, since window state is
        # enforced by the OS/WM rather than merely requested. Restored
        # automatically when the routine finishes (_on_backend_finished).
        self.showMinimized()
        self.plot_window.showMinimized()
        main_window = self.editor.window()
        if main_window is not None:
            main_window.raise_()
            main_window.activateWindow()
        needs_trace_init = not self.cmb_trace.currentText()
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
                    "existing stimuli before the search starts."
                )
            else:
                self._append_log(
                    f"Running an initial {duration:.3f} s simulation to "
                    "create trace monitors."
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
        stim = self._require_stim()
        config = RheobaseRunConfig(
            strategy=strategy,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            fixed_amplitude=self.spn_fixed_amplitude.value(),
            pulse_ms=self.spn_pulse_ms.value(),
            max_pulse_ms=self.spn_max_pulse_ms.value(),
            duration_tolerance_ms=self.spn_duration_tolerance_ms.value(),
            recovery_ms=self.spn_recovery_ms.value(),
            response_ms=self.spn_response_ms.value(),
            neuron_id=self.spn_neuron.value(),
            threshold_mv=self.spn_threshold_mv.value(),
            tolerance=self.spn_tolerance.value(),
            max_iterations=self.spn_max_iterations.value(),
            stim_dt=float(stim.dt),
            n_neurons=int(stim.values.shape[1]),
        )
        initial_estimate_seconds = (
            config.recovery_ms + config.pulse_ms + config.response_ms
        ) / 1000.0
        self._array_cursor = RoutineArrayCursor(
            self._append_start_sample, initial_estimate_seconds
        )
        if clear_log:
            self.log.clear()
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self._append_log(
            f"Estimated simulations for the current settings: ~{self._estimated_probe_count()}."
        )
        self._backend.begin_search(config)
        return

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
            return
        self._backend.simulation_finished()

    def _advance_domain(self) -> None:
        assert self.project is not None
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
        self._backend.record_probe_result(spike_count)

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
        # Count spikes over the pulse plus a fixed post-pulse margin, so the
        # window tracks the pulse instead of trailing a long fixed tail after a
        # short pulse (which could catch unrelated rebound activity).
        window_ms = self._current_duration_ms + self.spn_response_ms.value()
        return measure_spike_count(
            trace,
            self.spn_neuron.value(),
            self._pulse_onset_seconds,
            window_ms,
            self.spn_threshold_mv.value(),
        )

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
            (self.rheobase_axes, f"Input strength\n({units})"),
            (self.search_axes, f"Amplitude ({units})"),
        ):
            axes.set_facecolor(face)
            axes.set_ylabel(ylabel, color=text)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        self.rheobase_axes.set_title("")
        self.rheobase_axes.set_xlabel("Input duration (ms)", color=text)
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
        self._backend.finish(message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self._append_log(message)

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

    _low = backend_property("low")
    _high = backend_property("high")
    _pending_amplitude = backend_property("pending_amplitude")
    _pulse_onset_seconds = backend_property("pulse_onset_seconds")
    _duration_values_ms = backend_property("duration_values_ms")
    _duration_index = backend_property("duration_index")
    _current_duration_ms = backend_property("current_duration_ms")
    _duration_low_ms = backend_property("duration_low_ms")
    _duration_high_ms = backend_property("duration_high_ms")
    _duration_stage = backend_property("duration_stage")
    _samples = backend_property("samples")
    _thresholds = backend_property("thresholds")
    _rheobase = backend_property("rheobase")
    _chronaxie_ms = backend_property("chronaxie_ms")
    _fixed_spiking_duration_ms = backend_property("fixed_spiking_duration_ms")
    _iteration = backend_property("iteration")
    _logged_recovery_note = backend_property("logged_recovery_note")
    _resolution_warning_logged = backend_property("resolution_warning_logged")

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _on_measurement_changed(self) -> None:
        self._update_plot()

    def _on_backend_finished(self, message: str) -> None:
        # The backend can call finish() on itself (e.g. search converged)
        # without going through this dialog's _finish(), so this callback -
        # wired to RoutineBackend.on_finished - is the one hook guaranteed to
        # run on every completion path; it is what actually resets the
        # buttons and trims any padding placed ahead of the routine's last
        # block, so a stopped or completed run never leaves a long empty
        # tail. It is also what restores the dialog and its plots after
        # start() minimized them out of the way of the stim array/traces.
        self.showNormal()
        self.plot_window.showNormal()
        self.editor.reset_stim_array()
        self._array_cursor = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")

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
