from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import brian2.numpy_ as np
from matplotlib.ticker import MultipleLocator
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control.bss2_protocol import Bss2Settings
from neuroseq.device_control.syscon2_stimulation_host import routine_run_name
from neuroseq.qt_helpers import SelfRaisingDialog
from neuroseq.qt_helpers import WheelToScrollFilter
from neuroseq.qt_helpers import set_native_minimized_position
from neuroseq.qt_helpers import wrap_in_scroll_area
from neuroseq.stimulation.dynamic_input.dialog_helpers import confirm_start_routine
from neuroseq.stimulation.dynamic_input.dynamic_input_blocks import RoutineArrayCursor
from neuroseq.stimulation.dynamic_input.dynamic_input_blocks import run_to_sample
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    calcium_rise_fraction,
)
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    measure_calcium_spike_count,
)
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    measure_spike_count,
)
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import RoutineBackend
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import SimulationHost
from neuroseq.stimulation.dynamic_input.dynamic_input_routine import backend_property
from neuroseq.stimulation.dynamic_input.hardware_settings_widget import (
    HardwareSettingsWidget,
)
from neuroseq.stimulation.dynamic_input.hardware_targets import HostConnections
from neuroseq.stimulation.dynamic_input.hardware_targets import PerTargetValues
from neuroseq.stimulation.dynamic_input.hardware_targets import delivery_settings_note
from neuroseq.stimulation.dynamic_input.hardware_targets import hardware_start_sample
from neuroseq.stimulation.dynamic_input.rheobase import bisection_converged
from neuroseq.stimulation.dynamic_input.rheobase import next_bracket
from neuroseq.stimulation.stim_timing import format_milliseconds
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import snap_milliseconds_to_dt

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim_editor import StimEditor


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
        #: (shorter, longer) duration bracketing the chronaxie when only known
        #: to lie between an unfireable duration and a measured one.
        self.chronaxie_range: tuple[float, float] | None = None
        #: The chronaxie is at most this (shorter than the swept range).
        self.chronaxie_upper_bound_ms: float | None = None
        self.fixed_spiking_duration_ms: float | None = None
        self.iteration = 0
        self.logged_recovery_note = False
        self.resolution_warning_logged = False
        self.samples: list[tuple[float, float, int]] = []
        self.thresholds: list[tuple[float, float]] = []
        #: Durations at which even the maximum amplitude evoked no spike.
        self.unreached_durations: list[float] = []
        #: Spike counts of the trials of the probe being repeated.
        self.trial_counts: list[int] = []

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
        self.unreached_durations = []
        self.trial_counts = []
        self.rheobase = None
        self.chronaxie_ms = None
        self.chronaxie_range = None
        self.chronaxie_upper_bound_ms = None
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

    def _duration_measured(self, duration_ms: float) -> bool:
        """A threshold was found at this duration, or none could be reached: probing
        it again would only repeat the result."""
        return self._duration_threshold_exists(duration_ms) or any(
            abs(d - duration_ms) <= 1e-9 for d in self.unreached_durations
        )

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

    def probe(self, amplitude: float, *, repeat: bool = False) -> None:
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
        if not repeat:
            self._log(
                f"Probe {len(self.samples) + 1}: testing amplitude {amplitude:.6g} at {self.current_duration_ms:.6g} ms."
            )
        self.request_simulation(max(c.stim_dt, samples * c.stim_dt))

    def record_probe_result(self, spike_count: int) -> None:
        c = self._cfg()
        amplitude = self.pending_amplitude
        if c.repeats > 1:
            self.trial_counts.append(int(spike_count))
            if len(self.trial_counts) < c.repeats:
                self.probe(amplitude, repeat=True)
                return
            counts, self.trial_counts = self.trial_counts, []
            spiking = [count for count in counts if count > 0]
            # A response needs most trials to spike; its count is their median.
            spike_count = (
                int(np.median(spiking)) if 2 * len(spiking) > len(counts) else 0
            )
            self._log(
                f"Probe {len(self.samples) + 1}: {len(spiking)}/{len(counts)} trials "
                f"spiked (spikes per trial: {', '.join(map(str, counts))})."
            )
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
                self._handle_unreached_duration()
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

    def _handle_unreached_duration(self) -> None:
        """Even the maximum amplitude evokes no spike at the current duration.

        At the longest duration (or with a fixed duration) nothing can be measured.
        At a shorter duration of the strength-duration search the threshold is just
        above the range: keep the thresholds found so far, and carry on when that
        still decides the chronaxie (the maximum is at least twice the rheobase).
        """
        c = self._cfg()
        duration = self.current_duration_ms
        self.unreached_durations.append(duration)
        self._update_strength_duration_metrics()
        self._measurement_changed()
        message = (
            f"No spike at {duration:.6g} ms up to the maximum amplitude "
            f"{c.max_amplitude:.6g}; its threshold is above the range."
        )
        if self.duration_stage not in ("duration_min", "duration_chronaxie_bisect"):
            self.finish(f"{message} Increase the maximum amplitude.")
            return
        self._log(message)
        rheobase = min(t for _, t in self.thresholds)
        if c.max_amplitude >= 2.0 * rheobase:
            # Its threshold is above twice the rheobase: the chronaxie is longer.
            self._handle_strength_duration_threshold(above_twice_rheobase=True)
            return
        summary = (
            self.on_summarize() if self.on_summarize else "Duration sweep finished."
        )
        self.finish(
            f"{summary} To locate the chronaxie, raise the maximum amplitude to at "
            f"least {2.0 * rheobase:.6g} (twice the rheobase)."
        )

    def _handle_strength_duration_threshold(
        self, *, above_twice_rheobase: bool | None = None
    ) -> None:
        """Continue the strength-duration search after a duration's threshold
        (``above_twice_rheobase`` given instead when it was out of range)."""
        c = self._cfg()
        if above_twice_rheobase is None:
            threshold = self.thresholds[-1][1]
            above_twice_rheobase = threshold > 2.0 * min(t for _, t in self.thresholds)
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
            if not above_twice_rheobase:
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
            if above_twice_rheobase:
                self.duration_low_ms = self.current_duration_ms
            else:
                self.duration_high_ms = self.current_duration_ms
            self._bisect_strength_duration()

    def _bisect_strength_duration(self) -> None:
        if (
            self.duration_high_ms - self.duration_low_ms
            <= self._duration_tolerance_ms()
            or self._duration_measured(
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
        """Rheobase and chronaxie (where the threshold reaches twice the rheobase).

        The chronaxie is interpolated between two measured thresholds when they
        straddle twice the rheobase. When every measured threshold is below that:
        if a shorter duration could not fire even at a maximum of at least twice
        the rheobase, the chronaxie lies between it and the shortest measured
        duration (``chronaxie_range``, estimate its middle); otherwise only
        ``chronaxie_upper_bound_ms`` (the shortest measured duration) is known.
        """
        self.chronaxie_ms = None
        self.chronaxie_range = None
        self.chronaxie_upper_bound_ms = None
        if not self.thresholds:
            self.rheobase = None
            return
        pairs = sorted(self.thresholds)
        self.rheobase = min(value for _, value in pairs)
        target = 2.0 * self.rheobase
        for (d0, a0), (d1, a1) in zip(pairs, pairs[1:], strict=False):
            if a0 == target:
                self.chronaxie_ms = d0
                return
            if (a0 - target) * (a1 - target) <= 0.0 and a0 != a1:
                self.chronaxie_ms = d0 + (target - a0) * (d1 - d0) / (a1 - a0)
                return
        shortest, shortest_threshold = pairs[0]
        if shortest_threshold > target:
            return
        unreached = [
            d
            for d in self.unreached_durations
            if d < shortest and self._cfg().max_amplitude >= target
        ]
        if unreached:
            low = max(unreached)
            self.chronaxie_range = (low, shortest)
            self.chronaxie_ms = 0.5 * (low + shortest)
        else:
            self.chronaxie_upper_bound_ms = shortest


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
    #: Trials per probe amplitude; a probe counts as a response when most trials
    #: spike (noisy hardware: BrainScaleS-2, cells in a dish).
    repeats: int = 1


class RheobaseDialog(SelfRaisingDialog):
    """Closed-loop search for the smallest direct stimulus that evokes a spike."""

    simulation_requested = QtCore.Signal(float)
    AMPLITUDE_AND_DURATION_STRATEGY = "Amplitude and duration"
    FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY = "Fixed amplitude, varying duration"
    FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY = "Fixed duration, varying amplitude"
    SIMULATION_TARGET = "Brian2 simulation"
    SYSCON2_TARGET = "SysCon2 (calcium imaging)"
    BSS2_TARGET = "BrainScaleS-2 (neuromorphic)"
    DFF_READOUT = "Calcium response (ΔF/F)"
    SPIKES_READOUT = "Deconvolved spikes"
    #: Baseline before each probe pulse for the calcium response.
    CALCIUM_BASELINE_MS = 250.0
    #: Amplitudes (min, max, fixed, tolerance) a hardware target starts with: its
    #: amplitudes are fractions (0-1) of the target power / i_offset max.
    HARDWARE_AMPLITUDES = (0.1, 1.0, 0.5, 0.05)
    #: Trials per probe a hardware target starts with (its responses are noisy).
    HARDWARE_REPEATS = 3

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
        # SysCon2 target state: the stim-array range of the block in flight, and the
        # spike dataset used for the previous probe (to catch a forgotten re-import).
        self._block_range = (0, 0)
        self._last_measured_dataset: str | None = None
        self._data_before_block: dict[str, object] = {}
        #: What the host reported for the block that just completed (see
        #: SysCon2StimulationHost.block_completed); None for a manual import.
        self._block_payload: object = None
        #: End of the last block delivered to hardware (samples): the stim array
        #: keeps everything up to here when a hardware routine ends.
        self._delivered_end_sample = 0
        #: Names this SysCon2 run's folder and block files (see routine_run_name).
        self._run_name: str | None = None
        #: The (signal, slot) pairs this dialog connected, by stimulation host.
        self._host_connections = HostConnections()
        #: BrainScaleS-2 settings snapshot of the running routine (fixed scale).
        self._bss2_settings: Bss2Settings | None = None
        #: Data group collecting this run's BrainScaleS-2 readbacks.
        self._bss2_group: str | None = None
        self._bss2_warnings_logged = False
        self.setWindowTitle("Minimal input for spiking (rheobase)")
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
        self.spn_repeats = QtWidgets.QSpinBox(self)
        self.spn_repeats.setRange(1, 99)
        self.spn_repeats.setValue(1)
        self.spn_repeats.setToolTip(
            "Trials per probe amplitude. A probe counts as evoking a response when "
            "most of its trials spike, which makes the search robust to the "
            "trial-to-trial noise of hardware (odd numbers avoid ties)."
        )
        self.cmb_target = QtWidgets.QComboBox(self)
        self.cmb_target.addItems(
            [self.SIMULATION_TARGET, self.SYSCON2_TARGET, self.BSS2_TARGET]
        )
        self.cmb_target.setToolTip(
            "Brian2: simulate the probes. SysCon2: deliver each probe to the cells "
            "through SysCon2 and measure their calcium response. BrainScaleS-2: run "
            "each probe as a constant current on the neuromorphic chip (connect in "
            "Stimulation setup > BrainScaleS-2) and count its spikes."
        )
        #: Amplitude settings (min, max, fixed, tolerance) per target, so switching
        #: between Brian2 (stim units) and hardware (fractions) keeps both.
        self._target_values = PerTargetValues(
            (*self._amplitude_spins(), self.spn_repeats),
            self._target_defaults,
            self.cmb_target.currentText(),
        )
        self.cmb_target.currentTextChanged.connect(self._on_target_changed)
        self.cmb_spikes = QtWidgets.QComboBox(self)
        self.cmb_spikes.setToolTip(
            "Imported spike dataset (deconvolved calcium) holding the latest probe's "
            "recording. Its time axis must start when the sent block starts."
        )
        self.cmb_calcium_readout = QtWidgets.QComboBox(self)
        self.cmb_calcium_readout.addItems([self.DFF_READOUT, self.SPIKES_READOUT])
        self.cmb_calcium_readout.setToolTip(
            "How a probe delivered through SysCon2 counts as evoking a response: a "
            "calcium rise of at least the response threshold, or spikes deconvolved "
            "from the calcium signal (OASIS) after the pulse. Automatic blocks provide "
            "both; manual imports need the matching dataset."
        )
        self.cmb_calcium_readout.currentTextChanged.connect(
            self._update_target_controls
        )
        form.addRow("Stimulation target:", self.cmb_target)
        # BrainScaleS-2 connection, so a run need not start in the setup pane.
        self.bss2_row = QtWidgets.QWidget(self)
        bss2_layout = QtWidgets.QHBoxLayout(self.bss2_row)
        bss2_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_bss2_status = QtWidgets.QLabel("Not connected", self.bss2_row)
        self.btn_bss2_connect = QtWidgets.QPushButton("Connect", self.bss2_row)
        self.btn_bss2_connect.setToolTip(
            "Connect to BrainScaleS-2 with the settings of the BrainScaleS-2 tab "
            "(container image, setup, token)."
        )
        self.btn_bss2_connect.clicked.connect(self._connect_bss2)
        bss2_layout.addWidget(self.btn_bss2_connect)
        bss2_layout.addWidget(self.lbl_bss2_status)
        bss2_layout.addStretch(1)
        self.lbl_bss2_row = QtWidgets.QLabel("BrainScaleS-2:", self)
        form.addRow(self.lbl_bss2_row, self.bss2_row)
        form.addRow("Calcium readout:", self.cmb_calcium_readout)
        form.addRow("Trace to analyze:", self.cmb_trace)
        form.addRow("Spike data (calcium):", self.cmb_spikes)
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
        form.addRow("Repeats per probe:", self.spn_repeats)
        parameters_layout.addLayout(form)
        self.hardware = HardwareSettingsWidget(self, response_threshold=True)
        self.hardware.chk_automatic.toggled.connect(self._update_target_controls)
        parameters_layout.addWidget(self.hardware)
        parameters_layout.addStretch(1)

        self.parameters_scroll = QtWidgets.QScrollArea(self)
        self.parameters_scroll.setWidget(self.parameters_panel)
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameters_scroll.setMinimumHeight(80)
        layout.addWidget(self.parameters_scroll, 1)
        self._wheel_filter = WheelToScrollFilter(self.parameters_scroll)
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
        self.spn_repeats.valueChanged.connect(lambda _value: self._update_estimate())
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
        self.plot_window = SelfRaisingDialog()
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
        plot_layout.addWidget(wrap_in_scroll_area(self.rheobase_plot, self.plot_window))
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
        self.btn_data_imported = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Calcium data imported",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            ),
        )
        self.btn_data_imported.setToolTip(
            "SysCon2 target: click after you ran the sent block, acquired, imported the "
            "calcium data and selected the new spike dataset."
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
        self._update_sweeping_strategy_controls()
        self.refresh()
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
        set_native_minimized_position(
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
            selected_spikes = self.cmb_spikes.currentText()
            self.cmb_spikes.clear()
            self.cmb_spikes.addItems(self.project.get_data_names(SpikeData))
            self.cmb_spikes.setCurrentText(selected_spikes)
        syscon2_available = self.editor.stimulation_host is not None
        self.hardware.set_host(self.editor.stimulation_host)
        bss2_available = self.editor.bss2_host is not None
        model = cast(QtGui.QStandardItemModel, self.cmb_target.model())
        for row, available in ((1, syscon2_available), (2, bss2_available)):
            item = model.item(row)
            if item is not None:
                item.setEnabled(available)
        if (not syscon2_available and self._target_is_syscon2()) or (
            not bss2_available and self._target_is_bss2()
        ):
            self.cmb_target.setCurrentIndex(0)
        modifier = self.editor.current_stim_modifier
        if modifier is not None:
            stim = modifier.stim_array
            n_neurons = stim.values.shape[1]
            self.spn_neuron.setMaximum(max(0, n_neurons - 1))
            self.lbl_units.setText(self._amplitude_units(stim.units))
            self.lbl_fixed_units.setText(self._amplitude_units(stim.units))
            self.lbl_stim_name.setText(stim.name)
            self.lbl_stim_size.setText(f"{stim.values.shape[0]} samples")
            self.lbl_stim_units.setText(stim.units or "-")
            self.lbl_stim_dt.setText(f"{format_milliseconds(stim.dt * 1000.0)} ms")
            self.lbl_stim_neurons.setText(str(n_neurons))
        else:
            self.lbl_units.setText("")
            self.lbl_fixed_units.setText("")
            self.lbl_stim_name.setText("-")
            self.lbl_stim_size.setText("-")
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
        if self._target_is_syscon2():
            self._start_syscon2(choice, stim)
            return
        if self._target_is_bss2():
            self._start_bss2(choice)
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
        needs_trace_init = not self.cmb_trace.currentText()
        if needs_trace_init or prepare_seconds > 0.0:
            duration = max(prepare_seconds, 0.001)
            self._running = True
            self._update_target_controls()
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

    def _target_is_syscon2(self) -> bool:
        return self.cmb_target.currentText() == self.SYSCON2_TARGET

    def _target_is_bss2(self) -> bool:
        return self.cmb_target.currentText() == self.BSS2_TARGET

    def _amplitude_units(self, stim_units: str) -> str:
        """What amplitudes mean for the current target."""
        if self._target_is_bss2():
            return "× i_offset max"
        if self._target_is_syscon2():
            return "× target power"
        return stim_units

    def _amplitude_spins(self) -> tuple[QtWidgets.QDoubleSpinBox, ...]:
        return (
            self.spn_min_amplitude,
            self.spn_max_amplitude,
            self.spn_fixed_amplitude,
            self.spn_tolerance,
        )

    def _target_defaults(self, target: str) -> tuple[float, ...] | None:
        """Amplitudes and repeats a target starts with: fractions (0-1) and three
        trials on hardware; Brian2 keeps the values it was built with."""
        if target == self.SIMULATION_TARGET:
            return None
        return (*self.HARDWARE_AMPLITUDES, self.HARDWARE_REPEATS)

    def _on_target_changed(self, target: str) -> None:
        """Keep each target's amplitudes and repeats: Brian2 uses stim units, the
        hardware targets fractions (0-1), so one set never fits both."""
        self._target_values.switch(target)
        self.refresh()
        self._update_target_controls()

    def _hardware_name(self) -> str:
        return "BrainScaleS-2" if self._target_is_bss2() else "SysCon2"

    def _automatic(self) -> bool:
        """SysCon2 blocks are run and recorded without the user (issue #651)."""
        return self._target_is_syscon2() and self.hardware.automatic

    def _dff_readout(self) -> bool:
        return self.cmb_calcium_readout.currentText() == self.DFF_READOUT

    def _update_target_controls(self, *_args: object) -> None:
        syscon2 = self._target_is_syscon2()
        bss2 = self._target_is_bss2()
        automatic = self._automatic()
        self.btn_syscon_dt.setVisible(syscon2)
        self.btn_syscon_dt.setEnabled(not self._running)
        self.hardware.setVisible(syscon2)
        self.hardware.setEnabled(not self._running)
        # Requests and measurements both depend on the readout: fixed while running.
        self.cmb_calcium_readout.setEnabled(syscon2 and not self._running)
        dff = self._dff_readout()
        self.hardware.show_readout(dff)
        # Automatic blocks bring their own datasets; manual ones are picked here.
        # BrainScaleS-2 blocks bring their own spikes, so no dataset is picked.
        self.cmb_trace.setEnabled(not bss2 and (not syscon2 or (dff and not automatic)))
        self.spn_threshold_mv.setEnabled(not syscon2 and not bss2)
        self.cmb_spikes.setEnabled(syscon2 and not dff and not automatic)
        self.btn_data_imported.setVisible(syscon2 and not automatic)
        # The routine dispatches on the target; it must not change mid-run.
        self.cmb_target.setEnabled(not self._running)
        self.lbl_bss2_row.setVisible(bss2)
        self.bss2_row.setVisible(bss2)
        if syscon2 or bss2:
            self._connect_hosts()
        self._show_bss2_connection()

    def _hosts(self) -> list[Any]:
        """The hardware hosts available to this dialog (SysCon2, BrainScaleS-2)."""
        return [
            host
            for host in (self.editor.stimulation_host, self.editor.bss2_host)
            if host is not None
        ]

    def _host_slots(self, host: Any) -> list[tuple[Any, Callable]]:
        slots: list[tuple[Any, Callable]] = [
            (host.block_failed, self._on_block_failed),
            (host.block_completed, self._on_block_completed),
        ]
        syscon2 = self.editor.stimulation_host
        if syscon2 is not None and host is syscon2:
            # Only SysCon2 reports imported (not yet run) blocks.
            slots.append((syscon2.block_sent, self._on_block_sent))
        bss2 = self.editor.bss2_host
        if bss2 is not None and host is bss2:
            slots.append((bss2.connection_changed, self._show_bss2_connection))
        return slots

    def _connect_bss2(self) -> None:
        host = self.editor.bss2_host
        if host is not None:
            host.connect_hardware()
        self._show_bss2_connection()

    def _show_bss2_connection(self, *_args: object) -> None:
        host = self.editor.bss2_host
        if host is None:
            return
        self.lbl_bss2_status.setText(host.connection_status())
        self.btn_bss2_connect.setEnabled(host.can_connect and not self._running)

    def _connect_hosts(self) -> None:
        for host in self._hosts():
            self._host_connections.connect(host, self._host_slots(host))

    def _disconnect_hosts(self) -> None:
        self._host_connections.disconnect_all()

    def dispose(self) -> None:
        """Stop a running routine and stop following the stimulation hosts."""
        if self._running:
            self._finish("Routine closed.")
        self._disconnect_hosts()

    def _fix_syscon_dt(self) -> bool:
        if self._running:
            return False
        if not self.editor.prepare_syscon_timestep(self):
            return False
        self.refresh()
        return True

    def _start_syscon2(self, choice: str, stim: StimArray) -> None:
        """Begin a routine whose probes are delivered through SysCon2."""
        if not self._fix_syscon_dt():
            return
        host = self.editor.stimulation_host
        if host is None or host.phase != "idle":
            self._append_log("SysCon2 is not ready (a block is still in flight).")
            return
        if self._automatic():
            reason = host.recording_unavailable()
            if reason is not None:
                self._append_log(reason)
                return
        self._append_start_sample = hardware_start_sample(self.editor, choice)
        # Nothing delivered yet: ending now keeps the array up to here.
        self._delivered_end_sample = self._append_start_sample
        self._last_measured_dataset = None
        self._run_name = routine_run_name("rheobase")
        self._append_log(
            f"Block scripts are kept in {host.run_directory(self._run_name)}."
        )
        self._append_log(delivery_settings_note(host))
        if self._automatic():
            self._append_log(
                "SysCon2 target: amplitudes are fractions (0-1) of the export target "
                "power. Each probe is uploaded and run as its own block while ScanImage "
                "records it, triggered by the sequence; its ΔF/F traces are extracted "
                "with the project's Cellpose masks. The post-pulse window should span a "
                "few imaging frames."
            )
        else:
            self._append_log(
                "SysCon2 target: amplitudes are fractions (0-1) of the export target "
                "power. Each probe is sent as its own block; the imported data's time "
                "axis must start when that block starts, and the post-pulse window "
                "should span a few imaging frames."
            )
        self._begin_search(clear_log=False)

    def _start_bss2(self, choice: str) -> None:
        """Begin a routine whose probes run on BrainScaleS-2."""
        host = self.editor.bss2_host
        if host is None:
            self._append_log("BrainScaleS-2 is not available.")
            return
        reason = host.unavailable()
        if reason is not None:
            self._append_log(reason)
            return
        self._append_start_sample = hardware_start_sample(self.editor, choice)
        # Nothing delivered yet: ending now keeps the array up to here.
        self._delivered_end_sample = self._append_start_sample
        # One scale and speed-up for the whole search, whatever happens in the tab.
        self._bss2_settings = host.current_settings()
        self._bss2_warnings_logged = False
        self._bss2_group = f"BSS-2 rheobase {datetime.now():%Y-%m-%d %H-%M-%S}"
        self._append_log(
            "BrainScaleS-2 target: amplitudes are fractions (0-1) of the BrainScaleS-2 "
            f"tab's i_offset max ({self._bss2_settings.i_offset_max}). Each probe runs as its own "
            "constant-current block on the chip, starting from rest; stim neuron i is "
            "hardware neuron i, and its spikes after the pulse are counted."
        )
        self._begin_search(clear_log=False)

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
        if self._target_is_syscon2() or self._target_is_bss2():
            amplitudes = (
                [self.spn_fixed_amplitude.value()]
                if fixed_amplitude
                else [min_amplitude, max_amplitude]
            )
            if min(amplitudes) < 0.0 or max(amplitudes) > 1.0:
                self._finish(
                    "For SysCon2 the amplitudes must lie within 0-1 "
                    "(fraction of the export target power)."
                    if self._target_is_syscon2()
                    else "For BrainScaleS-2 the amplitudes must lie within 0-1 "
                    "(fraction of i_offset max)."
                )
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
            repeats=self.spn_repeats.value(),
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
        self._update_target_controls()
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
        start = self._array_cursor.place_block(self.editor, stim, block)
        self._block_range = (start, start + block.shape[0])
        return start

    def _request_simulation(self, seconds: float) -> None:
        self._backend.request_simulation(seconds)

    def request_simulation(self, seconds: float) -> None:
        if self._target_is_bss2():
            self._request_bss2_block()
            return
        if not self._target_is_syscon2():
            self.simulation_requested.emit(seconds)
            return
        host = self.editor.stimulation_host
        if host is None or self.project is None:
            self._finish("SysCon2 is not available.")
            return
        data_type = TraceData if self._dff_readout() else SpikeData
        self._data_before_block = {
            name: self.project.get_data(name, data_type)
            for name in self.project.get_data_names(data_type)
        }
        start, stop = self._block_range
        stim = self._require_stim()
        automatic = self._automatic()
        self._append_log(
            "Exporting the probe; SysCon2 runs it while ScanImage records..."
            if automatic
            else "Exporting probe and sending it to SysCon2..."
        )
        # A failure is reported through host.block_failed -> _on_block_failed.
        host.send_block(
            self.project,
            stim,
            start,
            stop,
            "rheobase",
            preserve_amplitude=True,
            owner=self,
            automatic=automatic,
            run_name=self._run_name,
            recording=(
                self.hardware.block_recording(stim, start, stop, "rheobase")
                if automatic
                else None
            ),
        )

    def _on_block_sent(self, owner: object, _result: object) -> None:
        if owner is not self or not self._running or self._automatic():
            return
        self.refresh()
        self._append_log(
            "Probe sent to SysCon2. Run it and acquire, import the calcium data, select "
            "the new dataset, then click 'Calcium data imported'."
        )
        self.btn_data_imported.setEnabled(True)

    def _request_bss2_block(self) -> None:
        host = self.editor.bss2_host
        if host is None or self.project is None:
            self._finish("BrainScaleS-2 is not available.")
            return
        start, stop = self._block_range
        self._append_log("Running the probe on BrainScaleS-2...")
        # A failure is reported through host.block_failed -> _on_block_failed.
        host.send_block(
            self.project,
            self._require_stim(),
            start,
            stop,
            "rheobase",
            owner=self,
            record_neuron=self.spn_neuron.value(),
            settings=self._bss2_settings,
            # One data group for the whole run, not one per probe.
            group=self._bss2_group,
        )

    def _on_block_failed(self, owner: object, message: str) -> None:
        if owner is self and self._running:
            self._finish(f"{self._hardware_name()} failed: {message}")

    def _on_block_completed(self, owner: object, payload: object) -> None:
        if owner is self and self._running:
            # Kept in the array when the routine ends (see _on_backend_finished).
            self._delivered_end_sample = self._block_range[1]
            self._block_payload = payload
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
        if self._target_is_syscon2():
            self._advance_syscon2()
            return
        if self._target_is_bss2():
            self._advance_bss2()
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

    def _advance_bss2(self) -> None:
        """Count the probed neuron's spikes in the block BrainScaleS-2 just ran."""
        assert self.project is not None
        payload = self._block_payload
        self._block_payload = None
        if isinstance(payload, dict) and not self._bss2_warnings_logged:
            for warning in payload.get("warnings") or ():
                self._append_log(f"BrainScaleS-2: {warning}")
            self._bss2_warnings_logged = True
        name = payload.get("spikes") if isinstance(payload, dict) else None
        spikes = payload.get("block_spikes") if isinstance(payload, dict) else None
        if not isinstance(spikes, SpikeData):
            self._finish(
                "The BrainScaleS-2 readback of this probe is missing from the "
                "project; the routine was stopped."
            )
            return
        self.refresh()
        # The readback's time axis starts with the block.
        onset = self._pulse_onset_seconds - self._block_range[0] * float(
            self._require_stim().dt
        )
        window_ms = self._current_duration_ms + self.spn_response_ms.value()
        count = measure_calcium_spike_count(
            spikes, self.spn_neuron.value(), onset, window_ms
        )
        self._append_log(f"Spikes after the probe on BrainScaleS-2: {count}.")
        self._last_measured_dataset = str(name)
        self._backend.record_probe_result(count)

    def _advance_syscon2(self) -> None:
        assert self.project is not None
        self.refresh()
        payload = self._block_payload
        self._block_payload = None
        dff = self._dff_readout()
        combo = self.cmb_trace if dff else self.cmb_spikes
        key = "trace" if dff else "spikes"
        if isinstance(payload, dict) and payload.get(key):
            # An automatic block names the datasets extracted from its recording.
            combo.setCurrentText(str(payload[key]))
        elif self._automatic():
            self._finish(
                "The recorded probe produced no calcium trace."
                if dff
                else "The recorded probe produced no deconvolved spikes."
            )
            return
        else:
            self._select_newest_if_stale(combo)
        name = combo.currentText()
        if not name:
            self.btn_data_imported.setEnabled(True)
            self._append_log(
                "Select the imported calcium trace first."
                if dff
                else "Select the imported spike dataset first."
            )
            self._reawait_acquisition()
            return
        if not self._is_fresh_dataset(name):
            self._append_log(
                f"'{name}' predates this probe or was already used; import the new "
                "recording, select it, and click again."
            )
            self._reawait_acquisition()
            return
        block_start_seconds = self._block_range[0] * float(self._require_stim().dt)
        onset = self._pulse_onset_seconds - block_start_seconds
        window_ms = self._current_duration_ms + self.spn_response_ms.value()
        if dff:
            trace = self.project.get_data(name, TraceData)
            if trace is None:
                self._finish(f"Trace '{name}' is unavailable.")
                return
            try:
                rise = calcium_rise_fraction(
                    trace,
                    self.spn_neuron.value(),
                    onset,
                    self.CALCIUM_BASELINE_MS,
                    window_ms,
                )
            except ValueError as exc:
                self._finish(f"Calcium measurement failed: {exc}")
                return
            self._append_log(f"Calcium rise after the probe: {100 * rise:.1f} % ΔF/F.")
            response = int(rise >= self.hardware.response_threshold)
        else:
            spikes = self.project.get_data(name, SpikeData)
            if spikes is None:
                self._finish(f"Spike dataset '{name}' is unavailable.")
                return
            frame_seconds = self._frame_period(
                str(payload["trace"])
                if isinstance(payload, dict) and payload.get("trace")
                else None,
                name,
            )
            response = measure_calcium_spike_count(
                spikes, self.spn_neuron.value(), onset, window_ms, frame_seconds
            )
            self._append_log(f"Deconvolved spikes after the probe: {response}.")
        self._last_measured_dataset = name
        self._backend.record_probe_result(response)

    def _frame_period(self, trace_name: str | None, spikes_name: str) -> float:
        """Imaging frame period of deconvolved spikes (they are timed at the start of
        their frame): from the block's trace, else from a trace in the spikes' data
        group (as the TIFF/Suite2p import adds both); 0 when unknown."""
        assert self.project is not None
        if trace_name:
            trace = self.project.get_data(trace_name, TraceData)
            if trace is not None:
                return float(trace.dt)
        experiment_name = spikes_name.split(" :: ", 1)[0]
        experiment = getattr(self.project, "experiments", {}).get(experiment_name)
        for datum in experiment.data if experiment is not None else ():
            if isinstance(datum, TraceData):
                return float(datum.dt)
        return 0.0

    def _is_fresh_dataset(self, name: str) -> bool:
        if self.project is None or not name or name.startswith("Simulation :: "):
            return False
        data_type = TraceData if self._dff_readout() else SpikeData
        data = self.project.get_data(name, data_type)
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
        if self._automatic():
            # Nobody imports automatic blocks by hand, so waiting would never end.
            self._finish(
                "The data recorded for this block is missing from the project; "
                "the routine was stopped."
            )
            return
        host = self.editor.stimulation_host
        if host is not None:
            host.reawait(self)
        self._backend.awaiting_simulation = True
        self.btn_data_imported.setEnabled(True)

    def _final_summary(self) -> str:
        if self._rheobase is None:
            return "Duration sweep finished without a rheobase estimate."
        summary = f"Duration sweep finished; rheobase ~= {self._rheobase:.6g} {self._unit_label()}"
        summary += self._chronaxie_text()
        unreached = self._backend.unreached_durations
        if unreached:
            summary += (
                "; no spike up to the maximum amplitude at "
                + ", ".join(f"{d:.6g}" for d in sorted(unreached))
                + " ms"
            )
        return summary + "."

    def _chronaxie_text(self) -> str:
        """The chronaxie as measured: a value, a bracket, or only a bound."""
        backend = self._backend
        if backend.chronaxie_range is not None:
            low, high = backend.chronaxie_range
            return (
                f", chronaxie between {low:.6g} and {high:.6g} ms (no spike at "
                f"{low:.6g} ms up to the maximum amplitude)"
            )
        if self._chronaxie_ms is not None:
            return f", chronaxie ~= {self._chronaxie_ms:.6g} ms"
        if backend.chronaxie_upper_bound_ms is not None:
            return (
                f", chronaxie <= {backend.chronaxie_upper_bound_ms:.6g} ms (below the "
                "swept range: even the shortest pulse needs less than twice the "
                "rheobase; lower the minimum pulse duration to locate it)"
            )
        return "; chronaxie is outside the swept duration range"

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
        unreached = self._backend.unreached_durations
        if unreached:
            self.rheobase_axes.scatter(
                unreached,
                [self.spn_max_amplitude.value()] * len(unreached),
                marker="^",
                color="#bd2c2c",
                s=60,
                label="Threshold above range",
            )
        if self._samples:
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
            if self._chronaxie_ms is not None or len(self._thresholds) > 1:
                text += self._chronaxie_text().replace(", ", "; ", 1)
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
        for host in self._hosts():
            host.cancel(owner=self)
        self.btn_data_imported.setEnabled(False)
        self._backend.finish(message)
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
        # tail. It is also what restores the dialog after start() minimized
        # it out of the way of the stim array/traces.
        self.showNormal()
        if self._target_is_syscon2() or self._target_is_bss2():
            # Delivered blocks are not simulated: keep them, drop only the
            # padding (and a block placed but never delivered) after them.
            dt = float(self._require_stim().dt)
            self.editor.trim_stim_array_after(self._delivered_end_sample * dt)
        else:
            self.editor.reset_stim_array()
        self._array_cursor = None
        self.btn_data_imported.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self._update_target_controls()

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
        return self.spn_repeats.value() * self._estimated_probe_amplitudes()

    def _estimated_probe_amplitudes(self) -> int:
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
