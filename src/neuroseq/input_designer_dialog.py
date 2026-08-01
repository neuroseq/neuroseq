from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brian2.numpy_ as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.resources.analysis_plugins.epsp_amplitudes import _to_mv
from neuroseq.resources.analysis_plugins.epsp_amplitudes import detect_epsps

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray
    from neuroseq.stim_editor import StimEditor


@dataclass(frozen=True)
class EPSPMeasurement:
    amplitude: float
    pulse_amplitudes: tuple[float, ...]
    units: str = "mV"


class InputDesignerDialog(QtWidgets.QDialog):
    """Closed-loop EPSP targeting with an STDP induction protocol."""

    simulation_requested = QtCore.Signal(float)
    PROBE_PULSES = 5
    PROBE_PERIOD_MS = 100.0
    PULSE_MS = 1.0
    LEAD_MS = 20.0
    RESPONSE_TAIL_MS = 50.0

    def __init__(
        self,
        editor: StimEditor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.editor = editor
        self.project = editor.project
        self._running = False
        self._phase = "idle"
        self._iteration = 0
        self._probe_pulse_times: list[float] = []
        self._baseline_epsp: float | None = None
        self._epsp_percentages: list[float] = []
        self.setWindowTitle("Input Designer (beta)")
        self.resize(650, 720)

        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Target a postsynaptic EPSP with five-pulse measurements and "
            "single pre/post STDP induction pairings.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QtWidgets.QFormLayout()
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.spn_presynaptic = QtWidgets.QSpinBox(self)
        self.spn_presynaptic.setMinimum(0)
        self.spn_presynaptic.setValue(0)
        self.spn_postsynaptic = QtWidgets.QSpinBox(self)
        self.spn_postsynaptic.setMinimum(0)
        self.spn_postsynaptic.setValue(1)
        self.spn_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_amplitude.setRange(0.0, 1e12)
        self.spn_amplitude.setValue(1.0)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_amplitude)
        amplitude_row.addWidget(self.lbl_units)
        self.spn_pairing_interval_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pairing_interval_ms.setRange(0.001, 1000.0)
        self.spn_pairing_interval_ms.setValue(10.0)
        self.spn_pairing_interval_ms.setSuffix(" ms")
        self.spn_target = QtWidgets.QDoubleSpinBox(self)
        self.spn_target.setRange(0.0, 10_000.0)
        self.spn_target.setValue(120.0)
        self.spn_target.setSuffix(" %")
        self.spn_tolerance = QtWidgets.QDoubleSpinBox(self)
        self.spn_tolerance.setRange(0.001, 1000.0)
        self.spn_tolerance.setValue(5.0)
        self.spn_tolerance.setSuffix(" %")
        self.spn_max_iterations = QtWidgets.QSpinBox(self)
        self.spn_max_iterations.setRange(1, 10_000)
        self.spn_max_iterations.setValue(20)
        self.chk_follow_input = QtWidgets.QCheckBox(
            "Follow new input with three blocks of history", self
        )
        self.chk_follow_input.setChecked(True)
        form.addRow("Trace to analyze:", self.cmb_trace)
        form.addRow("Presynaptic neuron ID:", self.spn_presynaptic)
        form.addRow("Postsynaptic neuron ID:", self.spn_postsynaptic)
        form.addRow("Pulse amplitude:", amplitude_row)
        form.addRow("Pre/post pulse interval:", self.spn_pairing_interval_ms)
        form.addRow("Target EPSP amplitude:", self.spn_target)
        form.addRow("Tolerance:", self.spn_tolerance)
        form.addRow("Maximum induction pairings:", self.spn_max_iterations)
        form.addRow("Live view:", self.chk_follow_input)
        layout.addLayout(form)

        self.epsp_plot = pg.PlotWidget(self)
        self.epsp_plot.setLabel("left", "EPSP amplitude", units="%")
        self.epsp_plot.setLabel("bottom", "Measurement")
        self.epsp_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.epsp_plot, 2)
        self.lbl_relative_difference = QtWidgets.QLabel(
            "Relative difference from target: waiting for baseline", self
        )
        layout.addWidget(self.lbl_relative_difference)
        self.difference_plot = pg.PlotWidget(self)
        self.difference_plot.setLabel("left", "Current − target", units="%")
        self.difference_plot.setLabel("bottom", "Measurement")
        self.difference_plot.showGrid(x=True, y=True, alpha=0.25)
        self.difference_plot.addItem(pg.InfiniteLine(angle=0, pen=pg.mkPen("k")))
        layout.addWidget(self.difference_plot, 1)
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.btn_start = buttons.addButton(
            "Start routine", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.btn_stop = buttons.addButton(
            "Stop", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        self.project = self.editor.project
        selected = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        if self.project is None:
            return
        self.cmb_trace.addItems(self.project.get_data_names(TraceData))
        self.cmb_trace.setCurrentText(selected)
        modifier = self.editor.current_stim_modifier
        if modifier is not None:
            stim = modifier.stim_array
            maximum = max(0, stim.values.shape[1] - 1)
            self.spn_presynaptic.setMaximum(maximum)
            self.spn_postsynaptic.setMaximum(maximum)
            self.lbl_units.setText(stim.units)

    def start(self) -> None:
        self.refresh()
        if self.project is None or self.editor.current_stim_modifier is None:
            self._append_log("No stimulation array is selected.")
            return
        if not self.cmb_trace.currentText():
            self._append_log("No simulation trace monitor is available.")
            return
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._append_log("Pre- and postsynaptic neuron IDs must differ.")
            return
        self._running = True
        self._iteration = 0
        self._baseline_epsp = None
        self._epsp_percentages = []
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()
        self._update_plots()
        self._append_probe()

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

    def _append_probe(self) -> None:
        stim = self._require_stim()
        pulse_n = max(1, round(self.PULSE_MS / 1000.0 / stim.dt))
        lead_n = max(1, round(self.LEAD_MS / 1000.0 / stim.dt))
        period_n = max(pulse_n + 1, round(self.PROBE_PERIOD_MS / 1000.0 / stim.dt))
        tail_n = max(1, round(self.RESPONSE_TAIL_MS / 1000.0 / stim.dt))
        samples = lead_n + (self.PROBE_PULSES - 1) * period_n + pulse_n + tail_n
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        pulse_offsets = [
            lead_n + index * period_n for index in range(self.PROBE_PULSES)
        ]
        for offset in pulse_offsets:
            block[offset : offset + pulse_n, self.spn_presynaptic.value()] = (
                self.spn_amplitude.value()
            )
        block_start = self._append_block(block)
        self._probe_pulse_times = [
            (block_start + offset) * stim.dt for offset in pulse_offsets
        ]
        self._phase = "baseline_probe" if self._baseline_epsp is None else "post_probe"
        self._append_log("Appended five presynaptic probe pulses.")
        self._run_to_sample(block_start + samples)

    def _append_pairing(self, potentiate: bool) -> None:
        stim = self._require_stim()
        pulse_n = max(1, round(self.PULSE_MS / 1000.0 / stim.dt))
        lead_n = max(1, round(self.LEAD_MS / 1000.0 / stim.dt))
        gap_n = max(1, round(self.spn_pairing_interval_ms.value() / 1000.0 / stim.dt))
        samples = (
            lead_n
            + gap_n
            + pulse_n
            + max(1, round(self.RESPONSE_TAIL_MS / 1000.0 / stim.dt))
        )
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        first = (
            self.spn_presynaptic.value()
            if potentiate
            else self.spn_postsynaptic.value()
        )
        second = (
            self.spn_postsynaptic.value()
            if potentiate
            else self.spn_presynaptic.value()
        )
        block[lead_n : lead_n + pulse_n, first] = self.spn_amplitude.value()
        block[lead_n + gap_n : lead_n + gap_n + pulse_n, second] = (
            self.spn_amplitude.value()
        )
        block_start = self._append_block(block)
        self._phase = "pairing"
        direction = (
            "potentiation (pre → post)" if potentiate else "depression (post → pre)"
        )
        self._append_log(f"Appended one {direction} pairing.")
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
            history_start = max(0, start - 3 * block.shape[0])
            self.editor.controller.focus_time_range(
                history_start * stim.dt, (start + block.shape[0]) * stim.dt
            )
        return start

    def _run_to_sample(self, stop_sample: int) -> None:
        stim = self._require_stim()
        current_time = self._require_project().b2_model.get_sim_time_secs()
        run_seconds = max(stim.dt, stop_sample * stim.dt - current_time)
        self.simulation_requested.emit(run_seconds)

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if self._phase == "pairing":
            self._append_probe()
            return
        trace = self.project.get_data(self.cmb_trace.currentText(), TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return
        try:
            measurement = self._measure_epsps(trace)
        except ValueError as exc:
            self._finish(f"EPSP measurement failed: {exc}")
            return
        if self._baseline_epsp is None:
            if measurement.amplitude <= 0.0:
                self._finish(
                    "Baseline EPSP must be positive before relative targeting."
                )
                return
            self._baseline_epsp = measurement.amplitude
            percentage = 100.0
            self._append_log(
                f"Baseline EPSP: {measurement.amplitude:.6g} {measurement.units} "
                "(five-pulse average = 100%)."
            )
        else:
            percentage = 100.0 * measurement.amplitude / self._baseline_epsp
            self._append_log(
                f"Current EPSP: {measurement.amplitude:.6g} {measurement.units} "
                f"({percentage:.3f}% of baseline; change from initial "
                f"{percentage - 100.0:+.3f}%)."
            )
        self._epsp_percentages.append(percentage)
        self._update_plots()
        difference = percentage - self.spn_target.value()
        if abs(difference) < self.spn_tolerance.value():
            self._finish("EPSP is within tolerance; target achieved.")
            return
        if self._iteration >= self.spn_max_iterations.value():
            self._finish("Maximum induction pairings reached.")
            return
        self._iteration += 1
        self._append_pairing(potentiate=difference < 0.0)

    def _measure_epsps(self, trace: TraceData) -> EPSPMeasurement:
        """Associate the shared EPSP plugin's detections with probe pulses."""
        neuron_id = self.spn_postsynaptic.value()
        matches = np.flatnonzero(trace.indices == neuron_id)
        if matches.size == 0:
            raise ValueError(
                f"Postsynaptic neuron {neuron_id} is not recorded in the "
                "selected trace."
            )
        trace_mv = _to_mv(trace.traces[int(matches[0])], trace.units)
        dt_ms = 1000.0 * trace.dt
        baseline_ms = 12.0
        response_ms = 40.0
        amplitudes: list[float] = []
        for pulse_time in self._probe_pulse_times:
            onset_ms = 1000.0 * pulse_time
            start_ms = max(0.0, onset_ms - baseline_ms)
            stop_ms = onset_ms + response_ms
            start = max(0, int(np.floor(start_ms / dt_ms)))
            stop = min(trace_mv.size, int(np.ceil(stop_ms / dt_ms)) + 1)
            detection = detect_epsps(trace_mv[start:stop], dt_ms)
            relative_onset_ms = onset_ms - start * dt_ms
            sample_tolerance_ms = 0.5 * dt_ms
            candidates = detection.amplitudes_mv[
                (detection.times_ms >= relative_onset_ms - sample_tolerance_ms)
                & (
                    detection.times_ms
                    <= relative_onset_ms + response_ms + sample_tolerance_ms
                )
            ]
            if candidates.size == 0:
                raise ValueError(
                    f"No EPSP was detected after the probe at {onset_ms:.3f} ms."
                )
            amplitudes.append(float(candidates[0]))
        return EPSPMeasurement(
            amplitude=float(np.mean(amplitudes)),
            pulse_amplitudes=tuple(amplitudes),
        )

    def _update_plots(self) -> None:
        x = np.arange(1, len(self._epsp_percentages) + 1)
        values = np.asarray(self._epsp_percentages)
        target = self.spn_target.value()
        self.epsp_plot.clear()
        self.epsp_plot.plot(x, values, pen=pg.mkPen("b", width=2), symbol="o")
        if values.size:
            self.target_line = pg.InfiniteLine(
                pos=target,
                angle=0,
                pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.epsp_plot.addItem(self.target_line)
        self.difference_plot.clear()
        self.difference_plot.addItem(pg.InfiniteLine(angle=0, pen=pg.mkPen("k")))
        self.difference_plot.plot(
            x, values - target, pen=pg.mkPen("m", width=2), symbol="o"
        )
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

    def simulation_failed(self, message: str) -> None:
        if self._running:
            self._finish(f"Simulation failed: {message}")

    def _finish(self, message: str) -> None:
        self._running = False
        self._phase = "idle"
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(message)

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)
