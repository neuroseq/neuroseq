from __future__ import annotations

from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.dynamic_input.rheobase import count_spikes
from neuroseq.resources.analysis_plugins.epsp_amplitudes import _to_mv
from neuroseq.resources.analysis_plugins.epsp_amplitudes import detect_epsps


@dataclass(frozen=True)
class EPSPMeasurement:
    amplitude: float
    pulse_amplitudes: tuple[float, ...]
    units: str = "mV"


class NoEpspDetected(ValueError):
    """Raised when no EPSP can be associated with a probe pulse."""


def _trace_row_mv(trace: TraceData, neuron_id: int) -> tuple[np.ndarray, float]:
    matches = np.flatnonzero(trace.indices == neuron_id)
    if matches.size == 0:
        raise ValueError(f"Neuron {neuron_id} is not recorded in the trace.")
    values = np.asarray(trace.traces[int(matches[0])], dtype=float)
    if str(trace.units).strip().lower() == "mv":
        # Simulated voltage traces are stored in SI volts but labelled "mV"
        # (the trace widget applies the same convention), so a "mV" trace
        # whose magnitudes are below 2 is really in volts.
        finite = np.abs(values[np.isfinite(values)])
        if finite.size and float(np.percentile(finite, 95)) < 2.0:
            values = values * 1000.0
        return values, 1000.0 * trace.dt
    return _to_mv(values, trace.units), 1000.0 * trace.dt


def measure_spike_count(
    trace: TraceData,
    neuron_id: int,
    onset_seconds: float,
    window_ms: float,
    threshold_mv: float,
) -> int:
    """Count threshold crossings in the requested trace window."""
    trace_mv, dt_ms = _trace_row_mv(trace, neuron_id)
    onset_ms = 1000.0 * onset_seconds
    start = max(0, int(np.floor(onset_ms / dt_ms)))
    stop = min(trace_mv.size, int(np.ceil((onset_ms + window_ms) / dt_ms)) + 1)
    return count_spikes(trace_mv[start:stop], threshold_mv)


def measure_epsp(
    trace: TraceData,
    neuron_id: int,
    probe_pulse_times_s: list[float],
    baseline_ms: float,
    response_ms: float,
) -> EPSPMeasurement:
    """Measure the dominant detected EPSP after each probe pulse."""
    trace_mv, dt_ms = _trace_row_mv(trace, neuron_id)
    amplitudes: list[float] = []
    for pulse_time in probe_pulse_times_s:
        onset_ms = 1000.0 * pulse_time
        start_ms = max(0.0, onset_ms - baseline_ms)
        stop_ms = onset_ms + response_ms
        start = max(0, int(np.floor(start_ms / dt_ms)))
        stop = min(trace_mv.size, int(np.ceil(stop_ms / dt_ms)) + 1)
        if start >= trace_mv.size:
            trace_duration_ms = trace_mv.size * dt_ms
            raise ValueError(
                f"The selected trace ends at {trace_duration_ms:.3f} ms, "
                f"before the probe at {onset_ms:.3f} ms. Select the current "
                "simulation voltage trace and try again."
            )
        detection = detect_epsps(trace_mv[start:stop], dt_ms)
        relative_onset_ms = onset_ms - start * dt_ms
        tolerance_ms = 0.5 * dt_ms
        candidates = detection.amplitudes_mv[
            (detection.times_ms >= relative_onset_ms - tolerance_ms)
            & (detection.times_ms <= relative_onset_ms + response_ms + tolerance_ms)
        ]
        if candidates.size == 0:
            raise NoEpspDetected(
                f"No EPSP was detected after the probe at {onset_ms:.3f} ms."
            )
        amplitudes.append(float(np.max(candidates)))
    return EPSPMeasurement(float(np.mean(amplitudes)), tuple(amplitudes))


def measure_calcium_spike_count(
    spikes: SpikeData,
    neuron_id: int,
    onset_seconds: float,
    window_ms: float,
) -> int:
    """Count deconvolved spikes of ``neuron_id`` in ``[onset, onset + window]``.

    ``onset_seconds`` is relative to the start of the acquisition that produced ``spikes``.
    """
    stop_seconds = onset_seconds + window_ms / 1000.0
    in_window = (
        (spikes.spike_indices == neuron_id)
        & (spikes.spike_times >= onset_seconds)
        & (spikes.spike_times <= stop_seconds)
    )
    return int(np.count_nonzero(in_window))


def measure_calcium_response(
    trace: TraceData,
    neuron_id: int,
    probe_pulse_times_s: list[float],
    baseline_ms: float,
    response_ms: float,
) -> EPSPMeasurement:
    """Mean calcium rise of ``neuron_id`` after each probe pulse.

    Per pulse: peak of the trace in ``[onset, onset + response_ms]`` minus the mean of the
    ``baseline_ms`` immediately before the onset. Times are relative to the start of the
    acquisition that produced ``trace``; the amplitude is in the trace's own units.
    """
    matches = np.flatnonzero(trace.indices == neuron_id)
    if matches.size == 0:
        raise ValueError(f"Neuron {neuron_id} is not recorded in the trace.")
    values = trace.traces[int(matches[0])]
    amplitudes: list[float] = []
    for onset in probe_pulse_times_s:
        onset_idx = int(np.floor(onset / trace.dt))
        if onset_idx >= values.size:
            raise ValueError(
                f"The selected trace ends at {values.size * trace.dt:.3f} s, "
                f"before the probe at {onset:.3f} s. Select the newly imported "
                "recording and try again."
            )
        onset_idx = max(onset_idx, 0)
        base_start = max(0, int(np.floor((onset - baseline_ms / 1000.0) / trace.dt)))
        baseline = values[base_start : max(onset_idx, base_start + 1)]
        stop = min(
            values.size, int(np.ceil((onset + response_ms / 1000.0) / trace.dt)) + 1
        )
        window = values[onset_idx : max(stop, onset_idx + 1)]
        amplitudes.append(float(np.max(window) - np.mean(baseline)))
    return EPSPMeasurement(
        float(np.mean(amplitudes)), tuple(amplitudes), units=trace.units
    )
