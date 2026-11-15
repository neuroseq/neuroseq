"""Detect EPSPs in membrane-voltage traces (used by dynamic input routines and
the EPSP amplitudes analysis plugin)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import brian2.numpy_ as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class EPSPDetection:
    times_ms: np.ndarray
    amplitudes_mv: np.ndarray


def to_mv(values: np.ndarray, units: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    unit = str(units).strip().lower().replace("µ", "u").replace("μ", "u")
    # Unit labels can originate in Brian2, Igor, or user-authored imports. Strip
    # harmless display punctuation and accept both symbols and full names.
    unit = re.sub(r"[\[\](){}\s._-]", "", unit)
    factors = {
        "v": 1000.0,
        "volt": 1000.0,
        "volts": 1000.0,
        "mv": 1.0,
        "mvolt": 1.0,
        "millivolt": 1.0,
        "millivolts": 1.0,
        "uv": 0.001,
        "uvolt": 0.001,
        "microvolt": 0.001,
        "microvolts": 0.001,
    }
    if not unit:
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            return values
        # Imported traces sometimes omit unit metadata. Membrane voltages are
        # commonly stored either in SI volts (~0.07) or millivolts (~70).
        factor = 1000.0 if float(np.percentile(finite, 95)) < 2.0 else 1.0
        return values * factor
    if unit not in factors:
        raise ValueError(
            "EPSP analysis requires a voltage trace "
            f"(V, mV, or µV); this trace is labelled {units!r}."
        )
    return values * factors[unit]


def detect_epsps(trace_mv: np.ndarray, dt_ms: float) -> EPSPDetection:
    """Return EPSP peak times and local-baseline amplitudes for one trace.

    The adaptive prominence is based on the median absolute first difference.
    Peaks reaching action-potential voltages, or rising >=20 mV above their
    local baseline, are discarded.
    """
    trace = np.asarray(trace_mv, dtype=float)
    if trace.size < 3 or dt_ms <= 0 or not np.all(np.isfinite(trace)):
        return EPSPDetection(np.asarray([], dtype=float), np.asarray([], dtype=float))

    noise = float(np.median(np.abs(np.diff(trace)))) / 0.6745
    prominence_mv = max(0.1, 3.0 * noise)
    min_distance = max(1, int(round(2.0 / dt_ms)))
    peak_indices, _ = find_peaks(trace, prominence=prominence_mv, distance=min_distance)

    baseline_window = max(2, int(round(10.0 / dt_ms)))
    baseline_gap = max(1, int(round(1.0 / dt_ms)))
    # An absolute-voltage trace sits near its (negative) resting potential; an
    # offset-removed / ΔV trace hovers around zero. Only apply the absolute
    # action-potential threshold when the trace is on an absolute scale, so
    # EPSPs on offset-removed traces are not all discarded by it.
    looks_absolute = float(np.median(trace)) < -30.0
    times: list[float] = []
    amplitudes: list[float] = []
    for peak in peak_indices:
        baseline_end = peak - baseline_gap
        baseline_start = max(0, baseline_end - baseline_window)
        if baseline_end <= baseline_start:
            continue
        baseline = float(np.median(trace[baseline_start:baseline_end]))
        amplitude = float(trace[peak] - baseline)
        # Reject action potentials: an AP rises tens of mV above baseline
        # (caught regardless of DC offset) and, on an absolute trace, also
        # crosses ~-20 mV.
        is_ap = amplitude >= 20.0 or (looks_absolute and trace[peak] >= -20.0)
        if amplitude < prominence_mv or is_ap:
            continue
        times.append(float(peak) * dt_ms)
        amplitudes.append(amplitude)
    return EPSPDetection(np.asarray(times), np.asarray(amplitudes))
