from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from neuroseq.example_projects import _FUER_ELISE_INTRO
from neuroseq.stim import DrawStimType


def _validate_common(
    n_neurons: int,
    pulse_ms: float,
    ipi_ms: float,
    repeats: int,
    dt_seconds: float,
) -> tuple[int, int]:
    if n_neurons < 1:
        raise ValueError("n_neurons must be positive")
    if pulse_ms <= 0.0:
        raise ValueError("pulse_ms must be positive")
    if ipi_ms < 0.0:
        raise ValueError("ipi_ms must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    pulse_n = int(round(pulse_ms / 1000.0 / dt_seconds))
    ipi_n = int(round(ipi_ms / 1000.0 / dt_seconds))
    return pulse_n, ipi_n


def pattern_total_samples(
    *,
    pulse_ms: float,
    ipi_ms: float,
    repeats: int,
    dt_seconds: float,
    events_per_repeat: int = 1,
    n_neurons: int = 1,
) -> int:
    pulse_n, ipi_n = _validate_common(
        n_neurons=n_neurons,
        pulse_ms=pulse_ms,
        ipi_ms=ipi_ms,
        repeats=repeats,
        dt_seconds=dt_seconds,
    )
    return repeats * events_per_repeat * (pulse_n + ipi_n)


def generate_sequential(
    n_neurons: int,
    pulse_ms: float,
    ipi_ms: float,
    magnitude: float,
    dt_seconds: float,
) -> np.ndarray:
    pulse_n, ipi_n = _validate_common(
        n_neurons=n_neurons,
        pulse_ms=pulse_ms,
        ipi_ms=ipi_ms,
        repeats=1,
        dt_seconds=dt_seconds,
    )
    total_n = n_neurons * (pulse_n + ipi_n)
    values = np.zeros((total_n, n_neurons), dtype=float)
    for neuron_index in range(n_neurons):
        t0 = neuron_index * (pulse_n + ipi_n)
        values[t0 : t0 + pulse_n, neuron_index] = magnitude
    return values


def generate_simultaneous(
    n_neurons: int,
    pulse_ms: float,
    ipi_ms: float,
    magnitude: float,
    repeats: int,
    dt_seconds: float,
) -> np.ndarray:
    pulse_n, ipi_n = _validate_common(
        n_neurons=n_neurons,
        pulse_ms=pulse_ms,
        ipi_ms=ipi_ms,
        repeats=repeats,
        dt_seconds=dt_seconds,
    )
    total_n = repeats * (pulse_n + ipi_n)
    values = np.zeros((total_n, n_neurons), dtype=float)
    for repeat_index in range(repeats):
        t0 = repeat_index * (pulse_n + ipi_n)
        values[t0 : t0 + pulse_n, :] = magnitude
    return values


def generate_fuer_elise(
    n_neurons: int,
    pulse_ms: float,
    ipi_ms: float,
    magnitude: float,
    repeats: int,
    dt_seconds: float,
    pulse_repetitions: int = 1,
) -> np.ndarray:
    pulse_n, ipi_n = _validate_common(
        n_neurons=max(1, n_neurons),
        pulse_ms=pulse_ms,
        ipi_ms=ipi_ms,
        repeats=repeats,
        dt_seconds=dt_seconds,
    )
    if pulse_repetitions < 1:
        raise ValueError("pulse_repetitions must be at least 1")
    n_events = len(_FUER_ELISE_INTRO)
    event_n = pulse_repetitions * (pulse_n + ipi_n)
    total_n = repeats * n_events * event_n
    values = np.zeros((total_n, n_neurons), dtype=float)
    for repeat_index in range(repeats):
        repeat_offset = repeat_index * n_events * event_n
        for event_index, event in enumerate(_FUER_ELISE_INTRO):
            pitches = event if isinstance(event, tuple) else (event,)
            for pulse_index in range(pulse_repetitions):
                t0 = (
                    repeat_offset
                    + event_index * event_n
                    + pulse_index * (pulse_n + ipi_n)
                )
                for pitch in pitches:
                    if 0 <= pitch < n_neurons:
                        values[t0 : t0 + pulse_n, pitch] = magnitude
    return values


def _per_pulse_magnitudes(
    profile: DrawStimType,
    value1: float,
    value2: float,
    repeats: int,
) -> np.ndarray:
    if profile == DrawStimType.Constant:
        return np.full(repeats, value1, dtype=float)
    if profile == DrawStimType.Linear:
        return np.linspace(value1, value2, repeats, dtype=float)
    if profile == DrawStimType.Logarithmic:
        if value1 <= 0.0 or value2 <= 0.0:
            raise ValueError("Logarithmic magnitudes must be positive")
        return np.geomspace(value1, value2, repeats, dtype=float)
    if profile == DrawStimType.Random:
        low = min(value1, value2)
        high = max(value1, value2)
        return np.random.uniform(low, high, repeats)
    raise ValueError(f"Unsupported profile: {profile!r}")


def generate_ramp(
    n_neurons: int,
    ramp_ms: float,
    ipi_ms: float,
    start_magnitude: float,
    end_magnitude: float,
    repeats: int,
    dt_seconds: float,
    neuron_index: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Continuous linear ramp(s) from start_magnitude to end_magnitude.

    Each ramp runs for ramp_ms; consecutive repeats are separated by ipi_ms
    of zero samples. A trailing IPI is included so the next pattern can
    start immediately after. If neuron_index is None, the same ramp is
    applied to every column; otherwise only the given column(s) get the ramp.
    """
    pulse_n, ipi_n = _validate_common(
        n_neurons=n_neurons,
        pulse_ms=ramp_ms,
        ipi_ms=ipi_ms,
        repeats=repeats,
        dt_seconds=dt_seconds,
    )
    indices = (
        None
        if neuron_index is None
        else _normalize_neuron_indices(neuron_index, n_neurons)
    )
    ramp = np.linspace(start_magnitude, end_magnitude, pulse_n, dtype=float)
    total_n = repeats * (pulse_n + ipi_n)
    values = np.zeros((total_n, n_neurons), dtype=float)
    for repeat_index in range(repeats):
        t0 = repeat_index * (pulse_n + ipi_n)
        if indices is None:
            values[t0 : t0 + pulse_n, :] = ramp[:, np.newaxis]
        else:
            for idx in indices:
                values[t0 : t0 + pulse_n, idx] = ramp
    return values


def generate_per_neuron(
    neuron_index: int | Sequence[int],
    n_neurons: int,
    pulse_ms: float,
    ipi_ms: float,
    profile: DrawStimType,
    value1: float,
    value2: float,
    repeats: int,
    dt_seconds: float,
) -> np.ndarray:
    pulse_n, ipi_n = _validate_common(
        n_neurons=n_neurons,
        pulse_ms=pulse_ms,
        ipi_ms=ipi_ms,
        repeats=repeats,
        dt_seconds=dt_seconds,
    )
    indices = _normalize_neuron_indices(neuron_index, n_neurons)
    magnitudes = _per_pulse_magnitudes(profile, value1, value2, repeats)
    total_n = repeats * (pulse_n + ipi_n)
    values = np.zeros((total_n, n_neurons), dtype=float)
    for repeat_index, magnitude in enumerate(magnitudes):
        t0 = repeat_index * (pulse_n + ipi_n)
        for idx in indices:
            values[t0 : t0 + pulse_n, idx] = magnitude
    return values


def _normalize_neuron_indices(
    neuron_index: int | Sequence[int],
    n_neurons: int,
) -> list[int]:
    indices = [neuron_index] if isinstance(neuron_index, int) else list(neuron_index)
    if not indices:
        raise ValueError("at least one neuron index is required")
    for idx in indices:
        if idx < 0 or idx >= n_neurons:
            raise ValueError("neuron_index out of range")
    return indices


def parse_neuron_indices(text: str, n_neurons: int) -> list[int]:
    """Parse a neuron selection string into a sorted list of unique indices.

    Accepts comma- (or whitespace-) separated indices and inclusive ranges,
    e.g. ``"0, 2, 5-8"`` -> ``[0, 2, 5, 6, 7, 8]``. Ranges may be written in
    either order (``"8-5"`` is the same as ``"5-8"``). Raises ValueError on
    empty input, unparseable tokens, or indices outside ``[0, n_neurons)``.
    """
    indices: list[int] = []
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Specify at least one neuron index.")
    for token in tokens:
        if "-" in token:
            lo_str, _, hi_str = token.partition("-")
            try:
                lo = int(lo_str.strip())
                hi = int(hi_str.strip())
            except ValueError:
                raise ValueError(f"Invalid neuron range: {token!r}") from None
            if lo > hi:
                lo, hi = hi, lo
            indices.extend(range(lo, hi + 1))
        else:
            try:
                indices.append(int(token))
            except ValueError:
                raise ValueError(f"Invalid neuron index: {token!r}") from None
    unique = sorted(set(indices))
    for idx in unique:
        if idx < 0 or idx >= n_neurons:
            raise ValueError(f"Neuron index {idx} is out of range (0–{n_neurons - 1}).")
    return unique


def is_finite_number(value: float) -> bool:
    return math.isfinite(value)
