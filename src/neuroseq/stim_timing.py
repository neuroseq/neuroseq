from __future__ import annotations

import math

import numpy as np

DEFAULT_STIM_DT_SECONDS = 0.001
STIM_DT_TOLERANCE_SECONDS = 1e-12


def ensure_2d_stim_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("Stimulus arrays must be 2-dimensional.")
    return values


def validate_stim_dt_seconds(dt_seconds: float | int) -> float:
    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Stimulus timestep must be a positive finite number.")
    return dt


def stim_dt_matches(dt_seconds_a: float | int, dt_seconds_b: float | int) -> bool:
    return math.isclose(
        float(dt_seconds_a),
        float(dt_seconds_b),
        rel_tol=0.0,
        abs_tol=STIM_DT_TOLERANCE_SECONDS,
    )


def samples_to_seconds(n_samples: int, dt_seconds: float | int) -> float:
    return float(int(n_samples)) * validate_stim_dt_seconds(dt_seconds)


def samples_to_milliseconds(n_samples: int, dt_seconds: float | int) -> float:
    return 1000.0 * samples_to_seconds(n_samples, dt_seconds)


def seconds_to_samples_ceil(
    duration_seconds: float | int, dt_seconds: float | int
) -> int:
    dt = validate_stim_dt_seconds(dt_seconds)
    duration = max(0.0, float(duration_seconds))
    return int(math.ceil(duration / dt))


def seconds_to_samples_nearest(
    duration_seconds: float | int, dt_seconds: float | int
) -> int:
    dt = validate_stim_dt_seconds(dt_seconds)
    duration = max(0.0, float(duration_seconds))
    return int(math.floor((duration / dt) + 0.5))


def milliseconds_to_samples_nearest(
    duration_milliseconds: float | int, dt_seconds: float | int
) -> int:
    return seconds_to_samples_nearest(float(duration_milliseconds) / 1000.0, dt_seconds)


def format_milliseconds(time_ms: float | int) -> str:
    value = float(time_ms)
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return f"{int(rounded)}"
    return f"{value:.7f}".rstrip("0").rstrip(".")


def remap_stim_values_to_dt(
    values: np.ndarray,
    old_dt_seconds: float | int,
    new_dt_seconds: float | int,
) -> np.ndarray:
    values = np.asarray(values)
    old_dt = validate_stim_dt_seconds(old_dt_seconds)
    new_dt = validate_stim_dt_seconds(new_dt_seconds)
    if stim_dt_matches(old_dt, new_dt):
        return np.array(values, copy=True)
    if values.ndim != 2:
        raise ValueError("Stimulus arrays must be 2-dimensional to be remapped.")

    n_old_samples = int(values.shape[0])
    if n_old_samples == 0:
        return np.zeros(values.shape, dtype=values.dtype)

    duration_seconds = samples_to_seconds(n_old_samples, old_dt)
    n_new_samples = seconds_to_samples_ceil(duration_seconds, new_dt)
    if n_new_samples == 0:
        return np.zeros((0, *values.shape[1:]), dtype=values.dtype)

    # Use the start of each new sample when remapping to preserve pulse onset timing.
    sample_start_seconds = np.arange(n_new_samples, dtype=np.float64) * new_dt
    source_indices = np.floor(sample_start_seconds / old_dt).astype(np.int64)
    np.clip(source_indices, 0, n_old_samples - 1, out=source_indices)
    return np.array(values[source_indices, :], copy=True)
