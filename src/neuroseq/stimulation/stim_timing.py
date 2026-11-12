from __future__ import annotations

import math

import numpy as np

DEFAULT_STIM_DT_SECONDS = 0.001
STIM_DT_TOLERANCE_SECONDS = 1e-12
# Sample positions (time / dt) within this many samples of an integer (or, for
# rounding, of a half-integer) are treated as exactly on it, so float error like
# 11 * 1e-3 / 1e-4 == 109.99999999999999 does not shift a sample by one. The
# relative term covers the accumulated error of very long arrays.
_SAMPLE_POSITION_ABS_TOLERANCE = 1e-9
_SAMPLE_POSITION_REL_TOLERANCE = 1e-12


def _sample_position_tolerance(position):
    return _SAMPLE_POSITION_ABS_TOLERANCE + _SAMPLE_POSITION_REL_TOLERANCE * np.abs(
        position
    )


def floor_sample_position(position):
    """Floor sample positions, treating values within float error as integers."""
    return np.floor(position + _sample_position_tolerance(position))


def ceil_sample_position(position):
    """Ceil sample positions, treating values within float error as integers."""
    return np.ceil(position - _sample_position_tolerance(position))


def round_sample_position(position):
    """Round sample positions half up, treating values within float error as ties."""
    return floor_sample_position(np.asarray(position) + 0.5)


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
    return int(ceil_sample_position(duration / dt))


def seconds_to_samples_nearest(
    duration_seconds: float | int, dt_seconds: float | int
) -> int:
    dt = validate_stim_dt_seconds(dt_seconds)
    duration = max(0.0, float(duration_seconds))
    return int(round_sample_position(duration / dt))


def milliseconds_to_samples_nearest(
    duration_milliseconds: float | int, dt_seconds: float | int
) -> int:
    return seconds_to_samples_nearest(float(duration_milliseconds) / 1000.0, dt_seconds)


def snap_milliseconds_to_dt(
    time_ms: float,
    dt_seconds: float,
    minimum_samples: int = 1,
) -> tuple[int, float]:
    """Return the dt-snapped sample count and effective millisecond duration."""
    n_samples = max(
        int(minimum_samples),
        milliseconds_to_samples_nearest(time_ms, dt_seconds),
    )
    return n_samples, samples_to_milliseconds(n_samples, dt_seconds)


def format_milliseconds(time_ms: float | int) -> str:
    value = float(time_ms)
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return f"{int(rounded)}"
    return f"{value:.7f}".rstrip("0").rstrip(".")


def zero_order_hold_sample_count(
    n_samples: int,
    old_dt_seconds: float | int,
    new_dt_seconds: float | int,
) -> int:
    """Number of ``new_dt`` samples needed to cover ``n_samples * old_dt``."""
    old_dt = validate_stim_dt_seconds(old_dt_seconds)
    new_dt = validate_stim_dt_seconds(new_dt_seconds)
    return int(ceil_sample_position(int(n_samples) * old_dt / new_dt))


def zero_order_hold_source_indices(
    n_new_samples: int,
    old_dt_seconds: float | int,
    new_dt_seconds: float | int,
) -> np.ndarray:
    """Index of the source sample active at the start of each new sample."""
    old_dt = validate_stim_dt_seconds(old_dt_seconds)
    new_dt = validate_stim_dt_seconds(new_dt_seconds)
    positions = np.arange(int(n_new_samples), dtype=np.float64) * new_dt / old_dt
    return floor_sample_position(positions).astype(np.int64)


def resample_zero_order_hold(
    values: np.ndarray,
    old_dt_seconds: float | int,
    new_dt_seconds: float | int,
) -> np.ndarray:
    """Resample along axis 0 by holding each source sample until the next one.

    Every new sample takes the source value active at its start time, which
    preserves pulse onsets (delayed by less than one new sample) and never
    invents intermediate values. The result covers at least the source duration.
    Always returns a new array.
    """
    values = np.asarray(values)
    old_dt = validate_stim_dt_seconds(old_dt_seconds)
    new_dt = validate_stim_dt_seconds(new_dt_seconds)
    if stim_dt_matches(old_dt, new_dt):
        return np.array(values, copy=True)
    n_old_samples = int(values.shape[0])
    if n_old_samples == 0:
        return np.zeros(values.shape, dtype=values.dtype)
    n_new_samples = zero_order_hold_sample_count(n_old_samples, old_dt, new_dt)
    source_indices = zero_order_hold_source_indices(n_new_samples, old_dt, new_dt)
    np.clip(source_indices, 0, n_old_samples - 1, out=source_indices)
    return np.array(values[source_indices], copy=True)


def remap_stim_values_to_dt(
    values: np.ndarray,
    old_dt_seconds: float | int,
    new_dt_seconds: float | int,
) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 and not stim_dt_matches(old_dt_seconds, new_dt_seconds):
        raise ValueError("Stimulus arrays must be 2-dimensional to be remapped.")
    return resample_zero_order_hold(values, old_dt_seconds, new_dt_seconds)


def padded_length_seconds(
    current_seconds: float,
    value_text: str,
    *,
    to_end_time: bool,
    dt_seconds: float,
) -> float:
    """Total input length after the Pad action, from the user's entry.

    ``value_text`` is the seconds to add, or with ``to_end_time`` the time
    point the arrays should end at (#647). Raises ValueError with a message
    for the user when the entry is not a number, not positive, or an end time
    that is not after the current end.
    """
    try:
        value = float(value_text)
    except ValueError:
        raise ValueError(
            f"Could not parse '{value_text}' as a number of seconds."
        ) from None
    if not math.isfinite(value):
        raise ValueError(f"Could not parse '{value_text}' as a number of seconds.")
    if not to_end_time:
        if value <= 0.0:
            raise ValueError("Padding seconds must be positive.")
        return current_seconds + value
    # Within half a sample of the current end pads nothing.
    if value <= current_seconds + 0.5 * validate_stim_dt_seconds(dt_seconds):
        raise ValueError(
            f"The end time {value:g} s is not after the current end of the input "
            f"array ({current_seconds:g} s). Enter a later end time, or use Trim "
            "input array to shorten the arrays."
        )
    return value
