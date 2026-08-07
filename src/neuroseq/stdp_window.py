from __future__ import annotations

import brian2.numpy_ as np


def delta_t_sweep(
    dt_min_ms: float,
    dt_max_ms: float,
    n_points: int,
    include_zero: bool = False,
) -> list[float]:
    if n_points < 1:
        return []
    if n_points == 1:
        return [0.5 * (dt_min_ms + dt_max_ms)]
    values = np.linspace(dt_min_ms, dt_max_ms, n_points, dtype=float)
    if not include_zero:
        step = abs(float(values[1] - values[0]))
        values = values[np.abs(values) > 0.5 * step]
    return [float(value) for value in values]


def percent_change(before: float, after: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (after - before) / baseline


def classify_window(delta_t_ms: float) -> str:
    if delta_t_ms > 0.0:
        return "LTP"
    if delta_t_ms < 0.0:
        return "LTD"
    return "none"
