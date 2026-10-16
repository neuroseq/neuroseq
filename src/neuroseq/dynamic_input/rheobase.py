from __future__ import annotations

import brian2.numpy_ as np


def count_spikes(trace_mv: np.ndarray, threshold_mv: float = -20.0) -> int:
    """Count upward threshold crossings in an already-mV voltage trace."""
    if trace_mv.size < 2:
        return 0
    crossings = np.flatnonzero(
        (trace_mv[:-1] < threshold_mv) & (trace_mv[1:] >= threshold_mv)
    )
    return int(crossings.size)


def next_bracket(
    low: float, high: float, mid: float, spiked: bool
) -> tuple[float, float]:
    """Update a rheobase bisection bracket after testing ``mid``."""
    if spiked:
        return low, mid
    return mid, high


def bisection_converged(low: float, high: float, tol: float) -> bool:
    """Return whether a bracket has reached the configured width tolerance."""
    return (high - low) <= tol
