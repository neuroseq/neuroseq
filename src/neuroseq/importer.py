from __future__ import annotations

from typing import Any
from typing import Literal
from typing import TypedDict

import brian2.numpy_ as np
import numpy.typing as npt

__all__ = [
    "Suite2pTracePayload",
    "Suite2pSpikePayload",
    "prepare_suite2p_traces",
    "prepare_suite2p_spikes",
    "compute_normalized_suite2p_traces",
    "compute_normalized_suite2p_spikes",
]


class Suite2pTracePayload(TypedDict):
    widget: Literal["trace"]
    timestamps: npt.NDArray[Any]
    traces: npt.NDArray[Any]
    dt: float
    sampling_rate: float


class Suite2pSpikePayload(TypedDict):
    widget: Literal["spike"]
    timestamps: npt.NDArray[Any]
    spikes: npt.NDArray[Any]
    dt: float
    threshold: float


def _normalize_array(
    data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], float, float]:
    """Return min/max normalized copy of the array along with the raw bounds."""
    if data.size == 0:
        raise ValueError("Cannot normalize empty array")

    raw_min = float(np.min(data))
    raw_max = float(np.max(data))

    if raw_max > raw_min:
        normalized = (data - raw_min) / (raw_max - raw_min)
    else:
        normalized = np.zeros_like(data)

    return normalized, raw_min, raw_max


def compute_normalized_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None = None,
) -> tuple[npt.NDArray[Any], float, float]:
    """
    Contrast Suite2p fluorescence data (optional neuropil subtraction) and normalize.
    Returns normalized traces along with the pre-normalization min/max values.
    """
    if fneu_data is not None and f_data.shape != fneu_data.shape:
        raise ValueError(
            f"F.npy shape {f_data.shape} does not match Fneu.npy shape {fneu_data.shape}"
        )

    contrasting_traces = f_data - 0.7 * fneu_data if fneu_data is not None else f_data
    return _normalize_array(contrasting_traces)


def compute_normalized_suite2p_spikes(
    spike_data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], float, float]:
    """Normalize Suite2p spiking activity to [0, 1]."""
    return _normalize_array(spike_data)


def prepare_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None,
    *,
    max_neurons: int,
    sampling_rate: float = 30.0,
) -> Suite2pTracePayload:
    """Create Neuroseq trace widget payload from Suite2p fluorescence data."""

    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    dt = 1.0 / float(sampling_rate)
    n_neurons, n_timepoints = f_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)

    return {
        "widget": "trace",
        "timestamps": np.arange(n_timepoints) * dt,
        "traces": normalized_traces[:n_neurons_to_import, :],
        "dt": dt,
        "sampling_rate": float(sampling_rate),
    }


def prepare_suite2p_spikes(
    spike_data: npt.NDArray[Any],
    *,
    dt: float,
    threshold: float,
    max_neurons: int,
) -> Suite2pSpikePayload:
    """
    Create Neuroseq spike widget payload from Suite2p spks.npy data.
    """
    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    n_neurons, n_timesteps = spike_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_spikes, _, _ = compute_normalized_suite2p_spikes(spike_data)
    trimmed_data = normalized_spikes[:n_neurons_to_import, :]

    mask = trimmed_data >= threshold
    neuron_indices, time_indices = np.nonzero(mask)
    spike_times = time_indices * dt

    return {
        "widget": "spike",
        "timestamps": spike_times,
        "spikes": neuron_indices,
        "dt": dt,
        "threshold": threshold,
    }
