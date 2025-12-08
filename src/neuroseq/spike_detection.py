from __future__ import annotations

from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt
from scipy.signal import find_peaks

from neuroseq.experiment import SpikeData


def detect_spikes_threshold(
    spike_data: npt.NDArray[Any],
    *,
    dt: float,
    threshold: float,
    max_neurons: int | None = None,
    normalize_per_neuron: bool = True,
) -> SpikeData:
    """
    Detect spikes using threshold-based method on preprocessed data.

    Args:
        spike_data: 2D array (neurons, timepoints) or 1D array (single neuron)
        dt: Time step between samples
        threshold: Detection threshold (0-1 if normalized)
        max_neurons: Maximum neurons to process (None = all)
        normalize_per_neuron: Whether to normalize each neuron's data to [0,1]

    Returns:
        SpikeData
    """

    # Handle 1D input (single neuron)
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(1, -1)
    elif spike_data.ndim > 2:
        raise ValueError("spike_data must be 1D or 2D array - you may have used the wrong file or e.g. format with soma within it. Convert that first")

    n_neurons, n_timesteps = spike_data.shape
    n_neurons_to_process = min(max_neurons or n_neurons, n_neurons)

    if normalize_per_neuron:
        processed_data = _normalize_per_neuron(spike_data[:n_neurons_to_process, :])
    else:
        processed_data = spike_data[:n_neurons_to_process, :].astype(np.float32)

    # Find spikes above threshold
    mask = processed_data >= threshold
    neuron_indices, time_indices = np.nonzero(mask)
    spike_times = time_indices.astype(np.float32) * dt

    return SpikeData("Detected spikes", neuron_indices.astype(np.int32), spike_times)


def _normalize_per_neuron(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    # maybe this should be abstract also to the other use in the codebase for normalizing.
    """Normalize each neuron's data to [0, 1] range (for each neuron, not for all neurons - massive variation exists0"""

    n_neurons, n_timepoints = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in range(n_neurons):
        neuron_data = data[i, :].astype(np.float32)
        data_min = float(np.min(neuron_data))
        data_max = float(np.max(neuron_data))

        if data_max > data_min:
            normalized[i, :] = (neuron_data - data_min) / (data_max - data_min)
        else:
            normalized[i, :] = np.zeros_like(neuron_data)

    return normalized
