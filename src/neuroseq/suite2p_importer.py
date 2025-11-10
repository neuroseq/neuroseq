from __future__ import annotations

import brian2.numpy_ as np
import numpy.typing as npt


def prepare_suite2p_traces(
    f_data: npt.NDArray,
    fneu_data: npt.NDArray,
    *,
    max_neurons: int,
    sampling_rate: float = 30.0,
) -> dict:
    """Create Neuroseq trace widget payload from Suite2p fluorescence data."""
    if f_data.shape != fneu_data.shape:
        raise ValueError(
            f"F.npy shape {f_data.shape} does not match Fneu.npy shape {fneu_data.shape}"
        )

    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    dt = 1.0 / float(sampling_rate)
    n_neurons, n_timepoints = f_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    contrasting_traces = f_data - 0.7 * fneu_data

    return {
        "widget": "trace",
        "timestamps": np.arange(n_timepoints) * dt,
        "traces": contrasting_traces[:n_neurons_to_import, :],
        "dt": dt,
        "sampling_rate": float(sampling_rate),
    }


def prepare_suite2p_spikes(
    spike_data: npt.NDArray,
    *,
    dt: float,
    threshold: float,
    max_neurons: int,
) -> dict:
    """
    Create Neuroseq spike widget payload from Suite2p spks.npy data.
    """
    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    n_neurons, n_timesteps = spike_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    trimmed_data = spike_data[:n_neurons_to_import, :]

    mask = trimmed_data >= threshold
    neuron_indices, time_indices = np.nonzero(mask)
    spike_times = time_indices * dt

    return {
        "widget": "spike",
        "timestamps": spike_times,
        "spikes": neuron_indices,
        "dt": dt,
        "n_neurons": n_neurons_to_import,
        "n_timesteps": n_timesteps,
        "threshold": threshold,
    }
