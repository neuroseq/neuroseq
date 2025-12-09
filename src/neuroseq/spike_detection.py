from __future__ import annotations

import logging
from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt
from scipy.signal import find_peaks

from neuroseq.data import SpikeData

logger = logging.getLogger(__name__)


def detect_spikes_threshold(
    spike_data: npt.NDArray[Any],
    *,
    dt: float,
    threshold: float,
    max_neurons: int | None = None,
    normalize_per_neuron: bool = True,
) -> SpikeData:
    """
    Detect spikes using find_peaks method on preprocessed data.

    Args:
        spike_data: 2D array (neurons, timepoints) or 1D array (single neuron)
        dt: Time step between samples
        threshold: Detection threshold (0-1 if normalized) - used as height_threshold multiplier
        max_neurons: Maximum neurons to process (None = all)
        normalize_per_neuron: Whether to normalize each neuron's data to [0,1]

    Returns:
        SpikeData
    """

    input_shape = getattr(spike_data, "shape", None)
    logger.debug(
        "Detecting spikes: input_shape=%s dt=%.6f threshold=%.4f max_neurons=%s normalize=%s",
        input_shape,
        dt,
        threshold,
        max_neurons,
        normalize_per_neuron,
    )

    # Handle 1D input (single neuron)
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(1, -1)
    elif spike_data.ndim > 2:
        raise ValueError(
            "spike_data must be 1D or 2D array - you may have used the wrong file or e.g. format with soma within it. Convert that first"
        )

    n_neurons, n_timesteps = spike_data.shape
    n_neurons_to_process = min(max_neurons or n_neurons, n_neurons)

    if normalize_per_neuron:
        processed_data = _normalize_per_neuron(spike_data[:n_neurons_to_process, :])
    else:
        processed_data = spike_data[:n_neurons_to_process, :].astype(np.float32)

    all_neuron_indices = []
    all_spike_times = []

    for neuron_idx in range(n_neurons_to_process):
        neuron_data = processed_data[neuron_idx, :]

        # Suite2p uses basic threshold agian now
        spike_result = detect_spikes_voltage_trace(
            neuron_data,
            neuron_id=neuron_idx,
            dt=dt,
            detection_method="threshold",
            threshold_multiplier=threshold,
        )

        if len(spike_result.spike_indices) > 0:
            all_neuron_indices.append(spike_result.spike_indices)
            all_spike_times.append(spike_result.spike_times)

    if all_neuron_indices:
        final_neuron_indices = np.concatenate(all_neuron_indices)
        final_spike_times = np.concatenate(all_spike_times)
    else:
        final_neuron_indices = np.array([], dtype=np.int32)
        final_spike_times = np.array([], dtype=np.float32)

    logger.debug(
        "Spike detection summary: processed_neurons=%d spikes_found=%d",
        n_neurons_to_process,
        len(final_spike_times),
    )

    return SpikeData("Detected spikes", final_neuron_indices, final_spike_times)


def _normalize_per_neuron(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Normalize each neuron's data to [0, 1] range (for each neuron, not for all neurons - massive variation exists)"""
    data = data.astype(np.float32)
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    ranges = max_vals - min_vals
    # Avoid division by zero for flat traces
    ranges[ranges == 0] = 1.0
    return (data - min_vals) / ranges


# todo: Refactor this into two functions again. no need to share parameters between threshold/peak detection here.
def detect_spikes_voltage_trace(
    voltage_trace: npt.NDArray[Any],
    *,
    neuron_id: int,
    dt: float,
    threshold_method: str = "mean_plus_std",
    threshold_multiplier: float = 3.0,
    min_spike_interval: float = 0.001,
    detection_method: str = "threshold",
    peak_height_threshold: float = 3.0,
    peak_prominence: float = 0.5,
    peak_distance: int | None = None,
    peak_width: float | None = None,
) -> SpikeData:
    """
    General spike detection based on data provided as trace, to derive spikes from.

    Args:
        voltage_trace: 1D voltage array for single neuron
        threshold_method: "mean_plus_std" or "absolute" (for threshold detection)
        peak_ - arguments for peak detection

    Returns:
        SpikeData object with detected spikes
    """
    if voltage_trace.ndim != 1:
        raise ValueError(
            "voltage_trace must be 1D array - run this function per Neuron"
        )
    if dt <= 0:
        raise ValueError("dt must be positive")
    if min_spike_interval < 0:
        raise ValueError("min_spike_interval must be non-negative")

    voltage_trace = voltage_trace.astype(np.float32)

    if detection_method == "peak":
        spike_indices = _detect_peaks_scipy(
            voltage_trace,
            dt,
            peak_height_threshold,
            peak_prominence,
            min_spike_interval,
            peak_distance,
            peak_width,
        )
    elif detection_method == "threshold":
        # Original threshold-based detection
        if threshold_method == "mean_plus_std":
            threshold = np.mean(voltage_trace) + threshold_multiplier * np.std(
                voltage_trace
            )
        elif threshold_method == "absolute":
            threshold = threshold_multiplier
        else:
            raise ValueError("threshold_method must be 'mean_plus_std' or 'absolute'")

        potential_spikes = np.where(voltage_trace > threshold)[0]
        min_spike_samples = int(min_spike_interval / dt)
        filtered_spikes = []

        if len(potential_spikes) > 0:
            filtered_spikes = [potential_spikes[0]]
            for spike_idx in potential_spikes[1:]:
                if spike_idx - filtered_spikes[-1] > min_spike_samples:
                    filtered_spikes.append(spike_idx)

        spike_indices = np.array(filtered_spikes, dtype=np.int32)
    else:
        raise ValueError("detection_method must be 'peak' or 'threshold'")

    spike_times = spike_indices.astype(np.float32) * dt
    neuron_indices = np.full(len(spike_indices), neuron_id, dtype=np.int32)

    return SpikeData(
        f"Voltage spikes (Neuron {neuron_id})", neuron_indices, spike_times
    )


def _detect_peaks_scipy(
    voltage_trace: npt.NDArray[Any],
    dt: float,
    peak_height_threshold: float,
    peak_prominence: float,
    min_spike_interval: float,
    peak_distance: int | None = None,
    peak_width: float | None = None,
) -> npt.NDArray[Any]:
    """
    Scipy-based peak detection for spike detection in voltage traces.

    Args:
        voltage_trace: 1D voltage array
        dt: Time step between samples
        peak_height_threshold: Minimum height above baseline (in std devs)
        peak_prominence: Minimum prominence for peak detection
        min_spike_interval: Minimum time between spikes (seconds)
        peak_distance: Minimum horizontal distance between peaks (samples)
        peak_width: Minimum width of peaks (samples)

    Returns:
        Array of spike indices
    """
    baseline = np.mean(voltage_trace)
    std_dev = np.std(voltage_trace)
    min_height = baseline + peak_height_threshold * std_dev
    min_spike_samples = int(min_spike_interval / dt)

    # Set up find_peaks parameters
    height = min_height
    prominence = peak_prominence * std_dev
    distance = peak_distance if peak_distance is not None else min_spike_samples
    width = peak_width

    # Use scipy find_peaks with these options  # think this works?
    spike_indices, _ = find_peaks(
        voltage_trace,
        height=height,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    return spike_indices.astype(np.int32)
