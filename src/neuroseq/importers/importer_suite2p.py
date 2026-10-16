from __future__ import annotations

import logging
from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.spike_detection import detect_spikes_threshold

logger = logging.getLogger(__name__)


def _normalize_array_per_neuron(
    data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Return per-neuron min/max normalized copy of the array along with the raw bounds."""
    if data.size == 0:
        raise ValueError("Cannot normalize empty array")

    if data.ndim != 2:
        raise ValueError("Data must be 2D array (neurons x timepoints)")

    n_neurons, n_timepoints = data.shape
    normalized = np.zeros_like(data)
    raw_mins = np.zeros(n_neurons, dtype=float)
    raw_maxs = np.zeros(n_neurons, dtype=float)

    for i in range(n_neurons):
        neuron_data = data[i, :]
        raw_min = float(np.min(neuron_data))
        raw_max = float(np.max(neuron_data))

        raw_mins[i] = raw_min
        raw_maxs[i] = raw_max

        if raw_max > raw_min:
            normalized[i, :] = (neuron_data - raw_min) / (raw_max - raw_min)
        else:
            normalized[i, :] = np.zeros_like(neuron_data)

    return normalized, raw_mins, raw_maxs


def compute_normalized_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Contrast Suite2p fluorescence data (optional neuropil subtraction) and normalize per-neuron.
    Returns normalized traces along with the per-neuron pre-normalization min/max values.
    """
    if fneu_data is not None and f_data.shape != fneu_data.shape:
        raise ValueError(
            f"F.npy shape {f_data.shape} does not match Fneu.npy shape {fneu_data.shape}"
        )

    contrasting_traces = f_data - 0.7 * fneu_data if fneu_data is not None else f_data
    return _normalize_array_per_neuron(contrasting_traces)


def compute_normalized_suite2p_spikes(
    spike_data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Normalize Suite2p spiking activity to [0, 1] per-neuron."""
    return _normalize_array_per_neuron(spike_data)


def _resolve_suite2p_indices(
    n_neurons_to_import: int,
    stimulus_to_cellpose_mapping: dict[int, int] | None,
) -> npt.NDArray[Any]:
    """
    Resolve Suite2p row indices (Cellpose IDs) to stimulus IDs. From Branch 417 with improvements.
    """
    if n_neurons_to_import <= 0:
        raise ValueError("n_neurons_to_import must be positive")

    if not stimulus_to_cellpose_mapping:
        return np.arange(n_neurons_to_import, dtype=np.uint32)

    cellpose_to_stim: dict[int, int] = {}
    duplicate_cellpose_ids: set[int] = set()
    for stim_id, cellpose_id in stimulus_to_cellpose_mapping.items():
        stim = int(stim_id)
        cellpose = int(cellpose_id)
        if cellpose in cellpose_to_stim:
            duplicate_cellpose_ids.add(cellpose)
            continue
        cellpose_to_stim[cellpose] = stim

    if duplicate_cellpose_ids:
        duplicate_ids = ", ".join(
            str(cell_id) for cell_id in sorted(duplicate_cellpose_ids)
        )
        raise ValueError(
            f"stimulus_to_cellpose_mapping contains duplicate Cellpose IDs: {duplicate_ids}"
        )

    # Suite2p rows are expected to be ordered by Cellpose ID 0..N-1.
    # Missing explicit mappings fall back to identity import (cellpose id -> same stim id).
    stim_ids = [
        cellpose_to_stim.get(cellpose_id, cellpose_id)
        for cellpose_id in range(n_neurons_to_import)
    ]
    if len(set(stim_ids)) != len(stim_ids):
        raise ValueError(
            "Resolved stimulus IDs are not unique; check stimulus_to_cellpose_mapping for collisions"
        )
    return np.asarray(stim_ids, dtype=np.uint32)


def remap_raw_cellpose_ids(
    raw_cellpose_ids: npt.NDArray[Any],
    stimulus_to_cellpose_mapping: dict[int, int] | None,
) -> npt.NDArray[Any]:
    """
    Resolve stimulus IDs for an array of raw (pre-mapping) Cellpose ROI ids, e.g. to
    re-derive ``TraceData.indices``/``SpikeData.spike_indices`` after
    ``stimulus_to_cellpose_mapping`` changes, without re-importing from disk.
    """
    raw_cellpose_ids = np.asarray(raw_cellpose_ids, dtype=np.uint32)
    if raw_cellpose_ids.size == 0:
        return raw_cellpose_ids.copy()

    n = int(raw_cellpose_ids.max()) + 1
    lookup = _resolve_suite2p_indices(n, stimulus_to_cellpose_mapping)
    return lookup[raw_cellpose_ids]


def prepare_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None,
    *,
    max_neurons: int,
    sampling_rate: float = 30.0,
    stimulus_to_cellpose_mapping: dict[int, int] | None = None,
) -> TraceData:
    """Create TraceData from Suite2p fluorescence data."""

    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    dt = 1.0 / float(sampling_rate)
    n_neurons, n_timepoints = f_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
    raw_cellpose_ids = np.arange(n_neurons_to_import, dtype=np.uint32)
    indices = remap_raw_cellpose_ids(raw_cellpose_ids, stimulus_to_cellpose_mapping)

    return TraceData(
        "Suite2p contrasting traces",
        normalized_traces[:n_neurons_to_import, :],
        indices,
        "units",
        dt,
        raw_cellpose_ids,
    )


def prepare_suite2p_spikes(
    spike_data: npt.NDArray[Any],
    *,
    dt: float,
    threshold: float,
    max_neurons: int,
    stimulus_to_cellpose_mapping: dict[int, int] | None = None,
) -> SpikeData:
    """
    Create SpikeData from Suite2p spks.npy data using shared spike detection.
    """
    logger.debug(
        "Preparing Suite2p spikes: shape=%s dt=%.6f threshold=%.4f max_neurons=%s",
        getattr(spike_data, "shape", None),
        dt,
        threshold,
        max_neurons,
    )
    normalized_spikes, _, _ = compute_normalized_suite2p_spikes(spike_data)

    result = detect_spikes_threshold(
        normalized_spikes,
        dt=dt,
        threshold=threshold,
        threshold_method="absolute",
        max_neurons=max_neurons,
        normalize_per_neuron=False,
    )
    # Validate/resolve against the full declared neuron count (not just the ids that
    # actually spiked) so a mapping collision among silent neurons is still caught.
    n_neurons = min(max_neurons, spike_data.shape[0])
    lookup = _resolve_suite2p_indices(n_neurons, stimulus_to_cellpose_mapping)
    raw_cellpose_ids = result.spike_indices.astype(np.uint32, copy=True)
    remapped_indices = (
        lookup[raw_cellpose_ids]
        if raw_cellpose_ids.size
        else raw_cellpose_ids.astype(lookup.dtype, copy=True)
    )
    result = SpikeData(
        result.name, remapped_indices, result.spike_times, raw_cellpose_ids
    )

    logger.debug(
        "Spike detection complete: %d spikes across %d neurons",
        len(result.spike_indices),
        len(np.unique(result.spike_indices)) if result.spike_indices.size else 0,
    )
    return result
