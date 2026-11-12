from __future__ import annotations

import logging
from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.importers.importer_tif import DEFAULT_NEUROPIL_COEFFICIENT
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
    neuropil_coefficient: float = DEFAULT_NEUROPIL_COEFFICIENT,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Contrast Suite2p fluorescence data (optional neuropil subtraction) and normalize per-neuron.
    Returns normalized traces along with the per-neuron pre-normalization min/max values.
    """
    if fneu_data is not None and f_data.shape != fneu_data.shape:
        raise ValueError(
            f"F.npy shape {f_data.shape} does not match Fneu.npy shape {fneu_data.shape}"
        )

    contrasting_traces = (
        f_data - float(neuropil_coefficient) * fneu_data
        if fneu_data is not None
        else f_data
    )
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
    # Unmapped Cellpose IDs keep their own number as stimulus ID, unless that
    # number is already a mapped stimulus ID (a sparse Custom mapping, or a
    # Linear one over Cellpose IDs with gaps). Those are paired, in sorted
    # order, with the mapped Cellpose IDs that are not themselves stimulus IDs,
    # so the mapping extends to a permutation: every Cellpose ID gets a unique
    # stimulus ID, and the result depends only on the ID and the mapping (not
    # on which other IDs are present), keeping remaps of subsets consistent.
    stim_ids_in_use = set(cellpose_to_stim.values())
    displaced_ids = sorted(stim_ids_in_use - cellpose_to_stim.keys())
    free_ids = sorted(cellpose_to_stim.keys() - stim_ids_in_use)
    fallback = dict(zip(displaced_ids, free_ids, strict=True))
    stim_ids = [
        cellpose_to_stim.get(cellpose_id, fallback.get(cellpose_id, cellpose_id))
        for cellpose_id in range(n_neurons_to_import)
    ]
    if len(set(stim_ids)) != len(stim_ids):
        raise ValueError(
            "Resolved input IDs are not unique; check stimulus_to_cellpose_mapping for collisions"
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


def _row_cellpose_ids(
    raw_cellpose_ids: npt.NDArray[Any] | None, n_rows: int
) -> npt.NDArray[Any]:
    if raw_cellpose_ids is None:
        return np.arange(n_rows, dtype=np.uint32)
    ids = np.asarray(raw_cellpose_ids, dtype=np.uint32)
    if ids.shape[0] < n_rows:
        raise ValueError(
            f"raw_cellpose_ids has {ids.shape[0]} entries for {n_rows} rows"
        )
    return ids[:n_rows]


def prepare_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None,
    *,
    max_neurons: int,
    sampling_rate: float = 30.0,
    stimulus_to_cellpose_mapping: dict[int, int] | None = None,
    neuropil_coefficient: float = DEFAULT_NEUROPIL_COEFFICIENT,
    raw_cellpose_ids: npt.NDArray[Any] | None = None,
) -> TraceData:
    """Create TraceData from Suite2p fluorescence data.

    ``raw_cellpose_ids`` gives the Cellpose ID of each row; by default row k is
    Cellpose ID k.
    """

    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    dt = 1.0 / float(sampling_rate)
    n_neurons, n_timepoints = f_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_traces, _, _ = compute_normalized_suite2p_traces(
        f_data, fneu_data, neuropil_coefficient
    )
    raw_cellpose_ids = _row_cellpose_ids(raw_cellpose_ids, n_neurons_to_import)
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
    raw_cellpose_ids: npt.NDArray[Any] | None = None,
) -> SpikeData:
    """
    Create SpikeData from Suite2p spks.npy data using shared spike detection.

    ``raw_cellpose_ids`` gives the Cellpose ID of each row; by default row k is
    Cellpose ID k.
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
    row_ids = _row_cellpose_ids(raw_cellpose_ids, n_neurons)
    row_stim_ids = remap_raw_cellpose_ids(row_ids, stimulus_to_cellpose_mapping)
    rows = result.spike_indices.astype(np.intp, copy=False)
    result = SpikeData(
        result.name, row_stim_ids[rows], result.spike_times, row_ids[rows]
    )

    logger.debug(
        "Spike detection complete: %d spikes across %d neurons",
        len(result.spike_indices),
        len(np.unique(result.spike_indices)) if result.spike_indices.size else 0,
    )
    return result
