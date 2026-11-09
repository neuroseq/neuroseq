"""Export traces or spikes as NumPy arrays of shape (neurons, time points).

Qt-free: :mod:`neuroseq.export.numpy_export_dialog` is the GUI. A ``.npy`` file holds
only the matrix; a ``.npz`` archive also holds the neuron id of each row, the time of
each column (seconds) and, for traces, their units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData


@dataclass
class NeuronTimeMatrix:
    """A (neurons, time points) matrix with what its rows and columns are."""

    values: npt.NDArray[np.float32]
    #: Neuron id of each row.
    neuron_ids: npt.NDArray[np.int64]
    #: Time of each column (seconds): a sample's time, or a bin's start.
    times: npt.NDArray[np.float64]
    units: str


def trace_matrix(trace: TraceData) -> NeuronTimeMatrix:
    """The trace's own (neurons, samples) matrix, rows ordered by neuron id."""
    order = np.argsort(trace.indices, kind="stable")
    values = np.asarray(trace.traces, dtype=np.float32)[order]
    return NeuronTimeMatrix(
        values=values,
        neuron_ids=np.asarray(trace.indices, dtype=np.int64)[order],
        times=np.arange(values.shape[1], dtype=np.float64) * float(trace.dt),
        units=trace.units,
    )


#: Largest matrix (in elements, 2 GB as float32) offered for export: a tiny spike
#: bin over a long recording would otherwise exhaust the memory.
MAX_MATRIX_ELEMENTS = 500_000_000


def _spike_columns(spikes: SpikeData, bin_s: float) -> npt.NDArray[np.int64]:
    times = np.asarray(spikes.spike_times, dtype=np.float64)
    return np.floor(times / bin_s + 1e-9).astype(np.int64).clip(min=0)


def spike_bin_count(spikes: SpikeData, bin_s: float, duration_s: float = 0.0) -> int:
    """Number of time bins (columns) :func:`spike_matrix` makes, without making it."""
    if bin_s <= 0:
        raise ValueError("The bin size must be greater than zero.")
    columns = _spike_columns(spikes, bin_s)
    return max(
        1,
        int(np.ceil(float(duration_s) / bin_s - 1e-9)),
        int(columns.max()) + 1 if columns.size else 0,
    )


def spike_matrix(
    spikes: SpikeData,
    bin_s: float,
    duration_s: float = 0.0,
    neuron_ids: npt.ArrayLike | None = None,
) -> NeuronTimeMatrix:
    """Spike counts per neuron and time bin of ``bin_s`` seconds.

    The bins cover ``duration_s`` (at least up to the last spike). Rows are
    ``neuron_ids`` (default: 0 up to the highest spiking neuron, so row ``i`` is
    neuron ``i``); spikes of neurons not listed are dropped.
    """
    n_bins = spike_bin_count(spikes, bin_s, duration_s)
    indices = np.asarray(spikes.spike_indices, dtype=np.int64)
    if neuron_ids is None:
        n_rows = int(indices.max()) + 1 if indices.size else 0
        ids = np.arange(n_rows, dtype=np.int64)
    else:
        ids = np.unique(np.asarray(neuron_ids, dtype=np.int64))
    if ids.size * n_bins > MAX_MATRIX_ELEMENTS:
        raise ValueError(
            f"{ids.size} neurons × {n_bins} bins is too large to export; "
            "use a larger bin size."
        )
    columns = _spike_columns(spikes, bin_s)
    counts = np.zeros((ids.size, n_bins), dtype=np.float32)
    rows = np.searchsorted(ids, indices)
    known = np.zeros(indices.shape, dtype=bool)
    if ids.size:
        known = (rows < ids.size) & (ids[np.minimum(rows, ids.size - 1)] == indices)
    np.add.at(counts, (rows[known], columns[known]), 1.0)
    return NeuronTimeMatrix(
        values=counts,
        neuron_ids=ids,
        times=np.arange(n_bins, dtype=np.float64) * bin_s,
        units="spikes",
    )


def save_matrix(matrix: NeuronTimeMatrix, path: str | Path) -> Path:
    """Write ``matrix`` to ``path``: the bare array for ``.npy``, the array with
    ``neuron_ids``, ``times`` and ``units`` for ``.npz`` (``values`` is the array)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, matrix.values)
    elif suffix == ".npz":
        np.savez(
            path,
            values=matrix.values,
            neuron_ids=matrix.neuron_ids,
            times=matrix.times,
            units=np.array(matrix.units),
        )
    else:
        raise ValueError(f"Use a .npy or .npz file name, not '{path.name}'.")
    return path
