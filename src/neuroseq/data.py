from collections.abc import Iterable
from dataclasses import dataclass

import brian2.numpy_ as np
from numpy.typing import NDArray


@dataclass
class TraceData:
    """A data class representing a set of neuron traces.

    Attributes:
        name (str): The name of the data set.
        traces (np.ndarray): A 2D array of traces where the first index is the neuron index and the second index is the time index.
        indices (np.ndarray): A 1D array of neuron indices corresponding to the first dimension of traces.
        units (str): The unit of the traces (e.g., 'mV', 'pA').
        dt (float): The time step in seconds between consecutive timestamps.
        raw_cellpose_ids (np.ndarray | None): The raw, pre-mapping Cellpose ROI id for
            each row of ``traces`` (row order == Cellpose ROI order), if this data was
            imported via a Cellpose-aware Suite2p pipeline. Used to re-resolve
            ``indices`` if the stimulus-to-Cellpose mapping changes later. ``None`` for
            data with no such association (e.g. patch-clamp).
    """

    name: str
    traces: NDArray[np.float32]
    indices: NDArray[np.uint32]
    units: str
    dt: float
    raw_cellpose_ids: NDArray[np.uint32] | None = None

    def __post_init__(self) -> None:
        self.traces = np.asarray(self.traces, dtype=np.float32)
        self.indices = np.asarray(self.indices, dtype=np.uint32)
        if self.raw_cellpose_ids is not None:
            self.raw_cellpose_ids = np.asarray(self.raw_cellpose_ids, dtype=np.uint32)

    def rows_for_neurons(
        self, neurons: slice | Iterable[int] | None
    ) -> NDArray[np.intp]:
        """Rows of ``traces`` holding the given neuron ids, ordered by id.

        ``neurons`` is a slice of neuron ids (as a plot selection produces), an
        explicit collection of ids, or None for every neuron. Rows follow
        ``indices``, which need not be ``arange(n)``; ids without a row are
        skipped.
        """
        ids = self.indices.astype(np.int64)
        if neurons is None:
            neurons = slice(None)
        if isinstance(neurons, slice):
            mask = np.ones(ids.shape[0], dtype=bool)
            if neurons.start is not None:
                mask &= ids >= neurons.start
            if neurons.stop is not None:
                mask &= ids < neurons.stop
            rows = np.flatnonzero(mask)
            return rows[np.argsort(ids[rows], kind="stable")]
        row_by_id = {int(neuron_id): row for row, neuron_id in enumerate(ids)}
        return np.asarray(
            [row_by_id[int(n)] for n in neurons if int(n) in row_by_id], dtype=np.intp
        )


@dataclass
class SpikeData:
    """A data class representing a set of neuron spikes.

    Attributes:
        name (str): The name of the data set.
        spike_indices (np.ndarray): A 1D array of spiking neuron indices.
        spike_times (np.ndarray): A 1D array of spiking timestamps.
        raw_cellpose_ids (np.ndarray | None): The raw, pre-mapping Cellpose ROI id for
            each spike event in ``spike_indices``, if this data was imported via a
            Cellpose-aware Suite2p pipeline. Used to re-resolve ``spike_indices`` if the
            stimulus-to-Cellpose mapping changes later. ``None`` for data with no such
            association (e.g. patch-clamp).
    """

    name: str
    spike_indices: NDArray[np.uint32]
    spike_times: NDArray[np.float32]
    raw_cellpose_ids: NDArray[np.uint32] | None = None

    def __post_init__(self) -> None:
        self.spike_indices = np.asarray(self.spike_indices, dtype=np.uint32)
        self.spike_times = np.asarray(self.spike_times, dtype=np.float32)
        if self.raw_cellpose_ids is not None:
            self.raw_cellpose_ids = np.asarray(self.raw_cellpose_ids, dtype=np.uint32)


@dataclass
class SynapseData:
    """A data class representing synapse data.

    Attributes:
        name (str): The name of the data set.
        values (np.ndarray): A 3D array of shape (n_neurons, n_neurons, n_timesteps).
        timestamps (np.ndarray): A 1D array of timestamps corresponding to the third dimension of values.
    """

    name: str
    values: NDArray[np.float32]
    timestamps: NDArray[np.float32]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float32)


Data = TraceData | SpikeData | SynapseData


@dataclass
class Experiment:
    """A data class representing a neuroscience experiment.

    Attributes:
        name (str): The name of the experiment.
        duration (float): The total duration of the experiment in seconds.
        timepoints (list[float]): A list of timepoints at which data was appended
        data (list[Data]): A list of data objects (TraceData, SpikeData, SynapseData) associated with the experiment.
    """

    name: str
    duration: float
    timepoints: list[float]
    data: list[Data]
