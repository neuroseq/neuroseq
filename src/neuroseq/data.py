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
    """

    name: str
    traces: NDArray[np.float32]
    indices: NDArray[np.uint32]
    units: str
    dt: float


@dataclass
class SpikeData:
    """A data class representing a set of neuron spikes.

    Attributes:
        name (str): The name of the data set.
        spike_indices (np.ndarray): A 1D array of spiking neuron indices.
        spike_times (np.ndarray): A 1D array of spiking timestamps.
    """

    name: str
    spike_indices: NDArray[np.uint32]
    spike_times: NDArray[np.float32]


@dataclass
class SynapseData:
    """A data class representing synapse data.

    Attributes:
        name (str): The name of the data set.
        values (np.ndarray): A 3D array of shape (n_neurons, n_neurons, n_timesteps).
        timestamps (np.ndarray): A 1D array of timestamps corresponding to the third dimension of values.
    """

    name: str
    values: np.ndarray
    timestamps: np.ndarray


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
