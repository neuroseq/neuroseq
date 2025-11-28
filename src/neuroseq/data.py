from dataclasses import dataclass

import brian2.numpy_ as np


@dataclass
class TraceData:
    """A data class representing a set of neuron traces.

    Attributes:
        name (str): The name of the data set.
        traces (np.ndarray): A 2D array of traces where the first index is the neuron index and the second index is the time index.
        units (str): The unit of the traces (e.g., 'mV', 'pA').
        dt (float): The time step in seconds between consecutive timestamps.
    """

    name: str
    traces: np.ndarray
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
    spike_indices: np.ndarray
    spike_times: np.ndarray


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
