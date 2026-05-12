# Example of an external simulation function that can be imported into neuroseq

import numpy as np
from neuroseq.data import TraceData, SpikeData, SynapseData, Experiment


def simulate(
    stims_dict: dict[str, np.ndarray],
    simulation_time: float,
    params: dict | None = None,
) -> Experiment:
    # This function is provided by neuroseq with a dict of stimuli, and the desired simulation time.
    # stims_dict is a dict of key:data pairs, where data contains
    #   - "array": 2d numpy array of timestamps x neurons
    #   - "dt": the time step of the stimulus in seconds as a float

    # ...do some simulation using stims_dict and simulation_time...

    # It should return an experiment object, which contains a list of simulation data objects
    data: list[TraceData | SpikeData | SynapseData] = []
    # Three types of data can be returned, which differ depending on the widget used to display the data in the GUI
    # Below are some example random data of each type:

    # TraceData attributes:
    #   - name (str): The name of the data set.
    #   - traces (np.ndarray): A 2D array of traces where the first index is the neuron index and the second index is the time index.
    #   - indices (np.ndarray): A 1D array of neuron indices corresponding to the first dimension of traces.
    #   - units (str): The unit of the traces (e.g., 'mV', 'pA').
    #   - dt (float): The time step in seconds between consecutive timestamps.
    n_neurons_trace = 32
    n_timestamps_trace = 2000
    data.append(
        TraceData(
            name="trace-example",
            traces=np.random.random_sample((n_neurons_trace, n_timestamps_trace)),
            indices=np.arange(n_neurons_trace, dtype=np.uint32),
            dt=simulation_time / n_timestamps_trace,
            units="mV",
        )
    )

    # SpikeData attributes:
    #   - name (str): The name of the data set.
    #   - spike_indices (np.ndarray): A 1D array of spiking neuron indices.
    #   - spike_times (np.ndarray): A 1D array of spiking timestamps.
    n_neurons_spike = 32
    n_spikes = 74
    data.append(
        SpikeData(
            name="spike-example",
            spike_indices=np.random.randint(n_neurons_spike, size=(n_spikes,)),
            spike_times=np.linspace(0, simulation_time, n_spikes),
        )
    )

    # SynapseData attributes:
    #   - name (str): The name of the data set.
    #   - values (np.ndarray): A 3D array of shape (n_neurons, n_neurons, n_timesteps).
    #   - timestamps (np.ndarray): A 1D array of timestamps corresponding to the third dimension of values.

    n_neurons_heatmap = 32
    n_timestamps_heatmap = 100
    data.append(
        SynapseData(
            name="synapse-example",
            values=np.random.rand(
                n_neurons_heatmap, n_neurons_heatmap, n_timestamps_heatmap
            ),
            timestamps=np.linspace(0, simulation_time, n_timestamps_heatmap),
        )
    )

    return Experiment("external simulation example", simulation_time, [], data)
