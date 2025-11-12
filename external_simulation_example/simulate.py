# Example of an external simulation function that can be imported into neuroseq

import numpy as np

SimData = dict[str, dict[str, str | float | int | np.ndarray]]


def simulate(stims_dict: dict[str, np.ndarray], simulation_time: float) -> SimData:
    # This function is provided by neuroseq with a dict of stimuli, and the desired simulation time.
    # stims_dict is a dict of key:data pairs, where data contains
    #   - "array": 2d numpy array of timestamps x neurons
    #   - "dt": the time step of the stimulus in seconds as a float

    # ...do some simulation using stims_dict and simulation_time...

    # It should return a dict of key:data pairs, where the key is the name to be displayed in the GUI.
    data: SimData = {}
    # Three types of data can be returned, which differ depending on the widget used to display the data in the GUI
    # Below are some example random data of each type:

    # "trace" data is a dict of the following format:
    #   - "widget": "trace"
    #   - "timestamps": 1d array of timestamps
    #   - "traces": 2d array of neurons x timestamps
    n_neurons_trace = 32
    n_timestamps_trace = 2000
    data["trace_example"] = {
        "widget": "trace",
        "timestamps": np.linspace(0, simulation_time, n_timestamps_trace),
        "traces": np.random.rand(n_neurons_trace, n_timestamps_trace),
    }

    # "spike" data is a dict of the following format:
    #   - "widget": "spike"
    #   - "timestamps": 1d array of spike timestamps
    #   - "spikes": 1d array of spike neuron indices
    #   - "dt": the simulation time step in seconds as a float
    # Neuroseq infers neuron/time counts from the spike payload.
    n_neurons_spike = 32
    n_timesteps_spike = 2000
    n_spikes_spike = 74
    dt_spike = simulation_time / n_timesteps_spike
    data["spike_example"] = {
        "widget": "spike",
        "timestamps": np.linspace(0, simulation_time, n_spikes_spike),
        "spikes": np.random.randint(n_neurons_spike, size=(n_spikes_spike,)),
        "dt": dt_spike,
    }

    # "heatmap" data is a dict of the following format:
    #   - "widget": "heatmap"
    #   - "timestamps": 1d array of timestamps
    #   - "matrix": 3d array of neurons x neurons x timestamps
    n_neurons_heatmap = 32
    n_timestamps_heatmap = 100
    data["heatmap_example"] = {
        "widget": "heatmap",
        "timestamps": np.linspace(0, simulation_time, n_timestamps_heatmap),
        "matrix": np.random.rand(
            n_neurons_heatmap, n_neurons_heatmap, n_timestamps_heatmap
        ),
    }

    return data
