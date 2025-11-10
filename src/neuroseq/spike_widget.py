from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import cmap_transform_neuron_indices


class SpikeWidget(TimeseriesWidget):
    """A widget that displays neuron spikes.

    spikes is a 1d array of spiking neuron indices, and timestamps is a 1d array of the corresponding timestamps of these spikes
    """

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(controller, parent)
        self.subplot.title = "Spikes"
        self.data_type = CustomAnalysisDataType.Trace

    def set_data(
        self,
        spike_indices: np.ndarray,
        spike_times: np.ndarray,
        dt: float,
        n_neurons: int,
        n_timesteps: int,
    ) -> None:
        self.data = {
            "spike_indices": spike_indices,
            "spike_times": spike_times,
            "dt": dt,
        }
        self.n_neurons = n_neurons
        self.n_timestamps = n_timesteps
        self.dt = dt
        self.subplot.clear()
        if spike_indices is None:
            return
        if spike_indices.size == 0 or spike_times.size == 0:
            self.logger.info("No spikes to display for current dataset.")
            self.graphic = None
            self.post_set_data()
            return
        data = np.stack(
            [
                spike_times.astype(np.float32) * np.float32(1000),
                spike_indices.astype(np.float32),
            ],
            axis=1,
        )
        self.graphic = self.subplot.add_scatter(
            data,
            cmap="jet",
            cmap_transform=cmap_transform_neuron_indices(self.n_neurons, spike_indices),
            sizes=7,
            size_space="screen",
            uniform_size=True,
        )
        self.controller.max_height = max(self.controller.max_height, self.n_neurons)
        self.controller.max_width = max(
            self.controller.max_width, n_timesteps * dt * 1000.0
        )  # convert to milliseconds
        self.post_set_data()
