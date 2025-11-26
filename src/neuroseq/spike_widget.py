from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtWidgets

from neuroseq.data import SpikeData
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

    def set_data(self, data: SpikeData | None) -> None:
        self.subplot.clear()
        if data is None or data.spike_times.shape[0] == 0:
            self.subplot.title = "Spikes"
            return
        self.subplot.title = f"Spikes: {data.name}"
        self.data = data
        self.n_neurons = max(data.spike_indices) + 1
        self.logger.info(
            f"{data.spike_indices.shape} spikes with {self.n_neurons} neurons"
        )
        # spikes don't have a dt, but we need one when selecting time slices,
        # so we choose dt=1ms so time slices have a resolution of 1ms
        self.dt_ms = 1.0
        self.n_timestamps = int(1000.0 * max(data.spike_times) / self.dt_ms) + 1
        spike_data = np.stack(
            [
                data.spike_times.astype(np.float32) * np.float32(1000),
                data.spike_indices.astype(np.float32),
            ],
            axis=1,
        )
        self.graphic = self.subplot.add_scatter(
            spike_data,
            cmap="jet",
            cmap_transform=cmap_transform_neuron_indices(
                self.n_neurons, data.spike_indices
            ),
            sizes=7,
            size_space="screen",
            uniform_size=True,
        )
        self.post_set_data()
