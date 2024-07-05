from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.timeseries_widget import TimeseriesWidget


class SpikeWidget(TimeseriesWidget):
    """A widget that displays neuron spikes.

    spikes is a 1d array of spiking neuron indices, and timestamps is a 1d array of the corresponding timestamps of these spikes
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

    def set_data(
        self,
        spike_indices: np.ndarray,
        spike_times: np.ndarray,
        dt: float,
        n_neurons: int,
        n_timesteps: int,
    ) -> None:
        self.clear_data()
        self.setLimits(xMin=0, xMax=n_timesteps * dt, yMin=-0.5, yMax=n_neurons - 0.5)
        for neuron_index in range(n_neurons):
            # filter the spikes for this neuron
            spikes = spike_times[spike_indices == neuron_index]
            # vertically offset the data for each neuron by its index
            y_offset = neuron_index - 0.5
            # give each neuron a unique color
            brush = (neuron_index, n_neurons)
            bar = SelectableBarPlotItem(
                x=spikes + 0.5 * dt,
                width=dt,
                y0=y_offset,
                y1=y_offset + 0.95,
                pen=None,
                brush=brush,
            )
            self.neuron_plots.append(bar)
            self.addItem(bar)

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        # since timestamps is a sparse array the plotting logic is different, and is implemented in set_data
        pass
