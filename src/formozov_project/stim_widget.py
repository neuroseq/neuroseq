from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.timeseries_widget import TimeseriesWidget


class StimWidget(TimeseriesWidget):
    """A widget that displays stimuli timeseries for neurons.

    stim should be a 2d array, where the first index is the neuron index and the second index is the time index.

    Assumes that the stimuli are non-negative.
    Applies global normalisation to the displayed stimuli such that the largest stimulus has an amplitude of 0.95.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

    def set_data(
        self,
        stims: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        super()._plot_data(stims, timestamps, dt)

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        assert isinstance(data, np.ndarray)
        stim_max = np.max(data)
        if stim_max == 0:
            stim_max = 1
        stim_norm = stim_max / 0.95
        for neuron_index, stim in enumerate(data):
            # vertically offset the data for each neuron by its index
            y_offset = neuron_index - 0.5
            # give each neuron a unique color
            brush = (neuron_index, len(data))
            bar = SelectableBarPlotItem(
                x=timestamps + 0.5 * dt,
                width=dt,
                y0=y_offset,
                y1=y_offset + stim / stim_norm,
                pen=None,
                brush=brush,
            )
            self.neuron_plots.append(bar)
            self.addItem(bar)
