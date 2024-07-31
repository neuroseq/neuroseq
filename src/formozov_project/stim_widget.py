from __future__ import annotations

import brian2 as b2
import brian2.numpy_ as np
from qtpy import QtWidgets

from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.timeseries_widget import TimeseriesWidget
from formozov_project.utils import shuffle_index


class StimWidget(TimeseriesWidget):
    """A widget that displays a brian2.TimedArray stimuli timeseries for neurons.

    Assumes that the stimuli are non-negative.
    Applies global normalisation to the displayed stimuli such that the largest stimulus has an amplitude of 1.0.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plotItem.showGrid(x=True, y=True)

    def set_data(self, stims: b2.TimedArray | None = None) -> None:
        if stims is None:
            self.clear_data()
            return
        super()._plot_data(stims.values.T, dt=stims.dt)

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        assert isinstance(data, np.ndarray)
        stim_max = np.max(data)
        if stim_max == 0:
            stim_max = 1
        stim_norm = stim_max
        for neuron_index, stim in enumerate(data):
            # vertically offset the data for each neuron by its index
            y_offset = neuron_index - 0.5
            # give each neuron a unique color
            brush = shuffle_index(neuron_index, len(data))
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
