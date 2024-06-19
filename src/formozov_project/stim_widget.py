from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from formozov_project.timeseries_widget import TimeseriesWidget


class StimWidget(TimeseriesWidget):
    """A widget that displays stimuli timeseries for neurons.

    stim should be a 2d array, where the first index is the neuron index and the second index is the time index.

    Assumes that the stimuli are non-negative.
    Applies global normalisation to the displayed stimuli such that the largest stimulus has an amplitude of 0.95.
    """

    def __init__(
        self,
        stim: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(stim, timestamps, dt, parent)

    def _plot_data(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        max_value = np.max(data)
        if max_value == 0:
            max_value = 1
        norm = max_value / 0.95
        for neuron_index, values in enumerate(data):
            # vertically offset the data for each neuron by its index
            y_offset = neuron_index - 0.5
            # give each neuron a unique color
            brush = (neuron_index, len(data))
            bar = pg.BarGraphItem(
                x=timestamps,
                width=1.0,
                y0=y_offset,
                y1=y_offset + values / norm,
                pen=None,
                brush=brush,
            )
            self.neuron_plots.append(bar)
            self.addItem(bar)
