from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from formozov_project.timeseries_widget import TimeseriesWidget


class SpikeWidget(TimeseriesWidget):
    """A widget that displays a set of spikes stacked vertically.

    Assumes that the spikes are non-negative.
    Applies global normalisation to the spikes such that the largest spike has an amplitude of 0.95.
    """

    def __init__(
        self,
        spikes: list[np.ndarray] | None = None,
        timestamps: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(spikes, timestamps, parent)

    def _plot_data(self, data: list[np.ndarray], timestamps: np.ndarray) -> None:
        spike_norm = np.max(data)
        for index, spike in enumerate(data):
            # offset each spike by its index, give each a unique color
            y_offset = index - 0.5
            bar = pg.BarGraphItem(
                x=timestamps,
                width=0.5,
                y0=y_offset,
                y1=y_offset + 0.95 * spike / spike_norm,
                pen=None,
                brush=(index, len(spike)),
            )
            self.neuron_plots.append(bar)
            self.addItem(bar)
