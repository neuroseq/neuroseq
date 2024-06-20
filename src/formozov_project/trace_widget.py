from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from formozov_project.timeseries_widget import TimeseriesWidget


class TraceWidget(TimeseriesWidget):
    """A widget that displays a set of traces stacked vertically.

    traces should be a 2d array, where the first index is the neuron index and the second index is the time index.

    Applies global normalisation to the traces so that they all lie in range [-0.5, 0.5],
    without changing their relative amplitudes or their offset from zero.
    """

    def __init__(
        self,
        traces: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(traces, timestamps, dt, parent)

    def _plot_data(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        trace_norm = 2.0 * np.max(np.abs(data))
        for index, trace in enumerate(data):
            # offset each trace by its index, give each trace a unique color
            line = pg.PlotDataItem(
                timestamps, trace / trace_norm + index, pen=(index, len(data))
            )
            self.neuron_plots.append(line)
            self.addItem(line)
