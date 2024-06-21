from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from formozov_project.timeline import Timeline


def set_alpha(
    graphics_item: list[QtWidgets.QGraphicsItem] | QtWidgets.QGraphicsItem, alpha: float
) -> None:
    if isinstance(graphics_item, list):
        for item in graphics_item:
            set_alpha(item, alpha)
    else:
        graphics_item.setOpacity(alpha)


class TimeseriesWidget(pg.PlotWidget):
    """A base class for widgets that display a set of timeseries data for a number of neurons, stacked vertically.

    data should be a 2d array or list of 2d arrays, where the first index of each
    array is the neuron index and the second index is the time index.

    Subclasses should call _plot_data() and implement _plot_neuron_data() to plot the data.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.timeline = Timeline()
        self.neuron_plots: (
            list[list[pg.BarGraphItem | pg.PlotDataItem]]
            | list[pg.BarGraphItem | pg.PlotDataItem]
        ) = []
        self.selected_neuron: int | None = None
        self.getAxis("top").show()

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        """Plot the data for each neuron.

        This method should create a plot for each neuron and store them in the self.neuron_plots list.

        The plotted data should be normalised so that each neuron's data lies in the range [-0.5, 0.5],
        and offset vertically by the neuron's index.
        """
        raise NotImplementedError

    def _plot_data(
        self,
        data: list[np.ndarray] | np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        self.neuron_plots.clear()
        self.selected_neuron = None
        self.clear()
        if data is None:
            return
        n_neurons = data[0].shape[0] if isinstance(data, list) else data.shape[0]
        n_timestamps = data[0].shape[1] if isinstance(data, list) else data.shape[1]
        if isinstance(data, list) and not all(
            d.shape[0] == n_neurons and d.shape[1] == n_timestamps for d in data
        ):
            raise ValueError("All data arrays must have the same shape")
        if dt is None:
            dt = 1.0
        if timestamps is not None:
            dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
        else:
            timestamps = dt * np.arange(n_timestamps)

        self.plotItem.getAxis("bottom").setLabel("Time (s)")
        # set the y-axis to display the trace index
        self.plotItem.getAxis("left").setLabel("Neuron")
        self.plotItem.getAxis("left").setTickSpacing(1, 1)
        # don't let user zoom/pan outside of the data
        self.setLimits(
            xMin=timestamps[0], xMax=timestamps[-1], yMin=-0.5, yMax=n_neurons - 0.5
        )
        self.addItem(self.timeline)
        self._plot_neuron_data(data, timestamps, dt)

    def mousePressEvent(self, ev: Any) -> None:
        # select neuron data on mouse click based on mouse y-location in plot
        pos = self.mapToView(ev.position())
        index_float = pos.y() + 0.5
        index = int(index_float) if 0 <= index_float < len(self.neuron_plots) else None
        self.select_neuron(index)

    def select_neuron(self, index: int | None) -> None:
        alpha_bold = 1.0
        alpha_faint = 0.3
        if index is None or self.selected_neuron == index:
            self.selected_neuron = None
            set_alpha(self.neuron_plots, alpha_bold)
        else:
            self.selected_neuron = index
            set_alpha(self.neuron_plots, alpha_faint)
            set_alpha(self.neuron_plots[index], alpha_bold)

    def connect_heatmap_widget(self, heatmap_widget: pg.ImageView) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.timeline.sigPositionChanged.connect(
            lambda: heatmap_widget.timeLine.setValue(self.timeline.getXPos())
        )
