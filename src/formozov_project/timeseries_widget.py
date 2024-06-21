from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from formozov_project.timeline import Timeline


class TimeseriesWidget(pg.PlotWidget):
    """A base class for widgets that display a set of timeseries data for a number of neurons, stacked vertically.

    data should be a 2d array, where the first index is the neuron index and the second index is the time index.

    _plot_data() must be implemented by subclasses to plot the data.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.timeline = Timeline()
        self.neuron_plots: list[pg.BarGraphItem | pg.PlotDataItem] = []
        self.selected_neuron: int | None = None
        self.getAxis("top").show()
        self.set_data(data, timestamps)

    def _plot_data(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Plot the data for each neuron.

        This method should create a plot for each neuron and store them in the self.neuron_plots list.

        The plotted data should be normalised so that each neuron's data lies in the range [-0.5, 0.5],
        and offset vertically by the neuron's index.
        """
        raise NotImplementedError

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        self.neuron_plots.clear()
        self.selected_neuron = None
        self.clear()
        if data is None or data.shape[0] == 0:
            return
        if dt is None:
            dt = 1.0
        if timestamps is None:
            timestamps = dt * np.arange(data.shape[1])
        self.plotItem.getAxis("bottom").setLabel("Time (s)")
        # set the y-axis to display the trace index
        self.plotItem.getAxis("left").setLabel("Neuron")
        self.plotItem.getAxis("left").setTickSpacing(1, 1)
        # don't let user zoom/pan outside of the data
        self.setLimits(
            xMin=timestamps[0], xMax=timestamps[-1], yMin=-0.5, yMax=data.shape[0] - 0.5
        )
        self.addItem(self.timeline)
        self._plot_data(data, timestamps)

    def mousePressEvent(self, ev: Any) -> None:
        # select neuron data on mouse click based on mouse y-location in plot
        pos = self.mapToView(ev.position())
        index_float = pos.y() + 0.5
        index = int(index_float) if 0 <= index_float < len(self.neuron_plots) else None
        self.select_neuron(index)

    def select_neuron(self, index: int | None) -> None:
        alpha_selected = 1.0
        alpha_deselected = 0.3
        if index is None or self.selected_neuron == index:
            # deselect the currently selected neuron plot
            self.selected_neuron = None
            for neuron_plot in self.neuron_plots:
                neuron_plot.setOpacity(alpha_selected)
            return
        for neuron_plot in self.neuron_plots:
            neuron_plot.setOpacity(alpha_deselected)
        self.selected_neuron = index
        self.neuron_plots[index].setOpacity(alpha_selected)

    def connect_heatmap_widget(self, heatmap_widget: pg.ImageView) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.timeline.sigPositionChanged.connect(
            lambda: heatmap_widget.timeLine.setValue(self.timeline.getXPos())
        )
