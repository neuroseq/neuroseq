from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from formozov_project.selectable_bar_line_plot_item import SelectableBarLinePlotItem
from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.selectable_line_plot_item import SelectableLinePlotItem
from formozov_project.timeseries_widget import TimeseriesWidget


class StimTraceWidget(TimeseriesWidget):
    """A widget that displays stimuli and trace timeseries for neurons.

    stims and traces should be 2d arrays, where the first index is the neuron index and the second index is the time index.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

    def set_data(
        self,
        stims: np.ndarray | None = None,
        traces: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        data = [stims, traces] if traces is not None and stims is not None else None
        super()._plot_data(data, timestamps, dt)

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        stims, traces = data
        max_stim = np.max(stims)
        if max_stim == 0:
            max_stim = 1
        stim_norm = max_stim / 0.95
        trace_norm = 2.0 * np.max(np.abs(traces)) if traces is not None else 1.0
        for neuron_index, [stim, trace] in enumerate(zip(stims, traces)):
            # vertically offset the data for each neuron by its index
            y_offset_center = neuron_index
            # give each neuron a unique color
            brush = (neuron_index, len(stims))
            bar = SelectableBarPlotItem(
                x=timestamps,
                width=dt,
                y0=y_offset_center - 0.5,
                y1=y_offset_center - 0.5 + stim / stim_norm,
                pen=None,
                brush=brush,
            )
            line = SelectableLinePlotItem(
                timestamps, y_offset_center + trace / trace_norm, pen=brush
            )
            self.addItem(line)
            self.addItem(bar)
            line_bar_plot_item = SelectableBarLinePlotItem(bar, line)
            self.neuron_plots.append(line_bar_plot_item)
