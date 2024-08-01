from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.selectable_line_plot_item import SelectableLinePlotItem
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import shuffle_index


class TraceWidget(TimeseriesWidget):
    """A widget that displays a set of traces stacked vertically.

    traces should be a 2d array, where the first index is the neuron index and the second index is the time index.

    Applies global normalisation to the traces so that they all lie in range [-0.5, 0.5],
    without changing their relative amplitudes or their offset from zero.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data_type = CustomAnalysisDataType.Trace
        self._create_custom_analysis_menu()
        self.mouse_clicked.connect(self._on_mouse_clicked)

    def set_data(
        self,
        traces: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        super()._plot_data(traces, timestamps, dt)

    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        assert isinstance(data, np.ndarray)
        trace_norm = 2.0 * np.max(np.abs(data))
        for index, trace in enumerate(data):
            # offset each trace by its index, give each trace a unique color
            line = SelectableLinePlotItem(
                timestamps,
                trace / trace_norm + index,
                pen=shuffle_index(index, len(data)),
            )
            self.neuron_plots.append(line)
            self.addItem(line)

    def _on_mouse_clicked(self, neuron_index: int, time_index: int):
        self.timeline.setPos(time_index * self.dt)
