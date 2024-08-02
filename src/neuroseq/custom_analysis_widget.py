from __future__ import annotations

import logging

import brian2.numpy_ as np
import numpy.typing as npt
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtWidgets

from neuroseq.utils import PrettyPrintEnum


class CustomAnalysisDataType(PrettyPrintEnum):
    Trace = 1
    Spike = 2


class CustomAnalysisType(PrettyPrintEnum):
    Histogram = "mdi6.chart-histogram"
    Histogram2d = "mdi6.data-matrix"


class CustomAnalysisWidget(MatplotlibWidget):
    """A custom analysis widget that analyses a slice of data and displays the results using matplotlib."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.subplot = self.getFigure().add_subplot(111)
        self.traces: npt.NDArray | None = None
        self.spike_times: npt.NDArray | None = None
        self.spike_indices: npt.NDArray | None = None
        self.dt: float = 1.0

    def _preprocess_data(
        self,
        data_type: CustomAnalysisDataType,
        analysis_type: CustomAnalysisType,
        neuron_slice: slice | None,
        time_slice: slice | None,
        data: dict[str, int | float | npt.NDArray],
    ) -> None:
        self.traces = None
        self.spike_times = None
        self.spike_indices = None
        self.dt = float(data["dt"])
        if neuron_slice is None:
            neuron_slice = slice(None, None)
            neuron_str = "all neurons"
        elif neuron_slice.stop == neuron_slice.start + 1:
            neuron_str = f"neuron {neuron_slice.start}"
        else:
            neuron_str = f"neurons {neuron_slice.start}-{neuron_slice.stop-1}"
        self.neuron_slice = neuron_slice
        if time_slice is None:
            time_slice = slice(None, None)
            time_str = "all time"
        else:
            time_str = (
                f"time {time_slice.start*self.dt:.4f}s-{time_slice.stop*self.dt:.4f}s"
            )
        self.time_slice = time_slice
        self.subplot.set_title(f"{data_type} {analysis_type}: {neuron_str}, {time_str}")
        if data_type == CustomAnalysisDataType.Trace:
            self.traces = np.asarray(data["data"])[neuron_slice, time_slice]
        elif data_type == CustomAnalysisDataType.Spike:
            spike_times = np.asarray(data["spike_times"])
            spike_indices = np.asarray(data["spike_indices"])
            if time_slice.start is not None and time_slice.stop is not None:
                times_mask = (spike_times >= time_slice.start * self.dt) & (
                    spike_times < time_slice.stop * self.dt
                )
                spike_indices = spike_indices[times_mask]
                spike_times = spike_times[times_mask]
            if neuron_slice.start is not None and neuron_slice.stop is not None:
                neurons_mask = (spike_indices >= neuron_slice.start) & (
                    spike_indices < neuron_slice.stop
                )
                spike_indices = spike_indices[neurons_mask]
                spike_times = spike_times[neurons_mask]
            self.spike_indices = spike_indices
            self.spike_times = spike_times

    def set_data(
        self,
        data_type: CustomAnalysisDataType | None,
        analysis_type: CustomAnalysisType | None,
        neuron_slice: slice | None,
        time_slice: slice | None,
        data: dict[str, int | float | npt.NDArray],
    ) -> None:
        self.subplot.clear()
        if data_type is None or analysis_type is None or len(data) == 0:
            return
        self.logger.info(
            f"Plotting {analysis_type} of {data_type} data {data.keys()} for neuron slice {neuron_slice} and time slice {time_slice}"
        )
        self._preprocess_data(data_type, analysis_type, neuron_slice, time_slice, data)
        if analysis_type == CustomAnalysisType.Histogram:
            if data_type == CustomAnalysisDataType.Trace:
                x = np.asarray(self.traces).flatten()
            elif data_type == CustomAnalysisDataType.Spike:
                x = self.spike_indices
            self.subplot.hist(x, bins=30, density=True)
            self.subplot.set_xlabel("Value")
            self.subplot.set_ylabel("Density")
        elif analysis_type == CustomAnalysisType.Histogram2d:
            if data_type == CustomAnalysisDataType.Trace:
                assert self.traces is not None
                x = np.tile(
                    self.dt * np.arange(self.traces.shape[1]), self.traces.shape[0]
                )
                if time_slice is not None and time_slice.start is not None:
                    x += time_slice.start * self.dt
                y = np.asarray(self.traces).flatten()
                self.subplot.set_ylabel("Values")
            elif data_type == CustomAnalysisDataType.Spike:
                x = self.spike_times
                y = self.spike_indices
                self.subplot.set_ylabel("Neurons")
            self.subplot.hist2d(x, y, bins=30, density=True)
            self.subplot.set_xlabel("Time")
        else:
            return
        self.draw()
