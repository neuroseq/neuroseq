from __future__ import annotations

import logging

import brian2.numpy_ as np
import numpy.typing as npt
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData
from neuroseq.pretty_print_enum import PrettyPrintEnum


class CustomAnalysisType(PrettyPrintEnum):
    Histogram = "mdi6.chart-histogram"
    Histogram2d = "mdi6.data-matrix"


class CustomAnalysisWindow(QtWidgets.QWidget):
    """A window that contains a CustomAnalysisWidget."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.setWindowTitle("Custom Analysis")
        self.resize(800, 600)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.analysis_widget = CustomAnalysisWidget(self)
        self.layout.addWidget(self.analysis_widget)


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
        self.spike_times_ms: npt.NDArray | None = None
        self.spike_indices: npt.NDArray | None = None
        self.dt_ms: float = 1.0

    def _preprocess_data(
        self,
        data: TraceData | SpikeData | SynapseData,
        analysis_type: CustomAnalysisType,
        neuron_slice: slice | None,
        time_slice: slice | None,
    ) -> None:
        self.traces = None
        self.spike_times_ms = None
        self.spike_indices = None
        if isinstance(data, TraceData):
            self.dt_ms = data.dt * 1000.0
        else:
            self.dt_ms = 1.0
        if neuron_slice is None:
            neuron_slice = slice(None, None)
            neuron_str = "all neurons"
        elif neuron_slice.stop == neuron_slice.start + 1:
            neuron_str = f"neuron {neuron_slice.start}"
        else:
            neuron_str = f"neurons {neuron_slice.start}-{neuron_slice.stop - 1}"
        self.neuron_slice = neuron_slice
        if time_slice is None:
            time_slice = slice(None, None)
            time_str = "all time"
        else:
            time_str = f"time {time_slice.start * self.dt_ms:.0f}ms-{time_slice.stop * self.dt_ms:.0f}ms"
        self.time_slice = time_slice
        self.subplot.set_title(f"{data.name} {analysis_type}: {neuron_str}, {time_str}")
        if isinstance(data, TraceData):
            self.traces = np.asarray(data.traces)[neuron_slice, time_slice]
        elif isinstance(data, SpikeData):
            spike_times_ms = (
                np.asarray(data.spike_times) * 1000.0
            )  # convert to milliseconds
            spike_indices = np.asarray(data.spike_indices)
            if time_slice.start is not None and time_slice.stop is not None:
                times_mask = (spike_times_ms >= time_slice.start * self.dt_ms) & (
                    spike_times_ms < time_slice.stop * self.dt_ms
                )
                spike_indices = spike_indices[times_mask]
                spike_times_ms = spike_times_ms[times_mask]
            if neuron_slice.start is not None and neuron_slice.stop is not None:
                neurons_mask = (spike_indices >= neuron_slice.start) & (
                    spike_indices < neuron_slice.stop
                )
                spike_indices = spike_indices[neurons_mask]
                spike_times_ms = spike_times_ms[neurons_mask]
            self.spike_indices = spike_indices
            self.spike_times_ms = spike_times_ms

    def set_data(
        self,
        data: TraceData | SpikeData | SynapseData | None,
        analysis_type: CustomAnalysisType | None,
        neuron_slice: slice | None,
        time_slice: slice | None,
    ) -> None:
        self.subplot.clear()
        if data is None or analysis_type is None:
            return
        self.logger.info(
            f"Plotting {analysis_type} of {type(data)} data for neuron slice {neuron_slice} and time slice {time_slice}"
        )
        self._preprocess_data(data, analysis_type, neuron_slice, time_slice)
        if analysis_type == CustomAnalysisType.Histogram:
            if isinstance(data, TraceData):
                x = np.asarray(self.traces).flatten()
            elif isinstance(data, SpikeData):
                x = self.spike_indices
            self.subplot.hist(x, bins=30, density=True)
            self.subplot.set_xlabel("Value")
            self.subplot.set_ylabel("Density")
        elif analysis_type == CustomAnalysisType.Histogram2d:
            if isinstance(data, TraceData):
                assert self.traces is not None
                x = np.tile(
                    self.dt_ms * np.arange(self.traces.shape[1]), self.traces.shape[0]
                )
                if time_slice is not None and time_slice.start is not None:
                    x += time_slice.start * self.dt_ms
                y = np.asarray(self.traces).flatten()
                self.subplot.set_ylabel("Values")
            elif isinstance(data, SpikeData):
                x = self.spike_times_ms
                y = self.spike_indices
                self.subplot.set_ylabel("Neurons")
            self.subplot.hist2d(x, y, bins=30, density=True)
            self.subplot.set_xlabel("Time")
        else:
            return
        self.draw()
