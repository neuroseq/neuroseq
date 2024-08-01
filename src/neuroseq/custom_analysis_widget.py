from __future__ import annotations

import logging

import brian2.numpy_ as np
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtWidgets

from neuroseq.utils import PrettyPrintEnum


class CustomAnalysisDataType(PrettyPrintEnum):
    Trace = 1
    Spike = 2


class CustomAnalysisType(PrettyPrintEnum):
    Histogram = 1


class CustomAnalysisWidget(MatplotlibWidget):
    """A custom analysis widget that analyses a slice of data and displays the results using matplotlib."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.subplot = self.getFigure().add_subplot(111)

    def set_data(
        self,
        data_type: CustomAnalysisDataType | None,
        analysis_type: CustomAnalysisType | None,
        neuron_slice: slice | None,
        time_slice: slice | None,
        data: dict[str, int | float | np.ndarray],
    ) -> None:
        self.subplot.clear()
        if data_type is None or analysis_type is None or len(data) == 0:
            return
        self.logger.info(
            f"Plotting {analysis_type} of {data_type} data {data.keys()} for neuron slice {neuron_slice} and time slice {time_slice}"
        )
        dt = data["dt"]
        if neuron_slice is None:
            neuron_slice = slice(None, None)
            neuron_str = "all neurons"
        elif neuron_slice.stop == neuron_slice.start + 1:
            neuron_str = f"neuron {neuron_slice.start}"
        else:
            neuron_str = f"neurons {neuron_slice.start}-{neuron_slice.stop-1}"
        if time_slice is None:
            time_slice = slice(None, None)
            time_str = "all time"
        else:
            time_str = f"time {time_slice.start*dt:.4f}s-{time_slice.stop*dt:.4f}s"
        self.subplot.set_title(f"{data_type} {analysis_type}: {neuron_str}, {time_str}")
        if analysis_type == CustomAnalysisType.Histogram:
            if data_type == CustomAnalysisDataType.Trace:
                arr = np.asarray(data["data"])
                x = arr[neuron_slice, time_slice].flatten()
            elif data_type == CustomAnalysisDataType.Spike:
                spike_times = np.asarray(data["spike_times"])
                spike_indices = np.asarray(data["spike_indices"])
                if time_slice.start is not None and time_slice.stop is not None:
                    spike_indices = spike_indices[
                        (spike_times >= time_slice.start * dt)
                        & (spike_times < time_slice.stop * dt)
                    ]
                if neuron_slice.start is not None and neuron_slice.stop is not None:
                    spike_indices = spike_indices[
                        (spike_indices >= neuron_slice.start)
                        & (spike_indices < neuron_slice.stop)
                    ]
                x = spike_indices
            else:
                return
            self.subplot.hist(x, bins=30, density=True)
            self.subplot.set_xlabel("Value")
            self.subplot.set_ylabel("Density")
        self.draw()
