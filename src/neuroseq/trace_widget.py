from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import cmap_transform_indices


class TraceWidget(TimeseriesWidget):
    """A widget that displays a set of traces stacked vertically.

    traces should be a 2d array, where the first index is the neuron index and the second index is the time index.

    Applies global normalisation to the traces so that they all lie in range [-0.5, 0.5],
    without changing their relative amplitudes or their offset from zero.
    """

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(controller, parent)
        self.data_type = CustomAnalysisDataType.Trace
        self.subplot.title = "Traces"

    def set_data(
        self,
        traces: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        assert isinstance(traces, np.ndarray)
        if not isinstance(timestamps, np.ndarray):
            if dt is None:
                dt = 1.0
            timestamps = dt * np.arange(traces.shape[1])
        self.data = {"data": traces, "timestamps": timestamps, "dt": 0.001}
        trace_norm = 2.0 * np.max(np.abs(traces))
        if trace_norm == 0:
            trace_norm = 1.0
        # convert timestamps to milliseconds and stack to produce a format suitable for line_stack
        # todo: should be possible to re-use the same timestamps for all traces?
        tiled_timestamps = np.tile(
            timestamps.astype(np.float32) * np.float32(1000.0), (traces.shape[0], 1)
        )
        data = np.stack(
            [tiled_timestamps, traces.astype(np.float32) / np.float32(trace_norm)],
            axis=2,
            dtype=np.float32,
        )
        self.subplot.clear()
        if traces is None:
            return
        self.graphic = self.subplot.add_line_collection(
            data,
            cmap="jet",
            cmap_transform=cmap_transform_indices(np.arange(traces.shape[0])),
            thickness=1,
        )
        for i, line in enumerate(self.graphic.graphics):
            line.offset = (line.offset[0], i, line.offset[2])
        self.controller.max_height = max(self.controller.max_height, traces.shape[0])
        self.n_neurons = traces.shape[0]
        if timestamps is not None:
            self.controller.max_width = max(
                self.controller.max_width, timestamps[-1] - timestamps[0]
            )
            self.n_timestamps = timestamps.shape[0]
        self.post_set_data()
