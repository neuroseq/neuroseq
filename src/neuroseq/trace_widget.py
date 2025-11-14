from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtWidgets

from neuroseq.experiment import TraceData
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import cmap_transform_neuron_indices


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
        self.subplot.title = "Traces"

    def set_data(
        self,
        data: TraceData | None = None,
    ) -> None:
        self.subplot.clear()
        if data is None:
            return
        self.logger.info(
            f"{data.traces.shape} traces with dt={data.dt}s and units={data.units}"
        )
        self.data = data
        self.n_neurons = data.traces.shape[0]
        self.dt_ms = data.dt * 1000.0  # convert to milliseconds
        self.n_timestamps = data.traces.shape[1]
        trace_norm = 2.0 * np.max(np.abs(data.traces))
        if trace_norm == 0:
            trace_norm = 1.0
        # convert timestamps to milliseconds and stack to produce a format suitable for line_stack
        # todo: should be possible to re-use the same timestamps for all traces?
        timestamps_ms = np.arange(data.traces.shape[1], dtype=np.float32) * np.float32(
            data.dt * 1000.0
        )
        tiled_timestamps_ms = np.tile(
            timestamps_ms,
            (self.n_neurons, 1),
        )
        data = np.stack(
            [
                tiled_timestamps_ms,
                data.traces.astype(np.float32) / np.float32(trace_norm),
            ],
            axis=2,
            dtype=np.float32,
        )
        self.graphic = self.subplot.add_line_collection(
            data,
            cmap="jet",
            cmap_transform=cmap_transform_neuron_indices(self.n_neurons),
            thickness=2,
        )
        for i, line in enumerate(self.graphic.graphics):
            line.offset = (line.offset[0], i, line.offset[2])
        self.controller.max_height = max(self.controller.max_height, self.n_neurons)
        self.controller.max_width = max(
            self.controller.max_width, self.n_timestamps * self.dt_ms
        )
        self.post_set_data()
