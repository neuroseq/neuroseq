from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import cmap_transform_neuron_indices


class TraceNormDialog(QtWidgets.QDialog):
    """A dialog to set the normalization factor for the traces."""

    def __init__(
        self,
        trace_widget: TraceWidget,
    ) -> None:
        super().__init__(trace_widget)
        self.setWindowTitle("Set Trace Normalization")
        self.trace_widget = trace_widget
        self.trace_norm_percent = self.trace_widget.trace_norm_percent
        self.offset_by_median = self.trace_widget.offset_by_median
        self.clip = self.trace_widget.clip

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label_norm = QtWidgets.QLabel(
            f"Normalization: {self.trace_norm_percent}%", self
        )
        self.layout.addWidget(self.label_norm)
        self.slider_norm = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider_norm.setRange(100, 10000)
        self.slider_norm.setValue(self.trace_norm_percent)
        self.slider_norm.valueChanged.connect(self.update_norm_input)
        self.layout.addWidget(self.slider_norm)

        self.chk_offset = QtWidgets.QCheckBox("Center median")
        self.chk_offset.setChecked(self.clip)
        self.chk_offset.clicked.connect(self.chk_offset_clicked)
        self.layout.addWidget(self.chk_offset)

        self.chk_clip = QtWidgets.QCheckBox("Clip values")
        self.chk_clip.setChecked(self.clip)
        self.chk_clip.clicked.connect(self.chk_clip_clicked)
        self.layout.addWidget(self.chk_clip)

        self.btn_reset_defaults = QtWidgets.QPushButton("Reset to defaults", self)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        self.layout.addWidget(self.btn_reset_defaults)

        self.btn_close = QtWidgets.QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        self.layout.addWidget(self.btn_close)

    def reset_defaults(self) -> None:
        self.slider_norm.setValue(100)
        self.offset_by_median = True
        self.clip = True
        self.chk_offset.setChecked(self.offset_by_median)
        self.chk_offset.setChecked(self.clip)
        self.update_norm_input(self.slider_norm.value())

    def chk_offset_clicked(self, checked: bool) -> None:
        self.offset_by_median = checked
        self.update_trace_widget()

    def chk_clip_clicked(self, checked: bool) -> None:
        self.clip = checked
        self.update_trace_widget()

    def update_norm_input(self, value: int) -> None:
        self.trace_norm_percent = value
        self.label_norm.setText(f"Normalization: {value}%")
        self.update_trace_widget()

    def update_trace_widget(self) -> None:
        self.trace_widget.trace_norm_percent = self.trace_norm_percent
        self.trace_widget.offset_by_median = self.offset_by_median
        self.trace_widget.clip = self.clip
        self.trace_widget.replot_traces()


class TraceWidget(TimeseriesWidget):
    """A widget that displays a set of traces stacked vertically.

    traces should be a 2d array, where the first index is the neuron index and the second index is the time index.
    """

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(controller, parent)
        self.subplot.title = "Traces"
        self.trace_norm_percent = 100
        self.offset_by_median = True
        self.clip = True

    def trace_norm_dialog(self):
        TraceNormDialog(self).exec()

    def get_normalized_trace_data(self) -> np.ndarray:
        if self.data is None or not isinstance(self.data, TraceData):
            return np.array([], dtype=np.float32)
        if self.offset_by_median:
            median_values = np.median(self.data.traces, axis=1)
            max_values = np.max(self.data.traces, axis=1)
            max_trace = np.max(np.abs(max_values - median_values))
        else:
            max_trace = np.max(np.abs(self.data.traces))
        norm = 0.5 if max_trace == 0 else 1.0 / (2.0 * max_trace)
        norm = np.float32((self.trace_norm_percent / 100.0) * norm)
        if self.offset_by_median:
            offset = (-median_values * norm).astype(np.float32)
        else:
            offset = np.zeros(self.n_neurons, dtype=np.float32)
        data = norm * self.data.traces.astype(np.float32) + offset[:, np.newaxis]
        if self.clip:
            np.clip(data, -0.5, 0.5, out=data)
        return data

    def replot_traces(self) -> None:
        """Update the plotted traces based on the current normalization settings."""
        if (
            self.data is not None
            and isinstance(self.data, TraceData)
            and self.graphic is not None
        ):
            normalized_traces = self.get_normalized_trace_data()
            for line_index, line_graphic in enumerate(self.graphic):
                line_graphic.data[:, 1] = normalized_traces[line_index, :]

    def set_data(
        self,
        data: TraceData | None = None,
    ) -> None:
        self.subplot.clear()
        if data is None:
            self.subplot.title = "Traces"
            return
        self.logger.info(
            f"{data.traces.shape} traces with dt={data.dt}s and units={data.units}"
        )
        self.subplot.title = f"Traces: {data.name}"
        self.data = data
        self.n_neurons = data.traces.shape[0]
        self.dt_ms = data.dt * 1000.0  # convert to milliseconds
        self.n_timestamps = data.traces.shape[1]
        # convert timestamps to milliseconds and stack to produce a format suitable for line_stack
        # todo: should be possible to re-use the same timestamps for all traces?
        timestamps_ms = np.arange(data.traces.shape[1], dtype=np.float32) * np.float32(
            data.dt * 1000.0
        )
        tiled_timestamps_ms = np.tile(
            timestamps_ms,
            (self.n_neurons, 1),
        )
        normalized_traces = self.get_normalized_trace_data()
        data = np.stack(
            [
                tiled_timestamps_ms,
                normalized_traces,
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
        self.post_set_data()
