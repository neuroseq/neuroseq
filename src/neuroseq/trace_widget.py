from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.experiment import TraceData
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
        self.clip = self.trace_widget.clip

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(
            f"Normalization: {self.trace_norm_percent}%", self
        )
        self.layout.addWidget(self.label)

        self.slider_norm = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider_norm.setRange(100, 10000)
        self.slider_norm.setValue(self.trace_norm_percent)
        self.slider_norm.valueChanged.connect(self.update_norm_input)
        self.layout.addWidget(self.slider_norm)

        self.btn_reset_norm = QtWidgets.QPushButton("Reset Normalization", self)
        self.btn_reset_norm.clicked.connect(self.reset_norm)
        self.layout.addWidget(self.btn_reset_norm)

        self.chk_clip = QtWidgets.QCheckBox("Clip values")
        self.chk_clip.setChecked(self.clip)
        self.chk_clip.clicked.connect(self.check_clip_clicked)
        self.layout.addWidget(self.chk_clip)

        self.btn_close = QtWidgets.QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        self.layout.addWidget(self.btn_close)

    def reset_norm(self) -> None:
        self.slider_norm.setValue(100)
        self.update_norm_input(self.slider_norm.value())

    def check_clip_clicked(self, checked: bool) -> None:
        self.clip = checked
        self.update_trace_widget()

    def update_norm_input(self, value: int) -> None:
        self.trace_norm_percent = value
        self.label.setText(f"Normalization: {value}%")
        self.update_trace_widget()

    def update_trace_widget(self) -> None:
        self.trace_widget.set_trace_norm(self.trace_norm_percent, self.clip)


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
        self.trace_norm_percent = 100
        self.clip = True

    def trace_norm_dialog(self):
        TraceNormDialog(self).exec()

    def set_trace_norm(self, trace_norm_percent: int, clip: bool) -> None:
        """Set the normalization factor for the traces."""
        self.trace_norm_percent = trace_norm_percent
        self.clip = clip
        if (
            self.data is not None
            and isinstance(self.data, TraceData)
            and self.graphic is not None
        ):
            max_trace = np.max(np.abs(self.data.traces))
            if max_trace == 0:
                norm = np.float32(1.0)
            else:
                norm = np.float32((self.trace_norm_percent / 100.0) / (2.0 * max_trace))
            for line_index, line_graphic in enumerate(self.graphic):
                data = (norm * self.data.traces[line_index, :]).astype(np.float32)
                if self.clip:
                    data = np.clip(data, -0.5, 0.5)
                line_graphic.data[:, 1] = data

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
        max_trace = np.max(np.abs(self.data.traces))
        if max_trace == 0:
            norm = np.float32(1.0)
        else:
            norm = np.float32((self.trace_norm_percent / 100.0) / (2.0 * max_trace))
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
                norm * data.traces.astype(np.float32),
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
