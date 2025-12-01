from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SynapseData


class HeatmapWidget(QtWidgets.QWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://www.fastplotlib.org/ver/dev/api/widgets/ImageWidget.html
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._empty_data = np.zeros((1, 1, 1), dtype=np.float32)
        self.image_widget: fpl.ImageWidget = fpl.ImageWidget(
            data=self._empty_data,
            names=[""],
            figure_kwargs={"canvas_kwargs": {"parent": self}},
        )
        # disable the bottom imgui toolbar
        self.image_widget.figure.guis["bottom"] = None
        self.subplot: fpl.layouts.Subplot = self.image_widget.figure[0, 0]
        self.timestamps: np.ndarray = np.array([])
        self.widget = self.image_widget.show()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.logger = logging.getLogger(__name__)
        self.configure_subplot()
        self.setMinimumSize(QtCore.QSize(100, 100))

    def clear_data(self) -> None:
        """Clear the data in the heatmap, setting it to a 1x1x1 array of zeros."""
        self.image_widget.set_data(self._empty_data)
        self.timestamps = np.array([])

    def set_data(
        self,
        data: SynapseData | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap."""
        self.clear_data()
        if data is None or data.values.shape[0] == 0:
            return
        self.logger.info(
            f"{data.values.shape} values with {data.timestamps.shape} timestamps"
        )
        # Data format used by the old HeatmapWidget: (n_rows, n_columns, n_timesteps)
        # rearrange to (n_timesteps, n_rows, n_columns) for fastplotlib
        values = data.values
        values = np.transpose(values, (2, 0, 1))
        self.image_widget.set_data(values.astype(np.float32))
        self.timestamps = data.timestamps

    def configure_subplot(self) -> None:
        self.subplot.toolbar = False
        self.subplot.axes.y.ticks = None
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.y.tick_format = lambda *a: ""
        self.subplot.axes.x.ticks = None
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.axes.x.tick_format = lambda *a: ""

    def set_time_ms(self, time_ms: float) -> None:
        """Set the time to display: data from the closest timestamp to the given time is displayed."""
        time_secs = float(time_ms) / 1000.0
        if self.timestamps.size == 0:
            return
        if time_secs <= self.timestamps[0]:
            idx = 0
        elif time_secs >= self.timestamps[-1]:
            idx = self.timestamps.shape[0] - 1
        else:
            idx = int((np.abs(self.timestamps - time_secs)).argmin())
        self.logger.debug(
            f"Nearest timestamp to {time_ms}ms: {self.timestamps[idx]:.3g}s (index {idx})"
        )
        self.image_widget.current_index = {"t": idx}

    def get_timestamp_index(self) -> int:
        """Get the current index of the heatmap."""
        return self.image_widget.current_index["t"]

    def closeEvent(self, event):
        if self.image_widget is not None:
            self.image_widget.figure.close()
            self.image_widget = None
        super().closeEvent(event)
