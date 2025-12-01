from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from neuroseq.data import SynapseData


class HeatmapWidget(QtWidgets.QWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.
    The widget is controlled via a QSlider, which controls the Heatmap index, connected TimeseriesWidgets linear selectors and the playback logic.

    see https://www.fastplotlib.org/ver/dev/api/widgets/ImageWidget.html
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        empty_data = np.zeros((1, 1, 1))
        self.image_widget: fpl.ImageWidget = fpl.ImageWidget(
            data=empty_data,
            names=[""],
            figure_kwargs={"canvas_kwargs": {"parent": self}},
        )
        self.image_widget.figure.guis["bottom"] = None
        self.subplot: fpl.layouts.Subplot = self.image_widget.figure[0, 0]
        self.timestamps: np.ndarray = np.array([0.0])
        self.widget = self.image_widget.show()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        self.slider = QtWidgets.QSlider(orientation=Qt.Horizontal, parent=self)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.playing = False
        self.relative_playback_speed = 1.0
        self.logger = logging.getLogger(__name__)
        self.configure_subplot()
        self.image_widget.figure.add_animations(self.update_heatmap)
        self.setMinimumSize(QtCore.QSize(100, 100))

    def clear_data(self) -> None:
        """Clear the data in the heatmap, setting it to a 1x1x1 array of zeros."""
        self.image_widget.set_data(np.zeros((1, 1, 1)))
        self.timestamps = np.array([0.0])
        self.slider.setValue(0)
        self.slider.setEnabled(False)

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
        self.image_widget.set_data(values)
        self.timestamps = data.timestamps
        self.slider.setMaximum(self.nframes() - 1)
        self.slider.setMinimum(0)
        self.slider.setEnabled(True)

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

    def set_relative_playback_speed(self, relative_playback_speed: float) -> None:
        """Set the relative playback speed, where 1 is real-time.
        Currently relative playback speed does not affect the playback rate.
        """
        self.relative_playback_speed = relative_playback_speed

    def nframes(self) -> int:
        """Return the number of frames in the heatmap."""
        return self.timestamps.shape[0]

    def connect_timeseries_widget_timeline(
        self, timeseries_widget: QtWidgets.QWidget
    ) -> None:
        """Connect the timeline in this widget to the timeline in the TimeseriesWidget."""
        self.slider.valueChanged.connect(timeseries_widget.on_slider_value_changed)

    def toggle_pause(self) -> None:
        """Toggle the playback state between playing and paused."""
        self.playing = not self.playing

    def stop(self) -> None:
        """Stop playback of the heatmap."""
        self.playing = False

    def set_current_index(self, idx: int) -> None:
        """Set the current index of the slider.
        This will also update the heatmap index,
        and via signals the connected TimeseriesWidget timelines.
        """
        self.slider.setValue(idx)
        self.slider.valueChanged.emit(idx)

    def set_heatmap_index(self, idx: int) -> None:
        """Set the current index of the heatmap.
        The heatmap index should only be modifed by the slider.
        """
        self.image_widget.current_index = {"t": idx}

    def get_heatmap_index(self) -> int:
        """Get the current index of the heatmap."""
        return self.image_widget.current_index["t"]

    def update_heatmap(self) -> None:
        """Update the heatmap to the next frame if playing."""
        if not self.playing:
            return
        current_idx = self.image_widget.current_index["t"]
        next_idx = current_idx + 1
        if next_idx >= self.nframes():
            next_idx = 0
        self.set_current_index(next_idx)

    def on_slider_value_changed(self, value: int) -> None:
        self.set_heatmap_index(value)

    def closeEvent(self, event):
        if self.image_widget is not None:
            self.image_widget.figure.close()
            self.image_widget = None
        super().closeEvent(event)
