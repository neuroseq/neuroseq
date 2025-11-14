from __future__ import annotations

import brian2.numpy_ as np
import pyqtgraph as pg
from qtpy import QtWidgets


class HeatmapWidget(pg.ImageView):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, discreteTimeLine=True)
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.setPredefinedGradient("viridis")
        self.relative_playback_speed = 1.0

    def clear_data(self) -> None:
        self.clear()

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap.
        data should be a 3d numpy array with shape (n_rows, n_columns, n_timesteps).
        timestamps should be a 1-d numpy array of timestamps in seconds.
        """
        self.clear()
        if data is not None and timestamps is not None:
            self.setImage(data.T, xvals=timestamps * 1000.0)
            self._update_playRate()

    def set_relative_playback_speed(self, relative_playback_speed: float) -> None:
        """Set the relative playback speed, where 1 is real-time."""
        self.relative_playback_speed = relative_playback_speed
        self._update_playRate()

    def _update_playRate(self) -> None:
        if hasattr(self, "tVals"):
            duration_msecs = self.tVals[-1] - self.tVals[0]
            duration_secs = duration_msecs / 1000
            realtime_fps = (self.nframes() - 1) / duration_secs
            self.playRate = self.relative_playback_speed * realtime_fps

    def connect_timeseries_widget_timeline(
        self, timeseries_widget: QtWidgets.QWidget
    ) -> None:
        """Connect the timeline in this widget to the timeline in the TimeseriesWidget."""

        def _on_sigTimeChanged(idx, time):
            print(f"Heatmap _on_sigTimeChanged: {idx}, {time}")
            timeseries_widget.timeline.selection = time

        self.sigTimeChanged.connect(_on_sigTimeChanged)
