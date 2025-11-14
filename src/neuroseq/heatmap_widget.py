from __future__ import annotations

import logging

import pyqtgraph as pg
from qtpy import QtWidgets

from neuroseq.experiment import SynapseData


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
        self.logger = logging.getLogger(__name__)

    def clear_data(self) -> None:
        self.clear()

    def set_data(
        self,
        data: SynapseData | None = None,
    ) -> None:
        self.clear()
        if data is None or data.values.shape[0] == 0:
            return
        self.logger.info(
            f"{data.values.shape} values with {data.timestamps.shape} timestamps"
        )
        self.setImage(data.values.T, xvals=data.timestamps * 1000.0)
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
        self.sigTimeChanged.connect(
            lambda idx, time: setattr(timeseries_widget.timeline, "selection", time)
        )
