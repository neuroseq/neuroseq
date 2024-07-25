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

    def clear_data(self) -> None:
        self.clear()

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.clear()
        if data is not None:
            self.setImage(data.T, xvals=timestamps)

    def connect_timeseries_widget_timeline(
        self, timeseries_widget: pg.PlotWidget
    ) -> None:
        """Connect the timeline in this widget to the timeline in the TimeseriesWidget."""
        self.sigTimeChanged.connect(
            lambda idx, time: timeseries_widget.timeline.setValue(time)
        )
