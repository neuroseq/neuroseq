from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets


class HeatmapWidget(pg.ImageView):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, discreteTimeLine=True)
        self.setPredefinedGradient("viridis")
        self.set_data(data, timestamps)

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.clear()
        if data is not None:
            self.setImage(data, xvals=timestamps)

    def connect_trace_widget(self, trace_widget: pg.PlotWidget) -> None:
        """Connect the timeline in this widget to the timeline in the TraceWidget."""
        self.sigTimeChanged.connect(
            lambda idx, time: trace_widget.timeline.setValue(time)
        )
