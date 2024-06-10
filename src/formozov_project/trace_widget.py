from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets


class TraceWidget(pg.PlotWidget):
    """A widget that displays a set of traces stacked vertically.

    Assumes the traces are normalised to the range [0-1]
    """

    def __init__(
        self,
        traces: Optional[list[np.ndarray]] = None,
        timestamps: Optional[np.ndarray] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.traces: list[pg.PlotDataItem] = []
        self.timeline = pg.InfiniteLine(pos=0, angle=90, movable=True)
        self.timeline.addMarker("v", 1.0)
        self.timeline.addMarker("^", 0.0)
        self.selected_trace: Optional[int] = None
        self.getAxis("top").show()
        self.set_data(traces, timestamps)

    def set_data(
        self,
        traces: Optional[list[np.ndarray]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> None:
        self.clear()
        self.traces.clear()
        if traces is None or len(traces) == 0:
            return
        if timestamps is None:
            timestamps = np.arange(traces[0].shape[0])
        for index, trace in enumerate(traces):
            # offset each trace by its index, give each trace a unique color
            line = pg.PlotDataItem(
                timestamps, trace + index - 0.5, pen=(index, len(traces))
            )
            line.setCurveClickable(True, width=20)
            line.sigClicked.connect(lambda item, event, i=index: self.select_trace(i))
            self.traces.append(line)
            self.addItem(line)
        self.plotItem.getAxis("bottom").setLabel("Time (ms)")
        # set the y-axis to display the trace index
        self.plotItem.getAxis("left").setLabel("Neuron")
        self.plotItem.getAxis("left").setTickSpacing(1, 1)
        # don't let user zoom/pan outside of the data
        self.setLimits(
            xMin=timestamps[0], xMax=timestamps[-1], yMin=-0.5, yMax=len(traces) - 0.5
        )
        self.addItem(self.timeline)

    def select_trace(self, index: int) -> None:
        alpha_selected = 1.0
        alpha_deselected = 0.25
        if self.selected_trace is not None and self.selected_trace == index:
            # deselect the currently selected trace
            self.selected_trace = None
            for trace in self.traces:
                trace.setAlpha(alpha_selected, True)
            return
        for trace in self.traces:
            trace.setAlpha(alpha_deselected, True)
        self.selected_trace = index
        self.traces[index].setAlpha(alpha_selected, True)

    def connect_heatmap_widget(self, heatmap_widget: pg.ImageView) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.timeline.sigPositionChanged.connect(
            lambda: heatmap_widget.timeLine.setValue(self.timeline.getXPos())
        )
