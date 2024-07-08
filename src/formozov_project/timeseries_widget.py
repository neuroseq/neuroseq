from __future__ import annotations

import abc
import logging
import math
from enum import Enum
from typing import Any
from typing import TypeAlias

import brian2.numpy_ as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from formozov_project.selectable_bar_line_plot_item import SelectableBarLinePlotItem
from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.selectable_line_plot_item import SelectableLinePlotItem
from formozov_project.timeline import Timeline
from formozov_project.utils import QtABCMeta

SelectablePlotItem: TypeAlias = (
    SelectableLinePlotItem | SelectableBarPlotItem | SelectableBarLinePlotItem
)


class MouseMode(Enum):
    NONE = 1
    SELECT = 2
    MOVE = 3


class TimeseriesWidget(pg.PlotWidget, metaclass=QtABCMeta):
    """A base class for widgets that display a set of timeseries data for a number of neurons, stacked vertically.

    data should be a 2d array or list of 2d arrays, where the first index of each
    array is the neuron index and the second index is the time index.

    Subclasses should call _plot_data() and implement _plot_neuron_data() to plot the data.
    """

    selection_moved = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.timeline = Timeline()
        self.timeline.setZValue(1e8)
        self.dt = 1.0
        self.mouse_mode = MouseMode.NONE
        self.mouse_press_pos: QtCore.QPointF | None = None
        self.mouse_selected_rect: QtCore.QRectF = QtCore.QRectF()
        self.capture_mouse_events = False
        self.selected_time_slice: slice | None = None
        self.selected_neuron_slice: slice | None = None
        self.neuron_plots: list[SelectablePlotItem] = []
        self.getAxis("top").show()
        self.roi = QtWidgets.QGraphicsRectItem(0, 0, 0, 0)
        self.roi.setPen(QtGui.QColor(255, 255, 255, 100))
        self.roi.setBrush(QtGui.QColor(200, 200, 255, 100))
        self.roi.setZValue(1e9)
        self.roi.hide()
        self.scene().addItem(self.roi)
        # in principle self.sigRangeChanged should be enough here, but it seems to get emitted before the view transform
        # used by self.mapFromView is updated, so we use the underlying ViewBox's sigRangeChanged signal instead
        self.plotItem.vb.sigRangeChanged.connect(self.update_roi)
        self.sigRangeChanged.connect(self.update_roi)

    def clear_data(self) -> None:
        self.neuron_plots.clear()
        self.unselect_neurons()
        self.clear()
        for ax_label in ["top", "bottom"]:
            ax = self.plotItem.getAxis(ax_label)
            ax.setLabel("Time (s)")
        # set the y-axis to display the trace index
        self.plotItem.getAxis("left").setLabel("Neuron")
        self.plotItem.getAxis("left").setTickSpacing(1, 1)
        self.addItem(self.timeline)

    @abc.abstractmethod
    def _plot_neuron_data(
        self, data: list[np.ndarray] | np.ndarray, timestamps: np.ndarray, dt: float
    ) -> None:
        """Plot the data for each neuron.

        This method should create a plot for each neuron and store them in the self.neuron_plots list.

        The plotted data should be normalised so that each neuron's data lies in the range [-0.5, 0.5],
        and offset vertically by the neuron's index.
        """
        # fallback for pyside6 which apparently does not fully support abstract methods in a metaclass
        raise NotImplementedError("Subclasses must implement _plot_neuron_data")

    def _plot_data(
        self,
        data: list[np.ndarray] | np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        self.clear_data()
        if data is None:
            return
        shape = data[0].shape if isinstance(data, list) else data.shape
        self.logger.debug(f"Plotting data with shape {shape}")
        if len(shape) != 2:
            return
        n_neurons, n_timestamps = shape
        if n_neurons == 0 or n_timestamps == 0:
            return
        if isinstance(data, list) and not all(
            d.shape[0] == n_neurons and d.shape[1] == n_timestamps for d in data
        ):
            raise ValueError("All data arrays must have the same shape")
        if timestamps is not None:
            # timestamps, if provided, take precedence over dt
            self.dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
        else:
            self.dt = dt if dt is not None else 1.0
            timestamps = self.dt * np.arange(n_timestamps)
        # don't let user zoom/pan outside of the data
        self.setLimits(
            xMin=timestamps[0],
            xMax=timestamps[-1] + self.dt,
            yMin=-0.5,
            yMax=n_neurons - 0.5,
        )
        self._plot_neuron_data(data, timestamps, self.dt)

    def mousePressEvent(self, ev: Any) -> None:
        if not self.capture_mouse_events:
            return super().mousePressEvent(ev)
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self.mouse_press_pos = ev.position()
        if self.roi.isVisible() and self.roi.rect().contains(ev.position()):
            self.mouse_mode = MouseMode.MOVE
        else:
            self.mouse_mode = MouseMode.SELECT
        self.logger.debug(f"Starting f{self.mouse_mode} from {self.mouse_press_pos}")

    def mouseMoveEvent(self, ev: Any) -> None:
        if not self.capture_mouse_events:
            return super().mouseMoveEvent(ev)
        if self.mouse_press_pos is None:
            return
        pos = ev.position()
        delta_pos = pos - self.mouse_press_pos
        self.logger.debug(f"Delta pos f{delta_pos}")
        self.logger.debug(
            f"{self.mouse_selected_rect} -> {self.mouse_selected_rect.translated(delta_pos)}"
        )
        if self.mouse_mode == MouseMode.MOVE:
            self.roi.setRect(self.mouse_selected_rect.translated(delta_pos))
        elif self.mouse_mode == MouseMode.SELECT:
            x = min(pos.x(), self.mouse_press_pos.x())
            y = min(pos.y(), self.mouse_press_pos.y())
            w = abs(delta_pos.x())
            h = abs(delta_pos.y())
            self.roi.setRect(QtCore.QRectF(x, y, w, h))
        self.logger.debug(f"Updating ROI to f{self.roi.rect()}")
        self.roi.show()

    def mouseReleaseEvent(self, ev: Any) -> None:
        if not self.capture_mouse_events:
            return super().mouseReleaseEvent(ev)
        if (
            self.mouse_press_pos is None
            or ev.button() != QtCore.Qt.MouseButton.LeftButton
        ):
            return
        pos = ev.position()
        if (
            self.mouse_mode == MouseMode.SELECT
            and self.mouse_press_pos == ev.position()
        ):
            self.unselect_neurons()
            self.timeline.setX(self.mapToView(pos).x())
            return
        if self.mouse_mode == MouseMode.SELECT:
            p0 = self.mapToView(self.roi.rect().topLeft())
            p1 = self.mapToView(self.roi.rect().bottomRight())
            x0 = max(0, math.ceil(p0.x() / self.dt))
            x1 = math.floor(p1.x() / self.dt)
            y0 = max(0, math.ceil(p1.y() + 0.5))
            y1 = math.floor(p0.y() + 0.5)
            selected_x_slice = slice(x0, x1)
            selected_y_slice = slice(y0, y1)
            self.logger.debug(
                f"Selecting neurons {selected_y_slice} and times {selected_x_slice}"
            )
            self.select_neurons(selected_y_slice, selected_x_slice)
        elif self.mouse_mode == MouseMode.MOVE:
            delta = self.mapToView(pos) - self.mapToView(self.mouse_press_pos)
            neuron_index_offset = round(delta.y())
            time_index_offset = round(delta.x() / self.dt)
            self.logger.debug(
                f"Selection moved by neuron index offset {neuron_index_offset}, time index offset {time_index_offset}"
            )
            self.selection_moved.emit(neuron_index_offset, time_index_offset)
        self.mouse_press_pos = None
        self.mouse_mode = MouseMode.NONE

    def unselect_neurons(self) -> None:
        self.mouse_press_pos = None
        self.roi.hide()
        self.selected_time_slice = None
        self.selected_neuron_slice = None
        for neuron_plot in self.neuron_plots:
            neuron_plot.set_selected_slice(slice(0, -1))

    def select_neurons(
        self,
        neuron_index_slice: slice | None = None,
        time_index_slice: slice | None = None,
    ) -> None:
        if neuron_index_slice is None:
            neuron_index_slice = slice(0, -1)
        if time_index_slice is None:
            time_index_slice = slice(0, -1)
        self.selected_time_slice = time_index_slice
        self.selected_neuron_slice = neuron_index_slice
        self.roi.show()
        self.update_roi()
        for neuron_index, neuron_plot in enumerate(self.neuron_plots):
            if neuron_index_slice.start <= neuron_index < neuron_index_slice.stop:
                neuron_plot.set_selected_slice(time_index_slice)
            else:
                neuron_plot.set_selected_slice(slice(0, 0))

    def update_roi(self) -> None:
        if (
            self.roi.isVisible()
            and self.selected_time_slice is not None
            and self.selected_neuron_slice is not None
        ):
            top_left_view = QtCore.QPointF(
                self.selected_time_slice.start * self.dt,
                self.selected_neuron_slice.stop - 0.5,
            )
            # in principle self.mapFromView should be enough here,
            # but the transform doesn't seem to get updated until after the corresponding signal is emitted,
            # so instead we do mapFromView for the underlying ViewBox and add its position inside the PlotItem
            self.mouse_selected_rect.setTopLeft(
                self.plotItem.vb.pos() + self.plotItem.vb.mapFromView(top_left_view)
            )
            bottom_right_view = QtCore.QPointF(
                self.selected_time_slice.stop * self.dt,
                self.selected_neuron_slice.start - 0.5,
            )
            self.mouse_selected_rect.setBottomRight(
                self.plotItem.vb.pos() + self.plotItem.vb.mapFromView(bottom_right_view)
            )
            self.roi.setRect(self.mouse_selected_rect)

    def connect_heatmap_widget(self, heatmap_widget: pg.ImageView) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.timeline.sigPositionChanged.connect(
            lambda: heatmap_widget.timeLine.setValue(self.timeline.getXPos())
        )
