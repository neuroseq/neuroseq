from __future__ import annotations

import abc
import logging
import math
from enum import Enum
from typing import Any
from typing import TypeAlias

import brian2.numpy_ as np
import pyqtgraph as pg
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.selectable_bar_line_plot_item import SelectableBarLinePlotItem
from neuroseq.selectable_bar_plot_item import SelectableBarPlotItem
from neuroseq.selectable_line_plot_item import SelectableLinePlotItem
from neuroseq.timeline import Timeline
from neuroseq.utils import QtABCMeta
from neuroseq.utils import add_action

SelectablePlotItem: TypeAlias = (
    SelectableLinePlotItem | SelectableBarPlotItem | SelectableBarLinePlotItem
)


class MouseMode(Enum):
    NONE = 1
    SELECT = 2
    MOVE = 3


class OffsetYAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_num_neurons(0)

    def set_num_neurons(self, n_neurons: int):
        major_ticks = []
        for neuron in range(n_neurons):
            major_ticks.append((float(neuron), str(neuron)))
            major_ticks.append((neuron + 0.5, ""))
        self.setTicks([major_ticks, []])


class MinDtXAxisItem(pg.AxisItem):
    def __init__(self, dt: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs, units="ms")
        self.dt = dt

    def set_dt(self, dt: float):
        self.dt = dt

    def tickSpacing(self, minVal, maxVal, size):
        levels = super().tickSpacing(minVal, maxVal, size)
        # filter out any levels that are less than the minimum spacing above dt
        min_spacing_above_dt = 7 * self.dt
        new_levels = [level for level in levels if level[0] > min_spacing_above_dt]
        # if there are less than 3 remaining levels, add a dt level
        if len(new_levels) < 3:
            new_levels.append((self.dt, 0))
        return new_levels


class TimeseriesWidget(pg.PlotWidget, metaclass=QtABCMeta):
    """A base class for widgets that display a set of timeseries data for a number of neurons, stacked vertically.

    data should be a 2d array or list of 2d arrays, where the first index of each
    array is the neuron index and the second index is the time index.

    Subclasses should call _plot_data() and implement _plot_neuron_data() to plot the data.
    """

    # these signals emit indices or slices of indices, neurons first and then times
    mouse_clicked = QtCore.Signal(int, int)
    selection_made = QtCore.Signal(slice, slice)
    selection_moved = QtCore.Signal(int, int)
    custom_analysis_requested = QtCore.Signal(
        CustomAnalysisDataType, CustomAnalysisType, slice, slice, dict
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            parent,
            axisItems={
                "bottom": MinDtXAxisItem(orientation="bottom"),
                "left": OffsetYAxisItem(orientation="left"),
            },
        )
        self.logger = logging.getLogger(__name__)
        self.data: dict[str, int | float | np.ndarray] = {}
        self.data_type: CustomAnalysisDataType | None = None
        self.timeline = Timeline()
        self.timeline.setZValue(1e8)
        self.dt = 1.0
        self.mouse_mode = MouseMode.NONE
        self.mouse_press_pos: QtCore.QPointF | None = None
        self.mouse_selected_rect: QtCore.QRectF = QtCore.QRectF()
        self.enable_mouse_move_roi = False
        self.selected_time_slice: slice | None = None
        self.selected_neuron_slice: slice | None = None
        self.neuron_plots: list[SelectablePlotItem] = []
        self.n_neurons = 0
        self.n_timestamps = 0
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

    # workaround for AttributeError: autoRangeEnabled when setClipToView is enabled,
    # see https://github.com/pyqtgraph/pyqtgraph/issues/1854 for more details
    def autoRangeEnabled(self):
        return self.plotItem.getViewBox().autoRangeEnabled()

    def _create_custom_analysis_menu(self) -> None:
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        self.menu_custom_analysis = QtWidgets.QMenu("Custom analysis", self)
        for custom_analysis_type in CustomAnalysisType:
            add_action(
                str(custom_analysis_type),
                lambda *args,
                _custom_analysis_type=custom_analysis_type: self.custom_analysis_requested.emit(
                    self.data_type,
                    _custom_analysis_type,
                    self.selected_neuron_slice,
                    self.selected_time_slice,
                    self.data,
                ),
                self.menu_custom_analysis,
                None,
                self,
                icon=qta.icon(custom_analysis_type.value),
            )

    def clear_data(self) -> None:
        self.neuron_plots.clear()
        self.data = {}
        self.n_neurons = 0
        self.n_timestamps = 0
        self.unselect_neurons()
        self.clear()
        # set the y-axis to display the trace index
        self.plotItem.getAxis("left").setLabel("Neuron")
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
        self.n_neurons, self.n_timestamps = shape
        if self.n_neurons == 0 or self.n_timestamps == 0:
            return
        if isinstance(data, list) and not all(
            d.shape[0] == self.n_neurons and d.shape[1] == self.n_timestamps
            for d in data
        ):
            raise ValueError("All data arrays must have the same shape")
        if timestamps is not None:
            # timestamps, if provided, take precedence over dt
            self.dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
        else:
            self.dt = dt if dt is not None else 1.0
            timestamps = self.dt * np.arange(self.n_timestamps)
        # don't let user zoom/pan outside of the data
        self.setLimits(
            xMin=timestamps[0],
            xMax=timestamps[-1] + self.dt,
            yMin=-0.5,
            yMax=self.n_neurons - 0.5,
        )
        self.plotItem.getAxis("bottom").set_dt(self.dt)
        self.plotItem.getAxis("left").set_num_neurons(self.n_neurons)
        self.data = {"data": data, "timestamps": timestamps, "dt": self.dt}
        self._plot_neuron_data(data, timestamps, self.dt)

    def mousePressEvent(self, ev: Any) -> None:
        if ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
            return super().mousePressEvent(ev)
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self.mouse_press_pos = ev.position()
        if (
            self.enable_mouse_move_roi
            and self.roi.isVisible()
            and self.roi.rect().contains(ev.position())
        ):
            self.mouse_mode = MouseMode.MOVE
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.mouse_mode = MouseMode.SELECT
        self.logger.debug(f"Starting f{self.mouse_mode} from {self.mouse_press_pos}")

    def mouseMoveEvent(self, ev: Any) -> None:
        if ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
            return super().mouseMoveEvent(ev)
        if self.mouse_press_pos is None:
            return
        pos = ev.position()
        delta_pos = pos - self.mouse_press_pos
        if self.mouse_mode == MouseMode.MOVE:
            self.roi.setRect(self.mouse_selected_rect.translated(delta_pos))
        elif self.mouse_mode == MouseMode.SELECT:
            x = min(pos.x(), self.mouse_press_pos.x())
            y = min(pos.y(), self.mouse_press_pos.y())
            w = abs(delta_pos.x())
            h = abs(delta_pos.y())
            self.roi.setRect(QtCore.QRectF(x, y, w, h))
        self.roi.show()

    def mouseReleaseEvent(self, ev: Any) -> None:
        if self.cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        if ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
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
            p = self.mapToView(pos)
            self.mouse_clicked.emit(
                math.floor(p.y() + 0.5), math.floor(p.x() / self.dt)
            )
            return
        if self.mouse_mode == MouseMode.SELECT:
            p0 = self.mapToView(self.roi.rect().topLeft())
            p1 = self.mapToView(self.roi.rect().bottomRight())
            x0 = max(0, math.ceil(p0.x() / self.dt))
            x1 = math.floor(p1.x() / self.dt)
            y0 = max(0, math.ceil(p1.y() + 0.5))
            y1 = math.floor(p0.y() + 0.5)
            selected_x_slice = slice(min(x0, x1), max(x0, x1))
            selected_y_slice = slice(min(y0, y1), max(y0, y1))
            self.logger.debug(
                f"Selecting neurons {selected_y_slice} and times {selected_x_slice}"
            )
            self.select_neurons(selected_y_slice, selected_x_slice)
            self.selection_made.emit(selected_y_slice, selected_x_slice)
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
            neuron_index_slice = slice(0, self.n_neurons)
        if time_index_slice is None:
            time_index_slice = slice(0, self.n_timestamps)
        if neuron_index_slice.stop is None:
            neuron_index_slice = slice(neuron_index_slice.start, self.n_neurons)
        if time_index_slice.stop is None:
            time_index_slice = slice(time_index_slice.start, self.n_timestamps)
        self.selected_time_slice = time_index_slice
        self.selected_neuron_slice = neuron_index_slice
        self.logger.debug(
            f"Selecting neurons {neuron_index_slice} and times {time_index_slice}"
        )
        self.roi.show()
        self.update_roi()
        for neuron_index, neuron_plot in enumerate(self.neuron_plots):
            if neuron_index >= neuron_index_slice.start and (
                neuron_index_slice.stop is None
                or neuron_index < neuron_index_slice.stop
            ):
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
