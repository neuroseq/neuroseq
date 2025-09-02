from __future__ import annotations

import logging
import math

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
import pyqtgraph as pg
import qtawesome as qta
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.gridlines import GridLines
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.region_of_interest import RegionOfInterest
from neuroseq.text import Text
from neuroseq.utils import InteractMode
from neuroseq.utils import QtABCMeta
from neuroseq.utils import add_action


class TimeseriesWidget(QtWidgets.QWidget, metaclass=QtABCMeta):
    """A base class for widgets that display a set of timeseries data for a number of neurons, stacked vertically.

    data should be a 2d array or list of 2d arrays, where the first index of each
    array is the neuron index and the second index is the time index.

    Subclasses should call post_set_data() after updating the data.
    """

    # these signals emit indices or slices of indices, neurons first and then times
    mouse_clicked = QtCore.Signal(
        int, int, int
    )  # neuron slice, time slice, mouse button (0 for no button, 1 for left, 2 for right, 3 for middle)
    mouse_dragged = QtCore.Signal(
        int, int, int, int, int
    )  # initial neuron index, current neuron index, initial time index, current time index, mouse button (0 for no button, 1 for left, 2 for right, 3 for middle)
    custom_analysis_requested = QtCore.Signal(
        CustomAnalysisDataType, CustomAnalysisType, slice, slice, dict
    )

    def __init__(
        self, controller: pygfx.Controller, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.heatmap_widget: HeatmapWidget | None = None
        self.controller = controller
        self.figure: fpl.Figure = fpl.Figure(controllers=controller)
        layout = QtWidgets.QVBoxLayout(self)
        self.widget = self.figure.show()
        layout.addWidget(self.widget)
        self.graphic: fpl.ImageGraphic | fpl.LineStack | fpl.ScatterGraphic | None = (
            None
        )
        self.mode = InteractMode.SELECT
        self.roi = RegionOfInterest(fill_color=(0.5, 0.5, 0.5, 0.2))
        self.hover_slice_graphic: fpl.Graphic | None = None
        self.text = Text()
        self.show_text = True
        self.timeline: fpl.LinearSelector | None = None
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        self.background_color = "k"
        self.foreground_color = "w"
        self.toggle_dark_mode()
        self.resize(800, 600)

        self.context_menu = QtWidgets.QMenu("Custom analysis", self)
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
                self.context_menu,
                None,
                self,
                icon=qta.icon(custom_analysis_type.value),
            )

        self.logger = logging.getLogger(__name__)
        self.data: dict[str, int | float | np.ndarray] = {}
        self.data_type: CustomAnalysisDataType | None = None
        self.dt = 1.0
        self.selection_in_progress = False
        self.mouse_pos: np.ndarray | None = None
        self.mouse_drag_time_index: int | None = None
        self.mouse_drag_neuron_index: int | None = None
        self.mouse_press_pos: np.ndarray | None = None
        self.mouse_press_button: int = (
            0  # 0 for no button pressed, 1 for left button, 2 for right button
        )
        self.selected_time_slice: slice | None = None
        self.selected_neuron_slice: slice | None = None
        self.n_neurons = 0
        self.n_timestamps = 0
        self.figure.renderer.add_event_handler(self.pointer_down, "pointer_down")
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.figure.renderer.add_event_handler(self.pointer_up, "pointer_up")

    def set_mode(self, mode: InteractMode) -> None:
        self.mode = mode
        self.mouse_press_pos = None
        self.mouse_press_button = 0
        self.selection_in_progress = False
        self.roi.visible = False
        self.unselect_neurons()

    def toggle_selection_text(self):
        self.show_text = not self.show_text
        self.roi.text.visible = self.show_text
        self.text.visible = self.show_text

    def toggle_dark_mode(self):
        """Toggle the colors of the plot between light and dark."""
        if self.background_color == "k":
            self.background_color = "w"
            self.foreground_color = "k"
        else:
            self.background_color = "k"
            self.foreground_color = "w"
        # some of these currently require accessing private attributes: should refactor when fastplotlib allows setting these colors directly
        self.subplot.background_color = self.background_color
        self.subplot.axes.colors = (
            self.foreground_color,
            self.foreground_color,
            self.foreground_color,
        )
        self.subplot.axes.grids.xy.axis_color = self.foreground_color
        self.subplot.axes.grids.xy.major_color = self.foreground_color
        self.subplot.axes.grids.xy.minor_color = self.foreground_color
        self.subplot.axes._x._text.material.color = self.foreground_color
        self.subplot.axes._y._text.material.color = self.foreground_color
        self.subplot.axes._x._line.material.color = self.foreground_color
        self.subplot.axes._y._line.material.color = self.foreground_color
        self.subplot.axes._x._points.material.color = self.foreground_color
        self.subplot.axes._y._points.material.color = self.foreground_color
        if self.graphic is not None:
            if self.graphic.cmap == "matlab:hot":
                self.graphic.cmap = "matlab:hot_r"
            elif self.graphic.cmap == "matlab:hot_r":
                self.graphic.cmap = "matlab:hot"
        self.roi.toggle_dark_mode()
        self.text.toggle_dark_mode()

    def contextMenuEvent(self, event):
        if (
            self.mode == InteractMode.SELECT
            and event.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.context_menu.exec(event.globalPos())
        else:
            super().contextMenuEvent(event)

    def post_set_data(self):
        self.subplot.auto_scale(zoom=1.0, maintain_aspect=False)
        self.subplot.grids = False
        self.subplot.axes.y.ticks = np.arange(0, self.n_neurons, 1)
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.axes.x.min_tick_distance = 100
        self.subplot.axes.x.tick_format = lambda x, x_min, x_max: f"{x:.0f}ms"
        self.subplot.toolbar = False
        self.subplot.add_graphic(self.roi, center=False)
        self.subplot.add_graphic(
            GridLines(self.n_neurons, self.n_timestamps * self.dt), center=False
        )
        self.subplot.add_graphic(self.text, center=False)
        self.roi.visible = False
        if hasattr(self.graphic, "add_linear_selector"):
            self.timeline = self.graphic.add_linear_selector(thickness=2)
            self.timeline.add_event_handler(
                lambda ev: self.heatmap_widget.timeLine.setValue(ev.info["value"]),
                "selection",
            )

    def clear_data(self) -> None:
        self.subplot.clear()
        self.hover_slice_graphic = None
        self.data = {}
        self.n_neurons = 0
        self.n_timestamps = 0
        self.unselect_neurons()

    def pointer_down(self, ev: pygfx.PointerEvent) -> None:
        self.mouse_press_pos = self.subplot.map_screen_to_world(ev)
        self.mouse_pos = self.mouse_press_pos
        self.mouse_press_button = ev.button
        self.logger.debug(
            f"pointer_down button {self.mouse_press_button} at {self.mouse_press_pos} in mode {self.mode}"
        )
        if self.mode == InteractMode.SELECT and self.mouse_press_button == 1:
            self.logger.debug("starting ROI selection")
            self.roi.visible = True
            self.selection_in_progress = True
        elif self.mode == InteractMode.DRAW:
            self.mouse_drag_time_index = math.floor(self.mouse_pos[0])
            self.mouse_drag_neuron_index = math.floor(self.mouse_pos[1] + 0.5)
            self.mouse_dragged.emit(
                self.mouse_drag_neuron_index,
                self.mouse_drag_neuron_index,
                self.mouse_drag_time_index,
                self.mouse_drag_time_index,
                self.mouse_press_button,
            )
        else:
            return

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        self.mouse_pos = self.subplot.map_screen_to_world(ev)
        if self.mouse_pos is None:
            self.text.visible = False
            if self.hover_slice_graphic is not None:
                self.hover_slice_graphic.visible = False
            return
        self.logger.debug(f"pointer_move at {self.mouse_pos} in mode {self.mode}")
        self.text.visible = self.show_text
        self.text.set_pos(self.mouse_pos)
        if self.hover_slice_graphic is not None:
            self.hover_slice_graphic.visible = True
            self.hover_slice_graphic.offset = (
                self.mouse_pos[0],
                self.mouse_pos[1],
                100,
            )
        if self.mouse_press_pos is None:
            return
        if self.mode == InteractMode.SELECT and self.selection_in_progress:
            x0 = min(self.mouse_pos[0], self.mouse_press_pos[0])
            x1 = max(self.mouse_pos[0], self.mouse_press_pos[0])
            y0 = min(self.mouse_pos[1] + 0.5, self.mouse_press_pos[1] + 0.5)
            y1 = max(self.mouse_pos[1] + 0.5, self.mouse_press_pos[1] + 0.5)
            time_slice = slice(max(0, math.floor(x0)), math.ceil(x1))
            neuron_slice = slice(math.floor(y0), math.ceil(y1))
            self.select_neurons(neuron_slice, time_slice)
            return
        elif self.mode == InteractMode.DRAW:
            drag_time_index = math.floor(self.mouse_pos[0])
            drag_neuron_index = math.floor(self.mouse_pos[1] + 0.5)
            if (
                drag_time_index != self.mouse_drag_time_index
                or drag_neuron_index != self.mouse_drag_neuron_index
            ):
                self.logger.debug(
                    f"Emitting mouse_dragged signal: neuron {self.mouse_drag_neuron_index} time {self.mouse_drag_time_index} -> neuron {drag_neuron_index} time {drag_time_index}"
                )
                self.mouse_dragged.emit(
                    self.mouse_drag_neuron_index,
                    drag_neuron_index,
                    self.mouse_drag_time_index,
                    drag_time_index,
                    self.mouse_press_button,
                )
                self.mouse_drag_time_index = drag_time_index

    def pointer_up(self, ev: pygfx.PointerEvent) -> None:
        if self.mouse_press_pos is None:
            self.mouse_press_button = 0
            return
        self.logger.debug(
            f"pointer_up at {self.mouse_pos} with button {self.mouse_press_button} in mode {self.mode}"
        )
        if (
            self.mode == InteractMode.SELECT
            and self.selection_in_progress
            and self.mouse_pos is not None
        ):
            if (
                self.mouse_press_pos[0] == self.mouse_pos[0]
                and self.mouse_press_pos[1] == self.mouse_pos[1]
            ):
                self.unselect_neurons()
                self.roi.visible = False
        elif self.mode == InteractMode.DRAW:
            self.mouse_clicked.emit(
                self.mouse_drag_neuron_index,
                self.mouse_drag_time_index,
                self.mouse_press_button,
            )
        self.mouse_press_pos = None
        self.mouse_press_button = 0
        self.selection_in_progress = False

    def unselect_neurons(self) -> None:
        self.mouse_press_pos = None
        self.selected_time_slice = None
        self.selected_neuron_slice = None
        self.roi.visible = False
        self.selection_in_progress = False

    def select_neurons(
        self,
        neuron_index_slice: slice | None = None,
        time_index_slice: slice | None = None,
    ) -> None:
        pass
        if neuron_index_slice is None:
            neuron_index_slice = slice(0, self.n_neurons)
        if time_index_slice is None:
            time_index_slice = slice(0, self.n_timestamps)
        if neuron_index_slice.stop is None:
            neuron_index_slice = slice(neuron_index_slice.start, self.n_neurons)
        if time_index_slice.stop is None:
            time_index_slice = slice(time_index_slice.start, self.n_timestamps)
        self.roi.visible = True
        self.roi.select_slice(neuron_index_slice, time_index_slice)
        self.selected_time_slice = time_index_slice
        self.selected_neuron_slice = neuron_index_slice
        self.logger.debug(
            f"Selecting neurons {neuron_index_slice} and times {time_index_slice}"
        )

    def connect_heatmap_widget(self, heatmap_widget: pg.ImageView) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.heatmap_widget = heatmap_widget
