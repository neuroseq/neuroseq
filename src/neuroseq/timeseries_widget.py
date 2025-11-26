from __future__ import annotations

import logging
import math

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
import qtawesome as qta
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData
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
        object, CustomAnalysisType, bool, slice, slice
    )  # data, analysis type, new window, neuron slice, time slice
    escape_pressed = QtCore.Signal()

    def __init__(
        self, controller: pygfx.Controller, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.heatmap_widget: HeatmapWidget | None = None
        self.controller = controller
        self.figure: fpl.Figure = fpl.Figure(
            controllers=controller, canvas_kwargs={"parent": self}
        )
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
        self.grid_lines: GridLines | None = None
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        self._set_light_mode()
        self.resize(800, 600)

        self.context_menu = QtWidgets.QMenu("Custom analysis", self)
        for custom_analysis_type in CustomAnalysisType:
            for new_window in [False, True]:
                add_action(
                    f"{custom_analysis_type} {'(new window)' if new_window else ''}",
                    lambda *args,
                    _custom_analysis_type=custom_analysis_type,
                    _new_window=new_window: self.custom_analysis_requested.emit(
                        self.data,
                        _custom_analysis_type,
                        _new_window,
                        self.selected_neuron_slice,
                        self.selected_time_slice,
                    ),
                    self.context_menu,
                    None,
                    self,
                    icon=qta.icon(custom_analysis_type.value),
                )

        self.logger = logging.getLogger(__name__)
        self.data: TraceData | SpikeData | SynapseData | None = None
        self.data_values: np.ndarray | None = None
        self.units = ""
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
        self.dt_ms = 1.0
        self.n_timestamps = 0
        self.figure.renderer.add_event_handler(self.pointer_down, "pointer_down")
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.figure.renderer.add_event_handler(self.pointer_up, "pointer_up")
        self.figure.renderer.add_event_handler(self.key_down, "key_down")

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

    def contextMenuEvent(self, event):
        if (
            self.mode == InteractMode.SELECT
            and event.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.context_menu.exec(event.globalPos())
        else:
            super().contextMenuEvent(event)

    def _set_light_mode(self) -> None:
        # some of these currently require accessing private attributes: should refactor when fastplotlib allows setting these colors directly
        self.subplot.background_color = "w"
        self.subplot.axes.colors = (
            "k",
            "k",
            "k",
        )
        self.subplot.axes.grids.xy.axis_color = "k"
        self.subplot.axes.grids.xy.major_color = "k"
        self.subplot.axes.grids.xy.minor_color = "k"
        self.subplot.axes._x._text.material.color = "k"
        self.subplot.axes._y._text.material.color = "k"
        self.subplot.axes._x._line.material.color = "k"
        self.subplot.axes._y._line.material.color = "k"
        self.subplot.axes._x._points.material.color = "k"
        self.subplot.axes._y._points.material.color = "k"

    def post_set_data(self):
        self.subplot.auto_scale(zoom=1.0, maintain_aspect=False)
        self.subplot.grids = False
        self.subplot.axes.y.ticks = np.arange(0, self.n_neurons, 1)
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.x.ticks = None
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.axes.x.min_tick_distance = 100
        self.subplot.axes.x.tick_format = lambda x, x_min, x_max: f"{x:.0f}ms"
        self.subplot.toolbar = False
        self.subplot.add_graphic(self.roi, center=False)
        self.grid_lines = GridLines(self.n_neurons, self.n_timestamps * self.dt_ms)
        self.subplot.add_graphic(self.grid_lines, center=False)
        self.controller.add_grid_lines(self.grid_lines)
        self.subplot.add_graphic(self.text, center=False)
        self._set_light_mode()
        self.roi.visible = False
        if hasattr(self.graphic, "add_linear_selector"):
            self.timeline = self.graphic.add_linear_selector(thickness=2)
            self.timeline.add_event_handler(
                self.on_timeline_change,
                "selection",
            )

    def clear_data(self) -> None:
        self.subplot.clear()
        self.subplot.grids = False
        self.subplot.axes.y.ticks = {}
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.axes.x.ticks = {}
        self.subplot.toolbar = False
        self.hover_slice_graphic = None
        self.data = None
        self.n_neurons = 0
        self.n_timestamps = 0
        self.dt_ms = 0.0
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
            self.mouse_drag_time_index = math.floor(self.mouse_pos[0] / self.dt_ms)
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
        hover_value_text = ""
        if self.data_values is not None:
            time_index = np.clip(
                math.floor(self.mouse_pos[0] / self.dt_ms),
                0,
                self.data_values.shape[0] - 1,
            )
            neuron_index = np.clip(
                math.floor(self.mouse_pos[1] + 0.5), 0, self.data_values.shape[1] - 1
            )
            hover_value = self.data_values[time_index, neuron_index]
            hover_value_text = f"{hover_value:.3g} {self.units}"
        self.text.set_pos(self.mouse_pos, hover_value_text)
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
            time_slice = slice(
                max(0, math.floor(x0 / self.dt_ms)), math.ceil(x1 / self.dt_ms)
            )
            neuron_slice = slice(math.floor(y0), math.ceil(y1))
            self.select_neurons(neuron_slice, time_slice)
            return
        elif self.mode == InteractMode.DRAW:
            drag_time_index = math.floor(self.mouse_pos[0] / self.dt_ms)
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

    def key_down(self, ev: pygfx.KeyboardEvent) -> None:
        if ev.key == "Escape":
            self.unselect_neurons()
            self.escape_pressed.emit()

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
        self.roi.select_slice(neuron_index_slice, time_index_slice, self.dt_ms)
        self.selected_time_slice = time_index_slice
        self.selected_neuron_slice = neuron_index_slice
        self.logger.debug(
            f"Selecting neurons {neuron_index_slice} and times {time_index_slice}"
        )

    def connect_heatmap_widget(self, heatmap_widget: HeatmapWidget) -> None:
        """Connect the timeline in this widget to the timeline in the HeatmapWidget."""
        self.heatmap_widget = heatmap_widget

    def on_slider_value_changed(self, value: int) -> None:
        """Slot for the heatmap widget slider.valueChanged signal.
        Updates the timeline in this widget.
        """
        if self.heatmap_widget is None:
            return
        if self.timeline is None:
            return
        t = self.heatmap_widget.timestamps[value] * 1000.0
        self.timeline.selection = t

    def on_timeline_change(self, ev):
        """Outgoing event handler for timeline selection changes."""
        # timestamp value to index
        idx = (
            np.abs(self.heatmap_widget.timestamps * 1000.0 - ev.info["value"])
        ).argmin()
        # parse to int
        try:
            idx = int(idx)
        except Exception as e:
            self.logger.error(f"Error converting index to int: {e}")
        try:
            self.heatmap_widget.slider.setValue(idx)
        except Exception as e:
            self.logger.error(f"Error setting current index: {e}")
