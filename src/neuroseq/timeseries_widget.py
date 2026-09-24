from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.analysis_plugins import AnalysisPlugin
from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.fpl_widget import FPLWidget
from neuroseq.gridlines import GridLines
from neuroseq.region_of_interest import RegionOfInterest
from neuroseq.stim_timing import format_milliseconds
from neuroseq.text import Text
from neuroseq.timeline import Timeline
from neuroseq.utils import InteractMode
from neuroseq.utils import add_action


class TimeseriesWidget(FPLWidget):
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
        object, object, bool, object, object
    )  # data, analysis type, new window, neuron slice, time slice
    escape_pressed = QtCore.Signal()
    neuron_hovered = QtCore.Signal(object)  # int | None - the real neuron/stim id

    SUPPORTS_DETACH = True

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
        default_controller: ConstrainedPanZoomController | None = None,
        show_axis_link_controls: bool = False,
    ) -> None:
        super().__init__(controller=controller, parent=parent)
        self.default_controller = default_controller
        self.default_x_scroll_bar: QtWidgets.QScrollBar | None = None
        self.default_y_scroll_bar: QtWidgets.QScrollBar | None = None
        self.x_scrollbar: QtWidgets.QScrollBar | None = None
        self.y_scrollbar: QtWidgets.QScrollBar | None = None
        self.x_btn_zoom_in: QtWidgets.QPushButton | None = None
        self.x_btn_zoom_out: QtWidgets.QPushButton | None = None
        self.y_btn_zoom_in: QtWidgets.QPushButton | None = None
        self.y_btn_zoom_out: QtWidgets.QPushButton | None = None
        self.x_controls_widget: QtWidgets.QWidget | None = None
        self.x_controls_layout: QtWidgets.QHBoxLayout | None = None
        self.plot_container: QtWidgets.QWidget | None = None
        self._axis_sync_in_progress = False
        self._has_axis_link_controls = show_axis_link_controls
        self.lock_x_checkbox: QtWidgets.QToolButton | None = None
        self.lock_y_checkbox: QtWidgets.QToolButton | None = None
        self._setup_scroll_bars()
        self._setup_axis_link_controls()
        self.graphic: fpl.ImageGraphic | fpl.LineStack | fpl.ScatterGraphic | None = (
            None
        )
        self.background_graphics: list[fpl.Graphic] = []
        self.show_background_graphics = True
        self.mode = InteractMode.SELECT
        overlay_colors = theme_module.plot_overlay_colors(self._theme)
        self.roi = RegionOfInterest(
            fill_color=overlay_colors["roi_fill"],
            text_color=overlay_colors["roi_text"],
        )
        self._greyed_neuron_ids: set[int] = set()
        self._all_neuron_ids: list[int] = []
        self._visible_neuron_ids: list[int] = []
        self._neuron_id_to_display_row: dict[int, int] = {}
        self.hover_slice_graphic: fpl.Graphic | None = None
        self._last_hovered_neuron_id: int | None = None
        self.text = Text()
        self.show_text = True
        self.timeline = Timeline()
        self.grid_lines: GridLines | None = None
        self.show_grid = True
        self._open_analysis_in_new_window = True

        self.context_menu = QtWidgets.QMenu("Custom analysis", self)

        self.logger = logging.getLogger(__name__)
        self.data: TraceData | SpikeData | SynapseData | None = None
        self.data_values: np.ndarray | None = None
        # multiplier applied to data_values when shown as a hover value
        self.data_values_scale: float = 1.0
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
        self.n_neurons_total = 0
        self.dt_ms = 1.0
        self.n_timestamps = 0
        self.figure.renderer.add_event_handler(self.pointer_down, "pointer_down")
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.figure.renderer.add_event_handler(self.pointer_up, "pointer_up")
        self.figure.renderer.add_event_handler(self.pointer_leave, "pointer_leave")
        self.figure.renderer.add_event_handler(self.key_down, "key_down")

    def apply_theme(self, theme: str) -> None:
        """Recolor the canvas and independent plot overlays."""
        super().apply_theme(theme)
        colors = theme_module.plot_overlay_colors(self._theme)
        self.roi.set_colors(colors["roi_fill"], colors["roi_text"])
        if self.grid_lines is not None:
            self.grid_lines.set_colors(colors["grid"], colors["shading"])

    def _setup_axis_link_controls(self) -> None:
        if not self._has_axis_link_controls:
            return

    def _setup_scroll_bars(self) -> None:
        if not self._has_axis_link_controls:
            return
        controller = self.controller
        if not isinstance(controller, ConstrainedPanZoomController):
            return
        root_layout = self.layout()
        if not isinstance(root_layout, QtWidgets.QBoxLayout):
            return

        root_layout.removeWidget(self.widget)

        self.plot_container = QtWidgets.QWidget(self)
        container_layout = QtWidgets.QGridLayout(self.plot_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.widget, 0, 0, 4, 1)

        self.y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.y_scrollbar.setFixedWidth(12)
        controller.connect_y_scroll_bar(self.y_scrollbar)
        container_layout.addWidget(self.y_scrollbar, 0, 1)

        self.y_btn_zoom_in = QtWidgets.QPushButton("+", self.plot_container)
        self.y_btn_zoom_in.setFixedSize(14, 24)
        self.y_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_in.setToolTip("Zoom in (Shift + Mouse Wheel Up)")
        self.y_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.0, 0.5), ()))
        container_layout.addWidget(self.y_btn_zoom_in, 1, 1)

        self.y_btn_zoom_out = QtWidgets.QPushButton("-", self.plot_container)
        self.y_btn_zoom_out.setFixedSize(14, 24)
        self.y_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_out.setToolTip("Zoom out (Shift + Mouse Wheel Down)")
        self.y_btn_zoom_out.clicked.connect(lambda: controller.zoom((0.0, -0.5), ()))
        container_layout.addWidget(self.y_btn_zoom_out, 2, 1)

        self.lock_y_checkbox = self._create_lock_button("Y")
        self.lock_y_checkbox.setToolTip("Lock Y pan/zoom to the default view")
        container_layout.addWidget(self.lock_y_checkbox, 3, 1)

        self.x_controls_widget = QtWidgets.QWidget(self.plot_container)
        self.x_controls_widget.setFixedHeight(14)
        self.x_controls_layout = QtWidgets.QHBoxLayout(self.x_controls_widget)
        self.x_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.x_controls_layout.setSpacing(0)

        self.x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.x_scrollbar.setFixedHeight(12)
        controller.connect_x_scroll_bar(self.x_scrollbar)
        self.x_controls_layout.addWidget(self.x_scrollbar, 1)

        self.x_btn_zoom_out = QtWidgets.QPushButton("-", self.plot_container)
        self.x_btn_zoom_out.setFixedSize(24, 14)
        self.x_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_out.setToolTip("Zoom out (Ctrl + Mouse Wheel Down)")
        self.x_btn_zoom_out.clicked.connect(lambda: controller.zoom((-0.5, 0.0), ()))
        self.x_controls_layout.addWidget(self.x_btn_zoom_out)

        self.x_btn_zoom_in = QtWidgets.QPushButton("+", self.plot_container)
        self.x_btn_zoom_in.setFixedSize(24, 14)
        self.x_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_in.setToolTip("Zoom in (Ctrl + Mouse Wheel Up)")
        self.x_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.5, 0.0), ()))
        self.x_controls_layout.addWidget(self.x_btn_zoom_in)

        self.lock_x_checkbox = self._create_lock_button("X")
        self.lock_x_checkbox.setToolTip("Lock X pan/zoom to the default view")
        self.x_controls_layout.addWidget(self.lock_x_checkbox)

        container_layout.addWidget(self.x_controls_widget, 4, 0, 1, 2)

        self.x_scrollbar.valueChanged.connect(self._on_local_x_scroll_changed)
        self.x_scrollbar.rangeChanged.connect(self._on_local_x_scroll_changed)
        self.y_scrollbar.valueChanged.connect(self._on_local_y_scroll_changed)
        self.y_scrollbar.rangeChanged.connect(self._on_local_y_scroll_changed)

        root_layout.insertWidget(0, self.plot_container, 1)

    def _create_lock_button(self, axis: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setCheckable(True)
        button.setChecked(True)
        button.setFixedSize(14, 14)
        button.setAutoRaise(True)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        button.setStyleSheet(
            "QToolButton { margin: 0px; padding: 0px; font-size: 12px; }"
        )
        button.toggled.connect(
            lambda checked, btn=button, axis_name=axis: self._update_lock_button(
                btn, axis_name, checked
            )
        )
        if axis == "X":
            button.toggled.connect(self._on_lock_x_toggled)
        else:
            button.toggled.connect(self._on_lock_y_toggled)
        self._update_lock_button(button, axis, True)
        return button

    def _make_lock_icon(self, locked: bool) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pen = QtGui.QPen(self.palette().buttonText().color())
        pen.setWidthF(1.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        shackle = QtGui.QPainterPath()
        if locked:
            shackle.moveTo(3.2, 5.8)
            shackle.arcTo(QtCore.QRectF(3.2, 1.0, 5.6, 6.4), 180, 180)
        else:
            shackle.moveTo(8.8, 5.8)
            shackle.arcTo(QtCore.QRectF(3.2, -1.4, 5.6, 6.4), 0, 180)
        painter.drawPath(shackle)
        painter.drawRoundedRect(QtCore.QRectF(2.2, 5.8, 7.6, 5.2), 1.2, 1.2)

        painter.end()
        return QtGui.QIcon(pixmap)

    def _update_lock_button(
        self,
        button: QtWidgets.QToolButton,
        axis: str,
        checked: bool,
    ) -> None:
        button.setText("")
        button.setIcon(self._make_lock_icon(checked))
        button.setIconSize(QtCore.QSize(12, 12))
        state = "linked" if checked else "independent"
        button.setToolTip(f"{axis} axis {state}")
        button.setStatusTip(f"{axis} axis {state}")

    @property
    def x_locked(self) -> bool:
        return self.lock_x_checkbox is None or self.lock_x_checkbox.isChecked()

    @property
    def y_locked(self) -> bool:
        return self.lock_y_checkbox is None or self.lock_y_checkbox.isChecked()

    def set_default_scroll_bars(
        self,
        x_scroll_bar: QtWidgets.QScrollBar,
        y_scroll_bar: QtWidgets.QScrollBar,
    ) -> None:
        self.default_x_scroll_bar = x_scroll_bar
        self.default_y_scroll_bar = y_scroll_bar
        self.default_x_scroll_bar.valueChanged.connect(
            self._on_default_x_scroll_changed
        )
        self.default_x_scroll_bar.rangeChanged.connect(
            self._on_default_x_scroll_changed
        )
        self.default_y_scroll_bar.valueChanged.connect(
            self._on_default_y_scroll_changed
        )
        self.default_y_scroll_bar.rangeChanged.connect(
            self._on_default_y_scroll_changed
        )
        self._sync_from_default_axes()

    def _on_lock_x_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_x_from_default()

    def _on_lock_y_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_y_from_default()

    def _on_default_x_scroll_changed(self, *args) -> None:
        if self.x_locked:
            self._sync_x_from_default()

    def _on_default_y_scroll_changed(self, *args) -> None:
        if self.y_locked:
            self._sync_y_from_default()

    def _on_local_x_scroll_changed(self, *args) -> None:
        if self.x_locked:
            self._sync_x_to_default()

    def _on_local_y_scroll_changed(self, *args) -> None:
        if self.y_locked:
            self._sync_y_to_default()

    def _sync_from_default_axes(self) -> None:
        if self.x_locked:
            self._sync_x_from_default()
        if self.y_locked:
            self._sync_y_from_default()

    def _sync_x_from_default(self) -> None:
        if self.default_controller is None:
            return
        self._sync_axis(
            source_controller=self.default_controller,
            target_controller=self.controller,
            axis="x",
        )

    def _sync_y_from_default(self) -> None:
        if self.default_controller is None:
            return
        self._sync_axis(
            source_controller=self.default_controller,
            target_controller=self.controller,
            axis="y",
        )

    def _sync_x_to_default(self) -> None:
        if self.default_controller is None:
            return
        self._sync_axis(
            source_controller=self.controller,
            target_controller=self.default_controller,
            axis="x",
        )

    def _sync_y_to_default(self) -> None:
        if self.default_controller is None:
            return
        self._sync_axis(
            source_controller=self.controller,
            target_controller=self.default_controller,
            axis="y",
        )

    def _sync_axis(
        self,
        source_controller: pygfx.Controller,
        target_controller: pygfx.Controller,
        axis: str,
    ) -> None:
        if self._axis_sync_in_progress:
            return
        self._axis_sync_in_progress = True
        try:
            try:
                source_state = source_controller._get_camera_state()
                target_state = target_controller._get_camera_state()
            except ValueError:
                return
            if axis == "x":
                new_state = {
                    "width": float(source_state["width"]),
                    "position": (
                        float(source_state["position"][0]),
                        float(target_state["position"][1]),
                        float(target_state["position"][2]),
                    ),
                }
            else:
                new_state = {
                    "height": float(source_state["height"]),
                    "position": (
                        float(target_state["position"][0]),
                        float(source_state["position"][1]),
                        float(target_state["position"][2]),
                    ),
                }
            # _set_camera_state() already pushes the new state to the
            # camera itself - an extra _update_cameras() here would just be
            # a redundant, wasted re-render on every sync.
            target_controller._set_camera_state(new_state)
        finally:
            self._axis_sync_in_progress = False

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

    def set_grid_visible(self, is_visible: bool) -> None:
        self.show_grid = is_visible
        if self.grid_lines is not None:
            self.grid_lines.visible = is_visible

    def _display_row_for_tick_y(self, y: float) -> int:
        """Map a y-axis tick candidate position to a display row.

        The base implementation assumes the default one-unit-per-row
        spacing; subclasses with non-uniform row spacing (e.g. TraceWidget's
        auto-extend-vertical-space mode) override this to look up the row
        that actually occupies that position instead.

        When there isn't enough vertical room to fit a tick per row, pygfx's
        ruler halves its step (e.g. to 0.5) rather than skipping rows, so it
        asks for labels at half-integer y's too. Naively rounding those to
        the nearest row (Python's round() included, since it rounds .5 to
        the nearest *even* integer) draws that row's label at the half-step
        position instead of centered on the row itself, off by up to half a
        row - and every other integer row's own on-row candidate then gets
        skipped as a same-label duplicate, hiding it entirely. Answering
        "invalid" for any non-integer y instead makes the formatter fall
        back to its normal "no real row here" blank, so a crowded axis shows
        fewer labels rather than misplaced ones.
        """
        row = round(y)
        if abs(y - row) > 1e-6:
            return -1
        return int(row)

    def _create_y_tick_formatter(self):
        # """Create a y-axis tick formatter that prevents label repetition."""
        state = {"last_label": None}

        def format_tick(y, y_min, y_max):
            # y is the display row (compacted position); labels always show the
            # real neuron/stim id at that row, so the axis stays consistent with
            # the cell segmentation's numbering regardless of any hidden/greyed
            # neurons compacting the rows above/below it.
            #
            # The axis can be asked for tick candidates beyond the currently
            # visible row count - e.g. the broader pannable extent
            # (controller.max_height) doesn't shrink just because some
            # neurons are hidden, only the current camera view does. Such a
            # position has no real neuron behind it at all: fabricating a
            # label from the raw row index there would draw a "ghost" tick
            # number with no corresponding grid line or data underneath it.
            display_row = self._display_row_for_tick_y(y)
            if not (0 <= display_row < len(self._visible_neuron_ids)):
                return ""
            neuron_id = self._visible_neuron_ids[display_row]
            label = f"{neuron_id}"
            if label == state["last_label"]:
                return ""
            state["last_label"] = label
            return label

        return format_tick

    def contextMenuEvent(self, event):
        if (
            self.mode == InteractMode.SELECT
            and event.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._rebuild_custom_analysis_menu()
            self.context_menu.exec(event.globalPos())
        else:
            super().contextMenuEvent(event)

    def _add_analysis_action(
        self,
        analysis_type: CustomAnalysisType | AnalysisPlugin,
        icon_name: str,
    ) -> None:
        add_action(
            str(analysis_type),
            lambda *args, _analysis_type=analysis_type: (
                self.custom_analysis_requested.emit(
                    self.data,
                    _analysis_type,
                    self._open_analysis_in_new_window,
                    self.selected_neuron_slice,
                    self.selected_time_slice,
                )
            ),
            self.context_menu,
            None,
            self,
            icon=icon_name,
        )

    def _set_open_analysis_in_new_window(self, checked: bool) -> None:
        self._open_analysis_in_new_window = checked

    def _rebuild_custom_analysis_menu(self) -> None:
        self.context_menu.clear()
        add_action(
            "Open analysis in a new window",
            self._set_open_analysis_in_new_window,
            self.context_menu,
            None,
            self,
            icon="mdi6.open-in-new",
            checkable=True,
            is_checked=self._open_analysis_in_new_window,
        )
        self.context_menu.addSeparator()
        for custom_analysis_type in CustomAnalysisType:
            self._add_analysis_action(
                custom_analysis_type,
                custom_analysis_type.value,
            )

        plugin_analyses = discover_analysis_plugins()
        if plugin_analyses:
            self.context_menu.addSeparator()
            for plugin_analysis in plugin_analyses:
                self._add_analysis_action(
                    plugin_analysis,
                    plugin_analysis.icon,
                )

        if self.SUPPORTS_DETACH:
            self.context_menu.addSeparator()
            detach_action = self.context_menu.addAction("Detach panel")
            assert detach_action is not None
            detach_action.triggered.connect(
                lambda: self._request_detach_from_window(keep_original=False)
            )
            copy_action = self.context_menu.addAction("Detach copy of panel")
            assert copy_action is not None
            copy_action.triggered.connect(
                lambda: self._request_detach_from_window(keep_original=True)
            )

    def _request_detach_from_window(self, *, keep_original: bool) -> None:
        detach_panel = getattr(self.window(), "detach_panel", None)
        if callable(detach_panel):
            detach_panel(self, keep_original=keep_original)

    def post_set_data(self):
        self.subplot.grids = False
        self.subplot.axes.y.ticks = None
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.y.min_tick_distance = 35
        self.subplot.axes.y.tick_format = self._create_y_tick_formatter()
        self.subplot.axes.x.ticks = None
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.axes.x.min_tick_distance = 100
        self.subplot.axes.x.tick_format = lambda x, x_min, x_max: (
            f"{format_milliseconds(x)}ms"
        )
        # Push axes (incl. tick labels) above gridline z=199 so labels render on top.
        self.subplot.axes.world_object.world.z = 300
        self.subplot.toolbar = False
        self.subplot.add_graphic(self.roi, center=False)
        self.subplot.add_graphic(self.timeline, center=False)
        max_time_ms = float(self.n_timestamps) * float(self.dt_ms)
        max_neurons = self.n_neurons
        if self.controller is not None:
            max_time_ms = max(max_time_ms, float(self.controller.max_width))
            max_neurons = max(max_neurons, int(self.controller.max_height))
        overlay_colors = theme_module.plot_overlay_colors(self._theme)
        self.grid_lines = GridLines(
            max_neurons,
            max_time_ms,
            dt_ms=self.dt_ms,
            grid_color=overlay_colors["grid"],
            shading_color=overlay_colors["shading"],
        )
        self.subplot.add_graphic(self.grid_lines, center=False)
        self.grid_lines.visible = self.show_grid
        self.controller.add_grid_lines(self.grid_lines)
        self.subplot.add_graphic(self.text, center=False)
        self._apply_fpl_theme()
        self.roi.visible = False

    def clear_data(self) -> None:
        if self.controller is not None and self.grid_lines is not None:
            self.controller.remove_grid_lines(self.grid_lines)
        super().clear_data()
        self.graphic = None
        self.hover_slice_graphic = None
        self.background_graphics = []
        self.data = None
        self.data_values = None
        self.data_values_scale = 1.0
        self.units = ""
        self.grid_lines = None
        self._all_neuron_ids = []
        self.n_neurons = 0
        self.n_neurons_total = 0
        self.n_timestamps = 0
        self.dt_ms = 0.0
        self.unselect_neurons()
        self.clear_greyed_neurons()

    def closeEvent(self, event) -> None:
        self._set_hovered_neuron_id(None)
        super().closeEvent(event)

    def _rebuild_visible_neuron_mapping(self) -> None:
        self._visible_neuron_ids = sorted(
            set(self._all_neuron_ids) - self._greyed_neuron_ids
        )
        self._neuron_id_to_display_row = {
            neuron_id: row for row, neuron_id in enumerate(self._visible_neuron_ids)
        }
        self.n_neurons = len(self._visible_neuron_ids)
        self.n_neurons_total = len(self._all_neuron_ids)

    def _display_row_to_neuron_id(self, display_row: int) -> int | None:
        if 0 <= display_row < len(self._visible_neuron_ids):
            return self._visible_neuron_ids[display_row]
        return None

    def _apply_visible_neuron_layout(self) -> None:
        """Subclasses override to reposition/hide/rewrite their graphics so only
        currently-visible neurons are shown, compacted to fill the gaps left by
        any greyed (hidden) ones."""

    def _refit_view_to_visible_neurons(self) -> None:
        """Shrink the grid lines and snap the camera to exactly the currently
        visible neuron count, so hidden rows leave no dead/blank space in the
        view (rather than merely being blanked out within the old extent)."""
        if self.grid_lines is not None:
            self.grid_lines.update_neuron_extent(self.n_neurons)
        if self.controller is not None and self.n_neurons > 0:
            self.controller.focus_neuron_range(0, self.n_neurons)

    def _display_row_for_mouse_y(self, y: float) -> int:
        """Map a pointer's world y position to a display row, clamped to the
        currently visible row range. See _display_row_for_tick_y()."""
        return int(np.clip(math.floor(y + 0.5), 0, self.n_neurons - 1))

    def _neuron_slice_from_y_range(self, y_a: float, y_b: float) -> slice:
        """Map an ROI drag's two endpoint y positions to a display-row slice.
        See _display_row_for_tick_y()."""
        y_min = min(y_a, y_b) + 0.5
        y_max = max(y_a, y_b) + 0.5
        return slice(math.floor(y_min), math.ceil(y_max))

    def set_neuron_greyed(self, neuron_index: int, greyed: bool) -> None:
        """Hide (or restore) one neuron's row, compacting the remaining visible
        rows to fill the gap. The underlying data is never touched."""
        if greyed:
            if neuron_index in self._greyed_neuron_ids:
                return
            self._greyed_neuron_ids.add(neuron_index)
        else:
            if neuron_index not in self._greyed_neuron_ids:
                return
            self._greyed_neuron_ids.discard(neuron_index)
        self._rebuild_visible_neuron_mapping()
        self._apply_visible_neuron_layout()
        self._refit_view_to_visible_neurons()

    def set_neurons_greyed(
        self,
        newly_hidden: Iterable[int],
        newly_shown: Iterable[int],
    ) -> None:
        """Batch version of set_neuron_greyed(): apply every change first,
        then rebuild/refit the view exactly once.

        A single selection change upstream (e.g. picking one cell in the
        Spatial Footprints panel) can hide or show many rows at once - up to
        every other neuron, for a segmentation with N cells. Calling
        set_neuron_greyed() once per affected row would rebuild this
        widget's entire graphic from scratch N times over for what is
        conceptually one update; this does it once instead.
        """
        changed = False
        for neuron_index in newly_hidden:
            if neuron_index not in self._greyed_neuron_ids:
                self._greyed_neuron_ids.add(neuron_index)
                changed = True
        for neuron_index in newly_shown:
            if neuron_index in self._greyed_neuron_ids:
                self._greyed_neuron_ids.discard(neuron_index)
                changed = True
        if not changed:
            return
        self._rebuild_visible_neuron_mapping()
        self._apply_visible_neuron_layout()
        self._refit_view_to_visible_neurons()

    def is_neuron_greyed(self, neuron_index: int) -> bool:
        return neuron_index in self._greyed_neuron_ids

    def clear_greyed_neurons(self) -> None:
        if not self._greyed_neuron_ids:
            return
        self._greyed_neuron_ids.clear()
        self._rebuild_visible_neuron_mapping()
        self._apply_visible_neuron_layout()
        self._refit_view_to_visible_neurons()

    def set_time_ms(self, time_ms: int) -> None:
        self.timeline.t = time_ms

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
            display_row = math.floor(self.mouse_pos[1] + 0.5)
            neuron_id = self._display_row_to_neuron_id(display_row)
            if neuron_id is None:
                return
            self.mouse_drag_time_index = math.floor(self.mouse_pos[0] / self.dt_ms)
            self.mouse_drag_neuron_index = neuron_id
            self.mouse_dragged.emit(
                self.mouse_drag_neuron_index,
                self.mouse_drag_neuron_index,
                self.mouse_drag_time_index,
                self.mouse_drag_time_index,
                self.mouse_press_button,
            )
        else:
            return

    def _position_hover_slice(self) -> None:
        """Place the hover-slice preview at the current pointer position.

        Subclasses may override this to snap the preview to discrete cells;
        the base implementation follows the raw pointer position.
        """
        if self.hover_slice_graphic is None or self.mouse_pos is None:
            return
        self.hover_slice_graphic.offset = (
            self.mouse_pos[0],
            self.mouse_pos[1],
            100,
        )

    def _set_hovered_neuron_id(self, neuron_id: int | None) -> None:
        if neuron_id == self._last_hovered_neuron_id:
            return
        self._last_hovered_neuron_id = neuron_id
        self.neuron_hovered.emit(neuron_id)

    def _clear_hover(self) -> None:
        self.mouse_pos = None
        self.text.visible = False
        if self.hover_slice_graphic is not None:
            self.hover_slice_graphic.visible = False
        self._set_hovered_neuron_id(None)

    def pointer_leave(self, ev: pygfx.PointerEvent) -> None:
        self._clear_hover()

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        self.mouse_pos = self.subplot.map_screen_to_world(ev)
        if self.mouse_pos is None:
            self._clear_hover()
            return
        self.logger.debug(f"pointer_move at {self.mouse_pos} in mode {self.mode}")
        self.text.visible = self.show_text
        hover_value_text = ""
        hovered_neuron_id: int | None = None
        if self.data_values is not None and self._visible_neuron_ids:
            time_index = np.clip(
                math.floor(self.mouse_pos[0] / self.dt_ms),
                0,
                self.data_values.shape[0] - 1,
            )
            display_row = self._display_row_for_mouse_y(self.mouse_pos[1])
            neuron_id = self._visible_neuron_ids[display_row]
            hovered_neuron_id = neuron_id
            hover_value = (
                self.data_values[time_index, neuron_id] * self.data_values_scale
            )
            hover_value_text = f"Neuron {neuron_id}\n{hover_value:.3g} {self.units}"
        self._set_hovered_neuron_id(hovered_neuron_id)
        self.text.set_pos(self.mouse_pos, hover_value_text)
        if self.hover_slice_graphic is not None:
            self.hover_slice_graphic.visible = True
            self._position_hover_slice()
        if self.mouse_press_pos is None:
            return
        if self.mode == InteractMode.SELECT and self.selection_in_progress:
            x0 = min(self.mouse_pos[0], self.mouse_press_pos[0])
            x1 = max(self.mouse_pos[0], self.mouse_press_pos[0])
            time_slice = slice(
                max(0, math.floor(x0 / self.dt_ms)), math.ceil(x1 / self.dt_ms)
            )
            neuron_slice = self._neuron_slice_from_y_range(
                self.mouse_pos[1], self.mouse_press_pos[1]
            )
            self.select_neurons(neuron_slice, time_slice)
            return
        elif self.mode == InteractMode.DRAW:
            drag_time_index = math.floor(self.mouse_pos[0] / self.dt_ms)
            drag_display_row = math.floor(self.mouse_pos[1] + 0.5)
            drag_neuron_index = self._display_row_to_neuron_id(drag_display_row)
            if drag_neuron_index is None:
                return
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

    def _display_slice_to_real_range(self, display_slice: slice) -> tuple[int, int]:
        """Translate a display-row slice (as produced by mouse-driven selection)
        into the smallest-to-largest real neuron id span it touches. Real ids are
        not necessarily contiguous once neurons are hidden/compacted, so this is
        an approximation - a real id that falls inside the span but is itself
        hidden gets included too."""
        if not self._visible_neuron_ids:
            return (0, 0)
        start_row = max(0, display_slice.start or 0)
        stop_row = min(
            len(self._visible_neuron_ids),
            display_slice.stop
            if display_slice.stop is not None
            else len(self._visible_neuron_ids),
        )
        if stop_row <= start_row:
            return (0, 0)
        real_ids_in_range = self._visible_neuron_ids[start_row:stop_row]
        return (real_ids_in_range[0], real_ids_in_range[-1] + 1)

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
        real_start, real_stop = self._display_slice_to_real_range(neuron_index_slice)
        self.roi.visible = True
        self.roi.select_slice(
            neuron_index_slice,
            time_index_slice,
            self.dt_ms,
            label_neuron_range=(real_start, real_stop),
        )
        self.selected_time_slice = time_index_slice
        self.selected_neuron_slice = slice(real_start, real_stop)
        self.logger.debug(
            f"Selecting neurons {self.selected_neuron_slice} (display {neuron_index_slice}) and times {time_index_slice}"
        )
