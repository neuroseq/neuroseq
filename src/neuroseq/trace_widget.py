from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import TraceData
from neuroseq.stdp_synaptic_tuning_dialog import _SelfRaisingDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_widget import apply_cmap_alpha
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.trace_widget_state import TraceWidgetState
from neuroseq.utils import get_cmap
from neuroseq.utils import neuron_indices_to_colors


class TraceNormDialog(_SelfRaisingDialog):
    """A dialog to set the normalization factor for the traces.

    No Qt parent (see _SelfRaisingDialog's docstring): a parent sets
    WM_TRANSIENT_FOR, and on X11/KWin a transient dialog can end up sharing
    its owner's "raise group" so clicking it never brings it above another
    sibling popup.
    """

    def __init__(
        self,
        trace_widget: TraceWidget,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Set Trace Normalization")
        self.trace_widget = trace_widget
        self.trace_norm_percent = self.trace_widget.trace_norm_percent
        self.offset_by_median = self.trace_widget.offset_by_median
        self.clip = self.trace_widget.clip

        self.content_layout = QtWidgets.QVBoxLayout(self)

        self.label_norm = QtWidgets.QLabel(
            f"Normalization: {self.trace_norm_percent}%", self
        )
        self.content_layout.addWidget(self.label_norm)
        norm_input_layout = QtWidgets.QHBoxLayout()
        self.slider_norm = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider_norm.setRange(50, 1000)
        self.slider_norm.setValue(self.trace_norm_percent)
        self.slider_norm.valueChanged.connect(self.update_norm_input)
        norm_input_layout.addWidget(self.slider_norm)
        self.spin_norm = QtWidgets.QSpinBox(self)
        self.spin_norm.setRange(50, 1000)
        self.spin_norm.setSuffix("%")
        self.spin_norm.setValue(self.trace_norm_percent)
        self.spin_norm.valueChanged.connect(self.update_norm_input)
        norm_input_layout.addWidget(self.spin_norm)
        self.content_layout.addLayout(norm_input_layout)

        self.chk_offset = QtWidgets.QCheckBox("Center median")
        self.chk_offset.setChecked(self.clip)
        self.chk_offset.clicked.connect(self.chk_offset_clicked)
        self.content_layout.addWidget(self.chk_offset)

        self.chk_clip = QtWidgets.QCheckBox("Clip values")
        self.chk_clip.setChecked(self.clip)
        self.chk_clip.clicked.connect(self.chk_clip_clicked)
        self.content_layout.addWidget(self.chk_clip)

        self.auto_extend_vertical_space = self.trace_widget.auto_extend_vertical_space
        self.chk_auto_extend = QtWidgets.QCheckBox("Auto-extend vertical space")
        self.chk_auto_extend.setToolTip(
            "When a trace's normalized amplitude would otherwise get clipped, "
            "grow that neuron's row just enough to fit it instead - without "
            "encroaching on the neighboring rows."
        )
        self.chk_auto_extend.setChecked(self.auto_extend_vertical_space)
        self.chk_auto_extend.clicked.connect(self.chk_auto_extend_clicked)
        self.content_layout.addWidget(self.chk_auto_extend)

        self.btn_reset_defaults = QtWidgets.QPushButton("Reset to defaults", self)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        self.content_layout.addWidget(self.btn_reset_defaults)

        self.btn_close = QtWidgets.QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        self.content_layout.addWidget(self.btn_close)

    def reset_defaults(self) -> None:
        self.slider_norm.setValue(100)
        self.spin_norm.setValue(100)
        self.offset_by_median = True
        self.clip = True
        self.auto_extend_vertical_space = False
        self.chk_offset.setChecked(self.offset_by_median)
        self.chk_offset.setChecked(self.clip)
        self.chk_auto_extend.setChecked(self.auto_extend_vertical_space)
        self.update_norm_input(self.slider_norm.value())

    def chk_offset_clicked(self, checked: bool) -> None:
        self.offset_by_median = checked
        self.update_trace_widget()

    def chk_clip_clicked(self, checked: bool) -> None:
        self.clip = checked
        self.update_trace_widget()

    def chk_auto_extend_clicked(self, checked: bool) -> None:
        self.auto_extend_vertical_space = checked
        self.update_trace_widget()

    def update_norm_input(self, value: int) -> None:
        self.trace_norm_percent = value
        self.label_norm.setText(f"Normalization: {value}%")
        self.slider_norm.setValue(value)
        self.spin_norm.setValue(value)
        self.update_trace_widget()

    def update_trace_widget(self) -> None:
        self.trace_widget.trace_norm_percent = self.trace_norm_percent
        self.trace_widget.offset_by_median = self.offset_by_median
        self.trace_widget.clip = self.clip
        self.trace_widget.auto_extend_vertical_space = self.auto_extend_vertical_space
        self.trace_widget.replot_traces()


class TraceWidget(TimeseriesWidget):
    """A widget that displays a set of traces stacked vertically.
    traces should be a 2d array, where the first index is the neuron index and the second index is the time index.
    """

    vector_export_requested = QtCore.Signal()

    # Line thickness is expressed in screen pixels. Below this fraction of a
    # neuron's pixel band, adjacent traces' strokes start to touch; above the
    # default thickness there's nothing to gain from thickening further.
    _THICKNESS_BAND_FRACTION = 0.7
    _MIN_LINE_THICKNESS = 1.0
    _MAX_LINE_THICKNESS = 2.0

    # Trace lines default to z=0; keep the stim overlay just behind them so
    # the line is always visible on top, without affecting the grid (z=199).
    STIM_OVERLAY_Z = -1.0
    STIM_OVERLAY_ALPHA = 0.6

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
        default_controller: ConstrainedPanZoomController | None = None,
        show_axis_link_controls: bool = False,
        state: TraceWidgetState | None = None,
    ) -> None:
        super().__init__(
            controller,
            parent,
            default_controller=default_controller,
            show_axis_link_controls=show_axis_link_controls,
        )
        self.trace_norm_percent = 100
        self.offset_by_median = True
        self.clip = True
        self.auto_extend_vertical_space = False
        # Per-display-row half-height (>= 0.5) and center position, only
        # populated (and only non-uniform) while auto_extend_vertical_space
        # is on - see _compute_row_layout(). Empty/unused otherwise, in
        # which case every row keeps the default one-unit spacing.
        self._row_half_heights: list[float] = []
        self._row_centers: list[float] = []
        self._row_boundaries: list[float] = []
        # The controller's neuron-count-driven height limit before auto-extend
        # inflated it to fit an over-scaled row - restored once auto-extend is
        # turned back off. None means auto-extend hasn't touched it (yet).
        self._base_max_height: float | None = None
        self._trace_norm_dialog: TraceNormDialog | None = None
        self.vector_export_requested.connect(self.preview_traces)
        # Whether to overlay a stim array on top of these traces, and which
        # one - a view-local display setting like trace normalization, not
        # part of TraceWidgetState: each detached copy of this panel can
        # independently show its own stim overlay.
        self._stim_overlay_enabled = False
        self._stim_overlay_array: StimArray | None = None
        self._stim_overlay_graphic: fpl.ImageGraphic | None = None
        if isinstance(self.controller, ConstrainedPanZoomController):
            self.controller.add_state_change_listener(self._update_line_thickness)
        self._state = state if state is not None else TraceWidgetState()
        # Keep the connection handles (not just the callables): a
        # signal-to-signal connection (`signal.connect(other_signal.emit)`,
        # as used elsewhere for this same shared-state pattern) can't be
        # disconnected by passing the callable again - only disconnecting
        # the exact handle .connect() returned works. None of this widget's
        # own connections are signal-to-signal today, but keep the same
        # robust pattern used for SpatialFootprintWidget's equivalent state.
        self._state_connections = [
            self._state.data_changed.connect(self._on_state_data_changed),
            self._state.greyed_changed.connect(self._on_state_greyed_changed),
        ]
        if self._state.data is not None:
            self._on_state_data_changed()

    def closeEvent(self, event) -> None:
        if isinstance(self.controller, ConstrainedPanZoomController):
            self.controller.remove_state_change_listener(self._update_line_thickness)
        # A detached view shares self._state with every other view (docked
        # or detached) bound to it. Qt is supposed to auto-disconnect a
        # destroyed receiver, but this widget's underlying render canvas can
        # be torn down well before its Python/Qt object is actually deleted,
        # leaving the connection live in between. A later mutation from
        # another view would then synchronously invoke this (now
        # GPU-resource-less) view's slot and segfault touching its dead
        # canvas - disconnect explicitly instead of relying on that timing.
        for connection in self._state_connections:
            with contextlib.suppress(TypeError, RuntimeError):
                self._state.disconnect(connection)  # type: ignore[call-arg]
        self._state_connections = []
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_line_thickness()

    def _update_line_thickness(self) -> None:
        """Shrink line thickness once a neuron's pixel band gets thinner than the stroke.

        At the default zoom (or zoomed into a handful of neurons) this clamps
        to the default thickness, so nothing changes. Once the view is zoomed
        out far enough that a neuron's band maps to fewer screen pixels than
        the stroke width, adjacent traces would otherwise bleed into each
        other's band; shrinking the stroke along with the band keeps them
        visually separated for as long as that's physically possible.
        """
        if self.graphic is None or not isinstance(
            self.controller, ConstrainedPanZoomController
        ):
            return
        # _get_camera_state() reads the live pygfx camera, which state-change
        # listeners can be notified ahead of (_update_cameras() applies pending
        # state after _notify_state_change() fires); _last_cam_state is the
        # pending state itself, so it's always current.
        camera_state = self.controller._last_cam_state
        if "height" not in camera_state:
            return
        visible_neurons = float(camera_state["height"])
        render_height = self.subplot.frame.get_render_rect()[3]
        if visible_neurons <= 0 or render_height <= 0:
            return
        pixels_per_band = render_height / visible_neurons
        thickness = float(
            np.clip(
                pixels_per_band * self._THICKNESS_BAND_FRACTION,
                self._MIN_LINE_THICKNESS,
                self._MAX_LINE_THICKNESS,
            )
        )
        for line in self.graphic.graphics:
            line.thickness = thickness

    def trace_norm_dialog(self) -> None:
        if self._trace_norm_dialog is None:
            self._trace_norm_dialog = TraceNormDialog(self)
        self._trace_norm_dialog.show()
        self._trace_norm_dialog.raise_()
        self._trace_norm_dialog.activateWindow()

    def get_normalized_trace_data(self, out: np.ndarray | None = None) -> np.ndarray:
        if self.data is None or not isinstance(self.data, TraceData):
            return np.array([], dtype=np.float32)
        traces = self.data.traces
        if self.offset_by_median:
            median_values = np.nanmedian(traces, axis=1)
            max_values = np.nanmax(traces, axis=1)
            min_values = np.nanmin(traces, axis=1)
            # A trace's excursion below its median can exceed its excursion
            # above it (e.g. a calcium baseline with an occasional deep dip
            # but only modest transients) - considering only max_values
            # here left the downward side completely unbounded, able to
            # swing many rows past its own and get cut off by whatever
            # camera edge it hit first (worst case: the only visible row).
            max_trace = np.max(
                np.maximum(max_values - median_values, median_values - min_values)
            )
        else:
            max_trace = max(
                abs(float(np.nanmin(traces))), abs(float(np.nanmax(traces)))
            )
        norm = 0.5 if max_trace == 0 else 1.0 / (2.0 * max_trace)
        norm = np.float32((self.trace_norm_percent / 100.0) * norm)
        if self.offset_by_median:
            offset = (-median_values * norm).astype(np.float32)
        else:
            offset = np.zeros(traces.shape[0], dtype=np.float32)
        if out is None:
            out = np.empty_like(traces)
        np.multiply(traces, norm, out=out)
        out += offset[:, np.newaxis]
        # When auto-extending, a row that would otherwise clip instead grows
        # to fit its own unclipped excursion (see _compute_row_layout()), so
        # clipping would only throw away the data auto-extend just made room
        # for.
        if self.clip and not self.auto_extend_vertical_space:
            np.clip(out, -0.5, 0.5, out=out)
        return out

    def replot_traces(self) -> None:
        """Update the plotted traces based on the current normalization settings."""
        if (
            self.data is not None
            and isinstance(self.data, TraceData)
            and self.graphic is not None
        ):
            normalized_traces = self.get_normalized_trace_data()
            for line_index, line_graphic in enumerate(self.graphic):
                line_graphic.data[:, 1] = normalized_traces[line_index, :]
            if self.auto_extend_vertical_space:
                # Row heights depend on the normalization settings just
                # applied above (e.g. dragging the percent slider can grow or
                # shrink which rows need extra room) - recompute the layout
                # and make sure the view's extent can fit it. Deliberately
                # *not* a full _refit_view_to_visible_neurons(): that also
                # snaps the camera to exactly the new total height, which
                # would rescale/recenter every row on screen just because a
                # normalization tweak changed one row's height - reading as
                # the trace itself changing even though its values didn't
                # (issue #551 review). Camera movement stays reserved for
                # actual visible-row changes (see _on_state_greyed_changed).
                self._apply_visible_neuron_layout()
                self._grow_view_extent_for_visible_neurons()
            elif self._base_max_height is not None:
                # auto-extend was just turned back off: restore the plain,
                # evenly-spaced layout and the controller height it displaced.
                # Same reasoning as above - update the extent, don't move
                # the camera out from under the user.
                if self.controller is not None:
                    self.controller.max_height = self._base_max_height
                self._base_max_height = None
                self._apply_visible_neuron_layout()
                if self.grid_lines is not None:
                    self.grid_lines.update_neuron_extent(self.n_neurons)

    def _compute_row_layout(self) -> None:
        """Recompute per-row half-heights/centers/boundaries so that every
        visible row is exactly tall enough to fit its own trace's current
        (unclipped) excursion, never encroaching on its neighbors.

        A row that doesn't need extra room keeps the default 0.5 half-height
        (i.e. the usual one-unit row), so this is a no-op change for any
        panel that never asked for auto-extend.
        """
        self._row_half_heights = []
        self._row_centers = []
        self._row_boundaries = []
        if not isinstance(self.data, TraceData) or not self._visible_neuron_ids:
            return
        normalized = self.get_normalized_trace_data()
        data_row_by_id = {int(idx): i for i, idx in enumerate(self.data.indices)}
        cumulative = -0.5
        for neuron_id in self._visible_neuron_ids:
            data_row = data_row_by_id.get(neuron_id)
            half_height = 0.5
            if data_row is not None and normalized.size:
                peak = float(np.nanmax(np.abs(normalized[data_row])))
                half_height = max(0.5, peak)
            upper = cumulative + 2.0 * half_height
            self._row_half_heights.append(half_height)
            self._row_centers.append(0.5 * (cumulative + upper))
            self._row_boundaries.append(upper)
            cumulative = upper

    def _display_row_for_tick_y(self, y: float) -> int:
        if not self.auto_extend_vertical_space or not self._row_boundaries:
            return super()._display_row_for_tick_y(y)
        # Unlike hover (which always resolves to the nearest row), a tick
        # candidate outside the actual row extent must map to an invalid row
        # so the formatter blanks it - see _create_y_tick_formatter()'s
        # "ghost tick" comment.
        if y < -0.5 or y >= self._row_boundaries[-1]:
            return -1
        return int(np.searchsorted(self._row_boundaries, y, side="right"))

    def _display_row_for_mouse_y(self, y: float) -> int:
        if not self.auto_extend_vertical_space or not self._row_boundaries:
            return super()._display_row_for_mouse_y(y)
        idx = int(np.searchsorted(self._row_boundaries, y, side="right"))
        return int(np.clip(idx, 0, len(self._row_boundaries) - 1))

    def _neuron_slice_from_y_range(self, y_a: float, y_b: float) -> slice:
        if not self.auto_extend_vertical_space or not self._row_boundaries:
            return super()._neuron_slice_from_y_range(y_a, y_b)
        boundaries = self._row_boundaries
        start = int(np.searchsorted(boundaries, min(y_a, y_b), side="right"))
        stop = int(np.searchsorted(boundaries, max(y_a, y_b), side="right")) + 1
        return slice(start, stop)

    def _refit_view_to_visible_neurons(self) -> None:
        if not self.auto_extend_vertical_space or not self._row_half_heights:
            super()._refit_view_to_visible_neurons()
            return
        self._grow_view_extent_for_visible_neurons()
        total_height = sum(2.0 * h for h in self._row_half_heights)
        if self.controller is not None and total_height > 0:
            self.controller.focus_neuron_range(0, math.ceil(total_height))

    def _grow_view_extent_for_visible_neurons(self) -> None:
        """Make sure the grid lines and the controller's zoom-out limit can
        fit the current (possibly grown) per-row layout, without moving the
        camera - see the auto-extend branch of replot_traces() for why that
        matters. _refit_view_to_visible_neurons() calls this too, then goes
        on to actually move the camera for genuine visible-row changes."""
        if not self._row_half_heights:
            return
        total_height = sum(2.0 * h for h in self._row_half_heights)
        if self.controller is not None:
            if self._base_max_height is None:
                self._base_max_height = self.controller.max_height
            self.controller.max_height = max(self._base_max_height, total_height)
        if self.grid_lines is not None:
            self.grid_lines.update_neuron_extent(max(1, math.ceil(total_height)))

    def _rebuild_custom_analysis_menu(self) -> None:
        super()._rebuild_custom_analysis_menu()
        self.context_menu.insertSeparator(self.context_menu.actions()[0])
        export_action = QtWidgets.QAction("Preview vector graphic…", self)
        export_action.setEnabled(isinstance(self.data, TraceData))
        export_action.triggered.connect(self.vector_export_requested.emit)
        self.context_menu.insertAction(self.context_menu.actions()[0], export_action)
        norm_action = QtWidgets.QAction("Normalization…", self)
        norm_action.triggered.connect(self.trace_norm_dialog)
        self.context_menu.insertAction(self.context_menu.actions()[0], norm_action)

    def preview_traces(self) -> QtWidgets.QDialog | None:
        """Preview the selected region, or the current view when nothing is selected."""
        controller = self.controller
        if (
            not isinstance(self.data, TraceData)
            or self.graphic is None
            or controller is None
        ):
            return None

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure

        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"Trace preview: {self.data.name}")
        dialog.resize(1000, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        figure = Figure(layout="tight", facecolor="none")
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, dialog))
        layout.addWidget(canvas)
        ax = figure.add_subplot(111, facecolor="none")
        colors = neuron_indices_to_colors(self.data.indices)
        for index, graphic, color in zip(
            self.data.indices, self.graphic.graphics, colors, strict=True
        ):
            if not graphic.visible:
                continue
            # Copy normalized, clipped coordinates and preserve neuron spacing.
            points = np.asarray(graphic.data[:]).copy()
            (line,) = ax.plot(
                points[:, 0] + graphic.offset[0],
                points[:, 1] + graphic.offset[1],
                color=color,
                linewidth=float(graphic.thickness) * 72 / figure.dpi,
            )
            line.set_gid(f"trace_{int(index)}")
        state = controller._get_camera_state()
        center_x, center_y = state["position"][:2]
        ax.set_xlim(center_x - state["width"] / 2, center_x + state["width"] / 2)
        ax.set_ylim(center_y - state["height"] / 2, center_y + state["height"] / 2)
        if self.selected_time_slice is not None:
            start, stop, _ = self.selected_time_slice.indices(self.n_timestamps)
            if stop > start:
                ax.set_xlim(start * self.dt_ms, stop * self.dt_ms)
        if self.selected_neuron_slice is not None:
            selection = self.selected_neuron_slice
            rows = [
                row
                for row, neuron_id in enumerate(self._visible_neuron_ids)
                if (selection.start is None or neuron_id >= selection.start)
                and (selection.stop is None or neuron_id < selection.stop)
            ]
            if rows:
                ax.set_ylim(rows[0] - 0.5, rows[-1] + 0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index (normalized traces)")
        ax.set_title(self.data.name)
        dialog.show()
        return dialog

    def set_data(
        self,
        data: TraceData | None = None,
    ) -> None:
        self._state.set_data(data)

    def _on_state_data_changed(self) -> None:
        data = self._state.data
        self.clear_data()
        if data is None:
            return
        self.logger.info(
            f"{data.traces.shape} traces with dt={data.dt}s and units={data.units}"
        )
        self.data = data
        self.data_values = data.traces.T
        self.data_values_scale = 1000.0
        self.units = data.units if data.units else "mV"
        self._all_neuron_ids = sorted({int(i) for i in data.indices})
        self._greyed_neuron_ids = set(self._state.greyed_neuron_ids)
        self._rebuild_visible_neuron_mapping()
        self.dt_ms = float(data.dt) * 1000.0  # convert to milliseconds
        self.n_timestamps = data.traces.shape[1]
        # convert timestamps to milliseconds and stack to produce a format suitable for line_stack
        # todo: should be possible to re-use the same timestamps for all traces?
        dt_ms = np.float32(self.dt_ms)
        timestamps_ms = np.arange(data.traces.shape[1], dtype=np.float32) * np.float32(
            dt_ms
        )
        n_neurons, n_timesteps = data.traces.shape
        line_data = np.empty((n_neurons, n_timesteps, 2), dtype=np.float32)
        line_data[:, :, 0] = timestamps_ms
        self.get_normalized_trace_data(out=line_data[:, :, 1])
        self.graphic = self.subplot.add_line_collection(
            line_data,
            colors=neuron_indices_to_colors(self.data.indices),
            thickness=2,
        )
        self._apply_visible_neuron_layout()
        self._update_line_thickness()
        self._rebuild_stim_overlay()
        self.post_set_data()

    def _on_state_greyed_changed(self) -> None:
        self._greyed_neuron_ids = set(self._state.greyed_neuron_ids)
        self._rebuild_visible_neuron_mapping()
        self._apply_visible_neuron_layout()
        self._rebuild_stim_overlay()
        self._refit_view_to_visible_neurons()

    def set_neuron_greyed(self, neuron_index: int, greyed: bool) -> None:
        self._state.set_neuron_greyed(neuron_index, greyed)

    def set_neurons_greyed(
        self,
        newly_hidden: Iterable[int],
        newly_shown: Iterable[int],
    ) -> None:
        self._state.set_neurons_greyed(newly_hidden, newly_shown)

    def clear_greyed_neurons(self) -> None:
        self._state.clear_greyed_neurons()

    def _apply_visible_neuron_layout(self) -> None:
        if self.graphic is None or not isinstance(self.data, TraceData):
            return
        if self.auto_extend_vertical_space:
            self._compute_row_layout()
        else:
            self._row_half_heights = []
            self._row_centers = []
            self._row_boundaries = []
        for i, line in enumerate(self.graphic.graphics):
            real_id = int(self.data.indices[i])
            display_row = self._neuron_id_to_display_row.get(real_id)
            line.visible = display_row is not None
            if display_row is not None:
                y = (
                    self._row_centers[display_row]
                    if self.auto_extend_vertical_space and self._row_centers
                    else float(display_row)
                )
                line.offset = (line.offset[0], y, line.offset[2])

    def clear_data(self) -> None:
        # Must be nulled *before* calling super(): TimeseriesWidget.clear_data()
        # wipes the whole subplot via FPLWidget.clear_data() and then, at its
        # own tail end, calls clear_greyed_neurons() - which (via
        # TraceWidgetState) can synchronously emit greyed_changed and
        # re-enter _rebuild_stim_overlay() before this method gets a chance
        # to run its own cleanup. If the stale reference were still set at
        # that point, _rebuild_stim_overlay() would try to remove a graphic
        # the subplot wipe already discarded (pygfx logs, but does not
        # raise, "Attempting to remove object that was not a child" for
        # this). _stim_overlay_enabled/_stim_overlay_array are left alone so
        # the setting survives a data refresh.
        self._stim_overlay_graphic = None
        if self._base_max_height is not None and self.controller is not None:
            self.controller.max_height = self._base_max_height
        self._base_max_height = None
        self._row_half_heights = []
        self._row_centers = []
        self._row_boundaries = []
        super().clear_data()

    def set_stim_overlay_enabled(self, enabled: bool) -> None:
        """Toggle showing a stim array's heatmap behind the trace lines
        (issue #530). Call set_stim_overlay_data() to pick/update which
        StimArray is shown."""
        if enabled == self._stim_overlay_enabled:
            return
        self._stim_overlay_enabled = enabled
        self._rebuild_stim_overlay()

    def set_stim_overlay_data(self, stim_array: StimArray | None) -> None:
        self._stim_overlay_array = stim_array
        self._rebuild_stim_overlay()

    def _rebuild_stim_overlay(self) -> None:
        if self._stim_overlay_graphic is not None:
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(self._stim_overlay_graphic)
            self._stim_overlay_graphic = None

        stim_array = self._stim_overlay_array
        if (
            not self._stim_overlay_enabled
            or stim_array is None
            or not isinstance(self.data, TraceData)
            or not self._visible_neuron_ids
        ):
            return

        n_timesteps, n_stim_neurons = stim_array.values.shape
        # Same-neuron overlay only (see issue #530): row i shows the stim
        # column for the same real neuron id as trace display row i, or is
        # left at zero if that id has no corresponding stim column.
        visible_values = np.zeros(
            (n_timesteps, len(self._visible_neuron_ids)),
            dtype=stim_array.values.dtype,
        )
        for column, neuron_id in enumerate(self._visible_neuron_ids):
            if 0 <= neuron_id < n_stim_neurons:
                visible_values[:, column] = stim_array.values[:, neuron_id]

        vmax = float(np.max(np.abs(stim_array.values)))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax if stim_array.stim_type == StimType.External else 0.0
        cmap = get_cmap(stim_array.stim_type, alpha=self.STIM_OVERLAY_ALPHA)
        self._stim_overlay_graphic = self.subplot.add_image(
            visible_values.T,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        apply_cmap_alpha(self._stim_overlay_graphic, cmap)
        dt_ms = float(stim_array.dt) * 1000.0
        self._stim_overlay_graphic.offset = (0.5 * dt_ms, 0.0, self.STIM_OVERLAY_Z)
        self._stim_overlay_graphic.world_object.world.scale = (dt_ms, 1.0, 1.0)
