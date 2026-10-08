"""Stimulus-array rendering helpers.

The primary `StimWidget` view renders stimulus data with an image/scatter path
that preserves amplitude-dependent alpha and true sample geometry. The
`MinimumStimulusOverlay` defined here is an auxiliary visibility layer for the
"keep short stimuli visible" mode: it expects the same `(time, neuron)` value
grid, treats each contiguous same-sign nonzero run as one pulse, and should be
used only when callers need a minimum on-screen footprint rather than the
faithful main heatmap/scatter rendering alone.
"""

from __future__ import annotations

import math

import fastplotlib as fpl
import numpy as np
import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import get_cmap


class MinimumStimulusOverlay(fpl.Graphic):
    """Marks every contiguous run of same-sign nonzero stimulus samples with
    two shapes, one per neuron row, batched into two `pygfx.Line` objects:

    - a "fill" segment spanning the run's true world-space extent (full
      row height via `thickness_space="world"`, thickness=1.0), so a wide
      pulse renders as one seamless block - the same shape the main heatmap
      image would show if it weren't drawn at (near-)zero alpha for a
      low-amplitude sample. Building one segment *per sample* instead of
      per run (an earlier version of this overlay did) leaves a visible
      gap between adjacent samples wherever they render narrower than the
      pixel spacing between them, breaking a single continuous pulse into
      what looks like a rapid train of short separate pulses.
    - a "marker" at the run's midpoint with a fixed on-screen width
      (`thickness_space="screen"`) spanning the same full row height, so
      the run stays visible even if its world-space extent collapses
      below one screen pixel when zoomed far out (the fill segment alone
      would vanish at that point, same as the main image does).

    Both materials disable depth test/write: this overlay sits above the
    main stimulus image in z so it isn't hidden by it, but the hover-value
    text (drawn elsewhere at a lower z) must remain readable on top of any
    stimulus representation, and must not be culled by this overlay having
    written a nearer depth value first.
    """

    def __init__(
        self,
        values: np.ndarray,
        dt_ms: float,
        stim_type: StimType,
        *,
        thickness: float,
        z: float,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        cmap = get_cmap(stim_type, alpha=alpha).to_mpl()
        positive_color = np.asarray(cmap(1.0), dtype=np.float32)
        negative_color = np.asarray(cmap(0.0), dtype=np.float32)

        n_time, n_neurons = values.shape
        fill_positions: list[list[float]] = []
        fill_colors: list[np.ndarray] = []
        marker_positions: list[list[float]] = []
        marker_colors: list[np.ndarray] = []
        for neuron in range(n_neurons):
            row = values[:, neuron]
            nonzero = row != 0
            if not nonzero.any():
                continue
            sign = np.sign(row)
            # A new run starts wherever the nonzero state changes, or - for
            # bidirectional (External) stim only - the sign flips without an
            # intervening zero sample.
            change = np.empty(n_time, dtype=bool)
            change[0] = True
            change[1:] = (nonzero[1:] != nonzero[:-1]) | (
                nonzero[1:] & nonzero[:-1] & (sign[1:] != sign[:-1])
            )
            run_starts = np.flatnonzero(change)
            run_stops = np.r_[run_starts[1:], n_time]
            for start, stop in zip(run_starts, run_stops, strict=True):
                if not nonzero[start]:
                    continue
                color = positive_color if sign[start] > 0 else negative_color
                x_start = start * dt_ms
                x_stop = stop * dt_ms
                fill_positions.append([x_start, float(neuron), z])
                fill_positions.append([x_stop, float(neuron), z])
                fill_colors.append(color)
                fill_colors.append(color)
                x_mid = 0.5 * (x_start + x_stop)
                marker_positions.append([x_mid, neuron - 0.5, z])
                marker_positions.append([x_mid, neuron + 0.5, z])
                marker_colors.append(color)
                marker_colors.append(color)

        group = pygfx.Group()
        if fill_positions:
            geometry = pygfx.Geometry(
                positions=np.asarray(fill_positions, dtype=np.float32),
                colors=np.asarray(fill_colors, dtype=np.float32),
            )
            material = pygfx.LineSegmentMaterial(
                thickness=1.0,
                thickness_space="world",
                color_mode="vertex",
                aa=True,
                depth_test=False,
                depth_write=False,
            )
            self.fill_line = pygfx.Line(geometry, material)
            group.add(self.fill_line)
        else:
            self.fill_line = None
        if marker_positions:
            geometry = pygfx.Geometry(
                positions=np.asarray(marker_positions, dtype=np.float32),
                colors=np.asarray(marker_colors, dtype=np.float32),
            )
            material = pygfx.LineSegmentMaterial(
                thickness=thickness,
                thickness_space="screen",
                color_mode="vertex",
                aa=True,
                depth_test=False,
                depth_write=False,
            )
            self.marker_line = pygfx.Line(geometry, material)
            group.add(self.marker_line)
        else:
            self.marker_line = None
        self._set_world_object(group)


def apply_cmap_alpha(image_graphic: fpl.ImageGraphic, cmap: object) -> None:
    """`add_image()` ignores a colormap's alpha channel; write it into the
    texture directly instead (e.g. so zero-valued stim stays transparent)."""
    image_graphic._material.map.texture.data[:] = cmap(  # type: ignore[operator]
        np.linspace(0, 255, 256, dtype=int)
    ).astype(np.float32)
    image_graphic._material.map.texture.update_range((0, 0, 0), size=(256, 1, 1))


def _active_span(column: np.ndarray, index: int) -> tuple[int, int] | None:
    """Return the first and last sample of the non-zero run containing ``index``.

    ``None`` when the sample at ``index`` is zero (the pointer is over empty space).
    """
    if column[index] == 0:
        return None
    zeros_before = np.flatnonzero(column[:index] == 0)
    first = int(zeros_before[-1]) + 1 if zeros_before.size else 0
    zeros_after = np.flatnonzero(column[index + 1 :] == 0)
    last = index + int(zeros_after[0]) if zeros_after.size else len(column) - 1
    return first, last


class StimWidget(TimeseriesWidget):
    """A widget that displays a StimArray stimuli timeseries for neurons.

    The colormap used to represent the intensity of the stimuli depends on the type of stimulus:
    - Excitatory stimuli are shown in a magenta colormap.
    - Inhibitory stimuli are shown in a blue colormap.
    - External stimuli (which can be both positive and negative) are shown in a diverging
    colormap ranging from purple (negative) to green (positive).
    """

    # Grid lines use z=199. Keep every stimulus representation above them.
    STIMULUS_Z = 200.0
    MINIMUM_SIZE_Z = 210.0
    HOVER_Z = 300.0
    SUPPORTS_DETACH = False

    def __init__(
        self,
        controller: pygfx.Controller,
        menu: QtWidgets.QMenu | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(controller, parent)
        self.context_menu = menu if menu is not None else QtWidgets.QMenu(self)
        self.preview_action = QtGui.QAction("Preview vector graphic…", self)
        self.context_menu.addAction(self.preview_action)
        self.preview_action.setEnabled(False)
        self.preview_action.triggered.connect(self.preview_stimuli)
        self._stim_name = ""
        self.subplot.title = "Stimuli"
        self.hover_slice_graphic: fpl.Graphic | None = None
        self._hover_pos_cache: tuple[int, float, float] | None = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.stim_type = StimType.Excitation
        self._background_stim_arrays: list[tuple[np.ndarray, str, float, StimType]] = []
        self.keep_short_stimuli_visible = True
        self.minimum_stimulus_pixels = 4.0
        self.minimum_size_graphic: MinimumStimulusOverlay | None = None
        self._background_minimum_size_graphics: list[MinimumStimulusOverlay] = []

    def contextMenuEvent(self, event) -> None:
        self.context_menu.exec(event.globalPos())

    def _configure_time_scaled_image(
        self,
        image_graphic: fpl.ImageGraphic,
        *,
        dt_ms: float | None = None,
        time_index: int = 0,
        y_offset: float = 0.0,
        z_offset: float = STIMULUS_Z,
    ) -> None:
        if dt_ms is None:
            dt_ms = self.dt_ms
        image_graphic.offset = (
            time_index * dt_ms + 0.5 * dt_ms,
            y_offset,
            z_offset,
        )
        image_graphic.world_object.world.scale = (dt_ms, 1.0, 1.0)

    def _hover_slice_position(self) -> tuple[int, float, float]:
        if self.mouse_pos is None:
            return 0, 0.0, self.HOVER_Z
        time_index = 0
        if self.n_timestamps > 0 and self.dt_ms > 0.0:
            time_index = int(
                np.clip(
                    np.floor(self.mouse_pos[0] / self.dt_ms),
                    0,
                    self.n_timestamps - 1,
                )
            )
        return time_index, float(np.rint(self.mouse_pos[1])), self.HOVER_Z

    def _update_hover_slice_position(self) -> None:
        if self.hover_slice_graphic is None:
            return
        pos = self._hover_slice_position()
        if pos == self._hover_pos_cache:
            return
        self._hover_pos_cache = pos
        time_index, y_offset, z_offset = pos
        self._configure_time_scaled_image(
            self.hover_slice_graphic,
            time_index=time_index,
            y_offset=y_offset,
            z_offset=z_offset,
        )

    def set_data(self, stim_array: StimArray | None = None) -> None:
        self.clear_data()
        if stim_array is None:
            return
        self._stim_name = stim_array.name
        self.preview_action.setEnabled(True)
        self.data_values = stim_array.values
        self.stim_type = stim_array.stim_type
        self.vmin = 0.0
        self.vmax = np.max(np.abs(stim_array.values))
        if self.vmax == 0:
            self.vmax = 1.0
        if self.stim_type == StimType.External:
            self.vmin = -self.vmax
        self.units = "" if stim_array.units == "1" else stim_array.units
        self.dt_ms = float(stim_array.dt) * 1000.0  # convert to milliseconds
        cmap = get_cmap(self.stim_type)
        self.graphic = self.subplot.add_image(
            stim_array.values.T,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=cmap,
        )
        # respect the colormap alpha so zero-valued stim is transparent (otherwise it
        # is drawn as opaque white, hiding the reduced-opacity background stims)
        apply_cmap_alpha(self.graphic, cmap)
        self._configure_time_scaled_image(self.graphic)
        self.minimum_size_graphic = self._add_minimum_size_graphic(
            stim_array.values, self.dt_ms, self.stim_type
        )
        self._all_neuron_ids = list(range(stim_array.values.shape[1]))
        self._rebuild_visible_neuron_mapping()
        self.n_timestamps = stim_array.values.shape[0]
        self.post_set_data()

    def _visible_values(self) -> np.ndarray:
        """The real data narrowed to just the currently-visible neurons' columns,
        compacted (no gaps, no padding) - shape (n_timesteps, n_visible). The
        real, untouched data lives in self.data_values."""
        assert self.data_values is not None
        if not self._visible_neuron_ids:
            return np.zeros(
                (self.data_values.shape[0], 0), dtype=self.data_values.dtype
            )
        return self.data_values[:, self._visible_neuron_ids]

    def _apply_visible_neuron_layout(self) -> None:
        if (
            self.graphic is None
            or self.data_values is None
            or not self._visible_neuron_ids
        ):
            # Leave the existing graphic in place rather than trying to build a
            # zero-height image if every neuron is currently hidden.
            return
        # ImageGraphic's buffer is fixed-size at creation (like the lasso-line
        # buffer issue) - it can't be resized via `.data = ...`, so rebuild it
        # at the new, smaller shape rather than zero-padding a full-size one
        # (which would leave dead/blank space where hidden neurons used to be).
        visible_values = self._visible_values()
        self.subplot.remove_graphic(self.graphic)
        cmap = get_cmap(self.stim_type)
        self.graphic = self.subplot.add_image(
            visible_values.T,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=cmap,
        )
        apply_cmap_alpha(self.graphic, cmap)
        self._configure_time_scaled_image(self.graphic)
        if self.minimum_size_graphic is not None:
            self.subplot.remove_graphic(self.minimum_size_graphic)
            self.minimum_size_graphic = None
        self.minimum_size_graphic = self._add_minimum_size_graphic(
            visible_values, self.dt_ms, self.stim_type
        )

    def post_set_data(self):
        super().post_set_data()
        if self.grid_lines is not None:
            self.grid_lines.update_dt_ms(self.dt_ms)

    def clear_data(self) -> None:
        self.preview_action.setEnabled(False)
        self._stim_name = ""
        self._background_stim_arrays = []
        self.minimum_size_graphic = None
        self._background_minimum_size_graphics = []
        super().clear_data()

    def preview_stimuli(self) -> QtWidgets.QDialog | None:
        """Preview stimulus blocks as vectors, using the selection or current view."""
        if self.data_values is None or self.graphic is None or self.controller is None:
            return None

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.collections import PolyCollection
        from matplotlib.figure import Figure

        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"Stimulus preview: {self._stim_name}")
        dialog.resize(1000, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        figure = Figure(layout="tight", facecolor="none")
        # Omit background patches entirely, including when savefig settings
        # override the figure's facecolor during toolbar export.
        figure.patch.set_visible(False)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, dialog))
        layout.addWidget(canvas)
        ax = figure.add_subplot(111, facecolor="none")
        ax.patch.set_visible(False)

        arrays: list[tuple[np.ndarray, float, StimType, float]] = []
        if self.show_background_graphics:
            arrays.extend(
                (values, dt_ms, stim_type, 0.25)
                for values, _, dt_ms, stim_type in self._background_stim_arrays
            )
        arrays.append((self.data_values, self.dt_ms, self.stim_type, 1.0))
        for values, dt_ms, stim_type, alpha in arrays:
            vertices = []
            amplitudes = []
            # Merge adjacent equal samples into one vector rectangle per pulse.
            for neuron, row in enumerate(values.T):
                edges = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
                for start, stop in zip(edges[:-1], edges[1:], strict=True):
                    if row[start] == 0:
                        continue
                    left, right = start * dt_ms, stop * dt_ms
                    vertices.append(
                        [
                            (left, neuron - 0.5),
                            (right, neuron - 0.5),
                            (right, neuron + 0.5),
                            (left, neuron + 0.5),
                        ]
                    )
                    amplitudes.append(row[start])
            vmax = float(np.max(np.abs(values))) or 1.0
            normalized = np.asarray(amplitudes) / vmax
            if stim_type == StimType.External:
                normalized = (normalized + 1.0) / 2.0
            collection = PolyCollection(
                vertices,
                facecolors=get_cmap(stim_type, alpha=alpha)(normalized),
                edgecolors="none",
                rasterized=False,
            )
            ax.add_collection(collection)

        state = self.controller._get_camera_state()
        cx, cy = state["position"][:2]
        ax.set_xlim(cx - state["width"] / 2, cx + state["width"] / 2)
        ax.set_ylim(cy - state["height"] / 2, cy + state["height"] / 2)
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
        ax.set_ylabel("Neuron index")
        ax.set_title(self._stim_name)
        dialog.show()
        return dialog

    def _add_minimum_size_graphic(
        self,
        values: np.ndarray,
        dt_ms: float,
        stim_type: StimType,
        *,
        alpha: float = 1.0,
    ) -> MinimumStimulusOverlay | None:
        """Overlay nonzero samples with fixed-screen-width, full-row-height
        vertical line segments (see MinimumStimulusOverlay).

        Colors render at a fixed alpha, decoupled from magnitude (mirroring
        _stim_pulse_rgba in stdp_synaptic_tuning_dialog.py): the whole point
        of this overlay is to guarantee a stimulus stays visible regardless
        of amplitude or zoom, so a low-amplitude sample must not fade toward
        transparent the way it legitimately does in the main heatmap image -
        that would silently defeat the guarantee for anything but
        near-maximum-amplitude pulses.
        """
        if not np.any(values):
            return None
        graphic = MinimumStimulusOverlay(
            values,
            dt_ms,
            stim_type,
            thickness=self.minimum_stimulus_pixels,
            z=self.MINIMUM_SIZE_Z,
            alpha=alpha,
        )
        self.subplot.add_graphic(graphic, center=False)
        graphic.visible = self.keep_short_stimuli_visible
        return graphic

    def set_keep_short_stimuli_visible(self, visible: bool) -> None:
        """Toggle the four-pixel minimum display size without changing stim data."""
        self.keep_short_stimuli_visible = visible
        if self.minimum_size_graphic is not None:
            self.minimum_size_graphic.visible = visible
        for graphic in self.background_graphics:
            graphic.visible = self.show_background_graphics
        for graphic in self._background_minimum_size_graphics:
            graphic.visible = visible and self.show_background_graphics

    def set_background_data(self, stim_array: StimArray | None = None) -> None:
        if stim_array is None:
            return
        vmax = np.max(np.abs(stim_array.values))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax if stim_array.stim_type == StimType.External else 0
        cmap = get_cmap(stim_array.stim_type, alpha=0.25)
        image_graphic = self.subplot.add_image(
            stim_array.values.T, vmin=vmin, vmax=vmax, cmap=cmap
        )
        self._configure_time_scaled_image(
            image_graphic,
            dt_ms=float(stim_array.dt) * 1000.0,
        )
        apply_cmap_alpha(image_graphic, cmap)
        image_graphic.visible = self.show_background_graphics
        self.background_graphics.append(image_graphic)
        minimum_graphic = self._add_minimum_size_graphic(
            stim_array.values,
            float(stim_array.dt) * 1000.0,
            stim_array.stim_type,
            alpha=0.25,
        )
        if minimum_graphic is not None:
            # _add_minimum_size_graphic only knows about "keep short inputs
            # visible"; a background array's indicator must also respect
            # "Show all input arrays", or switching input arrays brings every
            # other array's short-input marks back while that is off (#585).
            minimum_graphic.visible = (
                self.keep_short_stimuli_visible and self.show_background_graphics
            )
            self._background_minimum_size_graphics.append(minimum_graphic)
        units_str = "" if stim_array.units == "1" else str(stim_array.units)
        self._background_stim_arrays.append(
            (
                stim_array.values,
                units_str,
                float(stim_array.dt) * 1000.0,
                stim_array.stim_type,
            )
        )

    def set_background_data_visible(self, visible: bool) -> None:
        self.show_background_graphics = visible
        for graphic in self.background_graphics:
            graphic.visible = self.show_background_graphics
        for graphic in self._background_minimum_size_graphics:
            graphic.visible = (
                self.show_background_graphics and self.keep_short_stimuli_visible
            )

    def autoscale_colormap(self) -> None:
        """Autoscale the colormap based on the current data."""
        if self.graphic is not None:
            max_value = np.max(np.abs(self.graphic.data.value))
            if max_value == 0:
                max_value = 1.0
            self.vmax = max_value
            self.graphic.vmax = self.vmax
            if self.stim_type == StimType.External:
                self.vmin = -self.vmax
                self.graphic.vmin = self.vmin

    def _real_slice_to_display_slice(self, real_slice: slice) -> slice | None:
        """Translate a slice of real neuron ids into the (contiguous) slice of
        display rows they currently occupy, or None if any of them is hidden or
        the resulting rows aren't contiguous (shouldn't happen for the
        single-neuron slices live stim editing always uses)."""
        display_rows = []
        for real_id in range(real_slice.start, real_slice.stop):
            row = self._neuron_id_to_display_row.get(real_id)
            if row is None:
                return None
            display_rows.append(row)
        if not display_rows:
            return slice(0, 0)
        if display_rows != list(range(display_rows[0], display_rows[-1] + 1)):
            return None
        return slice(display_rows[0], display_rows[-1] + 1)

    def update_data_slice(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray | float
    ) -> None:
        """Update a slice of the existing data in the widget.

        Values can be a 2d array matching the shape of the slice defined by neuron_slice and time_slice,
        or a single float value to set the entire slice to. ``neuron_slice`` is expressed
        in real neuron ids; it's translated to the current display row(s) before touching
        the texture, since hidden (greyed) neurons compact the displayed rows.
        """
        texture_values = values.T if isinstance(values, np.ndarray) else values
        display_slice = self._real_slice_to_display_slice(neuron_slice)
        if self.graphic is not None and display_slice is not None:
            # directly update the relevant data in the graphic's texture
            self.graphic.data[display_slice, time_slice] = texture_values
            self.autoscale_colormap()
            # Rebuilding the screen-size overlay scans the whole array and
            # recreates a scatter graphic. Skip that while the overlay is hidden
            # so interactive drawing does not pay the cost for an invisible
            # graphic; it is refreshed on the next edit once shown again.
            if self.keep_short_stimuli_visible:
                if self.minimum_size_graphic is not None:
                    self.subplot.remove_graphic(self.minimum_size_graphic)
                    self.minimum_size_graphic = None
                if self.data_values is not None:
                    self.minimum_size_graphic = self._add_minimum_size_graphic(
                        self._visible_values(), self.dt_ms, self.stim_type
                    )

    def set_hover_slice(self, data: np.ndarray | None) -> None:
        """Set the slice of data (if any) to be displayed at the current mouse position."""
        if self.hover_slice_graphic is not None:
            self.subplot.remove_graphic(self.hover_slice_graphic)
        if data is None:
            self.hover_slice_graphic = None
            self.subplot.canvas.unsetCursor()
            return
        self.subplot.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.hover_slice_graphic = self.subplot.add_image(
            data.T, offset=(0, 0, self.HOVER_Z)
        )
        self._hover_pos_cache = None
        self._update_hover_slice_position()

    def _position_hover_slice(self) -> None:
        # Snap the preview to the hovered time/neuron cell instead of following
        # the raw pointer, so it does not jitter or drift within a cell.
        self._update_hover_slice_position()

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        super().pointer_move(ev)
        if (
            self.mouse_pos is None
            or self.data_values is None
            or not self._visible_neuron_ids
        ):
            return
        t_idx_fg = np.clip(
            math.floor(self.mouse_pos[0] / self.dt_ms),
            0,
            self.data_values.shape[0] - 1,
        )
        display_row = np.clip(
            math.floor(self.mouse_pos[1] + 0.5), 0, self.n_neurons - 1
        )
        neuron_id = self._visible_neuron_ids[int(display_row)]
        raw_fg = float(self.data_values[t_idx_fg, neuron_id])
        best_val = -raw_fg if self.stim_type == StimType.Inhibition else raw_fg
        best_units = self.units
        best_span = _active_span(self.data_values[:, neuron_id], t_idx_fg)
        best_dt_ms = self.dt_ms
        for arr, units, dt_ms, stim_type in self._background_stim_arrays:
            t_idx = np.clip(
                math.floor(self.mouse_pos[0] / dt_ms),
                0,
                arr.shape[0] - 1,
            )
            n_idx_bg = int(np.clip(neuron_id, 0, arr.shape[1] - 1))
            raw_val = float(arr[t_idx, n_idx_bg])
            val = -raw_val if stim_type == StimType.Inhibition else raw_val
            if abs(val) > abs(best_val):
                best_val = val
                best_units = units
                best_span = _active_span(arr[:, n_idx_bg], int(t_idx))
                best_dt_ms = dt_ms
        text = f"Neuron {neuron_id}\n{best_val:.3g} {best_units}"
        if best_span is not None:
            first, last = best_span
            start_ms = first * best_dt_ms
            end_ms = (last + 1) * best_dt_ms
            text += f"\nStart: {start_ms:.6g} ms\nEnd: {end_ms:.6g} ms"
        self.text.set_pos(self.mouse_pos, text)
