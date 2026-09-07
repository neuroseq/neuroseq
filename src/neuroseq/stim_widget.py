from __future__ import annotations

import math

import fastplotlib as fpl
import numpy as np
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import get_cmap


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
        self.subplot.title = "Stimuli"
        self.hover_slice_graphic: fpl.Graphic | None = None
        self._hover_pos_cache: tuple[int, float, float] | None = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.stim_type = StimType.Excitation
        self._background_stim_arrays: list[tuple[np.ndarray, str, float, StimType]] = []
        self.keep_short_stimuli_visible = True
        self.minimum_stimulus_pixels = 4.0
        self.minimum_size_graphic: fpl.ScatterGraphic | None = None
        self._background_minimum_size_graphics: list[fpl.ScatterGraphic] = []

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

    @staticmethod
    def _apply_cmap_alpha(image_graphic: fpl.ImageGraphic, cmap: object) -> None:
        # set the graphic material texture colormap values directly, as add_image ignores any alpha values
        image_graphic._material.map.texture.data[:] = cmap(  # type: ignore[operator]
            np.linspace(0, 255, 256, dtype=int)
        ).astype(np.float32)
        image_graphic._material.map.texture.update_range((0, 0, 0), size=(256, 1, 1))

    def set_data(self, stim_array: StimArray | None = None) -> None:
        self.clear_data()
        if stim_array is None:
            return
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
        self._apply_cmap_alpha(self.graphic, cmap)
        self._configure_time_scaled_image(self.graphic)
        self.minimum_size_graphic = self._add_minimum_size_graphic(
            stim_array.values, self.dt_ms, self.stim_type
        )
        self.n_neurons = stim_array.values.shape[1]
        self.n_timestamps = stim_array.values.shape[0]
        self.post_set_data()

    def post_set_data(self):
        super().post_set_data()
        if self.grid_lines is not None:
            self.grid_lines.update_dt_ms(self.dt_ms)

    def clear_data(self) -> None:
        self._background_stim_arrays = []
        self.minimum_size_graphic = None
        self._background_minimum_size_graphics = []
        super().clear_data()

    def _add_minimum_size_graphic(
        self,
        values: np.ndarray,
        dt_ms: float,
        stim_type: StimType,
        *,
        alpha: float = 1.0,
    ) -> fpl.ScatterGraphic | None:
        """Overlay nonzero samples with square, screen-sized markers."""
        time_indices, neuron_indices = np.nonzero(values)
        if time_indices.size == 0:
            return None
        sample_values = values[time_indices, neuron_indices]
        vmax = float(np.max(np.abs(values))) or 1.0
        if stim_type == StimType.External:
            normalized_values = (sample_values + vmax) / (2.0 * vmax)
        else:
            normalized_values = np.abs(sample_values) / vmax
        colors = get_cmap(stim_type, alpha=alpha)(normalized_values).astype(np.float32)
        positions = np.column_stack(
            (
                (time_indices + 0.5) * dt_ms,
                neuron_indices,
                np.full(time_indices.shape, self.MINIMUM_SIZE_Z),
            )
        ).astype(np.float32)
        graphic = self.subplot.add_scatter(
            positions,
            colors=colors,
            sizes=self.minimum_stimulus_pixels,
            size_space="screen",
            uniform_size=True,
            markers="s",
            uniform_marker=True,
            edge_width=0.0,
        )
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
        self._apply_cmap_alpha(image_graphic, cmap)
        image_graphic.visible = self.show_background_graphics
        self.background_graphics.append(image_graphic)
        minimum_graphic = self._add_minimum_size_graphic(
            stim_array.values,
            float(stim_array.dt) * 1000.0,
            stim_array.stim_type,
            alpha=0.25,
        )
        if minimum_graphic is not None:
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

    def update_data_slice(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray | float
    ) -> None:
        """Update a slice of the existing data in the widget.

        Values can be a 2d array matching the shape of the slice defined by neuron_slice and time_slice,
        or a single float value to set the entire slice to.
        """
        texture_values = values.T if isinstance(values, np.ndarray) else values
        if self.graphic is not None:
            # directly update the relevant data in the graphic's texture
            self.graphic.data[neuron_slice, time_slice] = texture_values
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
                        self.data_values, self.dt_ms, self.stim_type
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
            or (
                not self._background_stim_arrays
                and self.stim_type != StimType.Inhibition
            )
        ):
            return
        t_idx_fg = np.clip(
            math.floor(self.mouse_pos[0] / self.dt_ms),
            0,
            self.data_values.shape[0] - 1,
        )
        n_idx = np.clip(
            math.floor(self.mouse_pos[1] + 0.5),
            0,
            self.data_values.shape[1] - 1,
        )
        raw_fg = float(self.data_values[t_idx_fg, n_idx])
        best_val = -raw_fg if self.stim_type == StimType.Inhibition else raw_fg
        best_units = self.units
        for arr, units, dt_ms, stim_type in self._background_stim_arrays:
            t_idx = np.clip(
                math.floor(self.mouse_pos[0] / dt_ms),
                0,
                arr.shape[0] - 1,
            )
            n_idx_bg = np.clip(
                math.floor(self.mouse_pos[1] + 0.5),
                0,
                arr.shape[1] - 1,
            )
            raw_val = float(arr[t_idx, n_idx_bg])
            val = -raw_val if stim_type == StimType.Inhibition else raw_val
            if abs(val) > abs(best_val):
                best_val = val
                best_units = units
        self.text.set_pos(self.mouse_pos, f"{best_val:.3g} {best_units}")
