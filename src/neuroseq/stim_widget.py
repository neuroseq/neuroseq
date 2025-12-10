from __future__ import annotations

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

    def __init__(
        self,
        controller: pygfx.Controller,
        menu: QtWidgets.QMenu | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(controller, parent)
        self.context_menu = menu
        self.subplot.title = "Stimuli"
        self.hover_slice_graphic: fpl.Graphic | None = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.stim_type = StimType.Excitation

    def set_data(self, stim_array: StimArray | None = None) -> None:
        self.clear_data(reset_autoscale=False)
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
        self.graphic = self.subplot.add_image(
            stim_array.values.T,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=get_cmap(self.stim_type),
            offset=(0.5, 0, 0),
        )
        # todo: properly support dt_ms != 1 (currently we share the camera with other widgets, but add_image has fixed integer x-axis spacing, so the bounds of the shared camera is only correct if dt_ms=1)
        self.dt_ms = stim_array.dt * 1000.0  # convert to milliseconds
        self.n_neurons = stim_array.values.shape[1]
        self.n_timestamps = stim_array.values.shape[0]
        self.post_set_data()
        self.autoscale = False

    def set_background_data(self, stim_array: StimArray | None = None) -> None:
        if stim_array is None:
            return
        vmax = np.max(np.abs(stim_array.values))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax if stim_array.stim_type == StimType.External else 0
        cmap = get_cmap(stim_array.stim_type, alpha=0.25)
        image_graphic = self.subplot.add_image(
            stim_array.values.T, vmin=vmin, vmax=vmax, cmap=cmap, offset=(0.5, 0, 0)
        )
        # set the graphic material texture colormap values directly, as add_image ignores any alpha values
        image_graphic._material.map.texture.data[:] = cmap(
            np.linspace(0, 255, 256, dtype=int)
        ).astype(np.float32)
        image_graphic._material.map.texture.update_range((0, 0, 0), size=(256, 1, 1))
        image_graphic.visible = self.show_background_graphics
        self.background_graphics.append(image_graphic)

    def set_background_data_visible(self, visible: bool) -> None:
        self.show_background_graphics = visible
        for graphic in self.background_graphics:
            graphic.visible = self.show_background_graphics

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

    def set_hover_slice(self, data: np.ndarray | None) -> None:
        """Set the slice of data (if any) to be displayed at the current mouse position."""
        if self.hover_slice_graphic is not None:
            self.subplot.remove_graphic(self.hover_slice_graphic)
        if data is None:
            self.hover_slice_graphic = None
            self.subplot.canvas.unsetCursor()
            return
        self.subplot.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        if self.mouse_pos is not None:
            offset = (np.rint(self.mouse_pos[0]), np.rint(self.mouse_pos[1]), 100)
        else:
            offset = (0, 0, 100)
        self.hover_slice_graphic = self.subplot.add_image(data.T, offset=offset)
