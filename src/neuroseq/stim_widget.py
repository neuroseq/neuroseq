from __future__ import annotations

import brian2 as b2
import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import PrettyPrintEnum


class StimType(PrettyPrintEnum):
    Excitation = 1
    Inhibition = 2
    External = 3


class StimWidget(TimeseriesWidget):
    """A widget that displays a brian2.TimedArray stimuli timeseries for neurons.

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
        self.cmap = "chrisluts:I_Magenta"
        self.stim_type = StimType.Excitation

    def set_data(
        self, stims: b2.TimedArray | None = None, stim_type=StimType.Excitation
    ) -> None:
        self.subplot.clear()
        if stims is None:
            return
        self.data_values = stims.values
        self.vmin = 0
        self.vmax = np.max(np.abs(stims.values))
        self.stim_type = stim_type
        match self.stim_type:
            case StimType.Excitation:
                self.cmap = "chrisluts:I_Magenta"
            case StimType.Inhibition:
                self.cmap = "chrisluts:I_Blue"
            case StimType.External:
                self.vmin = -self.vmax
                self.cmap = "colorbrewer:PRGn"
        self.units = b2.get_unit(stims.dim).__str__()
        self.graphic = self.subplot.add_image(
            stims.values.T,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.cmap,
            offset=(0.5, 0, 0),
        )
        # todo: properly support dt_ms != 1 (currently we share the camera with other widgets, but add_image has fixed integer x-axis spacing, so the bounds of the shared camera is only correct if dt_ms=1)
        self.dt_ms = stims.dt * 1000.0  # convert to milliseconds
        self.n_neurons = stims.values.shape[1]
        self.n_timestamps = stims.values.shape[0]
        self.controller.max_height = max(self.controller.max_height, self.n_neurons)
        self.controller.max_width = max(
            self.controller.max_width, self.n_timestamps * self.dt_ms
        )
        self.post_set_data()

    def autoscale_colormap(self) -> None:
        """Autoscale the colormap based on the current data."""
        if self.graphic is not None:
            max_value = np.max(np.abs(self.graphic.data.value))
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
