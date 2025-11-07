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

    Assumes that the stimuli are non-negative.
    Applies global normalisation to the displayed stimuli such that the largest stimulus has an amplitude of 1.0.
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

    def set_data(
        self, stims: b2.TimedArray | None = None, stim_type=StimType.Excitation
    ) -> None:
        self.subplot.clear()
        if stims is None:
            return
        self.dt = stims.dt
        self.hover_values = stims.values
        vmin = 0
        vmax = np.max(np.abs(stims.values))
        match stim_type:
            case StimType.Excitation:
                cmap = "chrisluts:I_Magenta"
            case StimType.Inhibition:
                cmap = "chrisluts:I_Blue"
            case StimType.External:
                vmin = -vmax
                cmap = "colorbrewer:PRGn"
        self.units = b2.get_unit(stims.dim).__str__()
        self.graphic = self.subplot.add_image(
            stims.values.T,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            offset=(0.5, 0, 0),
        )
        self.n_neurons = stims.values.shape[1]
        self.n_timestamps = stims.values.shape[0]
        self.controller.max_height = max(self.controller.max_height, self.n_neurons)
        self.controller.max_width = max(self.controller.max_width, self.n_timestamps)
        self.post_set_data()

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
