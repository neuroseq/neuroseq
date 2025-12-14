from __future__ import annotations

import fastplotlib as fpl
import pygfx
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets


class FPLWidget(QtWidgets.QWidget):
    """A Qwidget implementing base class for widgets that use fastplotlib to display data."""

    def __init__(
        self,
        controller: pygfx.Controller | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.figure: fpl.Figure = fpl.Figure( # all of these objects hacve oen figure - parent self renders it correctly
            # - the subplot then controls the drawn part/graphic
            controllers=controller, canvas_kwargs={"parent": self}
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # no margin, no spacing
        layout.setSpacing(0)
        self.widget = self.figure.show()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        self._set_light_mode()
        self.setMinimumSize(QtCore.QSize(100, 100))
        self.resize(800, 600) # I am not sure why this does not resize e.g. the neuron ones, those adapt to screen size.
        # is this important or could this be changed?  # I think this is ignored by TimeseiresWidget because
        # the vertical space is limited/in a QLayout so it is reduced

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

    # todo: cellpose equivalent? Note these basically invert the grid/library colours (default black text on white, swap to white on black)
    # but they do mean things then appear which we might not need to appear fro e.g. CellposeOverlayWidget

    def clear_data(self) -> None:
        self.subplot.clear()
        for dock in self.subplot.docks.values():
            dock.clear()
        self.subplot.grids = False
        self.subplot.axes.y.ticks = {}
        self.subplot.axes.x.ticks = {}
        # Keep axes visible by default; subclasses can override visibility as needed
        self.subplot.axes.y.points.visible = True
        self.subplot.axes.y.line.visible = True
        self.subplot.axes.x.points.visible = True
        self.subplot.axes.x.line.visible = True
        self.subplot.toolbar = False

    def closeEvent(self, event):
        if self.figure is not None:
            # stop the renderer before closing the widget otherwise it may try to render to the deleted object
            self.figure.close()
            self.figure = None
        super().closeEvent(event)
