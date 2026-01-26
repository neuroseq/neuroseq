from __future__ import annotations

import fastplotlib as fpl
import pygfx
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets


class FPLWidget(QtWidgets.QWidget):
    """A base class for widgets that use fastplotlib to display data."""

    def __init__(
        self,
        controller: pygfx.Controller | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.figure: fpl.Figure = fpl.Figure(
            controllers=controller, canvas_kwargs={"parent": self}
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.widget = self.figure.show()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        self._set_light_mode()
        self.setMinimumSize(QtCore.QSize(100, 100))
        self.resize(800, 600)

    def hide_controls(self) -> None:
        """Hide any fastplotlib toolbars/menus for minimal displays."""
        for attr in ("toolbar", "_toolbar", "menubar", "_menubar"):
            widget = getattr(self.widget, attr, None)
            if widget is not None:
                widget.hide()

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

    def clear_data(self) -> None:
        self.subplot.clear()
        for dock in self.subplot.docks.values():
            dock.clear()
        self.subplot.grids = False
        self.subplot.axes.y.ticks = {}
        self.subplot.axes.x.ticks = {}
        self.subplot.axes.y.points.visible = False
        self.subplot.axes.y.line.visible = False
        self.subplot.axes.x.points.visible = False
        self.subplot.axes.x.line.visible = False
        self.subplot.toolbar = False

    def closeEvent(self, event):
        if self.figure is not None:
            # stop the renderer before closing the widget otherwise it may try to render to the deleted object
            self.figure.close()
            self.figure = None
        super().closeEvent(event)


class FullScreenFPLWidget(FPLWidget):
    """FPLWidget that sizes itself to the available screen on show."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        print("Available size:")
        print(available.size())
        self.setFixedSize(available.height(), available.width())
