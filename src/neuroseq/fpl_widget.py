from __future__ import annotations

import logging
import os

import fastplotlib as fpl
import pygfx
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme as theme_module


class FPLWidget(QtWidgets.QWidget):
    """A base class for widgets that use fastplotlib to display data."""

    def __init__(
        self,
        controller: pygfx.Controller | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self._theme = theme_module.LIGHT
        canvas_kwargs: dict[str, object] = {"parent": self}
        if os.environ.get("NEUROSEQ_HEADLESS_TEST", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            # Use rendercanvas bitmap presentation for headless Qt test runs.
            canvas_kwargs["present_method"] = "bitmap"
        self.figure: fpl.Figure = fpl.Figure(
            controllers=controller, canvas_kwargs=canvas_kwargs
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.widget = self.figure.show()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        # initially replace renderer with a guarded one that checks the viewport size
        # and restores the fastplotlib renderer once the viewport size is positive
        self._fpl_figure_render = None
        try:
            figure_render = self.figure._render
            self.figure.canvas.request_draw(self._fpl_guarded_render)
            self._fpl_figure_render = figure_render
        except (AttributeError, TypeError) as exc:
            self.logger.debug(
                "Could not install fastplotlib startup draw guard (%s); harmless "
                "wgpu viewport warnings may appear during startup.",
                exc,
            )
        self._apply_fpl_theme()
        self.setMinimumSize(QtCore.QSize(100, 100))
        self.resize(800, 600)

    def _fpl_canvas_ready(self) -> bool:
        """True once the render canvas is large enough to draw without producing a negative viewport."""
        if self.figure is None:
            return False
        width, height = self.figure.canvas.get_physical_size()
        if width <= 0 or height <= 0:
            return False
        for subplot in self.figure:
            *_, render_width, render_height = subplot.frame.get_render_rect()
            if render_width <= 0 or render_height <= 0:
                return False
        return True

    def _fpl_guarded_render(self, draw: bool = True) -> None:
        if self.figure is None or self._fpl_figure_render is None:
            return
        if not self._fpl_canvas_ready():
            return
        # Valid viewport: restore the real fastplotlib renderer for subsequent
        # frames and render this one.
        self.figure.canvas.request_draw(self._fpl_figure_render)
        self._fpl_figure_render(draw=draw)

    def _set_fpl_rendering(self, enabled: bool) -> None:
        """Pause or resume the fastplotlib render scheduler."""
        if self.figure is None:
            return
        try:
            if enabled:
                self.figure.canvas.set_update_mode("ondemand")
                self.figure.canvas.request_draw()
            else:
                self.figure.canvas.set_update_mode("manual")
        except (AttributeError, TypeError, RuntimeError) as exc:
            self.logger.debug("Could not toggle fastplotlib rendering (%s).", exc)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._set_fpl_rendering(True)

    def hideEvent(self, event) -> None:
        self._set_fpl_rendering(False)
        super().hideEvent(event)

    def hide_controls(self) -> None:
        """Hide any fastplotlib toolbars/menus for minimal displays."""
        for attr in ("toolbar", "_toolbar", "menubar", "_menubar"):
            widget = getattr(self.widget, attr, None)
            if widget is not None:
                widget.hide()

    def apply_theme(self, theme: str) -> None:
        """Switch this widget's canvas between light and dark themes."""
        self._theme = theme_module.normalize_theme(theme)
        self._apply_fpl_theme()

    def _apply_fpl_theme(self) -> None:
        # some of these currently require accessing private attributes: should refactor when fastplotlib allows setting these colors directly
        background, foreground = theme_module.fpl_colors(self._theme)
        # background_color setter splats its argument, so a multi-character hex
        # string must be wrapped in a tuple (a single corner color for all four).
        self.subplot.background_color = (background,)
        self.subplot.axes.colors = (
            foreground,
            foreground,
            foreground,
        )
        self.subplot.axes.grids.xy.axis_color = foreground
        self.subplot.axes.grids.xy.major_color = foreground
        self.subplot.axes.grids.xy.minor_color = foreground
        self.subplot.axes._x._text.material.color = foreground
        self.subplot.axes._y._text.material.color = foreground
        self.subplot.axes._x._line.material.color = foreground
        self.subplot.axes._y._line.material.color = foreground
        self.subplot.axes._x._points.material.color = foreground
        self.subplot.axes._y._points.material.color = foreground

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
            self._set_fpl_rendering(False)
            app = QtWidgets.QApplication.instance()
            if app is not None:
                try:
                    # wait for any pending renders to complete
                    self.figure.renderer.device._poll_wait()
                except Exception as exc:
                    self.logger.debug("Could not poll GPU device on close (%s).", exc)
                app.processEvents(
                    QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )
            # stop the renderer before closing the widget otherwise it may try to render to the deleted object
            if hasattr(self.figure, "close"):
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
        if not self.isWindow():
            return
        window = self.window()
        window_handle = window.windowHandle() if window is not None else None
        screen = window_handle.screen() if window_handle is not None else None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        scale = screen.devicePixelRatio()
        target_width = int((available.width() - 400) * scale)
        target_height = int((available.height() - 400) * scale)
        self.setFixedSize(
            target_width,
            target_height,
        )
