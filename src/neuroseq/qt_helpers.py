"""Generic Qt widgets and window helpers shared across NeuroSeq dialogs."""

from __future__ import annotations

import sys
from collections.abc import Callable

from qtpy import QtCore
from qtpy import QtWidgets


class SelfRaisingDialog(QtWidgets.QDialog):
    """A non-modal QDialog that raises itself above its siblings whenever it
    is activated (clicked into focus).

    Several window managers raise an entire transient-dialog chain as one
    unit on click, rather than reordering siblings within it - so clicking
    one of several sibling popups (all transient for the same parent
    routine dialog: the parameters dialog, its plots window, and its
    stim-pattern heatmap window) does not bring it in front of the others;
    the whole group just moves together, at whatever relative order the
    window manager already had. Explicitly re-raising on ActivationChange
    bypasses that window-manager heuristic and enforces the expected
    last-clicked-is-on-top order.
    """

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.raise_()


def set_native_minimized_position(hwnd: int, x: int, y: int) -> bool:
    """Set a Windows iconic-window position without changing restore geometry."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class WindowPlacement(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.UINT),
            ("flags", wintypes.UINT),
            ("show_cmd", wintypes.UINT),
            ("min_position", wintypes.POINT),
            ("max_position", wintypes.POINT),
            ("normal_position", wintypes.RECT),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_placement = user32.GetWindowPlacement
    get_placement.argtypes = [wintypes.HWND, ctypes.POINTER(WindowPlacement)]
    get_placement.restype = wintypes.BOOL
    set_placement = user32.SetWindowPlacement
    set_placement.argtypes = [wintypes.HWND, ctypes.POINTER(WindowPlacement)]
    set_placement.restype = wintypes.BOOL
    placement = WindowPlacement()
    placement.length = ctypes.sizeof(WindowPlacement)
    if not get_placement(hwnd, ctypes.byref(placement)):
        return False
    placement.flags |= 0x0001  # WPF_SETMINPOSITION
    placement.min_position = wintypes.POINT(int(x), int(y))
    return bool(set_placement(hwnd, ctypes.byref(placement)))


class WheelToScrollFilter(QtCore.QObject):
    """Redirect wheel events from value widgets to a scroll area.

    Spin boxes and combo boxes change their value on a wheel scroll, which makes
    scrolling the parameter panel accidentally edit fields. This filter forwards
    the wheel to the scroll area so the panel scrolls and values stay put.
    """

    def __init__(self, scroll_area: QtWidgets.QScrollArea) -> None:
        super().__init__(scroll_area)
        self._scroll_area = scroll_area

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if event is not None and event.type() == QtCore.QEvent.Type.Wheel:
            QtWidgets.QApplication.sendEvent(self._scroll_area.viewport(), event)
            return True
        return super().eventFilter(obj, event)


def wrap_in_scroll_area(
    widget: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None
) -> QtWidgets.QScrollArea:
    """Wrap a plot widget (already given its own minimum size) in a
    QScrollArea, shared by every routine dialog's plots window.

    setWidgetResizable(True) still lets the widget fill extra space when
    the window is larger than its minimum - but Qt never shrinks it below
    that minimum, so a window made smaller than the plot needs (e.g. a
    short screen) scrolls the content instead of squeezing the figure
    until axis labels overlap or get clipped off, which is what putting
    the widget directly in a layout does.
    """
    scroll = QtWidgets.QScrollArea(parent)
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    return scroll


class StableSizeTabWidget(QtWidgets.QTabWidget):
    """A QTabWidget whose size hint doesn't change when switching tabs.

    QTabWidget bases sizeHint()/minimumSizeHint() on the *current* page
    alone, so an enclosing QSplitter reflows every time the user switches to
    a tab whose content has a different natural size - e.g. the Cell
    segmentation and Video tabs visibly bounced around on every switch.
    Basing it on the largest page instead keeps the splitter's allocation
    stable regardless of which tab is showing.
    """

    def _combined_page_size(
        self, hint_of: Callable[[QtWidgets.QWidget], QtCore.QSize]
    ) -> QtCore.QSize:
        width = 0
        height = 0
        for index in range(self.count()):
            page = self.widget(index)
            if page is None:
                continue
            size = hint_of(page)
            width = max(width, size.width())
            height = max(height, size.height())
        bar = self.tabBar()
        bar_size = bar.sizeHint() if bar is not None else QtCore.QSize(0, 0)
        # The default (North) tab position stacks the bar above the pages.
        return QtCore.QSize(max(width, bar_size.width()), height + bar_size.height())

    def sizeHint(self) -> QtCore.QSize:
        return self._combined_page_size(lambda w: w.sizeHint())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self._combined_page_size(lambda w: w.minimumSizeHint())
