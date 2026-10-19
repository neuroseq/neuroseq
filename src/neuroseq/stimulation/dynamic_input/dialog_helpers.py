"""Shared prompts and window behavior for dynamic-input routines."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from qtpy import QtCore
from qtpy import QtWidgets

if TYPE_CHECKING:
    from neuroseq.stimulation.stim_editor import StimEditor


def confirm_start_routine(dialog: QtWidgets.QWidget, editor: StimEditor) -> str:
    """Ask how to start a dynamic-input routine, shared by all three routine
    dialogs' Start handlers.

    A stim array with no stimuli at all needs no prompt - just start. When
    it already holds stimuli (including from a previous run of this or
    another routine), ask whether to clear it and start from the
    beginning, or continue appending after the last existing stimulus -
    which keeps a previous run's progress. "Continue" only trims any
    unsimulated tail first (a manually drawn pattern, or a routine stopped
    mid-run), same as reset_stim_array() - it never touches simulated
    content. Returns "clear", "continue", or "cancel".
    """
    if editor.current_stim_modifier is None:
        return "continue"
    if not editor.current_stim_array_has_any_data():
        return "continue"
    name = editor.current_stim_modifier.stim_array.name
    # Parented to whichever top-level window is actually active, not to
    # `dialog` - see clear_stim_array()'s docstring for why: the routine
    # dialogs are created without a Qt parent (_show_routine_dialog), so a
    # box that is not explicitly raised/activated can end up shown behind
    # them, and this box is now shown far more often than the old
    # unsimulated-tail-only check it replaced (every routine start on an
    # array that already has any content, not just a rare edge case) - a
    # modal QMessageBox stuck behind another window blocks the whole
    # application with no visible way to dismiss it.
    box = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow() or dialog)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle("Existing input")
    box.setText(f"'{name}' already contains stimuli.")
    box.setInformativeText(
        "Clear the array and start from the beginning, continue after the "
        "last existing stimulus (keeping progress from a previous run), or "
        "cancel."
    )
    clear_button = box.addButton(
        "Clear and start over", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
    )
    continue_button = box.addButton(
        "Continue after last stimulus", QtWidgets.QMessageBox.ButtonRole.AcceptRole
    )
    cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel_button)
    box.show()
    box.raise_()
    box.activateWindow()
    box.exec()
    clicked = box.clickedButton()
    if clicked is clear_button:
        return "clear"
    if clicked is continue_button:
        return "continue"
    return "cancel"


def _set_native_minimized_position(hwnd: int, x: int, y: int) -> bool:
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


class _WheelToScrollFilter(QtCore.QObject):
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


def _wrap_in_scroll_area(
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
