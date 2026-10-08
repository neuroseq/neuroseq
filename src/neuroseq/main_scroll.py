"""Whole-window scrolling with nested-scroll priority.

The main workspace lives in a ``QScrollArea`` so it stays usable on small screens. A wheel
event over a nested scrollable widget (list, text box, panel scroll area...) is handled by
that widget until it reaches its end in the wheel direction; only then does the main
window scroll.
"""

from __future__ import annotations

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


def _can_scroll(bar: QtWidgets.QScrollBar, delta: int) -> bool:
    """Whether ``bar`` can still move for a wheel ``delta`` (positive = up/left)."""
    if bar.maximum() <= bar.minimum() or not bar.isEnabled():
        return False
    return bar.value() > bar.minimum() if delta > 0 else bar.value() < bar.maximum()


class MainScrollFilter(QtCore.QObject):
    """Application event filter forwarding exhausted wheel events to ``outer``."""

    def __init__(self, outer: QtWidgets.QScrollArea) -> None:
        super().__init__(outer)
        self._outer = outer

    def eventFilter(self, watched: QtCore.QObject | None, event: QtCore.QEvent | None):
        if event is None or event.type() != QtCore.QEvent.Type.Wheel:
            return False
        if not isinstance(watched, QtWidgets.QWidget) or not isinstance(
            event, QtGui.QWheelEvent
        ):
            return False
        if watched.window() is not self._outer.window() or not self._outer.isAncestorOf(
            watched
        ):
            return False
        dy, dx = event.angleDelta().y(), event.angleDelta().x()
        if dy == 0 and dx == 0:
            return False
        horizontal = abs(dx) > abs(dy) or bool(
            event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        delta = dx if abs(dx) > abs(dy) else dy

        widget: QtWidgets.QWidget | None = watched
        while widget is not None and widget is not self._outer:
            if isinstance(widget, QtWidgets.QAbstractScrollArea):
                bar = (
                    widget.horizontalScrollBar()
                    if horizontal
                    else widget.verticalScrollBar()
                )
                if bar is not None and _can_scroll(bar, delta):
                    return False  # nested widget still has room: let it scroll
            elif isinstance(widget, QtWidgets.QAbstractSlider):
                return False  # sliders own their wheel
            elif isinstance(widget, QtWidgets.QAbstractSpinBox | QtWidgets.QComboBox):
                if widget.hasFocus():
                    return False
            widget = widget.parentWidget()

        outer_bar = (
            self._outer.horizontalScrollBar()
            if horizontal
            else self._outer.verticalScrollBar()
        )
        if outer_bar is None or not _can_scroll(outer_bar, delta):
            return False
        outer_bar.setValue(
            outer_bar.value() - round(delta / 120 * 3 * outer_bar.singleStep())
        )
        event.accept()
        return True


def make_main_scroll_area(
    content: QtWidgets.QWidget, min_width: int, min_height: int
) -> QtWidgets.QScrollArea:
    """Wrap ``content`` in a scroll area and install the wheel-chaining filter."""
    content.setMinimumSize(min_width, min_height)
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidget(content)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.installEventFilter(MainScrollFilter(area))
    return area
