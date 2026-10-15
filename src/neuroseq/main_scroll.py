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


ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, ZOOM_DEFAULT = 50, 300, 10, 100


class ZoomControl(QtWidgets.QWidget):
    """``-`` / percent field / ``+`` controls; emits ``zoom_changed(percent)``."""

    zoom_changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)
        self.btn_minus = QtWidgets.QToolButton()
        self.btn_minus.setText("\u2212")
        self.btn_minus.setToolTip("Zoom out")
        self.btn_plus = QtWidgets.QToolButton()
        self.btn_plus.setText("+")
        self.btn_plus.setToolTip("Zoom in")
        self.field = QtWidgets.QLineEdit(f"{ZOOM_DEFAULT}%")
        self.field.setFixedWidth(52)
        self.field.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.field.setToolTip(f"Zoom ({ZOOM_MIN}-{ZOOM_MAX}%); press Enter to apply")
        for widget in (self.btn_minus, self.field, self.btn_plus):
            layout.addWidget(widget)
        self._zoom = ZOOM_DEFAULT
        self.btn_minus.clicked.connect(lambda: self.set_zoom(self._zoom - ZOOM_STEP))
        self.btn_plus.clicked.connect(lambda: self.set_zoom(self._zoom + ZOOM_STEP))
        self.field.editingFinished.connect(self._on_field_edited)

    @property
    def zoom(self) -> int:
        return self._zoom

    def set_zoom(self, percent: int) -> None:
        percent = max(ZOOM_MIN, min(ZOOM_MAX, int(percent)))
        self.field.setText(f"{percent}%")
        self.btn_minus.setEnabled(percent > ZOOM_MIN)
        self.btn_plus.setEnabled(percent < ZOOM_MAX)
        if percent != self._zoom:
            self._zoom = percent
            self.zoom_changed.emit(percent)

    def _on_field_edited(self) -> None:
        text = self.field.text().strip().rstrip("%").strip()
        try:
            self.set_zoom(round(float(text)))
        except ValueError:
            self.set_zoom(self._zoom)  # restore the display


class WorkspaceZoom(QtCore.QObject):
    """Scales a workspace's minimum size and fonts; the scroll area handles overflow."""

    def __init__(
        self, content: QtWidgets.QWidget, base_width: int, base_height: int
    ) -> None:
        super().__init__(content)
        self._content = content
        self._base_size = (base_width, base_height)
        font = content.font()
        self._base_font = QtGui.QFont(font)
        self._base_points = font.pointSizeF()
        self._base_pixels = font.pixelSize()
        self.apply(ZOOM_DEFAULT)

    def apply(self, percent: int) -> None:
        factor = percent / 100.0
        self._content.setMinimumSize(
            round(self._base_size[0] * factor), round(self._base_size[1] * factor)
        )
        font = QtGui.QFont(self._base_font)
        if self._base_points > 0:
            font.setPointSizeF(self._base_points * factor)
        elif self._base_pixels > 0:
            font.setPixelSize(max(1, round(self._base_pixels * factor)))
        self._content.setFont(font)


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
