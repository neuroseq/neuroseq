from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.main_scroll import make_main_scroll_area


def _wheel(widget: QtWidgets.QWidget, delta: int) -> None:
    pos = QtCore.QPointF(widget.rect().center())
    event = QtGui.QWheelEvent(
        pos,
        QtCore.QPointF(widget.mapToGlobal(widget.rect().center())),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, event)


def _build(qtbot: QtBot):
    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    inner = QtWidgets.QListWidget()
    inner.addItems([str(i) for i in range(200)])
    layout.addWidget(inner)
    area = make_main_scroll_area(content, 300, 2000)
    qtbot.addWidget(area)
    area.resize(400, 300)
    area.show()
    qtbot.waitExposed(area)
    return area, inner


def test_nested_widget_scrolls_first_then_main_window(qtbot: QtBot) -> None:
    area, inner = _build(qtbot)
    bar, outer = inner.verticalScrollBar(), area.verticalScrollBar()
    assert bar.maximum() > 0 and outer.maximum() > 0

    _wheel(inner.viewport(), -120)
    assert bar.value() > 0
    assert outer.value() == 0

    bar.setValue(bar.maximum())
    _wheel(inner.viewport(), -120)
    assert outer.value() > 0


def test_scrolling_up_returns_priority_at_nested_top(qtbot: QtBot) -> None:
    area, inner = _build(qtbot)
    area.verticalScrollBar().setValue(200)
    _wheel(inner.viewport(), 120)  # inner already at top -> main window scrolls up
    assert area.verticalScrollBar().value() < 200
