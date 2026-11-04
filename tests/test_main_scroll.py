from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.main_scroll import MainScrollFilter
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


def test_one_filter_serves_every_main_scroll_area(qtbot: QtBot) -> None:
    first_area, first_inner = _build(qtbot)
    second_area, second_inner = _build(qtbot)
    app = QtWidgets.QApplication.instance()
    assert app is not None
    assert len(app.findChildren(MainScrollFilter)) == 1

    second_bar = second_inner.verticalScrollBar()
    second_bar.setValue(second_bar.maximum())
    _wheel(second_inner.viewport(), -120)
    assert second_area.verticalScrollBar().value() > 0
    assert first_area.verticalScrollBar().value() == 0


def test_zoom_control_buttons_field_and_limits(qtbot: QtBot) -> None:
    from neuroseq.main_scroll import ZoomControl

    control = ZoomControl()
    qtbot.addWidget(control)
    seen: list[int] = []
    control.zoom_changed.connect(seen.append)
    assert control.field.text() == "100%"

    control.btn_plus.click()
    assert control.zoom == 110 and control.field.text() == "110%"
    control.btn_minus.click()
    control.btn_minus.click()
    assert control.zoom == 90

    control.field.setText("150")
    control.field.editingFinished.emit()
    assert control.zoom == 150
    control.field.setText("9999")
    control.field.editingFinished.emit()
    assert control.zoom == 300 and not control.btn_plus.isEnabled()
    control.field.setText("abc")
    control.field.editingFinished.emit()
    assert control.zoom == 300 and control.field.text() == "300%"
    assert seen == [110, 100, 90, 150, 300]


def test_workspace_zoom_scales_size_and_font(qtbot: QtBot) -> None:
    from neuroseq.main_scroll import WorkspaceZoom

    content = QtWidgets.QWidget()
    qtbot.addWidget(content)
    zoom = WorkspaceZoom(content, 1000, 800)
    base_points = content.font().pointSizeF()
    zoom.apply(150)
    assert content.minimumSize() == QtCore.QSize(1500, 1200)
    assert content.font().pointSizeF() > base_points
    zoom.apply(100)
    assert content.minimumSize() == QtCore.QSize(1000, 800)
    assert content.font().pointSizeF() == base_points


def test_gui_zoom_control_resizes_workspace(qtbot: QtBot) -> None:
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    assert gui.zoom_control.field.text() == "100%"
    content = gui._main_scroll.widget()
    gui.zoom_control.btn_plus.click()
    assert content.minimumSize() == QtCore.QSize(1650, 1045)
    gui.zoom_control.set_zoom(100)
    assert content.minimumSize() == QtCore.QSize(1500, 950)
    gui._close_confirmed = True
    gui.close()
