from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.project_workspace import DetachedPanelWindow


class _CloseTrackingWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> bool:
        self.close_calls += 1
        return super().close()


def test_close_event_calls_widget_close_and_on_close_callback(qtbot) -> None:
    widget = _CloseTrackingWidget()
    callbacks: list[None] = []
    window = DetachedPanelWindow(
        widget, "Title", on_close=lambda: callbacks.append(None)
    )
    qtbot.addWidget(window)

    window.close()

    assert widget.close_calls == 1
    assert callbacks == [None]


def test_scroll_area_wraps_widget_and_is_resizable(qtbot) -> None:
    widget = QtWidgets.QWidget()
    window = DetachedPanelWindow(widget, "Title")
    qtbot.addWidget(window)

    scroll_area = window.findChild(QtWidgets.QScrollArea)

    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert scroll_area.widget() is widget
