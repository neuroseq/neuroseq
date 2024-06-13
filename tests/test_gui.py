from __future__ import annotations

from formozov_project.gui import Gui
from pytest import approx
from pytestqt.qtbot import QtBot


def test_gui_click_ok(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_ok.click()
    qtbot.waitUntil(lambda: not gui.isVisible())


def test_gui_click_play(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    assert gui.heatmap_widget.currentIndex == 0
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    gui.btn_play.click()
    qtbot.waitUntil(
        lambda: gui.heatmap_widget.currentIndex == gui.data.timestamps.shape[0] - 1
    )
    assert gui.trace_widget.timeline.getXPos() == approx(gui.data.timestamps[-1])
