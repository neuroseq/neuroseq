from __future__ import annotations

from formozov_project.gui import Gui
from pytestqt.qtbot import QtBot


def test_gui_click_ok(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_ok.click()
    qtbot.waitUntil(lambda: not gui.isVisible())
