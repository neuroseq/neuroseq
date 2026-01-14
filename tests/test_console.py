from __future__ import annotations

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.console_widget import ConsoleWidget
from neuroseq.project import Project


def test_console(qtbot: QtBot, test_project: Project) -> None:
    signals: list[bool] = []
    console = ConsoleWidget()
    qtbot.addWidget(console)
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    console.console_command_entered.connect(lambda: signals.append(True))
    assert "np" in console.namespace()
    assert "b2" in console.namespace()
    assert console.namespace()["project"] is None
    # start with a valid project
    console.set_project(test_project)
    assert "np" in console.namespace()
    assert console.namespace()["project"] is not None
    # define a new variable in the console namespace
    console.input.setText("x = 5")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert console.namespace()["x"] == 5
    assert len(signals) == 1
    # modify a stim array in the project in a valid way
    target_a = np.ones((5, 20))
    console.input.setText("project.stim_arrays['stimulus'].values = np.ones((5,20))")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(test_project.stim_arrays["stimulus"].values, target_a)
    assert len(signals) == 2
