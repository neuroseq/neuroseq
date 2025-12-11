from __future__ import annotations

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.console_widget import ConsoleWidget
from neuroseq.model import Model


def test_console(qtbot: QtBot, test_model: Model) -> None:
    signals: list[bool] = []
    console = ConsoleWidget()
    qtbot.addWidget(console)
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    console.console_command_entered.connect(lambda: signals.append(True))
    assert "np" in console.namespace()
    assert "b2" in console.namespace()
    assert console.namespace()["model"] is None
    # start with a valid model
    console.set_model(test_model)
    assert "np" in console.namespace()
    assert console.namespace()["model"] is not None
    # define a new variable in the console namespace
    console.input.setText("x = 5")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert console.namespace()["x"] == 5
    assert len(signals) == 1
    # modify a stim array in the model in a valid way
    target_a = np.ones((5, 20))
    console.input.setText("model.stim_arrays['stimulus'].values = np.ones((5,20))")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(test_model.stim_arrays["stimulus"].values, target_a)
    assert len(signals) == 2
