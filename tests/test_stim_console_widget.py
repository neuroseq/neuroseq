from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from formozov_project.stim_console_widget import StimConsoleWidget
from pytestqt.qtbot import QtBot
from qtpy import QtCore


@pytest.mark.parametrize(
    "stim, valid, description",
    [
        (np.zeros((10, 10)), True, "2d numpy array"),
        (None, False, "None"),
        (5, False, "<class 'int'>"),
        (3.14, False, "<class 'float'>"),
        (np.zeros(10), False, "1d numpy array"),
        (np.zeros((10, 10, 10)), False, "3d numpy array"),
    ],
)
def test_stim_console_widget_validate_stim(
    stim: Any, valid: bool, description: str, qtbot: QtBot
) -> None:
    stim_console_widget = StimConsoleWidget(stim=stim)
    qtbot.addWidget(stim_console_widget)
    assert stim_console_widget.validate_stim() == (valid, description)


def test_stim_console_widget_repl(qtbot: QtBot) -> None:
    signals: list[tuple[np.ndarray, bool]] = []
    stim_console_widget = StimConsoleWidget(None)
    stim_console_widget.stim_updated.connect(
        lambda stim, valid: signals.append((stim, valid))
    )
    # start with (invalid) stim array and np
    assert "stim" in stim_console_widget.namespace()
    assert "np" in stim_console_widget.namespace()
    # define a new variable, see it added to the console widget variables, stim array still invalid
    stim_console_widget.input.setText("x = 5")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert stim_console_widget.namespace()["x"] == 5
    array, valid = signals[-1]
    assert not valid
    assert array is None
    # modify stim array in a valid way
    target_stim = np.ones((5, 20))
    stim_console_widget.input.setText("stim = np.ones((5,20))")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_console_widget.namespace()["stim"], target_stim)
    array, valid = signals[-1]
    assert valid
    assert np.allclose(array, target_stim)
    # modify stim array in an invalid way: StimWidget is not updated
    target_stim = np.ones(7)
    stim_console_widget.input.setText("stim = np.ones(7)")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_console_widget.namespace()["stim"], target_stim)
    array, valid = signals[-1]
    assert not valid
    assert np.allclose(array, target_stim)
