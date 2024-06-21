from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from formozov_project.stim_console_widget import StimConsoleWidget
from pytestqt.qtbot import QtBot
from qtpy import QtCore


@pytest.mark.parametrize(
    "stims, valid, description",
    [
        (np.zeros((10, 10)), True, "2d numpy array"),
        (None, False, "None"),
        (5, False, "<class 'int'>"),
        (3.14, False, "<class 'float'>"),
        (np.zeros(10), False, "1d numpy array"),
        (np.zeros((10, 10, 10)), False, "3d numpy array"),
    ],
)
def test_stim_console_widget_validate_stims(
    stims: Any, valid: bool, description: str, qtbot: QtBot
) -> None:
    stim_console_widget = StimConsoleWidget(stims=stims)
    qtbot.addWidget(stim_console_widget)
    assert stim_console_widget.validate_stims() == (valid, description)


def test_stim_console_widget_repl(qtbot: QtBot) -> None:
    signals: list[tuple[np.ndarray, bool]] = []
    stim_console_widget = StimConsoleWidget(None)
    stim_console_widget.stims_updated.connect(
        lambda stims, valid: signals.append((stims, valid))
    )
    # start with (invalid) stims array and np
    assert "stims" in stim_console_widget.namespace()
    assert "np" in stim_console_widget.namespace()
    # define a new variable, see it added to the console widget variables, stims array still invalid
    stim_console_widget.input.setText("x = 5")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert stim_console_widget.namespace()["x"] == 5
    array, valid = signals[-1]
    assert not valid
    assert array is None
    # modify stims array in a valid way
    target_stims = np.ones((5, 20))
    stim_console_widget.input.setText("stims = np.ones((5,20))")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_console_widget.namespace()["stims"], target_stims)
    array, valid = signals[-1]
    assert valid
    assert np.allclose(array, target_stims)
    # modify stims array in an invalid way: StimWidget is not updated
    target_stims = np.ones(7)
    stim_console_widget.input.setText("stims = np.ones(7)")
    qtbot.keyPress(stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_console_widget.namespace()["stims"], target_stims)
    array, valid = signals[-1]
    assert not valid
    assert np.allclose(array, target_stims)
