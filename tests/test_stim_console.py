from __future__ import annotations

from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import pytest
from formozov_project.b2_model import B2Model
from formozov_project.stim_console import StimConsole
from pytestqt.qtbot import QtBot
from qtpy import QtCore


@pytest.mark.parametrize(
    "stims, valid, description",
    [
        (None, False, "None"),
        (5, False, "<class 'int'>"),
        (3.14, False, "<class 'float'>"),
        ([], False, "<class 'list'>"),
        (np.zeros(10), False, "<class 'numpy.ndarray'>"),
        ({"a": None}, False, "dict with None for key 'a'"),
        ({"a": 3.14}, False, "dict with <class 'float'> for key 'a'"),
        ({"a": [1, 2, 3]}, False, "dict with <class 'list'> for key 'a'"),
        ({"a": np.zeros((10, 10, 10))}, False, "dict with 3d numpy array for key 'a'"),
        ({"a": np.zeros((10, 10))}, True, "dict of 1d or 2d numpy arrays"),
        (
            {"a": np.zeros(10), "b": np.zeros((2, 3))},
            True,
            "dict of 1d or 2d numpy arrays",
        ),
    ],
)
def test_stim_console_validate_stims(
    stims: Any, valid: bool, description: str, qtbot: QtBot
) -> None:
    stim_console = StimConsole()
    qtbot.addWidget(stim_console)
    assert stim_console.validate_stims() == (False, "None")
    stim_console.namespace()["stims"] = stims
    assert stim_console.validate_stims() == (valid, description)


def test_stim_console_repl(qtbot: QtBot) -> None:
    signals: list[dict[str, np.ndarray]] = []
    stim_console = StimConsole()
    stim_console.sig_stims_values_dict_updated.connect(
        lambda stims_values_dict: signals.append(stims_values_dict)
    )
    assert "np" in stim_console.namespace()
    assert "b2" in stim_console.namespace()
    assert stim_console.namespace()["stims"] is None
    assert stim_console.namespace()["model"] is None
    # start with a valid dict of stims and a valid model
    stims = {
        "a": b2.TimedArray(np.zeros((10, 10)), dt=1 * b2.ms),
        "b": b2.TimedArray(np.zeros((100, 3)), dt=5 * b2.ms),
    }
    stim_console.set_stims(stims)
    model = B2Model()
    model.parameters = {"x": 4}
    stim_console.set_model(model)
    assert "np" in stim_console.namespace()
    assert "stims" in stim_console.namespace()
    assert "a" in stim_console.namespace()["stims"]
    assert "b" in stim_console.namespace()["stims"]
    assert stim_console.namespace()["model"] is not None
    # define a new variable, see it added to the console widget variables and get a signal with the new dict of stims,
    # since they are (still) valid, and the signal is emitted every time the user runs a command and stims dict is valid
    stim_console.input.setText("x = 5")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_console.namespace()["x"] == 5
    assert len(signals) == 1
    assert np.allclose(signals[-1]["a"], stims["a"].values)
    assert np.allclose(signals[-1]["b"], stims["b"].values)
    # modify stims array in a valid way
    target_a = np.ones((5, 20))
    stim_console.input.setText("stims['a'] = np.ones((5,20))")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_console.namespace()["stims"]["a"], target_a)
    assert len(signals) == 2
    assert np.allclose(signals[-1]["a"], target_a)
    assert np.allclose(signals[-1]["b"], stims["b"].values)
    # modify stims array in an invalid way: no signal should be emitted
    stim_console.input.setText("stims['a'] = 5")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert len(signals) == 2
    # modify the stims array directly using the model (this bypasses the stim array validation checks)
    assert "a" not in model.parameters
    stim_console.input.setText(
        "model.parameters['a'] = b2.TimedArray([1,2,3], 0.1*b2.second)"
    )
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert "a" in model.parameters
    assert np.allclose(model.parameters["a"].values, [1, 2, 3])
    assert model.parameters["a"].dt == 0.1
