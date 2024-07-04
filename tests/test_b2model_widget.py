from __future__ import annotations

from typing import cast

import formozov_project.b2_model_widget as B2W
from brian2 import Quantity
from formozov_project.b2_model import B2Model
from numpy import isclose
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore


def test_dt_def(qtbot: QtBot) -> None:
    b2w = B2W.B2ModelWidget(B2Model(), None)

    qtbot.addWidget(b2w)

    qtbot.keyClicks(b2w.timestep, "0.2 ms")

    b2w.accept_timestep()

    assert b2w.model.dt is not None
    assert isclose(cast(Quantity, b2w.model.dt.base), 0.0002)
    assert b2w.timestep_button.isDown()
    assert b2w.timestep_button.text() == "saved"


def test_dt_def_exceptions(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    b2w = B2W.B2ModelWidget(B2Model(), None)

    qtbot.addWidget(b2w)

    qtbot.keyClicks(b2w.timestep, "0.2 ms")

    def raise_error(_: str) -> None:
        raise ValueError("something went wrong")

    monkeypatch.setattr(b2w, "_convert_string_to_quantity", raise_error)

    b2w.accept_timestep()

    qtbot.waitUntil(b2w.error_dialog.isVisible)

    assert (
        b2w.error_dialog.text()
        == "Error when trying to convert timestep 0.2 ms: something went wrong"
    )
    b2w.error_dialog.close()


def test_random_seed_def(qtbot: QtBot) -> None:
    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    b2w.show()

    qtbot.addWidget(b2w)

    b2w.seed.setText("2231")

    b2w.accept_seed()
    assert b2w.model.seed == 2231
    assert b2w.seed_button.text() == "saved"
    assert b2w.seed_button.isDown()

    b2w.seed.setText("cannot_convert_to_int123")

    b2w.accept_seed()

    qtbot.waitUntil(b2w.error_dialog.isVisible)

    assert (
        b2w.error_dialog.text()
        == "Error when trying to convert seed cannot_convert_to_int123: invalid literal for int() with base 10: 'cannot_convert_to_int123'"
    )

    b2w.error_dialog.close()


def test_open_module_dialogs(qtbot: QtBot) -> None:
    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    b2w.show()

    qtbot.addWidget(b2w)

    qtbot.mouseClick(b2w.add_neurongroup_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(b2w.neuron_dlg.isVisible)

    b2w.neuron_dlg.close()

    qtbot.waitUntil(lambda: b2w.neuron_dlg.isVisible() is False)

    qtbot.mouseClick(b2w.add_synapse_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(b2w.synapse_dlg.isVisible)
    b2w.synapse_dlg.close()

    qtbot.waitUntil(lambda: b2w.synapse_dlg.isVisible() is False)

    qtbot.mouseClick(b2w.add_parameter_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(b2w.param_dlg.isVisible)

    b2w.param_dlg.close()

    qtbot.waitUntil(lambda: b2w.param_dlg.isVisible() is False)


def test_open_initialization_dialog(qtbot: QtBot) -> None:
    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    b2w.show()

    qtbot.addWidget(b2w)

    qtbot.mouseClick(b2w.init_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(b2w.init_widget.isVisible)

    b2w.init_widget.close()
