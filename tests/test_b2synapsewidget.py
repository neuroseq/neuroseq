from __future__ import annotations

from typing import Any

import formozov_project.b2_model_widget as B2W
from formozov_project import b2_model
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_synapsedialog_enter_close(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    cancel_button = sdiag.button_box.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: sdiag.isVisible() is False)


def test_synapsedialog_define_name(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    qtbot.keyClicks(sdiag.name_entry, "x")

    assert sdiag.name_entry.text() == "x"

    assert "x" not in sdiag.model.synapses

    qtbot.mouseClick(sdiag.name_button, QtCore.Qt.MouseButton.LeftButton)

    assert "x" in sdiag.model.synapses


def test_synapse_input(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())
    qtbot.addWidget(sdiag)
    sdiag.show()

    qtbot.keyClicks(sdiag.name_entry, "x")
    qtbot.mouseClick(sdiag.name_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.origin, "s")
    qtbot.mouseClick(sdiag.origin_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.target, "t")
    qtbot.mouseClick(sdiag.target_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.equation, "dv/dt = -v : volt")
    qtbot.mouseClick(sdiag.equation_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.pre, "a * b: siemens")
    qtbot.mouseClick(sdiag.pre_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.post, "c * d: siemens")
    qtbot.mouseClick(sdiag.post_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.connect, "s !=  t")
    qtbot.mouseClick(sdiag.connect_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.keyClicks(sdiag.connect_prob, "1.0")
    qtbot.mouseClick(sdiag.connect_prob_button, QtCore.Qt.MouseButton.LeftButton)

    assert sdiag.model.synapses["x"] == {
        "name": "x",
        "origin": "s",
        "target": "t",
        "model": "dv/dt = -v : volt\n",
        "on_pre": "a * b: siemens\n",
        "on_post": "c * d: siemens\n",
        "connection_condition": "s !=  t",
        "p": 1.0,
    }


def test_accept_synapse(
    qtbot: QtBot, mfx_with_synapse: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    model, _ = mfx_with_synapse

    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    sdiag.model = model
    sdiag.name = (
        "Ce"  # artificially make sure a name has been set for the synapse in the gui
    )
    sdiag.accept()

    # assertion that the network has an object of name "Ce" with internal name "Ce" is enough to
    # assert that assembly was successful
    assert sdiag.model.synapses["Ce"] is not None


def test_accept_synapse_exceptions(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    sdiag.model.synapses["x"] = {
        "origin": "s",
        "model": "dv/dt = -v : volt\n",
        "on_pre": "a * b: siemens\n",
        "on_post": "c * d: siemens\n",
        "connection_condition": "s !=  t",
        "p": 1.0,
    }

    sdiag.accept()

    # qtbot.waitUntil(sdiag.error_dialog.isVisible)
    assert (
        sdiag.error_dialog.text()
        == "Error, first define a synapse system by giving it a name"
    )

    sdiag.error_dialog.close()

    sdiag.name = "x"

    sdiag.accept()

    # qtbot.waitUntil(sdiag.error_dialog.isVisible)
    assert (
        sdiag.error_dialog.text()
        == "The model equation and the origin- and target neurongroups need to be defined"
    )
    sdiag.error_dialog.close()
