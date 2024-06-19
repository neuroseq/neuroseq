from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_neurondialog_enter_close(qtbot: QtBot) -> None:
    neuron_dlg = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(neuron_dlg)

    neuron_dlg.show()

    cancel_button = neuron_dlg.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: neuron_dlg.isVisible() is False)


def test_neurondialog_define_name(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    qtbot.keyClicks(ndiag.name_entry, "x")

    assert ndiag.name_entry.text() == "x"

    assert "x" not in ndiag.model.neurongroups

    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.LeftButton)

    assert "x" in ndiag.model.neurongroups


def test_neurondialog_define_equation(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    qtbot.keyClicks(ndiag.name_entry, "x")
    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.LeftButton)
    assert "x" in ndiag.model.neurongroups

    qtbot.keyClicks(ndiag.equation, "dv/dt = -v : mV")

    qtbot.mouseClick(ndiag.equation_button, QtCore.Qt.LeftButton)

    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\n"

    qtbot.keyClicks(ndiag.equation, "dv/dt = -v : mV")
    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\n"

    qtbot.mouseClick(ndiag.equation_button, QtCore.Qt.LeftButton)

    assert (
        ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\ndv/dt = -v : mV\n"
    )
