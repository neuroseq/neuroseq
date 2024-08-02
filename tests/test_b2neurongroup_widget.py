from __future__ import annotations

import neuroseq.b2_model_widget as B2W
from brian2 import ms
from neuroseq.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_neurondialog_enter_close(qtbot: QtBot) -> None:
    neuron_dlg = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(neuron_dlg)

    neuron_dlg.show()

    cancel_button = neuron_dlg.button_box.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: neuron_dlg.isVisible() is False)


def test_neurondialog_define_name(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    qtbot.keyClicks(ndiag.name_entry, "x")

    assert ndiag.name_entry.text() == "x"

    assert "x" not in ndiag.model.neurongroups

    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.MouseButton.LeftButton)

    assert "x" in ndiag.model.neurongroups

    qtbot.keyClicks(ndiag.name_entry, "y")

    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(ndiag.error_dialog.isVisible)

    assert ndiag.error_dialog.text() == "A name has already been defined"

    qtbot.mouseClick(
        ndiag.error_dialog.button(QtWidgets.QMessageBox.Ok),
        QtCore.Qt.MouseButton.LeftButton,
    )

    assert ndiag.error_dialog.isVisible() is False


def test_neurondialog_define_equation(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    qtbot.mouseClick(ndiag.equation_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(ndiag.error_dialog.isVisible)

    assert ndiag.error_dialog.text() == "Please define a name first!"

    ndiag.error_dialog.close()

    qtbot.keyClicks(ndiag.name_entry, "x")
    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.MouseButton.LeftButton)
    assert "x" in ndiag.model.neurongroups

    qtbot.keyClicks(ndiag.equation, "dv/dt = -v : mV")

    qtbot.mouseClick(ndiag.equation_button, QtCore.Qt.MouseButton.LeftButton)

    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV"


def test_neurondialog_define_dt(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    qtbot.keyClicks(ndiag.name_entry, "x")

    assert ndiag.name_entry.text() == "x"

    assert "x" not in ndiag.model.neurongroups

    qtbot.mouseClick(ndiag.name_button, QtCore.Qt.MouseButton.LeftButton)

    assert "x" in ndiag.model.neurongroups

    qtbot.keyClicks(ndiag.dt, "0.1*ms")

    assert ndiag.dt.text() == "0.1*ms"

    assert "dt" not in ndiag.model.neurongroups["x"]

    qtbot.mouseClick(ndiag.timestep_button, QtCore.Qt.MouseButton.LeftButton)

    assert "dt" in ndiag.model.neurongroups["x"]
    assert ndiag.model.neurongroups["x"]["dt"] == 0.1 * ms


def test_neurongroup_assembling(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()
    qtbot.waitUntil(lambda: ndiag.isVisible() is True)

    ndiag.model.neurongroups["x"] = {}
    ndiag.model.neurongroups["x"]["model"] = "dv/dt = -v : volt\n"
    ndiag.model.neurongroups["x"]["neuronnumber"] = 64
    ndiag.model.neurongroups["x"]["dt"] = 0.1 * ms
    ndiag.name = "x"

    ndiag.accept()

    qtbot.waitUntil(lambda: ndiag.isVisible() is False)

    assert ndiag.model.neurongroups["x"] == {
        "model": "dv/dt = -v : volt\n",
        "neuronnumber": 64,
        "dt": 0.1 * ms,
    }


def test_neurongroup_assembling_exceptions(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    ndiag.accept()
    qtbot.waitUntil(ndiag.error_dialog.isVisible)
    assert (
        ndiag.error_dialog.text()
        == "Error, first define a neurongroup by giving it a name"
    )
    ndiag.error_dialog.close()

    ndiag.model.neurongroups["x"] = {}
    ndiag.model.neurongroups["x"]["model"] = "dv/dt = -v : mV\n"
    ndiag.name = "x"

    ndiag.accept()

    qtbot.waitUntil(ndiag.error_dialog.isVisible)
    assert (
        ndiag.error_dialog.text()
        == "The model equation and the neuronnumber need to be defined"
    )
    ndiag.error_dialog.close()
