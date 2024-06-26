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

    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\n"

    qtbot.keyClicks(ndiag.equation, "dv/dt = -v : mV")
    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\n"

    qtbot.mouseClick(ndiag.equation_button, QtCore.Qt.MouseButton.LeftButton)

    assert (
        ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\ndv/dt = -v : mV\n"
    )


def test_neurongroup_assembling(qtbot: QtBot) -> None:
    ndiag = B2W.NeuronGroupDialog(B2Model())

    qtbot.addWidget(ndiag)

    ndiag.show()

    ndiag.model.neurongroups["x"] = {}
    ndiag.model.neurongroups["x"]["model"] = "dv/dt = -v : volt\n"
    ndiag.model.neurongroups["x"]["n"] = 64
    ndiag.name = "x"

    ndiag.accept()

    assert ndiag.model.network["x"].name == "x"
    assert ndiag.model.network["x"]._N == 64


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

    ndiag.model.neurongroups["x"]["n"] = 64

    ndiag.accept()

    qtbot.waitUntil(ndiag.error_dialog.isVisible)

    assert (
        ndiag.error_dialog.text()
        == "Error when assembling neurongroup x: Error in assembling neuron group x: Error parsing the unit specification for variable 'v': Unit specification refers to 'mV', but this is not a base unit. Use 'volt' instead."
    )
    ndiag.error_dialog.close()
