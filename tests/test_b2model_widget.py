from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_name_def_save_not_implemented(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    b2w = B2W.B2ModelWidget(B2Model(), None, {})

    qtbot.addWidget(b2w)

    b2w.show()

    assert b2w.isVisible()

    qtbot.keyClicks(b2w.name_entry, "modelname")
    assert b2w.name_entry.text() == "modelname"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    qtbot.mouseClick(b2w.name_button, QtCore.Qt.LeftButton)

    # Wait until the error dialog is visible
    qtbot.waitUntil(b2w.error_dialog.isVisible)

    assert b2w.error_dialog.isVisible()
    # Verify error message box content
    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()


def test_dt_def_save_not_implemented(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    b2w = B2W.B2ModelWidget(B2Model(), None, {})

    qtbot.addWidget(b2w)

    b2w.show()

    qtbot.keyClicks(b2w.timestep, "0.2 ms")
    assert b2w.timestep.text() == "0.2 ms"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    qtbot.mouseClick(b2w.timestep_button, QtCore.Qt.LeftButton)

    # Wait until the error dialog is visible
    qtbot.waitUntil(b2w.error_dialog.isVisible)

    # Verify error message box content
    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()


def test_random_seed_def_save_not_implemented(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    b2w = B2W.B2ModelWidget(B2Model(), None, {})

    b2w.show()

    qtbot.addWidget(b2w)

    qtbot.keyClicks(b2w.seed, "2231")

    assert b2w.seed.text() == "2231"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    qtbot.mouseClick(b2w.seed_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(b2w.error_dialog.isVisible)

    assert b2w.error_dialog.isVisible()

    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()


def test_parameterdialog_nothing_defined_errors(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    param_dlg = B2W.ParameterDialog(B2Model())

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert param_dlg.error_dialog.text() == "Please enter a name!"

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    # Verify main window is still visible
    assert param_dlg.isVisible()

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert (
        param_dlg.error_dialog.text()
        == "Please define a parameter first by giving it a name!"
    )

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    assert param_dlg.isVisible()

    # set name, then try to save parameter that hasn´t been defined

    qtbot.keyClicks(param_dlg.name_entry, "param")

    assert param_dlg.name_entry.text() == "param"

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.LeftButton)

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.LeftButton)
    assert (
        param_dlg.error_dialog.text()
        == "Please assign a value or expression before saving!"
    )

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    # nothing is happening
    assert param_dlg.model.parameters["param"] == {}


def test_parameterdialog_enter_fields(qtbot: QtBot) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    # open dialog
    qtbot.keyClicks(param_dlg.name_entry, "param")

    assert param_dlg.name_entry.text() == "param"

    assert "param" not in param_dlg.model.parameters

    assert param_dlg.name_button.text() == "save"

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.LeftButton)

    assert param_dlg.name_button.text() == "saved"

    assert "param" in param_dlg.model.parameters
    assert param_dlg.model.parameters["param"] == {}

    qtbot.keyClicks(param_dlg.value_entry, "42")

    assert param_dlg.value_entry.toPlainText() == "42"
    assert param_dlg.model.parameters["param"] == {}  # nothing has been saved yet

    assert param_dlg.value_button.text() == "save"

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.LeftButton)

    assert param_dlg.value_button.text() == "saved"

    assert param_dlg.model.parameters["param"] == "42"  # nothing has been saved yet


def test_display_parameters_empty(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    monkeypatch.setattr(B2W.QtWidgets.QDialog, "exec_", B2W.QtWidgets.QDialog.show)

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    qtbot.mouseClick(param_dlg.show_parameters_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(param_dlg.dictwidget.isVisible)

    assert param_dlg.dictwidget.table.text() == "No parameters defined"


def test_display_parameters(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(B2W.QtWidgets.QDialog, "exec_", B2W.QtWidgets.QDialog.show)

    parameter_dlg = B2W.ParameterDialog(B2Model())

    qtbot.addWidget(parameter_dlg)

    parameter_dlg.show()

    parameter_dlg.model.parameters = {}

    parameter_dlg.model.parameters["p1"] = 12

    parameter_dlg.model.parameters["p2"] = "v*w : mC"

    qtbot.mouseClick(parameter_dlg.show_parameters_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(parameter_dlg.dictwidget.isVisible)

    qtbot.keyClick(parameter_dlg.dictwidget, QtCore.Qt.Key_Escape)

    header = parameter_dlg.dictwidget.table.headerItem()

    columns = [
        header.text(i) for i in range(0, parameter_dlg.dictwidget.table.columnCount())
    ]

    assert parameter_dlg.dictwidget.table.columnCount() == 2
    assert columns == ["Key", "Value"]
    assert parameter_dlg.dictwidget.table.topLevelItemCount() == 2


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

    assert ndiag.model.neurongroups["x"]["model"] == "dv/dt = -v : mV\ndv/dt = -v : mV\n"
