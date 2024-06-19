from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_display_parameters_empty(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    # monkeypatch.setattr(B2W.QtWidgets.QDialog, "exec_", B2W.QtWidgets.QDialog.show)

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    qtbot.mouseClick(param_dlg.show_parameters_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(param_dlg.dictwidget.isVisible)

    assert param_dlg.dictwidget.table.text() == "No parameters defined"


def test_display_parameters(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    # monkeypatch.setattr(B2W.QtWidgets.QDialog, "exec_", B2W.QtWidgets.QDialog.show)

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
