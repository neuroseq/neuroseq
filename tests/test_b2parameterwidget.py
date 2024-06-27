from __future__ import annotations

from typing import Any
from typing import Dict
from typing import cast

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_display_parameters_empty(qtbot: QtBot) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    param_dlg.show()

    qtbot.addWidget(param_dlg)

    qtbot.mouseClick(param_dlg.show_parameters_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.dictwidget.isVisible)

    assert param_dlg.dictwidget.table.text() == "No parameters defined"


def test_display_parameters(qtbot: QtBot) -> None:
    parameter_dlg = B2W.ParameterDialog(B2Model())

    parameter_dlg.show()

    qtbot.addWidget(parameter_dlg)

    parameter_dlg.model.parameters = {}

    parameter_dlg.model.parameters["p1"] = 12

    parameter_dlg.model.parameters["p2"] = "v*w : mC"

    qtbot.mouseClick(
        parameter_dlg.show_parameters_button, QtCore.Qt.MouseButton.LeftButton
    )

    qtbot.waitUntil(parameter_dlg.dictwidget.isVisible)

    qtbot.keyClick(parameter_dlg.dictwidget, QtCore.Qt.Key_Escape)

    header = parameter_dlg.dictwidget.table.headerItem()

    columns = [
        header.text(i) for i in range(0, parameter_dlg.dictwidget.table.columnCount())
    ]

    assert parameter_dlg.dictwidget.table.columnCount() == 2
    assert columns == ["Key", "Value"]
    assert parameter_dlg.dictwidget.table.topLevelItemCount() == 2


def test_parameterdialog_nothing_defined_errors(qtbot: QtBot) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert param_dlg.error_dialog.text() == "Please enter a name!"

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.MouseButton.LeftButton)

    param_dlg.model.define_parameter("x", 1 * B2W.b2.ms)

    param_dlg.name_entry.clear()
    qtbot.keyClicks(param_dlg.name_entry, "x")

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert (
        param_dlg.error_dialog.text()
        == "The name x has already been defined in parameters"
    )

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.MouseButton.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    # Verify main window is still visible
    assert param_dlg.isVisible()

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert (
        param_dlg.error_dialog.text()
        == "Please define a parameter first by giving it a name!"
    )

    ok_button = param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.MouseButton.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    assert param_dlg.isVisible()

    # set name, then try to save parameter that hasn´t been defined
    param_dlg.name_entry.clear()

    qtbot.keyClicks(param_dlg.name_entry, "param")

    assert param_dlg.name_entry.text() == "param"

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)
    qtbot.mouseClick(
        param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok),
        QtCore.Qt.MouseButton.LeftButton,
    )

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert (
        param_dlg.error_dialog.text()
        == "Please assign a value or expression before saving!"
    )

    qtbot.mouseClick(
        param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok),
        QtCore.Qt.MouseButton.LeftButton,
    )
    param_dlg.name = None

    param_dlg.accept()

    qtbot.waitUntil(param_dlg.error_dialog.isVisible)

    assert param_dlg.error_dialog.text() == "No valid name has been defined"

    qtbot.mouseClick(
        param_dlg.error_dialog.button(QtWidgets.QMessageBox.Ok),
        QtCore.Qt.MouseButton.LeftButton,
    )

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not param_dlg.error_dialog.isVisible())

    # nothing is happening
    assert cast(Dict[str, Any], param_dlg.model.parameters)["param"] == {}


def test_parameterdialog_enter_fields(qtbot: QtBot) -> None:
    param_dlg = B2W.ParameterDialog(B2Model())

    qtbot.addWidget(param_dlg)

    param_dlg.show()

    # open dialog
    qtbot.keyClicks(param_dlg.name_entry, "param")

    assert param_dlg.name_entry.text() == "param"

    assert "param" not in cast(Dict[str, Any], param_dlg.model.parameters)

    assert param_dlg.name_button.text() == "save"

    qtbot.mouseClick(param_dlg.name_button, QtCore.Qt.MouseButton.LeftButton)

    assert param_dlg.name_button.text() == "saved"

    assert "param" in cast(Dict[str, Any], param_dlg.model.parameters)
    assert cast(Dict[str, Any], param_dlg.model.parameters)["param"] == {}

    qtbot.keyClicks(param_dlg.value_entry, "42")

    assert param_dlg.value_entry.toPlainText() == "42"
    assert (
        cast(Dict[str, Any], param_dlg.model.parameters)["param"] == {}
    )  # nothing has been saved yet

    assert param_dlg.value_button.text() == "save"

    qtbot.mouseClick(param_dlg.value_button, QtCore.Qt.MouseButton.LeftButton)

    assert param_dlg.value_button.text() == "saved"

    assert (
        cast(Dict[str, Any], param_dlg.model.parameters)["param"] == 42
    )  # nothing has been saved yet
