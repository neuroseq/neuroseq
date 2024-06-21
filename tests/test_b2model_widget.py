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
    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    qtbot.addWidget(b2w)

    b2w.show()

    assert b2w.isVisible()

    qtbot.keyClicks(b2w.name_entry, "modelname")
    assert b2w.name_entry.text() == "modelname"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
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
    b2w = B2W.B2ModelWidget(B2Model(), None)

    qtbot.addWidget(b2w)

    b2w.show()

    qtbot.keyClicks(b2w.timestep, "0.2 ms")
    assert b2w.timestep.text() == "0.2 ms"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
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
    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    b2w.show()

    qtbot.addWidget(b2w)

    qtbot.keyClicks(b2w.seed, "2231")

    assert b2w.seed.text() == "2231"

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
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


def test_open_module_dialogs(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(B2W.QtWidgets.QDialog, "exec", B2W.QtWidgets.QDialog.show)

    b2w = B2W.B2ModelWidget(
        B2Model(),
        None,
    )

    b2w.show()

    qtbot.addWidget(b2w)

    qtbot.mouseClick(b2w.add_neurongroup_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(b2w.neuron_dlg.isVisible)

    b2w.neuron_dlg.close()

    qtbot.waitUntil(lambda: b2w.neuron_dlg.isVisible() is False)

    qtbot.mouseClick(b2w.add_synapse_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(b2w.synapse_dlg.isVisible)
    b2w.synapse_dlg.close()

    qtbot.waitUntil(lambda: b2w.synapse_dlg.isVisible() is False)

    qtbot.mouseClick(b2w.add_parameter_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(b2w.param_dlg.isVisible)

    b2w.param_dlg.close()

    qtbot.waitUntil(lambda: b2w.param_dlg.isVisible() is False)

    qtbot.mouseClick(b2w.display_system_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(b2w.dictwidget.isVisible)
