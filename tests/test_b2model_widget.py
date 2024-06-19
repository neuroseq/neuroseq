from __future__ import annotations
from pytestqt.qtbot import QtBot
import pytest
from typing import Type

from qtpy import QtWidgets
from qtpy import QtCore


import numpy as np

import brian2 as b2

from formozov_project.b2_model import B2Model
import formozov_project.b2_model_widget as B2W


@pytest.fixture()
def ornstein_model(tmp_path):
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    model = B2Model(seed=4321, logpath=tmp_path)
    model.define_neuronnumber("neurongroup", 2)
    model.define_dt(0.2 * b2.ms)
    model.define_solver_method("neurongroup", "euler")
    model.define_equation("neurongroup", eqs)
    model.assemble_neurongroup(name="neurongroup")
    model.define_monitor("M", b2.StateMonitor, "neurongroup", "y")

    # init
    model.network["neurongroup"].tau = np.array([1, 0.1]) * b2.second
    model.network["neurongroup"].sigma = np.array([0.1, 0.31622])
    model.network["neurongroup"].y = 1

    return model


def handle_dialog(window, app):
    # https://github.com/pytest-dev/pytest-qt/issues/256
    while window._test_dialog is None:
        app.processEvents()


@pytest.fixture()
def standalone_model_app(
    qtbot: QtBot,
):
    modeldict = {}

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    b2w = B2W.B2ModelWidget(B2Model(), None, modeldict)

    b2w.show()

    qtbot.addWidget(b2w)
    qtbot.waitUntil(lambda: b2w.isVisible())

    yield qtbot, b2w, modeldict

    app.shutdown()
    app = None


def test_name_def_save_not_implemented(
    standalone_model_app,
    mocker,
):
    bot, b2w, _ = standalone_model_app

    assert b2w.isVisible()

    bot.keyClicks(b2w.name_entry, "modelname")
    assert b2w.name_entry.text() == "modelname"

    mocker.patch.object(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    bot.mouseClick(b2w.name_button, QtCore.Qt.LeftButton)

    # Wait until the error dialog is visible
    bot.waitUntil(lambda: b2w.error_dialog.isVisible())

    assert b2w.error_dialog.isVisible()
    # Verify error message box content
    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    bot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    bot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()


def test_dt_def_save_not_implemented(standalone_model_app, mocker):
    bot, b2w, _ = standalone_model_app

    bot.keyClicks(b2w.timestep, "0.2 ms")
    assert b2w.timestep.text() == "0.2 ms"

    mocker.patch.object(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    bot.mouseClick(b2w.timestep_button, QtCore.Qt.LeftButton)

    # Wait until the error dialog is visible
    bot.waitUntil(lambda: b2w.error_dialog.isVisible())

    assert b2w.error_dialog.isVisible()

    # Verify error message box content
    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    bot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    bot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()


def test_random_seed_def_save_not_implemented(standalone_model_app, mocker): 
    bot, b2w, _ = standalone_model_app 

    bot.keyClicks(b2w.seed, "2231")

    assert b2w.seed.text() == "2231"

    mocker.patch.object(
        B2W.QtWidgets.QMessageBox, "exec_", B2W.QtWidgets.QMessageBox.show
    )

    bot.mouseClick(b2w.seed_button, QtCore.Qt.LeftButton)

    bot.waitUntil(lambda: b2w.error_dialog.isVisible())
    
    assert b2w.error_dialog.isVisible()

    assert b2w.error_dialog.text() == "Not connected yet!"

    ok_button = b2w.error_dialog.button(QtWidgets.QMessageBox.Ok)
    bot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    bot.waitUntil(lambda: not b2w.error_dialog.isVisible())

    # Verify main window is still visible
    assert b2w.isVisible()