from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_synapsedialog_enter_close(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    cancel_button = sdiag.button_box.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: sdiag.isVisible() is False)


def test_synapsedialog_define_name(qtbot: QtBot) -> None:
    sdiag = B2W.SynapseDialog(B2Model())

    qtbot.addWidget(sdiag)

    sdiag.show()

    qtbot.keyClicks(sdiag.name_entry, "x")

    assert sdiag.name_entry.text() == "x"

    assert "x" not in sdiag.model.synapses

    qtbot.mouseClick(sdiag.name_button, QtCore.Qt.LeftButton)

    assert "x" in sdiag.model.synapses


def test_synapsedialog_define_equation(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    sdiag = B2W.SynapseDialog(B2Model())
    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
    )
    qtbot.addWidget(sdiag)

    sdiag.show()

    qtbot.keyClicks(sdiag.name_entry, "x")
    qtbot.mouseClick(sdiag.name_button, QtCore.Qt.LeftButton)
    assert "x" in sdiag.model.synapses

    qtbot.keyClicks(sdiag.equation, "dv/dt = -v : mV")

    qtbot.mouseClick(sdiag.equation_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(sdiag.error_dialog.isVisible)

    assert sdiag.error_dialog.isVisible()

    assert sdiag.error_dialog.text() == "Not connected yet!"

    ok_button = sdiag.error_dialog.button(QtWidgets.QMessageBox.Ok)
    qtbot.mouseClick(ok_button, QtCore.Qt.LeftButton)

    # Ensure the dialog is no longer visible
    qtbot.waitUntil(lambda: not sdiag.error_dialog.isVisible())

    # Verify main window is still visible
    assert sdiag.isVisible()
