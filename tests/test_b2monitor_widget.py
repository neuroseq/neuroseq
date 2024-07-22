from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from brian2 import StateMonitor
from brian2 import ms
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_monitor_enter_close(qtbot: QtBot) -> None:
    mdlg = B2W.MonitorDialog(B2Model())

    qtbot.addWidget(mdlg)

    mdlg.show()

    qtbot.waitUntil(mdlg.isVisible)

    cancel_button = mdlg.button_box.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: mdlg.isVisible() is False)


def test_monitor_accept_variables(qtbot: QtBot) -> None:
    mdlg = B2W.MonitorDialog(B2Model())

    qtbot.addWidget(mdlg)

    mdlg.show()

    qtbot.waitUntil(mdlg.isVisible)

    mdlg.type.setText("StateMonitor")

    mdlg.variable.setText("var")

    mdlg.source.setText("P")

    mdlg.accept_type()
    assert mdlg.tp == "StateMonitor"

    mdlg.accept_source()
    assert mdlg.src == "P"

    mdlg.accept_variable()
    assert mdlg.var == "var"

    mdlg.variable.setText("")

    mdlg.accept_variable()

    qtbot.waitUntil(mdlg.error_dialog.isVisible)

    assert mdlg.error_dialog.text() == "Please define variable first"

    mdlg.error_dialog.close()


def test_monitor_accept(qtbot: QtBot) -> None:
    mdlg = B2W.MonitorDialog(B2Model(seed=4321))

    qtbot.addWidget(mdlg)

    mdlg.show()

    # use ornstein process from the b2model tests

    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    mdlg.model.define_neuronnumber("neurongroup", 2)
    mdlg.model.define_dt(0.2 * ms)
    mdlg.model.define_solver_method("neurongroup", "euler")
    mdlg.model.define_equation("neurongroup", eqs)
    mdlg.model._assemble_neurongroup(name="neurongroup")

    # define the same monitor as in the b2 model test
    mdlg.name = "M"
    mdlg.src = "neurongroup"
    mdlg.var = "y"
    mdlg.tp = "StateMonitor"

    mdlg.accept()

    assert mdlg.model.network["M"] is not None


def test_monitor_parameterdisplay(qtbot: QtBot) -> None:
    mdlg = B2W.MonitorDialog(B2Model(), None)
    qtbot.add_widget(mdlg)

    mdlg.name = "M"
    mdlg.var = "v"
    mdlg.src = "P"
    mdlg.tp = StateMonitor

    mdlg.display_parameters()

    qtbot.waitUntil(mdlg.parameterwindow.isVisible)
    mdlg.parameterwindow.close()

    table = mdlg.parameterwindow.findChild(QtWidgets.QTableWidget)

    assert table.item(0, 0).text() == "name"
    assert table.item(0, 1).text() == "M"

    assert table.item(1, 0).text() == "type"
    assert (
        table.item(1, 1).text() == "<class 'brian2.monitors.statemonitor.StateMonitor'>"
    )

    assert table.item(2, 0).text() == "source"
    assert table.item(2, 1).text() == "P"

    assert table.item(3, 0).text() == "variable"
    assert table.item(3, 1).text() == "v"
