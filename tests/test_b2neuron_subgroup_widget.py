from __future__ import annotations

import formozov_project.b2_model_widget as B2W
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_neuronsubgroupdialog_enter_close(qtbot: QtBot) -> None:
    neuron_sdlg = B2W.NeuronSubgroupDialog(B2Model())

    qtbot.addWidget(neuron_sdlg)

    neuron_sdlg.show()

    qtbot.waitUntil(neuron_sdlg.isVisible)

    cancel_button = neuron_sdlg.button_box.button(QtWidgets.QDialogButtonBox.Cancel)

    qtbot.mouseClick(cancel_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: neuron_sdlg.isVisible() is False)


def test_neuronsubgroupdialog_arg_conversion(qtbot: QtBot) -> None:
    neuron_sdlg = B2W.NeuronSubgroupDialog(B2Model())
    qtbot.add_widget(neuron_sdlg)

    neuron_sdlg.name = "x1"
    neuron_sdlg.target_name = "x"

    neuron_sdlg.definition.setText("1")
    neuron_sdlg.get_subgroup_def()
    assert neuron_sdlg.subgroup_def == 1

    neuron_sdlg.definition.setText("(1,2)")
    neuron_sdlg.get_subgroup_def()
    assert neuron_sdlg.subgroup_def == (1, 2)

    neuron_sdlg.definition.setText("[1,2,3]")
    neuron_sdlg.get_subgroup_def()
    assert neuron_sdlg.subgroup_def == [1, 2, 3]

    neuron_sdlg.definition.setText("wrong")
    neuron_sdlg.get_subgroup_def()

    qtbot.waitUntil(neuron_sdlg.error_dialog.isVisible)

    assert (
        "Error when trying to build subgroup x1 of neurongroup x: malformed node or string"
        in neuron_sdlg.error_dialog.text()
    )
    neuron_sdlg.error_dialog.close()

    neuron_sdlg.definition.setText("{1:2,3:'a'}")
    neuron_sdlg.get_subgroup_def()

    qtbot.waitUntil(neuron_sdlg.error_dialog.isVisible)

    assert (
        neuron_sdlg.error_dialog.text()
        == "subgroup x1 of neurongroup x is defined with a wrong type: <class 'dict'>"
    )
    neuron_sdlg.error_dialog.close()


def test_neuronsubgroup_targetname(qtbot: QtBot) -> None:
    neuron_sdlg = B2W.NeuronSubgroupDialog(B2Model())
    qtbot.add_widget(neuron_sdlg)

    neuron_sdlg.model.neurongroups["x"] = {}

    neuron_sdlg.target.setText("x")

    assert neuron_sdlg.target_name is None

    neuron_sdlg.get_targetname()

    assert neuron_sdlg.target_name == "x"

    assert neuron_sdlg.target_button.isDown()

    neuron_sdlg.target.setText("y")
    neuron_sdlg.get_targetname()

    qtbot.waitUntil(neuron_sdlg.error_dialog.isVisible)

    assert (
        neuron_sdlg.error_dialog.text()
        == "Define the neurongroup first before you define a subgroup for it. y is not known"
    )

    neuron_sdlg.error_dialog.close()


def test_neuronsubgroup_accept(qtbot: QtBot) -> None:
    neuron_sdlg = B2W.NeuronSubgroupDialog(B2Model())
    qtbot.add_widget(neuron_sdlg)

    neuron_sdlg.model.neurongroups["x"] = {}
    neuron_sdlg.model.neurongroups["x"]["model"] = "dv/dt = -v : volt\n"
    neuron_sdlg.model.neurongroups["x"]["neuronnumber"] = 64
    neuron_sdlg.name = "x1"
    neuron_sdlg.target_name = "x"
    neuron_sdlg.subgroup_def = 1

    neuron_sdlg.accept()

    neuron_sdlg.name = None

    neuron_sdlg.accept()

    qtbot.waitUntil(neuron_sdlg.error_dialog.isVisible)

    assert (
        neuron_sdlg.error_dialog.text()
        == "Error, every neuron subgroup needs a name and a defined target"
    )

    neuron_sdlg.error_dialog.close()


def test_neuronsubgroupdialog_parameterdisplay(qtbot: QtBot) -> None:
    neuron_sdlg = B2W.NeuronSubgroupDialog(B2Model())
    qtbot.add_widget(neuron_sdlg)

    neuron_sdlg.name = "x1"
    neuron_sdlg.target_name = "x"
    neuron_sdlg.subgroup_def = (1, 5)

    neuron_sdlg.display_parameters()

    qtbot.waitUntil(neuron_sdlg.parameterwindow.isVisible)
    neuron_sdlg.parameterwindow.close()

    table = neuron_sdlg.parameterwindow.findChild(QtWidgets.QTableWidget)

    assert table.item(0, 0).text() == "name"
    assert table.item(0, 1).text() == "x1"

    assert table.item(1, 0).text() == "target"
    assert table.item(1, 1).text() == "x"

    assert table.item(2, 0).text() == "indices"
    assert table.item(2, 1).text() == "(1, 5)"
