from typing import Any

import brian2 as b2
from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import NeuronGroupDialog
from formozov_project.b2_model_widget import NeuronSubgroupDialog
from formozov_project.b2_model_widget import ParameterDialog
from formozov_project.b2_model_widget import SynapseDialog
from formozov_project.model_editor import EditableNeurongroupTree
from formozov_project.model_editor import EditableParameterTree
from formozov_project.model_editor import ModelEditorDialog
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot


def test_etree_constr(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    stree = EditableParameterTree(mfx_nmda, SynapseDialog, "synapses")
    qtbot.addWidget(stree)

    assert stree.states == mfx_nmda.synapses

    assert len(stree.root.children()) == 1
    for p in stree.root.children():
        assert p.name() == "S"

    synapsenode = stree.root.children()[0]

    for p in synapsenode.children():
        assert p.name() in [
            "origin",
            "target",
            "model",
            "on_pre",
        ]


def test_etree_edit_item(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    stree = EditableParameterTree(mfx_nmda, SynapseDialog, "synapses")
    qtbot.addWidget(stree)

    synapsenode = stree.root.children()[0]
    normal = synapsenode.child("target")
    textfield = synapsenode.child("model")

    # edit normal parameter
    with qtbot.waitSignal(normal.sigValueChanged, timeout=1000, raising=True):
        normal.setValue("other")

    assert stree.model.synapses["S"]["target"] == "other"

    # edit textfield parameter
    with qtbot.waitSignal(textfield.sigValueChanged, timeout=1000, raising=True):
        textfield.setValue("dv/dt = -v: 1")

    assert stree.model.synapses["S"]["model"] == "dv/dt = -v: 1"


def test_tree_add_element(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(EditableParameterTree, "itemAt", lambda _, __: None)
    # patch the itemAt function to make sure we get the item adder dialog

    def exec_pdiag(pdiag: ParameterDialog) -> Any:
        pdiag.name_entry.setText("p")
        pdiag.value_entry.setText("2")
        pdiag.accept_name("parameters")
        pdiag.accept_value()
        pdiag.accept()
        return True

    monkeypatch.setattr(ParameterDialog, "exec", exec_pdiag)

    ptree = EditableParameterTree(mfx_nmda, ParameterDialog, "parameters")
    qtbot.addWidget(ptree)

    ptree.add_item()

    assert ptree.model.parameters["p"] == 2  # type: ignore


def test_ptree_edit_b2quantity(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    ptree = EditableParameterTree(mfx_nmda, ParameterDialog, "parameters")
    qtbot.addWidget(ptree)

    pnode = ptree.root

    a_node = pnode.child("a")

    with qtbot.waitSignal(a_node.sigValueChanged, timeout=1000, raising=True):
        a_node.setValue("1/60 ms")

    assert type(ptree.model.parameters["a"]) is b2.Quantity  # type: ignore
    assert ptree.model.parameters["a"] == ((1 / 60) * b2.msecond)  # type: ignore


def test_neurontree_add_neuronsubgroup(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    def exec_ndiag(ndiag: NeuronSubgroupDialog) -> Any:
        ndiag.name_entry.setText("sub")
        ndiag.target.setText("neuron_input")
        ndiag.definition.setText("(1,2)")
        ndiag.get_targetname()
        ndiag.get_subgroup_def()
        ndiag.accept_name(target="neurongroups")
        ndiag.accept()
        return True

    monkeypatch.setattr(NeuronSubgroupDialog, "exec", exec_ndiag)

    ntree = EditableNeurongroupTree(
        mfx_nmda,
    )

    qtbot.addWidget(ntree)

    assert ntree.states == mfx_nmda.neurongroups

    ntree._add_neuronsubgroup()

    assert "sub" in ntree.model.neurongroups
    assert ntree.model.neurongroups["sub"]["name"] == "sub"
    assert ntree.model.neurongroups["sub"]["target"] == "neuron_input"
    assert ntree.model.neurongroups["sub"]["idx"] == (1, 2)


def test_neurontree_add_neurongroup(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    def exec_ndiag(ndiag: NeuronSubgroupDialog) -> Any:
        ndiag.name_entry.setText("p1")
        ndiag.equation.setText("dv/dt = -v : siemens")
        ndiag.threshold.setText("v < -10 mV")
        ndiag.neuronnumber.setText("25")
        ndiag.solver.setText("euler")

        ndiag.accept_name("neurongroups")
        ndiag.accept_parameter_with_savebutton(
            ndiag.threshold,
            "threshold",
            ndiag.model.define_threshold,
            ndiag.threshold_button,
        )
        ndiag.accept_parameter_with_savebutton(
            ndiag.solver,
            "solver",
            ndiag.model.define_solver_method,
            ndiag.solver_button,
        )
        ndiag.accept_parameter_with_savebutton(
            ndiag.neuronnumber,
            "neuronnumber",
            ndiag.model.define_neuronnumber,
            ndiag.neuronnumber_button,
            lambda v: int(v),
        )
        ndiag.accept_parameter_with_savebutton(
            ndiag.equation, "model", ndiag.model.define_equation, ndiag.equation_button
        )
        ndiag.accept()
        return True

    monkeypatch.setattr(NeuronGroupDialog, "exec", exec_ndiag)

    ntree = EditableNeurongroupTree(
        mfx_nmda,
    )
    qtbot.addWidget(ntree)

    assert ntree.states == mfx_nmda.neurongroups

    ntree.add_item()

    assert "p1" in ntree.model.neurongroups
    assert ntree.model.neurongroups["p1"]["model"] == "dv/dt = -v : siemens\n"
    assert ntree.model.neurongroups["p1"]["th"] == "v < -10 mV"
    assert ntree.model.neurongroups["p1"]["method"] == "euler"
    assert ntree.model.neurongroups["p1"]["n"] == 25


def test_modeldialog(mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(diag)
    assert diag.model.must_be_rebuilt is False

    diag.model.define_neuronnumber("p1", 25)
    diag.model.define_solver_method("p1", "euler")
    diag.model.define_equation("p1", "dv/dt=-v: 1")

    assert diag.model.must_be_rebuilt is True
    diag.accept()

    assert "p1" in diag.model.neurongroups

    assert "p1" in diag.model.network.get_states()
