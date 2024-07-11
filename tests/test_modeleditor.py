import brian2 as b2
from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import ParameterDialog
from formozov_project.b2_model_widget import SynapseDialog
from formozov_project.model_editor import EditableParameterTree
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


# def test_stree_add_element(mfx_nmda: B2Model, qtbot: QtBot) -> None:
#     stree = EditableParameterTree(mfx_nmda, SynapseDialog, "synapses")
#     qtbot.addWidget(stree)


def test_ptree_edit_b2quantity(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    ptree = EditableParameterTree(mfx_nmda, ParameterDialog, "parameters")
    qtbot.addWidget(ptree)

    pnode = ptree.root

    print([p.name() for p in pnode])

    a_node = pnode.child("a")

    with qtbot.waitSignal(a_node.sigValueChanged, timeout=1000, raising=True):
        a_node.setValue("1/20 ms")

    assert type(ptree.model.parameters["a"]) is b2.Quantity
    assert ptree.model.parameters["a"] == ((1 / 20) * b2.msecond)


# def test_neurontree_add_neurongroup(mfx_nmda: B2Model) -> None:
#     stree = EditableNeurongroupTree(
#         mfx_nmda,
#     )

#     assert stree.states == mfx_nmda.neurongroups


# def test_neurontree_add_neuronsubgroup(mfx_nmda: B2Model) -> None:
#     diag = ModelEditorDialog(mfx_nmda)


# def test_modeldialog_constr(mfx_nmda: B2Model) -> None:
#     diag = ModelEditorDialog(mfx_nmda)


# def test_modeldialog_accept(mfx_nmda: B2Model) -> None:
#     diag = ModelEditorDialog(mfx_nmda)
