from typing import Any

import brian2 as b2
from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import NeuronGroupDialog
from formozov_project.b2_model_widget import NeuronSubgroupDialog
from formozov_project.b2_model_widget import ParameterDialog
from formozov_project.b2_model_widget import SynapseDialog
from formozov_project.model_editor import EagerQComboBox
from formozov_project.model_editor import EditableMonitorTree
from formozov_project.model_editor import EditableNeurongroupTree
from formozov_project.model_editor import EditableParameterTree
from formozov_project.model_editor import EditableSynapseTree
from formozov_project.model_editor import ModelEditorDialog
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore


def test_EagerComboBox(qtbot: QtBot) -> None:
    widget = EagerQComboBox()
    signal_emitted = False

    def on_clicked():
        nonlocal signal_emitted
        signal_emitted = True

    widget.clicked.connect(on_clicked)

    assert signal_emitted is False

    qtbot.mousePress(
        widget, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(10, 10)
    )
    assert signal_emitted is True


def test_etree_constr(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    stree = EditableSynapseTree(mfx_nmda)
    qtbot.addWidget(stree)

    assert stree.states == mfx_nmda.synapses

    assert len(stree.root.children()) == 1
    for p in stree.root.children():
        assert p.name() == "S"

    synapsenode = stree.root.children()[0]

    listnodes = ["method", "origin", "target"]

    for p in synapsenode.children():
        assert p.name() in [
            "origin",
            "target",
            "model",
            "on_pre",
        ]

        if p.name() in listnodes:
            assert p.type() == "updatinglist"


def test_etree_edit_item(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    stree = EditableParameterTree(mfx_nmda, SynapseDialog, "synapses")
    qtbot.addWidget(stree)

    synapsenode = stree.root.children()[0]
    normal = synapsenode.child("target")
    textfield = synapsenode.child("model")

    # edit normal parameter
    with qtbot.waitSignal(normal.sigValueChanged, timeout=1000, raising=True):
        normal.setValue("other")

    assert stree.b2model.synapses["S"]["target"] == "other"

    assert normal.value() == "other"

    # edit textfield parameter
    with qtbot.waitSignal(textfield.sigValueChanged, timeout=1000, raising=True):
        textfield.setValue("dv/dt = -v: 1")

    assert stree.b2model.synapses["S"]["model"] == "dv/dt = -v: 1"


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
    assert ptree.b2model.must_be_rebuilt is False

    ptree.add_item()

    assert ptree.b2model.parameters["p"] == 2  # type: ignore
    assert ptree.b2model.must_be_rebuilt is False


def test_ptree_edit_b2quantity(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    ptree = EditableParameterTree(mfx_nmda, ParameterDialog, "parameters")
    qtbot.addWidget(ptree)

    pnode = ptree.root

    a_node = pnode.child("a")

    with qtbot.waitSignal(a_node.sigValueChanged, timeout=1000, raising=True):
        a_node.setValue("1/60 * ms")

    assert ptree.b2model.parameters["a"] == ((1 / 60) * b2.msecond)  # type: ignore
    assert type(ptree.b2model.parameters["a"]) is b2.Quantity  # type: ignore
    assert ptree.b2model.must_be_rebuilt is False


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

    assert "sub" in ntree.b2model.neurongroups
    assert ntree.b2model.neurongroups["sub"]["name"] == "sub"
    assert ntree.b2model.neurongroups["sub"]["target"] == "neuron_input"
    assert ntree.b2model.neurongroups["sub"]["idxs"] == (1, 2)
    assert ntree.b2model.must_be_rebuilt is True


def test_rename_tree_element(
    mfx_nmda: B2Model, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    ptree = EditableParameterTree(mfx_nmda, ParameterDialog, "parameters")
    qtbot.addWidget(ptree)
    assert ptree.b2model.must_be_rebuilt is False

    pnode = ptree.root

    node = pnode.child("a")

    with qtbot.waitSignal(node.sigNameChanged, timeout=1000, raising=True):
        node.setName("x")

    item = node.makeTreeItem(0)

    assert item.text(0) == "x"

    assert "x" in ptree.b2model.parameters
    assert "a" not in ptree.b2model.parameters
    assert ptree.b2model.must_be_rebuilt is False

    node = pnode.child("x")

    with qtbot.waitSignal(node.sigNameChanged, timeout=1000, raising=True):
        node.setName("y")

    item = node.makeTreeItem(0)
    assert item.text(0) == "y"

    assert "y" in ptree.b2model.parameters
    assert "x" not in ptree.b2model.parameters


def test_neurontree_add_neurongroup_list_update(
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

    # add synpase tree to assure later that the data has been updated
    stree = EditableSynapseTree(mfx_nmda)
    qtbot.addWidget(stree)

    # add monitor tree to make sure that the update system for dropdown menus works
    mtree = EditableMonitorTree(mfx_nmda)
    qtbot.addWidget(mtree)
    assert mtree.states == mfx_nmda.monitors

    # get out the origin subwidget for synapses
    origin = stree.findItems(
        "origin",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    all_names = [origin.widget.itemText(i) for i in range(origin.widget.count())]

    assert set(all_names) == set(["neurons", "neuron_input", "input_subgroup"])

    assert ntree.states == mfx_nmda.neurongroups

    # get the variables widget for the monitor and check that it is filled as it should
    mn_item = mtree.findItems(
        "Mn",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    source = mn_item.child(1).value()
    typename = mn_item.child(2).value()
    var_selection = mn_item.child(3)
    variables = mtree._make_selection_list(typename, source)

    # make expected list of selectable items
    vs = []
    for obj in mtree.b2model.network.objects:
        if isinstance(obj, b2.StateMonitor | b2.EventMonitor):
            continue
        else:
            vs.extend(list(obj.equations.names) + list(obj.variables.keys()))

    vs = list(set(vs)) + [
        "None",
    ]

    entries = [
        var_selection.widget.itemText(i) for i in range(0, var_selection.widget.count())
    ]

    assert set(vs) == set(entries)

    assert set(variables) == set(list(mtree.b2model.network["neurons"].equations.names))

    variables = mtree._make_selection_list("SpikeMonitor", source)
    assert set(variables) == set(
        list(mtree.b2model.network["neurons"].variables)
        + [
            "None",
        ]
    )

    ntree.add_item()

    assert "p1" in ntree.b2model.neurongroups
    assert ntree.b2model.neurongroups["p1"]["model"] == "dv/dt = -v : siemens\n"
    assert ntree.b2model.neurongroups["p1"]["th"] == "v < -10 mV"
    assert ntree.b2model.neurongroups["p1"]["method"] == "euler"
    assert ntree.b2model.neurongroups["p1"]["neuronnumber"] == 25
    assert ntree.b2model.must_be_rebuilt is True

    # trigger list update for synapse origin
    origin._update_list()
    all_names = [origin.widget.itemText(i) for i in range(origin.widget.count())]
    assert set(all_names) == set(["neurons", "neuron_input", "p1", "input_subgroup"])

    var_selection.setSelected(True)
    assert var_selection.isSelected()

    updated_list = mtree._update_element_list()

    assert set(updated_list) == set(
        list(mtree.b2model.network["neurons"].equations.names)
    )


def test_modeldialog_accept(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(diag)
    assert diag.b2model.must_be_rebuilt is False
    specs = mfx_nmda.get_model_definition()
    assert diag.backup == specs
    assert list(diag.functions_backup.keys()) == list(mfx_nmda.functions.keys())

    diag.b2model.define_neuronnumber("p1", 25)
    diag.b2model.define_solver_method("p1", "euler")
    diag.b2model.define_equation("p1", "dv/dt=-v: 1")

    assert diag.b2model.must_be_rebuilt is True

    diag.accept()

    assert "p1" in diag.b2model.neurongroups
    assert "p1" in [obj.name for obj in diag.b2model.network.objects]

    def raise_error(_):
        raise RuntimeError("dummy")

    monkeypatch.setattr(B2Model, "assemble_network", raise_error)
    diag.b2model.define_neuronnumber("p1", 35)

    diag.accept()

    assert qmessagebox_calls[0] == {
        "name": "critical",
        "text": "Error when rebuilding the model after editing: dummy",
        "title": "Model rebuilding error",
    }


def test_modeldialog_reject(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.add_widget(diag)
    specs = mfx_nmda.get_model_definition()
    assert diag.backup == specs
    assert list(diag.functions_backup.keys()) == list(mfx_nmda.functions.keys())

    diag.b2model.functions["monitors"]["Mn"]["delete"]()

    assert "Mn" not in diag.b2model.monitors
    assert "Mn" in [obj.name for obj in diag.b2model.network.objects]

    diag.reject()

    assert "Mn" in diag.b2model.monitors
    assert "Mn" in [obj.name for obj in diag.b2model.network.objects]

    def raise_error(_, __):
        raise RuntimeError("dummy")

    monkeypatch.setattr(B2Model, "define_b2seed", raise_error)

    diag.reject()

    assert qmessagebox_calls == [
        {
            "name": "critical",
            "text": "Error when restoring old model: dummy",
            "title": "Model specification error",
        },
    ]


def test_monitordialog_with_run(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
):
    mfx_nmda.define_monitor_dt("M", 200 * b2.ms)
    mfx_nmda.assemble_network()
    assert mfx_nmda.monitors["M"]["dt"] == 200 * b2.ms

    diag = EditableMonitorTree(mfx_nmda)

    m = diag.findItems(
        "M",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    mfx_nmda.run_simulation(1 * b2.second)

    assert mfx_nmda.network["M"].g.shape[1] == 5
    dtparam = m.child(4)

    mfx_nmda.assemble_network()

    with qtbot.waitSignal(dtparam.param.sigValueChanged, timeout=1000, raising=True):
        dtparam.param.setValue("100 * ms")

    assert mfx_nmda.monitors["M"]["dt"] == 100 * b2.ms

    mfx_nmda.assemble_network()
    assert mfx_nmda.network["M"].clock.dt == 100 * b2.ms

    mfx_nmda.run_simulation(1 * b2.second)

    assert mfx_nmda.network["M"].g.shape[1] == 10


def test_initial_conditions(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.add_widget(diag)

    # get the initial condition tree

    # modify an existing element

    # add a new elemnet -> strin expression

    # add a new elemnet -> numpy based brian2 quantity

    # add a new element -> scalar brian2 quantity
