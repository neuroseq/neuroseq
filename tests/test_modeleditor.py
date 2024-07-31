import brian2 as b2
from formozov_project.b2_model import B2Model
from formozov_project.model_editor import EagerQComboBox
from formozov_project.model_editor import EditableMonitorTree
from formozov_project.model_editor import EditableNeurongroupTree
from formozov_project.model_editor import EditableParameterTree
from formozov_project.model_editor import EditableSynapseTree
from formozov_project.model_editor import ModelEditorDialog
from pyqtgraph.parametertree import Parameter
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
    stree = EditableParameterTree(mfx_nmda, "synapses", mfx_nmda.define_default_synapse)
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


def test_tree_add_new_parameter(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    ptree = EditableParameterTree(
        mfx_nmda, "parameters", mfx_nmda.define_default_parameter
    )
    qtbot.addWidget(ptree)

    assert (
        len(
            ptree.findItems(
                "new_parameter",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 0
    )

    assert "new_parameter" not in mfx_nmda.parameters
    ptree._add_item()
    assert "new_parameter" in mfx_nmda.parameters

    assert mfx_nmda.parameters["new_parameter"] is None

    assert (
        len(
            ptree.findItems(
                "new_parameter",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 1
    )


def test_tree_add_new_neurongroups(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableNeurongroupTree(
        mfx_nmda,
    )
    qtbot.addWidget(ntree)

    assert (
        len(
            ntree.findItems(
                "new_neurongroup",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 0
    )

    assert "new_neurongroup" not in mfx_nmda.neurongroups
    ntree._add_item()
    assert "new_neurongroup" in mfx_nmda.neurongroups
    assert (
        len(
            ntree.findItems(
                "new_neurongroup",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 1
    )

    assert (
        len(
            ntree.findItems(
                "new_neuronsubgroup",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 0
    )
    assert "new_neuronsubgroup" not in mfx_nmda.neurongroups
    ntree._add_neuronsubgroup()

    assert (
        len(
            ntree.findItems(
                "new_neuronsubgroup",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                1,
            )
        )
        == 0
    )
    assert "new_neuronsubgroup" in mfx_nmda.neurongroups


def test_tree_add_new_synapses(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableSynapseTree(
        mfx_nmda,
    )
    qtbot.addWidget(ntree)

    assert (
        len(
            ntree.findItems(
                "new_synapse",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 0
    )

    assert "new_synapse" not in mfx_nmda.synapses
    ntree._add_item()
    assert "new_synapse" in mfx_nmda.synapses
    assert (
        len(
            ntree.findItems(
                "new_synapse",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 1
    )


def test_tree_add_new_monitor(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableMonitorTree(
        mfx_nmda,
    )
    qtbot.addWidget(ntree)

    assert (
        len(
            ntree.findItems(
                "new_monitor",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 0
    )

    assert "new_monitor" not in mfx_nmda.monitors
    ntree._add_item()
    assert "new_monitor" in mfx_nmda.monitors
    assert (
        len(
            ntree.findItems(
                "new_monitor",
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
        )
        == 1
    )


def test_add_new_init(mfx_nmda, qtbot: QtBot) -> None:
    itree = EditableParameterTree(
        mfx_nmda, "initials", mfx_nmda.define_default_init, add_to_groups=True
    )

    qtbot.add_widget(itree)

    assert "group" not in mfx_nmda.initials
    itree._add_item()
    assert "group" in mfx_nmda.initials


def test_add_new_init_element(mfx_nmda, qtbot: QtBot) -> None:
    itree = EditableParameterTree(
        mfx_nmda, "initials", mfx_nmda.define_default_init, add_to_groups=True
    )

    qtbot.add_widget(itree)

    sn = itree.findItems(
        "S",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    new_param = Parameter.create(
        **{
            "name": "new_element",  # dummy values
        }
    )

    assert "variable" not in mfx_nmda.initials["S"]

    with qtbot.waitSignal(sn.param.sigAddNew, timeout=1000, raising=True):
        sn.param.addNew(new_param)

    assert "variable" in mfx_nmda.initials["S"]
    assert "value" in mfx_nmda.initials["S"]["variable"]


def test_delete_neurongroup(mfx_nmda, qtbot: QtBot) -> None:
    # send delete signal -> then check that group is gone
    ntree = EditableNeurongroupTree(mfx_nmda)
    qtbot.add_widget(ntree)

    # find neurongroup element with name 'neurongroup
    assert "neurons" in mfx_nmda.neurongroups

    ng = ntree.findItems(
        "neurons",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]
    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "neurons" not in mfx_nmda.neurongroups


def test_delete_neurongroup_element(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableNeurongroupTree(mfx_nmda)
    qtbot.add_widget(ntree)

    # find neurongroup element with name 'neurons
    assert "neuronnumber" in mfx_nmda.neurongroups["neurons"]

    ngs = ntree.findItems(
        "neuronnumber",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )

    ng = [p.param for p in ngs if p.parent().param.name() == "neurons"][0]

    with qtbot.waitSignal(ng.sigRemoved, timeout=1000, raising=True):
        ng.remove()

    assert "neuronnumber" not in mfx_nmda.neurongroups["neurons"]


def test_delete_synapse(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableSynapseTree(mfx_nmda)
    qtbot.add_widget(ntree)

    assert "S" in mfx_nmda.synapses

    ng = ntree.findItems(
        "S",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]
    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "S" not in mfx_nmda.synapses


def test_delete_synapse_element(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableSynapseTree(mfx_nmda)
    qtbot.add_widget(ntree)

    assert "origin" in mfx_nmda.synapses["S"]

    ngs = ntree.findItems(
        "origin",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )

    ng = [p.param for p in ngs if p.parent().param.name() == "S"][0]

    with qtbot.waitSignal(ng.sigRemoved, timeout=1000, raising=True):
        ng.remove()

    assert "origin" not in mfx_nmda.synapses["S"]


def test_delete_monitor(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableMonitorTree(mfx_nmda)
    qtbot.add_widget(ntree)

    assert "Mn" in mfx_nmda.monitors

    ng = ntree.findItems(
        "Mn",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]
    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "Mn" not in mfx_nmda.monitors


def test_delete_monitor_element(mfx_nmda, qtbot: QtBot) -> None:
    ntree = EditableMonitorTree(mfx_nmda)
    qtbot.add_widget(ntree)

    assert "source" in mfx_nmda.monitors["M"]

    ngs = ntree.findItems(
        "source",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )

    ng = [p.param for p in ngs if p.parent().param.name() == "M"][0]

    with qtbot.waitSignal(ng.sigRemoved, timeout=1000, raising=True):
        ng.remove()

    assert "source" not in mfx_nmda.monitors["M"]


def test_delete_parameter(mfx_nmda, qtbot: QtBot) -> None:
    tree = EditableParameterTree(
        mfx_nmda, "parameters", mfx_nmda.define_default_parameter
    )
    qtbot.add_widget(tree)

    assert "a" in mfx_nmda.parameters

    ng = tree.findItems(
        "a",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "a" not in mfx_nmda.parameters


def test_delete_init(mfx_nmda, qtbot: QtBot) -> None:
    tree = EditableParameterTree(
        mfx_nmda, "initials", mfx_nmda.define_default_init, add_to_groups=True
    )
    qtbot.add_widget(tree)

    assert "S" in mfx_nmda.initials

    ng = tree.findItems(
        "S",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "S" not in mfx_nmda.initials


def test_delete_init_element(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    itree = EditableParameterTree(
        mfx_nmda, "initials", mfx_nmda.define_default_init, add_to_groups=True
    )
    qtbot.add_widget(itree)

    assert "w" in mfx_nmda.initials["S"]

    ng = itree.findItems(
        "w",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    with qtbot.waitSignal(ng.param.sigRemoved, timeout=1000, raising=True):
        ng.param.remove()

    assert "w" not in mfx_nmda.initials["S"]


def test_ptree_edit_b2quantity(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    ptree = EditableParameterTree(
        mfx_nmda, "parameters", mfx_nmda.define_default_parameter
    )
    qtbot.addWidget(ptree)

    pnode = ptree.root

    a_node = pnode.child("a")

    with qtbot.waitSignal(a_node.sigValueChanged, timeout=1000, raising=True):
        a_node.setValue("1/60 * ms")

    assert ptree.b2model.parameters["a"] == ((1 / 60) * b2.msecond)  # type: ignore
    assert type(ptree.b2model.parameters["a"]) is b2.Quantity  # type: ignore
    assert ptree.b2model.must_be_rebuilt is False


def test_rename_tree_element(mfx_nmda: B2Model, qtbot: QtBot) -> None:
    ptree = EditableParameterTree(
        mfx_nmda, "parameters", mfx_nmda.define_default_parameter
    )
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
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    ntree = EditableNeurongroupTree(
        mfx_nmda,
    )
    qtbot.addWidget(ntree)

    ntree._add_neuronsubgroup()

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

    assert set(all_names) == set(
        ["neurons", "neuron_input", "new_neuronsubgroup", "input_subgroup"]
    )

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

    ntree._add_item()

    assert "new_neurongroup" in ntree.b2model.neurongroups
    assert ntree.b2model.must_be_rebuilt is True

    # trigger list update for synapse origin
    origin._update_list()
    all_names = [origin.widget.itemText(i) for i in range(origin.widget.count())]
    assert set(all_names) == set(
        [
            "neurons",
            "neuron_input",
            "new_neurongroup",
            "input_subgroup",
            "new_neuronsubgroup",
        ]
    )

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

    diag.b2model.define_neuronnumber("p1", 25)
    diag.b2model.define_solver_method("p1", "euler")
    diag.b2model.define_equation("p1", "dv/dt=-v: 1")
    assert diag.b2model.must_be_rebuilt is True
    diag.accept()
    assert "p1" in diag.b2model.neurongroups
    assert "p1" in [obj.name for obj in diag.b2model.network.objects]


def test_modeldialog_accept_exception(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(diag)

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

    diag.b2model.functions["monitors"]["Mn"]["delete"]()

    assert "Mn" not in diag.b2model.monitors
    assert "Mn" in [obj.name for obj in diag.b2model.network.objects]

    diag.reject()

    assert "Mn" in diag.b2model.monitors
    assert "Mn" in [obj.name for obj in diag.b2model.network.objects]


def test_modeldialog_reject_exception(
    mfx_nmda: B2Model, qtbot: QtBot, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.add_widget(diag)

    def raise_error(_):
        raise RuntimeError("dummy")

    monkeypatch.setattr(B2Model, "load_model", raise_error)

    diag.reject()

    assert qmessagebox_calls == [
        {
            "name": "critical",
            "text": "Error when restoring old model: dummy",
            "title": "Model specification error",
        },
    ]


def test_method_selection(
    mfx_nmda: B2Model,
    qtbot: QtBot,
):
    diag = ModelEditorDialog(mfx_nmda)
    qtbot.add_widget(diag)

    # find tree
    forest = diag.findChildren(EditableParameterTree)
    tree = [t for t in forest if isinstance(t, EditableNeurongroupTree)][0]

    # find dropdown for method for the neurongroup "neuron_input"
    method_dropdown_elements = tree.findItems(
        "method",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )

    method_dropdown = list(
        filter(
            lambda w: w.param.parent().name() == "neuron_input",
            method_dropdown_elements,
        )
    )[0]

    with qtbot.waitSignal(
        method_dropdown.param.sigValueChanged, timeout=1000, raising=True
    ):
        method_dropdown.param.setValue("exponential_euler")

    assert mfx_nmda.neurongroups["neuron_input"]["method"] == "exponential_euler"

    assert mfx_nmda.network["Mn"].g.shape == (1, 0)
    # try to run
    mfx_nmda.run_simulation(1.0 * b2.second)

    exp_euler_data_shape = mfx_nmda.network["Mn"].g.shape

    with qtbot.waitSignal(
        method_dropdown.param.sigValueChanged, timeout=1000, raising=True
    ):
        method_dropdown.param.setValue("euler")
    assert mfx_nmda.neurongroups["neuron_input"]["method"] == "euler"

    mfx_nmda.assemble_network()
    mfx_nmda.run_simulation(1.0 * b2.second)

    euler_data_shape = mfx_nmda.network["Mn"].g.shape

    assert exp_euler_data_shape == euler_data_shape


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
