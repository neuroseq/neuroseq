import brian2 as b2
from neuroseq.b2_model import B2Model
from neuroseq.model_editor import EagerQComboBox
from neuroseq.model_editor import EditableMonitorTree
from neuroseq.model_editor import EditableNeurongroupTree
from neuroseq.model_editor import EditableParameterTree
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog
from pyqtgraph.parametertree import Parameter
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


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


def test_default_neurongroup_addition(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    mdiag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(mdiag)

    # find add neurongroup button
    button = next(
        filter(
            lambda p: p.text() == "New Neurongroup",
            mdiag.findChildren(QtWidgets.QPushButton),
        )
    )
    assert button is not None
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    ntree = mdiag.findChild(EditableNeurongroupTree)

    assert "new_neurongroup" in mdiag.b2model.neurongroups
    for name, should in zip(
        [
            "neuronnumber",
            "model",
            "method",
            "dt",
            "reset",
            "th",
            "refractory",
        ],
        [0, "dx/dt = y : 1", "-", 10 * b2.msecond, "0", "0", "0"],
        strict=True,
    ):
        assert mdiag.b2model.neurongroups["new_neurongroup"][name] == should
        o = [
            o
            for o in ntree.findItems(
                name,
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
            if o.parent().param.name() == "new_neurongroup"
        ][0]

        if isinstance(o.widget, QtWidgets.QLineEdit):
            assert (
                o.widget.text() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        elif isinstance(o.widget, QtWidgets.QTextEdit):
            assert (
                o.widget.toPlainText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        else:
            assert (
                o.widget.currentText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )


def test_default_synapse_addition(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    mdiag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(mdiag)

    button = next(
        filter(
            lambda p: p.text() == "New Synapse",
            mdiag.findChildren(QtWidgets.QPushButton),
        )
    )
    assert button is not None
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    stree = mdiag.findChild(EditableSynapseTree)

    assert "new_synapse" in mdiag.b2model.synapses

    for name, should in zip(
        [
            "origin",
            "target",
            "method",
            "model",
            "on_pre",
            "on_post",
            "dt",
            "delay",
            "p",
            "connection_condition",
        ],
        [
            "-",
            "-",
            "-",
            "dx/dt = y : 1",
            "x=y",
            "x=y",
            10 * b2.ms,
            0 * b2.ms,
            0.0,
            "x!=y",
        ],
        strict=True,
    ):
        assert mdiag.b2model.synapses["new_synapse"][name] == should

        o = [
            o
            for o in stree.findItems(
                name,
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
            if o.parent().param.name() == "new_synapse"
        ][0]

        if isinstance(o.widget, QtWidgets.QLineEdit):
            assert (
                o.widget.text() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        elif isinstance(o.widget, QtWidgets.QTextEdit):
            assert (
                o.widget.toPlainText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        else:
            assert (
                o.widget.currentText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )


def test_default_monitor_addition(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    mdiag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(mdiag)

    button = next(
        filter(
            lambda p: p.text() == "New Monitor",
            mdiag.findChildren(QtWidgets.QPushButton),
        )
    )
    assert button is not None
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    stree = mdiag.findChild(EditableMonitorTree)

    assert "new_monitor" in mdiag.b2model.monitors

    for name, should in zip(
        ["type", "source", "variable", "dt"],
        ["-", "-", "-", None],
        strict=True,
    ):
        assert mdiag.b2model.monitors["new_monitor"][name] == should

        o = [
            o
            for o in stree.findItems(
                name,
                QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
                0,
            )
            if o.parent().param.name() == "new_monitor"
        ][0]

        if isinstance(o.widget, QtWidgets.QLineEdit):
            assert (
                o.widget.text() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        elif isinstance(o.widget, QtWidgets.QTextEdit):
            assert (
                o.widget.toPlainText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )
        else:
            assert (
                o.widget.currentText() == should.__repr__()
                if isinstance(should, b2.Quantity)
                else str(should)
            )


def test_default_parameter_addition(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    mdiag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(mdiag)

    button = next(
        filter(
            lambda p: p.text() == "New Parameter",
            mdiag.findChildren(QtWidgets.QPushButton),
        )
    )
    assert button is not None
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    forest = mdiag.findChildren(
        EditableParameterTree,
    )
    stree = [tree for tree in forest if tree.name == "parameters"][0]

    assert "new_parameter" in mdiag.b2model.parameters

    assert mdiag.b2model.parameters["new_parameter"] is None

    o = stree.findItems(
        "new_parameter",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    assert o.widget.text() == "None"


def test_default_init_addition(
    mfx_nmda: B2Model,
    qtbot: QtBot,
) -> None:
    mdiag = ModelEditorDialog(mfx_nmda)
    qtbot.addWidget(mdiag)

    button = next(
        filter(
            lambda p: p.text() == "New Initial Condition",
            mdiag.findChildren(QtWidgets.QPushButton),
        )
    )
    assert button is not None

    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    forest = mdiag.findChildren(
        EditableParameterTree,
    )
    stree = [tree for tree in forest if tree.name == "initials"][0]

    assert "group" in mdiag.b2model.initials

    assert mdiag.b2model.initials["group"]["value"] is None

    o = [
        o
        for o in stree.findItems(
            "value",
            QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
            0,
        )
        if o.parent().param.name() == "group"
    ][0]

    assert o.widget.text() == "None"


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
    widgets = [
        o
        for o in stree.findItems(
            "origin",
            QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
            0,
        )
    ]

    origin = widgets[0]

    all_names = [origin.widget.itemText(i) for i in range(origin.widget.count())]

    assert set(all_names) == set(
        ["neurons", "neuron_input", "new_neuronsubgroup", "input_subgroup", "-"]
    )
    assert ntree.states == mfx_nmda.neurongroups

    # get the variables widget for the monitor and check that it is filled as it should
    mn_item = mtree.findItems(
        "Mn",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    source = mn_item.child(0).value()
    typename = mn_item.child(1).value()
    var_selection = mn_item.child(2)
    variables = mtree._make_selection_list(typename, source)

    # make expected list of selectable items
    vs = []
    for obj in mtree.b2model.network.objects:
        if isinstance(obj, b2.StateMonitor | b2.EventMonitor):
            continue
        else:
            vs.extend(list(obj.equations.names) + list(obj.variables.keys()))

    vs = list(set(vs)) + ["None", "-"]

    entries = [
        var_selection.widget.itemText(i) for i in range(0, var_selection.widget.count())
    ]

    assert set(vs) == set(entries)

    assert set(variables) == set(
        list(mtree.b2model.network["neurons"].equations.names)
        + [
            "-",
        ]
    )

    variables = mtree._make_selection_list("SpikeMonitor", source)
    assert set(variables) == set(
        list(mtree.b2model.network["neurons"].variables) + ["None", "-"]
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
            "-",
        ]
    )

    var_selection.setSelected(True)
    assert var_selection.isSelected()

    updated_list = mtree._update_element_list()

    assert set(updated_list) == set(
        list(mtree.b2model.network["neurons"].equations.names)
        + [
            "-",
        ]
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

    # FIXME: this signal connection is wrong. It needs to start with the selection of the menu item, which should then trigger the 'sigValueChanged' signal in the end.
    # it does so in the real app, but not in the tests

    with qtbot.waitSignal(
        method_dropdown.param.sigValueChanged, timeout=1000, raising=True
    ):
        method_dropdown.param.setValue("exponential_euler")

    assert diag.b2model.neurongroups["neuron_input"]["method"] == "exponential_euler"
    assert mfx_nmda.neurongroups["neuron_input"]["method"] == "euler"

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


def test_monitordialog_modify_with_run(
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
    dtparam = m.child(3)

    mfx_nmda.assemble_network()

    with qtbot.waitSignal(dtparam.param.sigValueChanged, timeout=1000, raising=True):
        dtparam.param.setValue("100 * ms")

    assert mfx_nmda.monitors["M"]["dt"] == 100 * b2.ms

    mfx_nmda.assemble_network()
    assert mfx_nmda.network["M"].clock.dt == 100 * b2.ms

    mfx_nmda.run_simulation(1 * b2.second)

    assert mfx_nmda.network["M"].g.shape[1] == 10
