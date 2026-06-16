"""Tests for `EditableExternalSimParamsTree` and the model editor dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pyqtgraph.parametertree import Parameter
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui

from neuroseq.model_editor import EditableExternalSimParamsTree
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.project import Project


@pytest.fixture
def sample_params() -> dict[str, dict]:
    return {
        "num_lat": {
            "value": 50,
            "default": 50,
            "type": "int",
            "group": "Network",
            "description": "Latent neurons.",
        },
        "C_v": {
            "value": 1.0,
            "default": 1.0,
            "type": "float",
            "group": "Neuron",
            "description": "Dendritic capacitance.",
        },
        "E_l": {
            "value": -70.0,
            "default": -70.0,
            "type": "float",
            "group": "Neuron",
            "description": "Leak reversal.",
        },
        "learn": {
            "value": True,
            "default": True,
            "type": "bool",
            "group": "Simulation",
            "description": "Enable plasticity.",
        },
    }


def _find_leaf(tree: EditableExternalSimParamsTree, name: str):
    for group in tree.root.children():
        for leaf in group.children():
            if leaf.name() == name:
                return leaf
    raise AssertionError(f"Parameter {name!r} not found in tree")


def test_tree_groups_and_types(qtbot: QtBot, sample_params):
    tree = EditableExternalSimParamsTree(sample_params)
    qtbot.addWidget(tree)

    group_names = [g.name() for g in tree.root.children()]
    # Groups appear in insertion order of params_dict.
    assert group_names == ["Network", "Neuron", "Simulation"]

    assert _find_leaf(tree, "num_lat").type() == "int"
    assert _find_leaf(tree, "C_v").type() == "float"
    assert _find_leaf(tree, "learn").type() == "bool"


def test_tree_edit_writes_back_to_dict(qtbot: QtBot, sample_params):
    tree = EditableExternalSimParamsTree(sample_params)
    qtbot.addWidget(tree)

    leaf = _find_leaf(tree, "num_lat")
    leaf.setValue(12)
    assert sample_params["num_lat"]["value"] == 12

    bool_leaf = _find_leaf(tree, "learn")
    bool_leaf.setValue(False)
    assert sample_params["learn"]["value"] is False


def test_dialog_external_mode_when_path_set(qtbot: QtBot, sample_params):
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = sample_params

    dialog = ModelEditorDialog(project.b2_model, parent=None, project=project)
    qtbot.addWidget(dialog)

    # External-mode dialog gets a deep copy and exposes it as
    # external_simulate_params; the original on Project is untouched until accept.
    assert dialog.external_simulate_params is not None
    assert dialog.external_simulate_params == sample_params
    assert dialog.external_simulate_params is not sample_params

    # Editing through the tree mutates the dialog's copy only.
    leaf = _find_leaf(dialog.external_sim_tree, "num_lat")
    leaf.setValue(7)
    assert dialog.external_simulate_params["num_lat"]["value"] == 7
    assert project.external_simulate_params["num_lat"]["value"] == 50


def test_dialog_b2_mode_when_path_unset(qtbot: QtBot):
    project = Project()
    assert project.external_simulate_path is None

    dialog = ModelEditorDialog(project.b2_model, parent=None, project=project)
    qtbot.addWidget(dialog)

    assert dialog.external_simulate_params is None
    # b2_model copy populated as usual
    assert dialog.b2model is not None


def test_dialog_b2_mode_when_project_omitted(qtbot: QtBot):
    """Backward compatibility: caller without `project=` still works."""
    project = Project()
    dialog = ModelEditorDialog(project.b2_model, parent=None)
    qtbot.addWidget(dialog)

    assert dialog.external_simulate_params is None
    assert dialog.project is None


def test_dialog_accept_in_external_mode_does_not_touch_b2(
    qtbot: QtBot, sample_params, monkeypatch
):
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = sample_params

    dialog = ModelEditorDialog(project.b2_model, parent=None, project=project)
    qtbot.addWidget(dialog)

    # If accept() in external mode tried to call into b2model, the b2model
    # attribute has must_be_rebuilt=False and assemble_network would still get
    # called. Set a sentinel that fails if that path is taken.
    def _boom():
        raise AssertionError("b2 path should not be executed in external mode")

    monkeypatch.setattr(dialog.b2model, "assemble_network", _boom)
    dialog.b2model.must_be_rebuilt = True  # would trigger the boom in b2 path

    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted


def test_connect_external_sim_params_skips_non_group_children(
    qtbot: QtBot, sample_params
) -> None:
    tree = EditableExternalSimParamsTree(sample_params)
    qtbot.addWidget(tree)
    connected: list[str] = []
    tree._connect_leaf = lambda leaf: connected.append(leaf.name())  # type: ignore[method-assign]

    fake_root = SimpleNamespace(
        children=lambda: [
            SimpleNamespace(type=lambda: "str", children=lambda: []),
            SimpleNamespace(
                type=lambda: "group",
                children=lambda: [SimpleNamespace(name=lambda: "leaf_a")],
            ),
        ]
    )

    tree._connect_external_sim_params(fake_root)
    assert connected == ["leaf_a"]


def test_context_menu_event_passes_through_on_item_hit(
    qtbot: QtBot, sample_params, monkeypatch
) -> None:
    tree = EditableExternalSimParamsTree(sample_params)
    qtbot.addWidget(tree)

    calls: list[object] = []
    monkeypatch.setattr(tree, "itemAt", lambda pos: object())
    monkeypatch.setattr(
        Parameter,
        "contextMenuEvent",
        lambda self, ev: calls.append(ev),
        raising=False,
    )
    monkeypatch.setattr(
        "pyqtgraph.parametertree.ParameterTree.contextMenuEvent",
        lambda self, ev: calls.append(ev),
    )

    event = QtGui.QContextMenuEvent(
        QtGui.QContextMenuEvent.Reason.Mouse,
        QtCore.QPoint(1, 1),
    )
    tree.contextMenuEvent(event)
    assert calls == [event]
