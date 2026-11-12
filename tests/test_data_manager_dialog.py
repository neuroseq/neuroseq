import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.data_manager_dialog import DESCRIPTION_COLUMN
from neuroseq.data_manager_dialog import NAME_COLUMN
from neuroseq.data_manager_dialog import DataManagerDialog
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _answer_question(monkeypatch: pytest.MonkeyPatch, confirmed: bool) -> None:
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: (
            QtWidgets.QMessageBox.StandardButton.Yes
            if confirmed
            else QtWidgets.QMessageBox.StandardButton.No
        ),
    )


def _project(test_experiments: list[Experiment]) -> Project:
    project = Project()
    first, second = test_experiments
    project.experiments = {
        "first": first,
        "empty": Experiment("Empty recording", 1.0, [], []),
        "second": second,
    }
    return project


def _group(dialog: DataManagerDialog, key: str) -> QtWidgets.QTreeWidgetItem:
    item = dialog.group_item(key)
    assert item is not None
    return item


def _edit(
    qtbot: QtBot,
    dialog: DataManagerDialog,
    item: QtWidgets.QTreeWidgetItem,
    column: int,
    text: str,
) -> None:
    """Edit ``item`` the way the delegate commits a finished editor."""
    changed: list[bool] = []
    dialog.data_changed.connect(lambda: changed.append(True))
    item.setText(column, text)
    qtbot.waitUntil(lambda: bool(changed))


@pytest.mark.parametrize("selected_rows", [[0], [0, 2]])
@pytest.mark.parametrize("confirmed", [False, True])
def test_delete_selected_data_groups(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    test_experiments: list[Experiment],
    selected_rows: list[int],
    confirmed: bool,
) -> None:
    project = _project(test_experiments)
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)
    changed = []
    dialog.data_changed.connect(lambda: changed.append(True))
    _answer_question(monkeypatch, confirmed)

    for row in selected_rows:
        dialog.tree.topLevelItem(row).setSelected(True)
    qtbot.mouseClick(dialog.delete_button, QtCore.Qt.MouseButton.LeftButton)

    expected_keys = [
        key
        for row, key in enumerate(["first", "empty", "second"])
        if not confirmed or row not in selected_rows
    ]
    assert list(project.experiments) == expected_keys
    assert dialog.tree.topLevelItemCount() == len(expected_keys)
    assert dialog.info_label.text() == f"Total: {len(expected_keys)} data group(s)"
    assert project.dirty is confirmed
    assert changed == ([True] if confirmed else [])
    if confirmed:
        assert len(project.history) == 1
        project.history.undo()
        assert list(project.experiments) == ["first", "empty", "second"]
        project.history.redo()
        assert list(project.experiments) == expected_keys


def test_deleting_one_data_set_keeps_the_rest_of_the_group_and_inputs(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    test_experiments: list[Experiment],
) -> None:
    """#652: deleting the calcium traces must not delete other data or inputs."""
    project = _project(test_experiments)
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, np.ones((10, 3)), "mW", 0.01)
    )
    traces, spikes = project.experiments["first"].data
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)
    _answer_question(monkeypatch, True)

    group = _group(dialog, "first")
    assert group is not None and group.childCount() == 2
    group.child(0).setSelected(True)
    qtbot.mouseClick(dialog.delete_button, QtCore.Qt.MouseButton.LeftButton)

    assert list(project.experiments) == ["first", "empty", "second"]
    assert project.experiments["first"].data == [spikes]
    assert list(project.stim_arrays) == ["stim"]
    group = _group(dialog, "first")
    assert group is not None and group.childCount() == 1
    project.history.undo()
    assert project.experiments["first"].data[0] is traces
    assert project.experiments["first"].data[1] is spikes
    project.history.redo()
    assert project.experiments["first"].data == [spikes]


def test_selecting_a_group_and_its_data_set_deletes_the_group_once(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    test_experiments: list[Experiment],
) -> None:
    project = _project(test_experiments)
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)
    _answer_question(monkeypatch, True)
    group = _group(dialog, "second")
    assert group is not None
    group.setSelected(True)
    group.child(1).setSelected(True)
    _group(dialog, "first").child(1).setSelected(True)

    qtbot.mouseClick(dialog.delete_button, QtCore.Qt.MouseButton.LeftButton)

    assert list(project.experiments) == ["first", "empty"]
    assert len(project.experiments["first"].data) == 1
    assert len(project.history) == 1


def test_rename_and_describe_group_update_project_and_survive_save(
    qtbot: QtBot,
    test_experiments: list[Experiment],
    tmp_path,
) -> None:
    """#654: edits in the dialog are written to the project, undoable and saved."""
    project = _project(test_experiments)
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)

    _edit(qtbot, dialog, _group(dialog, "first"), NAME_COLUMN, "  Calcium  ")
    assert list(project.experiments) == ["Calcium", "empty", "second"]
    assert project.experiments["Calcium"].name == "Calcium"
    trace_name = project.experiments["Calcium"].data[0].name
    assert f"Calcium :: {trace_name}" in project.get_data_names(TraceData)

    _edit(qtbot, dialog, _group(dialog, "Calcium"), DESCRIPTION_COLUMN, "slice 3")
    assert project.experiments["Calcium"].description == "slice 3"
    assert _group(dialog, "Calcium").text(DESCRIPTION_COLUMN) == "slice 3"

    datum = project.experiments["Calcium"].data[0]
    _edit(qtbot, dialog, _group(dialog, "Calcium").child(0), NAME_COLUMN, "dF/F")
    assert datum.name == "dF/F"
    assert project.get_data("Calcium :: dF/F", TraceData) is datum

    path = tmp_path / "renamed.nseq"
    project.save(path, SaveLoadOptions(True, True, False, True))
    loaded = Project(path)
    assert list(loaded.experiments) == ["Calcium", "empty", "second"]
    assert loaded.experiments["Calcium"].description == "slice 3"
    assert loaded.experiments["Calcium"].data[0].name == "dF/F"

    assert len(project.history) == 3
    project.history.undo()
    assert datum.name == trace_name
    project.history.undo()
    assert project.experiments["Calcium"].description == ""
    project.history.undo()
    assert list(project.experiments) == ["first", "empty", "second"]
    assert project.experiments["first"].name == "e1"
    project.history.redo()
    assert list(project.experiments) == ["Calcium", "empty", "second"]


@pytest.mark.parametrize("bad_name", ["", "second", "Simulation", "a :: b"])
def test_invalid_group_names_are_rejected(
    qtbot: QtBot,
    qmessagebox_calls,
    test_experiments: list[Experiment],
    bad_name: str,
) -> None:
    project = _project(test_experiments)
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)

    _edit(qtbot, dialog, _group(dialog, "first"), NAME_COLUMN, bad_name)

    assert list(project.experiments) == ["first", "empty", "second"]
    assert dialog.group_item("first") is not None
    assert len(project.history) == 0
    assert any(call["name"] == "warning" for call in qmessagebox_calls)


def test_duplicate_data_set_name_is_rejected(
    qtbot: QtBot,
    make_trace_data,
    test_experiments: list[Experiment],
) -> None:
    project = _project(test_experiments)
    project.experiments["empty"].data = [
        make_trace_data(name="a"),
        make_trace_data(name="b"),
    ]
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)

    _edit(qtbot, dialog, _group(dialog, "empty").child(1), NAME_COLUMN, "a")

    assert [d.name for d in project.experiments["empty"].data] == ["a", "b"]
    assert len(project.history) == 0


def test_buttons_follow_selection_and_open_editors(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    test_experiments: list[Experiment],
) -> None:
    dialog = DataManagerDialog(_project(test_experiments))
    qtbot.addWidget(dialog)
    edited: list[tuple[str, int]] = []
    monkeypatch.setattr(
        dialog.tree,
        "editItem",
        lambda item, column: edited.append((item.text(0), column)),
    )
    assert not dialog.delete_button.isEnabled()
    assert not dialog.rename_button.isEnabled()

    group = _group(dialog, "first")
    group.setSelected(True)
    assert dialog.rename_button.isEnabled()
    assert dialog.description_button.isEnabled()
    dialog.rename_button.click()
    dialog.description_button.click()
    dialog._on_item_double_clicked(group, DESCRIPTION_COLUMN)
    dialog._on_item_double_clicked(group, 2)  # duration is read-only

    group.setSelected(False)
    group.child(0).setSelected(True)
    assert dialog.rename_button.isEnabled()
    assert not dialog.description_button.isEnabled()
    dialog._on_item_double_clicked(group.child(0), DESCRIPTION_COLUMN)
    assert edited == [
        ("first", NAME_COLUMN),
        ("first", DESCRIPTION_COLUMN),
        ("first", DESCRIPTION_COLUMN),
    ]
