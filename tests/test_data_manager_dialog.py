import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data_manager_dialog import DataManagerDialog
from neuroseq.project import Project


@pytest.mark.parametrize("selected_rows", [[0], [0, 2]])
@pytest.mark.parametrize("confirmed", [False, True])
def test_delete_selected_data_groups(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    test_experiments: list[Experiment],
    selected_rows: list[int],
    confirmed: bool,
) -> None:
    project = Project()
    first, second = test_experiments
    project.experiments = {
        "first": first,
        "empty": Experiment("Empty recording", 1.0, [], []),
        "second": second,
    }
    dialog = DataManagerDialog(project)
    qtbot.addWidget(dialog)
    changed = []
    dialog.data_changed.connect(lambda: changed.append(True))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: (
            QtWidgets.QMessageBox.StandardButton.Yes
            if confirmed
            else QtWidgets.QMessageBox.StandardButton.No
        ),
    )

    for row in selected_rows:
        dialog.table.selectRow(row)
    qtbot.mouseClick(dialog.delete_button, QtCore.Qt.MouseButton.LeftButton)

    expected_keys = [
        key
        for row, key in enumerate(["first", "empty", "second"])
        if not confirmed or row not in selected_rows
    ]
    assert list(project.experiments) == expected_keys
    assert dialog.table.rowCount() == len(expected_keys)
    assert dialog.info_label.text() == f"Total: {len(expected_keys)} data group(s)"
    assert project.dirty is confirmed
    assert changed == ([True] if confirmed else [])
    if confirmed:
        assert len(project.history) == 1
        project.history.undo()
        assert list(project.experiments) == ["first", "empty", "second"]
        project.history.redo()
        assert list(project.experiments) == expected_keys
