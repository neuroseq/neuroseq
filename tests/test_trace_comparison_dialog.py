from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.visualization.trace_comparison_dialog import TraceComparisonDialog
from neuroseq.visualization_support.trace_comparison import ComparisonView

ARRAYS = ["Rep 1 :: dF/F", "Rep 2 :: dF/F", "Rep 3 :: dF/F"]


def _dialog(qtbot: QtBot, **kwargs) -> TraceComparisonDialog:
    dialog = TraceComparisonDialog(list(range(12)), ARRAYS, **kwargs)
    qtbot.addWidget(dialog)
    return dialog


def _shown_ids(dialog: TraceComparisonDialog) -> list[int]:
    items = (dialog.neuron_list.item(r) for r in range(dialog.neuron_list.count()))
    return [
        int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        for item in items
        if item is not None and not item.isHidden()
    ]


def test_preselection_and_ok_enabled(qtbot: QtBot) -> None:
    dialog = _dialog(
        qtbot, checked_neuron_ids=[3, 4], checked_array_names=["Rep 2 :: dF/F"]
    )
    assert dialog.selected_neuron_ids == [3, 4]
    assert dialog.selected_array_names == ["Rep 2 :: dF/F"]
    assert dialog.ok_button.isEnabled()


def test_ok_disabled_until_neuron_and_array_are_chosen(qtbot: QtBot) -> None:
    dialog = _dialog(qtbot)
    assert not dialog.ok_button.isEnabled()
    dialog.neuron_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
    assert not dialog.ok_button.isEnabled()
    dialog.array_list.item(2).setCheckState(QtCore.Qt.CheckState.Checked)
    assert dialog.ok_button.isEnabled()
    dialog.select_none_button.click()
    assert not dialog.ok_button.isEnabled()


def test_filter_limits_select_all_and_none(qtbot: QtBot) -> None:
    dialog = _dialog(qtbot, checked_neuron_ids=[0])
    dialog.filter_edit.setText("1")
    assert _shown_ids(dialog) == [1]
    dialog.filter_edit.setText("2-4")
    assert _shown_ids(dialog) == [2, 3, 4]
    dialog.select_all_button.click()
    assert dialog.selected_neuron_ids == [0, 2, 3, 4]
    dialog.filter_edit.setText("3")
    dialog.select_none_button.click()
    assert dialog.selected_neuron_ids == [0, 2, 4]
    # Not an id range: falls back to substring matching on the label.
    dialog.filter_edit.setText("ron 1")
    assert _shown_ids(dialog) == [1, 10, 11]
    dialog.filter_edit.setText("")
    assert len(_shown_ids(dialog)) == 12


def test_range_field_ticks_ids_and_reports_problems(qtbot: QtBot) -> None:
    dialog = _dialog(qtbot)
    dialog.range_edit.setText("1-3, 9")
    dialog.range_button.click()
    assert dialog.selected_neuron_ids == [1, 2, 3, 9]
    assert dialog.range_error.isHidden()

    dialog.range_edit.setText("10-14")
    dialog.range_button.click()
    assert dialog.selected_neuron_ids == [1, 2, 3, 9, 10, 11]
    assert "3 id(s)" in dialog.range_error.text()
    assert not dialog.range_error.isHidden()

    dialog.range_edit.setText("x")
    dialog.range_button.click()
    assert "not a neuron id" in dialog.range_error.text()
    assert dialog.selected_neuron_ids == [1, 2, 3, 9, 10, 11]


def test_array_select_all_and_none(qtbot: QtBot) -> None:
    dialog = _dialog(qtbot)
    dialog.select_all_arrays_button.click()
    assert dialog.selected_array_names == ARRAYS
    dialog.select_no_arrays_button.click()
    assert dialog.selected_array_names == []


def _dts(values: dict[str, float]):
    return lambda name: values.get(name)


def test_view_options_are_enabled_only_for_their_view(qtbot: QtBot) -> None:
    dialog = _dialog(
        qtbot,
        checked_neuron_ids=[0],
        checked_array_names=ARRAYS[:2],
        array_dt=_dts(dict.fromkeys(ARRAYS, 0.1)),
    )
    assert dialog.view is ComparisonView.OVERLAY
    assert not dialog.lock_time_checkbox.isEnabled()
    assert not dialog.gap_spin.isEnabled()
    assert not dialog.add_to_project_checkbox.isEnabled()
    assert dialog.lock_time and not dialog.lock_value

    dialog.view_buttons[ComparisonView.SIDE_BY_SIDE].setChecked(True)
    assert dialog.view is ComparisonView.SIDE_BY_SIDE
    assert not dialog.lock_value_checkbox.isEnabled()

    dialog.view_buttons[ComparisonView.SEPARATE_WINDOWS].setChecked(True)
    assert dialog.lock_time_checkbox.isEnabled()
    assert dialog.lock_value_checkbox.isEnabled()
    assert not dialog.gap_spin.isEnabled()

    dialog.view_buttons[ComparisonView.CONCATENATED].setChecked(True)
    assert not dialog.lock_time_checkbox.isEnabled()
    assert dialog.gap_spin.isEnabled()
    assert dialog.add_to_project_checkbox.isEnabled()
    assert not dialog.name_edit.isEnabled()
    assert not dialog.add_to_project
    dialog.add_to_project_checkbox.setChecked(True)
    assert dialog.add_to_project
    assert dialog.name_edit.isEnabled()
    assert dialog.new_array_name == "Concatenated: Rep 1 + Rep 2"
    dialog.gap_spin.setValue(50)
    assert dialog.gap_ms == 50.0

    # Switching away from Concatenated keeps the tick but stops adding.
    dialog.view_buttons[ComparisonView.OVERLAY].setChecked(True)
    assert not dialog.add_to_project
    assert dialog.ok_button.isEnabled()


def test_add_to_project_needs_equal_dt(qtbot: QtBot) -> None:
    dts = {ARRAYS[0]: 0.1, ARRAYS[1]: 0.1, ARRAYS[2]: 0.05}
    dialog = _dialog(
        qtbot,
        checked_neuron_ids=[0],
        checked_array_names=ARRAYS[:2],
        array_dt=_dts(dts),
        view=ComparisonView.CONCATENATED,
    )
    assert dialog.add_to_project_checkbox.isEnabled()
    dialog.add_to_project_checkbox.setChecked(True)
    dialog.array_list.item(2).setCheckState(QtCore.Qt.CheckState.Checked)
    assert not dialog.add_to_project_checkbox.isEnabled()
    assert not dialog.add_to_project
    tooltip = dialog.add_to_project_checkbox.toolTip()
    assert "same sampling interval" in tooltip
    assert "Rep 3: 50 ms" in tooltip
    dialog.array_list.item(2).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert dialog.add_to_project

    dialog.select_no_arrays_button.click()
    assert not dialog.add_to_project_checkbox.isEnabled()
    assert "Tick the trace arrays" in dialog.add_to_project_checkbox.toolTip()

    # Without dt information adding is never offered.
    unknown = _dialog(
        qtbot, checked_array_names=ARRAYS[:1], view=ComparisonView.CONCATENATED
    )
    assert not unknown.add_to_project_checkbox.isEnabled()
    assert "unknown" in unknown.add_to_project_checkbox.toolTip()


def test_new_array_name_follows_selection_until_edited(qtbot: QtBot) -> None:
    dialog = _dialog(
        qtbot,
        checked_neuron_ids=[0],
        checked_array_names=ARRAYS[:2],
        array_dt=_dts(dict.fromkeys(ARRAYS, 0.1)),
        existing_group_names=["Rep 1", "Taken"],
        view=ComparisonView.CONCATENATED,
    )
    dialog.add_to_project_checkbox.setChecked(True)
    dialog.array_list.item(2).setCheckState(QtCore.Qt.CheckState.Checked)
    assert dialog.new_array_name == "Concatenated: Rep 1 + Rep 2 + Rep 3"

    dialog.name_edit.clear()
    qtbot.keyClicks(dialog.name_edit, "Taken")
    assert not dialog.ok_button.isEnabled()
    assert "already exists" in dialog.name_error.text()
    assert not dialog.name_error.isHidden()
    dialog.name_edit.setText("A :: B")
    assert not dialog.ok_button.isEnabled()
    dialog.name_edit.setText("Joined")
    assert dialog.ok_button.isEnabled()
    assert dialog.name_error.isHidden()
    # A user-edited name is kept when the selection changes.
    dialog.array_list.item(2).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert dialog.new_array_name == "Joined"


def test_arrays_can_be_reordered(qtbot: QtBot) -> None:
    dialog = _dialog(
        qtbot, checked_array_names=ARRAYS, view=ComparisonView.CONCATENATED
    )
    assert dialog.selected_array_names == ARRAYS
    dialog.array_list.setCurrentRow(2)
    dialog.move_up_button.click()
    assert dialog.selected_array_names == [ARRAYS[0], ARRAYS[2], ARRAYS[1]]
    assert dialog.array_list.currentRow() == 1
    assert dialog.new_array_name == "Concatenated: Rep 1 + Rep 3 + Rep 2"
    dialog.array_list.setCurrentRow(0)
    dialog.move_down_button.click()
    assert dialog.selected_array_names == [ARRAYS[2], ARRAYS[0], ARRAYS[1]]
    # Moving past either end does nothing.
    dialog.array_list.setCurrentRow(0)
    dialog.move_up_button.click()
    assert dialog.selected_array_names == [ARRAYS[2], ARRAYS[0], ARRAYS[1]]
    dialog.array_list.setCurrentRow(2)
    dialog.move_down_button.click()
    assert dialog.selected_array_names == [ARRAYS[2], ARRAYS[0], ARRAYS[1]]
    assert (
        dialog.array_list.dragDropMode()
        == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
    )
