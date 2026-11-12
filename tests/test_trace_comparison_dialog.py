from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.visualization.trace_comparison_dialog import TraceComparisonDialog

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
