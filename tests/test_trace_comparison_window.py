from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq import theme
from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.project import Project
from neuroseq.visualization.trace_comparison_views import ConcatenatedTraceWindow
from neuroseq.visualization.trace_comparison_views import LinkedArrayWindow
from neuroseq.visualization.trace_comparison_window import TraceComparisonWindow
from neuroseq.visualization.trace_comparison_window import open_trace_comparison
from neuroseq.visualization_support.trace_comparison import ARRAY_COLORS
from neuroseq.visualization_support.trace_comparison import AxisLinkGroup
from neuroseq.visualization_support.trace_comparison import ComparisonView
from neuroseq.visualization_support.trace_comparison import build_trace_comparison
from neuroseq.visualization_support.trace_comparison import build_trace_concatenation
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _trace(ids, n_samples: int, dt: float, seed: int = 0) -> TraceData:
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    return TraceData("dF/F", rng.random((len(ids), n_samples)), ids, "dF/F", dt)


def _project(*traces: TraceData) -> Project:
    project = Project()
    for number, trace in enumerate(traces, start=1):
        project.add_experiment(
            Experiment(f"Rep {number}", trace.traces.shape[1] * trace.dt, [], [trace])
        )
    return project


def _window(qtbot: QtBot, comparison, **kwargs) -> TraceComparisonWindow:
    window = TraceComparisonWindow(comparison, **kwargs)
    return window


def test_window_overlays_two_arrays_for_three_neurons(qtbot: QtBot) -> None:
    a = _trace([0, 1, 2], 50, 0.01, seed=1)
    b = _trace([0, 1, 2], 50, 0.01, seed=2)
    comparison = build_trace_comparison(
        [("Rep 1 :: dF/F", a), ("Rep 2 :: dF/F", b)], [0, 1, 2]
    )
    window = _window(qtbot, comparison)

    axes = window.figure.axes
    assert len(axes) == 3
    for ax, neuron_id in zip(axes, [0, 1, 2], strict=True):
        assert ax.get_ylabel() == f"Neuron {neuron_id}\n(dF/F)"
        lines = ax.get_lines()
        assert [line.get_label() for line in lines] == ["Rep 1", "Rep 2"]
        assert [line.get_color() for line in lines] == list(ARRAY_COLORS[:2])
    np.testing.assert_array_equal(axes[1].get_lines()[1].get_ydata(), b.traces[1])
    # One linked time axis across rows.
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[2])
    assert axes[0].get_xlim() == (0.0, 490.0)
    legend_texts = [t.get_text() for t in window.figure.legends[0].get_texts()]
    assert legend_texts == ["Rep 1", "Rep 2"]
    assert window.note_label.isHidden()

    axes[0].set_xlim(0, 400)
    ticks = axes[-1].xaxis.get_major_formatter()(250.0, 0)
    assert ticks == "250 ms"


def test_window_side_by_side_gives_each_array_a_column(qtbot: QtBot) -> None:
    a = _trace([0, 1, 2], 20, 0.01)
    b = _trace([0, 1], 20, 0.01)
    comparison = build_trace_comparison([("A :: dF/F", a), ("B :: dF/F", b)], [0, 2])
    window = _window(qtbot, comparison, theme=theme.DARK)

    window.side_by_side_checkbox.setChecked(True)
    axes = window.figure.axes
    assert len(axes) == 4
    assert [ax.get_title() for ax in axes[:2]] == ["A", "B"]
    assert len(axes[0].get_lines()) == 1
    assert len(axes[1].get_lines()) == 1
    # Neuron 2 is not in B: that cell is empty and says so.
    assert axes[3].get_lines() == []
    assert any("not in this array" in t.get_text() for t in axes[3].texts)
    assert "B: 2" in window.note_label.text()
    assert not window.note_label.isHidden()

    window.side_by_side_checkbox.setChecked(False)
    assert len(window.figure.axes) == 2


def test_window_handles_arrays_with_different_dt(qtbot: QtBot) -> None:
    fast = _trace([0], 300, 1 / 30)
    slow = _trace([0], 5, 2.0)
    comparison = build_trace_comparison([("F :: x", fast), ("S :: x", slow)], [0])
    window = _window(qtbot, comparison)

    fast_line, slow_line = window.figure.axes[0].get_lines()
    assert fast_line.get_xdata()[-1] == np.float64(299 * 1000 / 30)
    np.testing.assert_allclose(slow_line.get_xdata(), [0, 2000, 4000, 6000, 8000])
    assert window.figure.axes[0].get_xlim()[1] == fast_line.get_xdata()[-1]
    assert window.figure.axes[0].xaxis.get_major_formatter()(4000.0, 0) == "4 s"


def test_window_without_rows_shows_message(qtbot: QtBot) -> None:
    comparison = build_trace_comparison([("A :: x", _trace([0], 5, 0.1))], [9])
    window = _window(qtbot, comparison)
    assert any(
        "None of the chosen" in t.get_text() for t in window.figure.axes[0].texts
    )
    assert "A :: x: 9" in window.note_label.text()


def test_open_trace_comparison_preselects_current_array_and_opens_popup(
    qtbot: QtBot,
) -> None:
    rep1 = _trace([0, 1, 2], 40, 0.01, seed=1)
    rep2 = _trace([0, 1, 2], 20, 0.02, seed=2)
    rep3 = _trace([0, 2], 40, 0.01, seed=3)
    project = _project(rep1, rep2, rep3)

    history_length = len(project.history)
    dirty = project.dirty
    dialog = open_trace_comparison(project, rep2, [1], title_suffix="Tab")
    qtbot.addWidget(dialog)
    assert dialog.isVisible()
    assert dialog.selected_array_names == ["Rep 2 :: dF/F"]
    assert dialog.selected_neuron_ids == [1]
    assert dialog.array_list.count() == 3

    dialog.select_all_arrays_button.click()
    dialog.range_edit.setText("0-2")
    dialog.range_button.click()
    dialog.accept()

    window = dialog.comparison_window
    assert isinstance(window, TraceComparisonWindow)
    assert window.isVisible()
    assert window.windowTitle() == "Trace comparison - Tab"
    assert window.windowFlags() & QtCore.Qt.WindowType.Window
    assert not window.isModal()
    assert [row.neuron_id for row in window.comparison.rows] == [0, 1, 2]
    assert [label for label, _ in window.comparison.legend] == [
        "Rep 1",
        "Rep 2",
        "Rep 3",
    ]
    assert window.comparison.missing == {"Rep 3": [1]}
    # Read-only: project data, undo history and dirty flag are untouched.
    assert project.experiments["Rep 2"].data[0] is rep2
    assert len(project.history) == history_length
    assert project.dirty == dirty


def test_open_trace_comparison_cancel_opens_nothing(qtbot: QtBot) -> None:
    project = _project(_trace([0], 5, 0.1))
    dialog = open_trace_comparison(project, None)
    qtbot.addWidget(dialog)
    assert dialog.selected_array_names == []
    assert dialog.neuron_list.count() == 0
    dialog.reject()
    assert dialog.comparison_window is None


def _linked_windows(qtbot: QtBot, lock_time=True, lock_value=False):
    a = _trace([0, 1], 50, 0.01, seed=1)
    b = _trace([0, 1, 2], 50, 0.01, seed=2)
    comparison = build_trace_comparison(
        [("Rep 1 :: dF/F", a), ("Rep 2 :: dF/F", b)], [0, 1, 2]
    )
    group: AxisLinkGroup[LinkedArrayWindow] = AxisLinkGroup()
    windows = [
        LinkedArrayWindow(
            comparison, label, group, lock_time=lock_time, lock_value=lock_value
        )
        for label in ("Rep 1", "Rep 2")
    ]
    for window in windows:
        qtbot.addWidget(window)
    return windows, group


def test_linked_array_window_stacks_one_arrays_neurons(qtbot: QtBot) -> None:
    (rep1, rep2), group = _linked_windows(qtbot)
    assert group.members == [rep1, rep2]
    assert len(rep1.axes) == 3
    assert [ax.get_ylabel() for ax in rep1.axes] == [
        f"Neuron {i}\n(dF/F)" for i in range(3)
    ]
    assert rep1.axes[0].get_title() == "Rep 1"
    assert rep1.axes[0].get_lines()[0].get_color() == ARRAY_COLORS[0]
    assert rep2.axes[0].get_lines()[0].get_color() == ARRAY_COLORS[1]
    # Neuron 2 is not in Rep 1: its row says so and the note lists it.
    assert rep1.axes[2].get_lines() == []
    assert any("not in this array" in t.get_text() for t in rep1.axes[2].texts)
    assert rep1.note_label.text() == "Not in this array: 2"
    assert rep2.note_label.isHidden()
    assert rep1.time_range == (0.0, 490.0)
    assert rep1.lock_time_checkbox.isChecked()
    assert not rep1.lock_value_checkbox.isChecked()


def test_linked_windows_propagate_time_and_unlink(qtbot: QtBot) -> None:
    (rep1, rep2), group = _linked_windows(qtbot)
    rep1.axes[0].set_xlim(100, 200)
    assert rep2.time_range == (100.0, 200.0)
    # Rows inside a window share time, so every row follows.
    assert rep2.axes[2].get_xlim() == (100.0, 200.0)
    # Value axes are not locked by default.
    rep1.axes[1].set_ylim(-5, 5)
    assert rep2.value_range(1) != (-5.0, 5.0)

    rep2.lock_time_checkbox.setChecked(False)
    rep1.axes[0].set_xlim(0, 50)
    assert rep2.time_range == (100.0, 200.0)
    rep2.axes[0].set_xlim(300, 400)
    assert rep1.time_range == (0.0, 50.0)

    # Locking again snaps to the group's current range.
    rep2.lock_time_checkbox.setChecked(True)
    assert rep2.time_range == (0.0, 50.0)


def test_linked_windows_value_lock_pairs_rows(qtbot: QtBot) -> None:
    (rep1, rep2), _group = _linked_windows(qtbot, lock_time=False, lock_value=True)
    rep2.axes[2].set_ylim(-1, 3)
    assert rep1.value_range(2) == (-1.0, 3.0)
    assert rep1.value_range(0) != (-1.0, 3.0)
    rep1.axes[0].set_xlim(10, 20)
    assert rep2.time_range == (0.0, 490.0)
    assert rep1.value_range(9) is None

    rep1.lock_value_checkbox.setChecked(False)
    rep2.axes[0].set_ylim(7, 8)
    assert rep1.value_range(0) != (7.0, 8.0)
    rep1.lock_value_checkbox.setChecked(True)
    assert rep1.value_range(0) == (7.0, 8.0)
    assert rep1.value_range(2) == (-1.0, 3.0)


def test_closing_a_linked_window_keeps_the_others_working(qtbot: QtBot) -> None:
    a = _trace([0], 20, 0.01)
    comparison = build_trace_comparison(
        [("R1 :: dF/F", a), ("R2 :: dF/F", a), ("R3 :: dF/F", a)], [0]
    )
    group: AxisLinkGroup[LinkedArrayWindow] = AxisLinkGroup()
    windows = [
        LinkedArrayWindow(comparison, label, group) for label in ("R1", "R2", "R3")
    ]
    for window in windows:
        window.show()
    first, middle, last = windows
    with qtbot.waitSignal(middle.destroyed, timeout=WAIT_TIMEOUT_MS):
        middle.close()
    assert group.members == [first, last]
    first.axes[0].set_xlim(30, 60)
    assert last.time_range == (30.0, 60.0)
    # Applying to a closed window is a no-op rather than an error.
    middle_like = LinkedArrayWindow(comparison, "R1", group)
    middle_like.close()
    middle_like.apply_time_range((0, 1))
    middle_like.apply_value_range(0, (0, 1))
    assert middle_like not in group.members


def test_linked_window_without_rows(qtbot: QtBot) -> None:
    comparison = build_trace_comparison([("A :: x", _trace([0], 5, 0.1))], [9])
    window = LinkedArrayWindow(comparison, "A :: x", AxisLinkGroup())
    qtbot.addWidget(window)
    assert window.axes == []
    assert window.time_range is None
    window.apply_time_range((0, 1))
    window.lock_time_checkbox.setChecked(False)
    window.lock_time_checkbox.setChecked(True)


def test_concatenated_window_draws_segments_and_boundaries(qtbot: QtBot) -> None:
    a = _trace([0, 1], 10, 0.1, seed=1)
    b = _trace([0], 5, 0.2, seed=2)
    concatenation = build_trace_concatenation(
        [("Rep 1 :: dF/F", a), ("Rep 2 :: dF/F", b)], [0, 1]
    )
    window = ConcatenatedTraceWindow(concatenation, theme=theme.DARK)
    qtbot.addWidget(window)
    axes = window.figure.axes
    assert len(axes) == 2
    lines = [line for line in axes[0].get_lines() if line.get_linestyle() == "-"]
    assert [line.get_color() for line in lines] == list(ARRAY_COLORS[:2])
    np.testing.assert_allclose(lines[1].get_xdata(), [1000, 1200, 1400, 1600, 1800])
    boundaries = [line for line in axes[1].get_lines() if line.get_linestyle() == "--"]
    assert [line.get_xdata()[0] for line in boundaries] == [1000.0]
    assert [t.get_text() for t in axes[0].texts] == ["Rep 1", "Rep 2"]
    assert axes[0].get_xlim() == (0.0, 2000.0)
    # Neuron 1 is not in Rep 2: that segment is NaN.
    assert np.isnan(axes[1].get_lines()[1].get_ydata()).all()
    assert "Rep 2: 1" in window.note_label.text()
    window.add_note("Added")
    assert window.note_label.text().endswith("\nAdded")


def test_concatenated_window_without_rows(qtbot: QtBot) -> None:
    concatenation = build_trace_concatenation([("A :: x", _trace([0], 5, 0.1))], [9])
    window = ConcatenatedTraceWindow(concatenation)
    qtbot.addWidget(window)
    assert any(
        "None of the chosen" in t.get_text() for t in window.figure.axes[0].texts
    )


def _open(qtbot: QtBot, project: Project, view: ComparisonView, **kwargs):
    dialog = open_trace_comparison(
        project, project.get_data("Rep 1 :: dF/F", TraceData), **kwargs
    )
    qtbot.addWidget(dialog)
    dialog.select_all_arrays_button.click()
    dialog.select_all_button.click()
    dialog.view_buttons[view].setChecked(True)
    return dialog


def test_open_side_by_side_view(qtbot: QtBot) -> None:
    project = _project(_trace([0, 1], 20, 0.01), _trace([0, 1], 20, 0.01))
    dialog = open_trace_comparison(
        project, project.get_data("Rep 1 :: dF/F", TraceData)
    )
    qtbot.addWidget(dialog)
    dialog.select_all_button.click()
    dialog.select_all_arrays_button.click()
    dialog.view_buttons[ComparisonView.SIDE_BY_SIDE].setChecked(True)
    dialog.accept()
    window = dialog.comparison_window
    assert isinstance(window, TraceComparisonWindow)
    assert window.side_by_side
    assert len(window.figure.axes) == 4


def test_open_separate_windows_cascades_and_links(qtbot: QtBot) -> None:
    project = _project(*(_trace([0, 1], 20, 0.01, seed=i) for i in range(3)))
    dialog = _open(qtbot, project, ComparisonView.SEPARATE_WINDOWS, title_suffix="T")
    dialog.lock_value_checkbox.setChecked(True)
    dialog.accept()
    windows = dialog.comparison_windows
    assert len(windows) == 3
    assert all(isinstance(window, LinkedArrayWindow) for window in windows)
    assert dialog.comparison_window is windows[0]
    assert [window.windowTitle() for window in windows] == [
        "Trace array Rep 1 - T",
        "Trace array Rep 2 - T",
        "Trace array Rep 3 - T",
    ]
    assert all(window.lock_value_checkbox.isChecked() for window in windows)
    positions = [window.pos() for window in windows]
    assert len({(p.x(), p.y()) for p in positions}) == 3
    assert positions[2] - positions[0] == QtCore.QPoint(64, 64)
    windows[2].axes[1].set_xlim(5, 15)
    assert windows[0].time_range == (5.0, 15.0)
    windows[0].axes[1].set_ylim(0, 2)
    assert windows[1].value_range(1) == (0.0, 2.0)


def test_open_concatenated_adds_undoable_trace_array(qtbot: QtBot) -> None:
    project = _project(_trace([0, 1], 10, 0.1, seed=1), _trace([0], 5, 0.1, seed=2))
    added: list[str] = []
    dialog = _open(
        qtbot, project, ComparisonView.CONCATENATED, on_data_added=added.append
    )
    # Join Rep 2 first.
    dialog.array_list.setCurrentRow(1)
    dialog.move_up_button.click()
    dialog.gap_spin.setValue(200)
    dialog.add_to_project_checkbox.setChecked(True)
    assert dialog.new_array_name == "Concatenated: Rep 2 + Rep 1"
    history_length = len(project.history)
    dialog.accept()

    window = dialog.comparison_window
    assert isinstance(window, ConcatenatedTraceWindow)
    assert [s.label for s in window.concatenation.segments] == ["Rep 2", "Rep 1"]
    assert added == ["Concatenated: Rep 2 + Rep 1"]
    assert "Added to the project" in window.note_label.text()
    name = "Concatenated: Rep 2 + Rep 1 :: dF/F"
    assert name in project.get_data_names(TraceData)
    joined = project.get_data(name, TraceData)
    assert joined is not None
    assert joined.traces.shape == (2, 17)
    assert joined.dt == 0.1
    np.testing.assert_array_equal(joined.indices, [0, 1])
    assert np.isnan(joined.traces[1, :7]).all()  # neuron 1 not in Rep 2, gap
    experiment = project.experiments["Concatenated: Rep 2 + Rep 1"]
    assert experiment.duration == pytest.approx(1.7)
    assert "Rep 2, Rep 1" in experiment.description
    assert len(project.history) == history_length + 1

    project.history.undo()
    assert name not in project.get_data_names(TraceData)
    project.history.redo()
    assert name in project.get_data_names(TraceData)


def test_open_concatenated_without_adding_leaves_project_alone(qtbot: QtBot) -> None:
    project = _project(_trace([0], 10, 0.1), _trace([0], 10, 0.05))
    dialog = _open(qtbot, project, ComparisonView.CONCATENATED)
    assert not dialog.add_to_project_checkbox.isEnabled()
    history_length = len(project.history)
    dialog.accept()
    assert len(project.experiments) == 2
    assert len(project.history) == history_length


def test_open_concatenated_reports_name_clash(qtbot: QtBot, qmessagebox_calls) -> None:
    project = _project(_trace([0], 10, 0.1), _trace([0], 10, 0.1))
    dialog = _open(qtbot, project, ComparisonView.CONCATENATED)
    dialog.add_to_project_checkbox.setChecked(True)
    # The group appears after the dialog opened (e.g. from another popup).
    project.add_experiment(
        Experiment(dialog.new_array_name, 1.0, [], [_trace([0], 10, 0.1)])
    )
    dialog.accept()
    assert "Added" not in dialog.comparison_window.note_label.text()
    assert [call["title"] for call in qmessagebox_calls] == [
        "Could not add trace array"
    ]


def test_workspace_trace_context_menu_opens_comparison_dialog(qtbot: QtBot) -> None:
    from neuroseq.project_workspace import ProjectWorkspace
    from neuroseq.visualization.trace_comparison_dialog import TraceComparisonDialog

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = _project(_trace([0, 1, 2], 40, 0.01), _trace([0, 1, 2], 40, 0.01))
    gui.update_project(project)
    gui._apply_trace_selection("Rep 2 :: dF/F")

    gui.trace_widget._last_hovered_neuron_id = 2
    gui.trace_widget._rebuild_custom_analysis_menu()
    action = next(
        action
        for action in gui.trace_widget.context_menu.actions()
        if action.text() == "Compare traces…"
    )
    action.trigger()

    dialogs = [w for w in gui.findChildren(TraceComparisonDialog) if w.isVisible()]
    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.selected_array_names == ["Rep 2 :: dF/F"]
    assert dialog.selected_neuron_ids == [2]
    assert dialog.array_list.count() == 2
    dialog.accept()
    window = dialog.comparison_window
    assert isinstance(window, TraceComparisonWindow)
    assert window.parent() is gui
    assert window.windowTitle().startswith("Trace comparison - ")

    # Joining into a new array lists it in the data selectors; undo removes it.
    shown_before = gui.cmb_trace.currentText()
    dialog = gui._compare_traces(gui.project.get_data("Rep 1 :: dF/F", TraceData), [0])
    dialog.select_all_arrays_button.click()
    dialog.view_buttons[ComparisonView.CONCATENATED].setChecked(True)
    dialog.add_to_project_checkbox.setChecked(True)
    dialog.accept()
    new_name = "Concatenated: Rep 1 + Rep 2 :: dF/F"
    combo_items = [gui.cmb_trace.itemText(i) for i in range(gui.cmb_trace.count())]
    assert new_name in combo_items
    assert gui.cmb_trace.currentText() == shown_before
    gui.project.history.undo()
    combo_items = [gui.cmb_trace.itemText(i) for i in range(gui.cmb_trace.count())]
    assert new_name not in combo_items

    # Detached copies of the trace panel and the all-data panel are wired too.
    fresh = gui._build_fresh_trace_widget()
    qtbot.addWidget(fresh)
    assert fresh.receivers(fresh.compare_traces_requested) == 1
    data_trace = gui.data_widget.trace_widget
    assert data_trace.receivers(data_trace.compare_traces_requested) == 1
