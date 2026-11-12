from __future__ import annotations

import numpy as np
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq import theme
from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.project import Project
from neuroseq.visualization.trace_comparison_window import TraceComparisonWindow
from neuroseq.visualization.trace_comparison_window import open_trace_comparison
from neuroseq.visualization_support.trace_comparison import ARRAY_COLORS
from neuroseq.visualization_support.trace_comparison import build_trace_comparison


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

    # Detached copies of the trace panel and the all-data panel are wired too.
    fresh = gui._build_fresh_trace_widget()
    qtbot.addWidget(fresh)
    assert fresh.receivers(fresh.compare_traces_requested) == 1
    data_trace = gui.data_widget.trace_widget
    assert data_trace.receivers(data_trace.compare_traces_requested) == 1
