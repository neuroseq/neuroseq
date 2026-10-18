from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.visualization.trace_widget import TraceWidget


def test_discover_analysis_plugins_contains_example_plugin() -> None:
    plugins = discover_analysis_plugins()
    plugin_ids = {plugin.id for plugin in plugins}
    assert "mean_activity" in plugin_ids


def test_trace_widget_context_menu_contains_plugin_actions(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget._rebuild_custom_analysis_menu()
    actions = trace_widget.context_menu.actions()
    action_texts = [action.text() for action in actions]

    assert "Open analysis in a new window" in action_texts
    open_in_new_action = next(
        action for action in actions if action.text() == "Open analysis in a new window"
    )
    assert open_in_new_action.isChecked()
    assert "Mean activity" in action_texts
    assert "Mean activity (new window)" not in action_texts


def test_trace_widget_analysis_action_allows_no_selection(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget._rebuild_custom_analysis_menu()
    mean_activity_action = next(
        action
        for action in trace_widget.context_menu.actions()
        if action.text() == "Mean activity"
    )

    with qtbot.waitSignal(trace_widget.custom_analysis_requested) as blocker:
        mean_activity_action.trigger()

    assert blocker.args[3:] == [None, None]


def test_custom_analysis_widget_runs_example_plugin(
    qtbot: QtBot, make_trace_data
) -> None:
    plugins = discover_analysis_plugins()
    example_plugin = next(plugin for plugin in plugins if plugin.id == "mean_activity")
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())

    data = make_trace_data(n_neurons=10, n_timestamps=200, dt=0.001)
    custom_analysis_widget.set_data(
        data,
        example_plugin,
        slice(0, 5),
        slice(10, 100),
    )
    qtbot.wait(20)

    assert custom_analysis_widget.traces is not None
    custom_analysis_widget.clear_data()
    assert custom_analysis_widget.traces is None
    assert custom_analysis_widget.spike_times_ms is None
    assert custom_analysis_widget.spike_indices is None
    assert custom_analysis_widget.spike_neuron_indices is None
