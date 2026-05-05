from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.trace_widget import TraceWidget


def test_discover_analysis_plugins_contains_example_plugin() -> None:
    plugins = discover_analysis_plugins()
    plugin_ids = {plugin.id for plugin in plugins}
    assert "mean_activity" in plugin_ids


def test_trace_widget_context_menu_contains_plugin_actions(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget._rebuild_custom_analysis_menu()
    action_texts = [action.text() for action in trace_widget.context_menu.actions()]

    assert "Mean activity" in action_texts
    assert "Mean activity (new window)" in action_texts


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
