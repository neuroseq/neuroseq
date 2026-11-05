from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.visualization.trace_widget import TraceWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from tests.wait_timeout import WAIT_TIMEOUT_MS


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

    with qtbot.waitSignal(
        trace_widget.custom_analysis_requested, timeout=WAIT_TIMEOUT_MS
    ) as blocker:
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)

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


USER_PLUGIN = """PLUGIN_ID = "{plugin_id}"
PLUGIN_NAME = "User plugin"

def render_analysis(widget, data, neuron_slice, time_slice):
    pass
"""


def test_discover_includes_user_plugins_without_shadowing_builtins(
    isolate_user_plugin_dir, caplog
) -> None:
    from neuroseq import analysis_plugins

    isolate_user_plugin_dir.mkdir(parents=True, exist_ok=True)
    (isolate_user_plugin_dir / "my_user_plugin.py").write_text(
        USER_PLUGIN.format(plugin_id="my_user_plugin"), encoding="utf-8"
    )
    # Same filename as a built-in plugin.
    (isolate_user_plugin_dir / "mean_activity.py").write_text(
        USER_PLUGIN.format(plugin_id="shadow_by_name"), encoding="utf-8"
    )
    # Different filename but a built-in PLUGIN_ID.
    (isolate_user_plugin_dir / "sneaky.py").write_text(
        USER_PLUGIN.format(plugin_id="mean_activity"), encoding="utf-8"
    )
    (isolate_user_plugin_dir / "_private.py").write_text(
        USER_PLUGIN.format(plugin_id="private"), encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        plugins = analysis_plugins.discover_analysis_plugins()

    ids = [plugin.id for plugin in plugins]
    assert "my_user_plugin" in ids
    assert "shadow_by_name" not in ids
    assert "private" not in ids
    assert ids.count("mean_activity") == 1
    mean = next(p for p in plugins if p.id == "mean_activity")
    assert mean.label != "User plugin"
    assert "collides with a built-in plugin" in caplog.text


def test_user_plugins_directory_env_override_and_default(monkeypatch, tmp_path):
    from neuroseq import analysis_plugins

    monkeypatch.setenv(analysis_plugins.USER_PLUGIN_DIR_ENV, str(tmp_path / "u"))
    assert analysis_plugins.user_plugins_directory() == tmp_path / "u"
    assert not (tmp_path / "u").exists()
    assert analysis_plugins.user_plugins_directory(create=True).is_dir()

    monkeypatch.delenv(analysis_plugins.USER_PLUGIN_DIR_ENV)
    default = analysis_plugins.user_plugins_directory()
    assert default.parts[-2:] == ("neuroseq", "analysis_plugins")
    assert not default.is_relative_to(analysis_plugins._plugins_directory())
