from __future__ import annotations

import threading

import numpy as np
import pytest
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.background_jobs import SimulationGuard
from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.gui import Gui
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.workspace_devices import WorkspaceDevices
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _project() -> Project:
    project = Project()
    project.add_stim_array(
        StimArray("input", StimType.Excitation, np.zeros((500, 2)), "nS", 0.01)
    )
    project.dirty = False
    return project


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(
        GuiSettings, "save_config_settings", lambda self, only=None: None
    )
    widget = Gui(_project())
    qtbot.addWidget(widget)
    yield widget
    for workspace in widget.workspaces:
        workspace.project.dirty = False
    widget.close()


def test_tabs_keep_complete_workspaces_and_undo_history(window, monkeypatch):
    first = window.active_workspace
    first.console_widget.namespace()["local_value"] = "first"
    first.time_slider.set_current_time_ms(1200)
    first.controller.focus_time_range(0.5, 1.5)
    first.static_input_controller.open_pattern_designer()
    dialog = first.static_input_controller._pattern_dialog
    dialog.spin_pulse_ms.setValue(250)
    first.stim_editor.replace_stim_values_with_undo("input", np.ones((500, 2)))
    modifier = first.stim_editor.current_stim_modifier
    first.dynamic_input_controller._open_routine_dialog_for(
        first.dynamic_input_controller.RHEOBASE_ROUTINE
    )
    dynamic_dialog = first.dynamic_input_controller._rheobase_dialog
    expected_range = first.controller.current_time_range()
    monkeypatch.setattr(
        first,
        "update_project",
        lambda *args, **kwargs: pytest.fail("tab switch rebuilt widgets"),
    )

    second = window._add_workspace(_project())
    assert second.project is not first.project
    for name in (
        "stim_editor",
        "controller",
        "trace_widget",
        "spike_widget",
        "heatmap_widget",
        "time_slider",
        "console_widget",
        "chat_widget",
        "syscon2_panel",
        "custom_analysis_widget",
        "static_input_controller",
        "dynamic_input_controller",
    ):
        assert getattr(first, name) is not getattr(second, name)
    assert "local_value" not in second.console_widget.namespace()
    assert second.time_slider.t_ms == 0
    assert not second.stim_editor.current_stim_modifier.action_stack

    window.project_tabs.setCurrentWidget(first)
    assert first.time_slider.t_ms == 1200
    assert first.controller.current_time_range() == expected_range
    assert first.console_widget.namespace()["local_value"] == "first"
    assert first.static_input_controller._pattern_dialog is dialog
    assert dialog.spin_pulse_ms.value() == 250
    assert first.dynamic_input_controller._rheobase_dialog is dynamic_dialog
    assert first.stim_editor.current_stim_modifier is modifier
    modifier.undo()
    assert not first.project.stim_arrays["input"].values.any()
    modifier.redo()
    assert first.project.stim_arrays["input"].values.all()


def test_shared_data_history_refreshes_workspace_and_detaches_replaced_project(window):
    workspace = window.active_workspace
    project = workspace.project
    project.add_experiment(
        Experiment(
            "imported",
            1.0,
            [],
            [TraceData("voltage", np.ones((2, 10)), np.arange(2), "mV", 0.1)],
        )
    )
    workspace.update_project(project)
    data_name = next(
        name for name in project.get_data_names(TraceData) if "imported" in name
    )
    assert workspace.cmb_trace.findText(data_name) >= 0
    workspace.cmb_trace.setCurrentText(data_name)
    assert workspace.stim_editor.action_undo.isEnabled()
    workspace._set_simulation_actions_enabled(False)
    assert not workspace.stim_editor.action_undo.isEnabled()
    workspace.stim_editor.undo()
    assert "imported" in project.experiments
    workspace._set_simulation_actions_enabled(True)
    workspace.stim_editor.action_undo.trigger()
    assert "imported" not in project.experiments
    assert workspace.cmb_trace.findText(data_name) == -1
    rebuilds: list[object] = []
    original_set_project = workspace.stim_editor.set_project

    def counting_set_project(*args, **kwargs):
        rebuilds.append(args)
        return original_set_project(*args, **kwargs)

    workspace.stim_editor.set_project = counting_set_project  # type: ignore[method-assign]
    workspace.stim_editor.action_redo.trigger()
    del workspace.stim_editor.set_project
    assert len(rebuilds) == 1  # one stim-editor rebuild per undo/redo
    assert "imported" in project.experiments
    assert workspace.cmb_trace.findText(data_name) >= 0
    replacement = _project()
    workspace.update_project(replacement)
    project.history.undo()
    assert workspace.project is replacement
    assert workspace.cmb_trace.findText(data_name) == -1
    assert not workspace.stim_editor.action_undo.isEnabled()


def test_hidden_workspace_receives_its_own_background_result(
    window, qtbot, monkeypatch
):
    first = window.active_workspace
    started, release = threading.Event(), threading.Event()

    def simulate(*args, **kwargs):
        started.set()
        release.wait(WAIT_TIMEOUT_MS / 1000)
        return Experiment("result", 0.1, [], [])

    first.project.external_simulate_function = simulate
    monkeypatch.setattr(
        first, "_append_experimental_data", first.project.add_experiment
    )
    first._do_external_simulation(0.1)
    qtbot.waitUntil(started.is_set, timeout=WAIT_TIMEOUT_MS)
    second = window._add_workspace(_project())
    try:
        assert window.project_tabs.isEnabled()
        assert first._background_job.busy
    finally:
        release.set()
    qtbot.waitUntil(lambda: not first._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    assert "result" in first.project.experiments
    assert "result" not in second.project.experiments
    assert window.active_workspace is second


def test_tab_reordering_keeps_project_identity_and_close_releases_only_its_owner(
    window,
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    second.time_slider.toggle_pause()
    window.project_tabs.tabBar().moveTab(1, 0)
    assert window.active_workspace is second
    assert second.time_slider.timer.isActive()
    assert window.workspaces == [second, first]
    window._close_tab(0)
    assert second._disposed
    assert not first._disposed
    assert window.active_workspace is first
    assert not first.syscon2_panel._manager._closed


def test_hidden_playback_pauses_and_workspaces_share_the_scanimage_connection(
    window, monkeypatch
):
    first = window.active_workspace
    first.time_slider.set_current_time_ms(1200)
    first.time_slider.toggle_pause()
    assert first.time_slider.timer.isActive()
    second = window._add_workspace(_project())
    assert not first.time_slider.timer.isActive()
    assert first.time_slider.t_ms == 1200
    assert first.scanimage_connection is second.scanimage_connection


def test_close_application_checks_unsaved_hidden_projects(window, monkeypatch):
    first = window.active_workspace
    first.project.mark_dirty()
    second = window._add_workspace(_project())
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    assert not window.close()
    assert window.active_workspace is first
    assert not first._disposed and not second._disposed
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )
    assert window.close()
    assert first._disposed and second._disposed


def test_close_tab_waits_for_save_and_tracks_workspace_after_reordering(
    window, monkeypatch
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    second.project.mark_dirty()
    callbacks = []
    monkeypatch.setattr(
        second, "_save_project", lambda on_success: callbacks.append(on_success)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Save,
    )
    window._close_tab(1)
    assert not second._disposed
    window.project_tabs.tabBar().moveTab(1, 0)
    second.project.dirty = False  # what the real save does on success
    callbacks[0](None)
    assert second._disposed and not first._disposed


def test_workspace_shortcuts_only_apply_within_their_workspace(window):
    # So Qt only triggers a shortcut in the workspace that has the focus.
    actions = [
        action
        for action in window.active_workspace.actions()
        if not action.shortcut().isEmpty()
    ]
    assert actions
    for action in actions:
        assert (
            action.shortcutContext()
            == QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )


def test_shortcut_from_the_shell_goes_to_the_current_workspace(window, monkeypatch):
    first = window.active_workspace
    second = window._add_workspace(_project())
    calls = []
    monkeypatch.setattr(first, "_create_new_project", lambda: calls.append("first"))
    monkeypatch.setattr(second, "_create_new_project", lambda: calls.append("second"))
    for workspace in (second, first):
        window.project_tabs.setCurrentWidget(workspace)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_N,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        window.keyPressEvent(event)
        assert event.isAccepted()
    assert calls == ["second", "first"]


def test_close_keeps_all_workspaces_alive_when_a_hidden_job_cannot_stop(
    window, monkeypatch
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    with monkeypatch.context() as patch:
        patch.setattr(first, "stop_background_job", lambda: False)
        assert not window.close()
        assert not first._disposed and not second._disposed
        assert window.active_workspace is second


def test_failed_save_during_application_close_restores_the_window(window, monkeypatch):
    workspace = window.active_workspace
    workspace.project.mark_dirty()
    callbacks = {}

    def save(**kwargs):
        callbacks.update(kwargs)
        return True

    monkeypatch.setattr(workspace, "_save_project", save)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Save,
    )
    assert not window.close()
    assert not window.isEnabled()
    callbacks["on_failure"]("save failed")
    assert window.isEnabled()
    assert workspace.project.dirty and not workspace._disposed


def test_recording_workspace_stays_open_until_its_acquisition_ends(
    window, mocker, qmessagebox_calls, tmp_path
):
    workspace = window.active_workspace
    connection = window.devices.scanimage
    mocker.patch.object(connection.process, "write")
    connection.connected = True
    try:
        workspace.acquisition_controller.start_scanimage_recording(
            {
                "run_id": "pending-run",
                "name": "Run",
                "directory": str(tmp_path),
                "stem": "2026-10-22",
                "counter": 1,
                "frames": 4,
            }
        )
        assert workspace.acquisition_controller.acquisition_in_progress
        window._close_tab(0)
        assert not workspace._disposed
        assert window.active_workspace is workspace
        assert "image acquisition" in qmessagebox_calls[-1]["text"]
        # A failed run is reported only through status_changed; it must still
        # release the workspace (it used to stay "in progress" forever).
        connection._reply({"id": connection._serial, "error": "volume unsupported"})
        assert connection.run_id is None
        assert not workspace.acquisition_controller.run_ids
        assert workspace.project.imaging_acquisitions[0]["status"] == "failed"
        workspace.project.dirty = False
        window._close_tab(0)
        assert workspace._disposed
    finally:
        connection.connected = False


def test_disconnected_unknown_run_does_not_block_closing(window, mocker):
    workspace = window.active_workspace
    connection = window.devices.scanimage
    workspace.acquisition_controller.run_ids.add("lost")
    connection.run_id = "lost"
    try:
        # The helper died: the run stays reconcilable on reconnect, but
        # nothing can progress while disconnected.
        assert not workspace.close_blockers()
    finally:
        connection.run_id = None
        workspace.acquisition_controller.run_ids.clear()


def _start_job(workspace, *, cancellable=True, honour_cancel=True):
    release = threading.Event()
    results = []
    assert workspace._dispatch_io_task(
        lambda report: release.wait(WAIT_TIMEOUT_MS / 1000) and "done",
        "Long operation...",
        results.append,
        cancel_callback=(
            (release.set if honour_cancel else lambda: None) if cancellable else None
        ),
    )
    return release, results


def test_application_close_lists_all_blockers_and_stops_nothing(
    window, qtbot, qmessagebox_calls
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    release, results = _start_job(first)
    second.syscon2_panel._execution_phase = "running"
    try:
        assert not window.close()
        assert len(qmessagebox_calls) == 1
        text = qmessagebox_calls[0]["text"]
        assert "HOLO4D" in text and "Nothing was stopped" in text
        # The cancellable job in the other tab was not cancelled.
        assert first._background_job.busy and not release.is_set()
        assert not first._disposed and not second._disposed
    finally:
        second.syscon2_panel._execution_phase = "idle"
        release.set()
        qtbot.waitUntil(lambda: not first._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    assert results == ["done"]


def test_uncancellable_job_blocks_tab_close(window, qtbot, qmessagebox_calls):
    workspace = window.active_workspace
    release, _results = _start_job(workspace, cancellable=False)
    try:
        window._close_tab(0)
        assert not workspace._disposed
        assert (
            "Long operation is running and cannot be cancelled"
            in (qmessagebox_calls[-1]["text"])
        )
    finally:
        release.set()
        qtbot.waitUntil(
            lambda: not workspace._background_job.busy, timeout=WAIT_TIMEOUT_MS
        )


def test_application_close_cancels_jobs_then_closes_without_blocking(window, qtbot):
    first = window.active_workspace
    second = window._add_workspace(_project())
    release, results = _start_job(first)
    assert not window.close()  # returns at once; the job is still winding down
    assert first.progress_dialog.labelText() == "Cancelling..."
    qtbot.waitUntil(
        lambda: first._disposed and second._disposed, timeout=WAIT_TIMEOUT_MS
    )
    # The job's outcome was delivered rather than dropped.
    assert results == ["done"]


def test_close_timeout_keeps_workspaces_open_and_says_so(
    window, qtbot, monkeypatch, qmessagebox_calls
):
    monkeypatch.setattr("neuroseq.project_workspace.CLOSE_JOB_TIMEOUT_MS", 50)
    workspace = window.active_workspace
    release, _results = _start_job(workspace, honour_cancel=False)
    try:
        assert not window.close()
        qtbot.waitUntil(lambda: bool(qmessagebox_calls), timeout=WAIT_TIMEOUT_MS)
        assert "still stopping" in qmessagebox_calls[-1]["text"]
        assert not workspace._disposed and workspace._background_job.busy
    finally:
        release.set()
        qtbot.waitUntil(
            lambda: not workspace._background_job.busy, timeout=WAIT_TIMEOUT_MS
        )
    assert not window._close_gate.waiting


class _Manager(QtCore.QObject):
    connected = QtCore.Signal()
    disconnected = QtCore.Signal(str)
    state_changed = QtCore.Signal(dict)
    state_poll_failed = QtCore.Signal(str)
    command_finished = QtCore.Signal(str, object)
    command_failed = QtCore.Signal(str, str)

    def __init__(self):
        super().__init__()
        self.requests = []

    def submit(self, name, fn):
        self.requests.append(name)

    def shutdown(self):
        pass


def test_shared_syscon_connection_routes_replies_to_requesting_workspace(
    qapp, monkeypatch
):
    monkeypatch.setattr("neuroseq.workspace_devices.SysCon2ConnectionManager", _Manager)
    devices = WorkspaceDevices()
    first, second = devices.syscon_session(devices), devices.syscon_session(devices)
    replies, failures = [], []
    first.command_finished.connect(
        lambda name, result: replies.append(("first", name, result))
    )
    second.command_finished.connect(
        lambda name, result: replies.append(("second", name, result))
    )
    second.command_failed.connect(lambda name, error: failures.append((name, error)))
    first.submit("sequence_manager.import.1", lambda client: None)
    second.submit("sequence_manager.import.1", lambda client: None)
    devices.syscon.command_finished.emit(devices.syscon.requests[0], "A")
    devices.syscon.command_finished.emit(devices.syscon.requests[1], "B")
    devices.syscon.command_failed.emit(devices.syscon.requests[1], "error")
    assert replies == [
        ("first", "sequence_manager.import.1", "A"),
        ("second", "sequence_manager.import.1", "B"),
    ]
    assert failures == [("sequence_manager.import.1", "error")]
    # A callback already queued before disconnect may still reach the session.
    QtCore.QTimer.singleShot(
        0, lambda: first._finished(devices.syscon.requests[0], "queued reply")
    )
    first.shutdown()
    devices.syscon.command_finished.emit(devices.syscon.requests[0], "late")
    qapp.processEvents()
    assert len(replies) == 2
    second.shutdown()
    devices.shutdown()


def test_brian2_simulations_are_serialized_across_workspaces(
    window, qtbot, monkeypatch, qmessagebox_calls
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    release = threading.Event()
    runs = []
    monkeypatch.setattr(
        first.project,
        "run_simulation",
        lambda *args, **kwargs: release.wait(WAIT_TIMEOUT_MS / 1000),
    )
    monkeypatch.setattr(
        second.project, "run_simulation", lambda *args, **kwargs: runs.append(1)
    )
    first._do_brian2_simulation_threaded(0.1)
    assert first._background_job.busy
    try:
        second._do_brian2_simulation_threaded(0.1)
        assert not second._background_job.busy
        label = first._tab_label_for(first.project)
        assert f"'{label}' is already simulating" in qmessagebox_calls[-1]["text"]
        # External simulators may drive Brian2 too and take the same guard.
        second.project.external_simulate_function = lambda *args, **kwargs: None
        second._do_external_simulation(0.1)
        assert not second._background_job.busy
        # I/O stays per workspace.
        assert second._dispatch_io_task(lambda report: None, "Saving", lambda _: None)
        qtbot.waitUntil(
            lambda: not second._background_job.busy, timeout=WAIT_TIMEOUT_MS
        )
    finally:
        release.set()
    qtbot.waitUntil(lambda: not first._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    second._do_brian2_simulation_threaded(0.1)
    qtbot.waitUntil(lambda: not second._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    assert runs == [1]
    second.project.external_simulate_function = None


def test_simulation_guard_is_released_on_failure_cancel_and_close(
    window, qtbot, monkeypatch
):
    workspace = window.active_workspace

    def fail(*args, **kwargs):
        raise RuntimeError("diverged")

    monkeypatch.setattr(workspace.project, "run_simulation", fail)
    workspace._do_brian2_simulation_threaded(0.1)
    assert SimulationGuard.held_by() is workspace
    qtbot.waitUntil(lambda: not workspace._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    assert SimulationGuard.held_by() is None

    cancelled = threading.Event()
    monkeypatch.setattr(
        workspace.project,
        "run_simulation",
        lambda *a, **k: cancelled.wait(WAIT_TIMEOUT_MS / 1000),
    )
    monkeypatch.setattr(workspace.project, "request_cancel_simulation", cancelled.set)
    workspace._do_brian2_simulation_threaded(0.1)
    workspace._cancel_background_job()
    qtbot.waitUntil(lambda: not workspace._background_job.busy, timeout=WAIT_TIMEOUT_MS)
    assert SimulationGuard.held_by() is None

    closing = threading.Event()
    monkeypatch.setattr(
        workspace.project,
        "run_simulation",
        lambda *a, **k: closing.wait(WAIT_TIMEOUT_MS / 1000),
    )
    monkeypatch.setattr(workspace.project, "request_cancel_simulation", closing.set)
    workspace._do_brian2_simulation_threaded(0.1)
    window._close_tab(0)
    qtbot.waitUntil(lambda: workspace._disposed, timeout=WAIT_TIMEOUT_MS)
    assert SimulationGuard.held_by() is None


def test_first_frame_timeout_belongs_to_the_workspace_that_started_the_run(
    window, mocker, tmp_path
):
    first = window.active_workspace
    second = window._add_workspace(_project())
    connection = window.devices.scanimage
    command = mocker.patch.object(connection, "command", return_value=True)
    first.acquisition_controller.start_scanimage_recording(
        {
            "run_id": "a",
            "name": "Run",
            "directory": str(tmp_path),
            "stem": "2026-10-22",
            "counter": 1,
            "frames": 4,
        }
    )
    connection.run_id = "a"
    try:
        connection.status_changed.emit(
            {
                "acq_state": "grab",
                "run": {"id": "a", "state": "waiting_for_first_frame"},
            }
        )
        assert first.scanimage_controls._waiting_since is not None
        assert second.scanimage_controls._waiting_since is None
        # Even a stale timer in another workspace must not abort this run.
        second.scanimage_controls._waiting_since = -1e9
        second.scanimage_controls._check_first_frame_timeout()
        assert all(call.args[0] != "abort" for call in command.call_args_list)
        first.scanimage_controls._waiting_since = -1e9
        first.scanimage_controls._check_first_frame_timeout()
        command.assert_called_with("abort")
    finally:
        connection.run_id = None
        first.project.dirty = False
