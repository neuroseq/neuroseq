import json
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np
import pytest
import tifffile

from neuroseq.data import Experiment
from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.device_control.scanimage_engine_worker import EngineAdapter
from neuroseq.gui_settings import GuiSettings
from neuroseq.importers.importer_scanimage import index_recording
from tests.wait_timeout import WAIT_TIMEOUT_MS


@pytest.fixture
def recording(tmp_path, monkeypatch):
    metadata = {
        "FrameData": {
            "SI.hChannels.channelSave": [1, 2],
            "SI.hRoiManager.scanFrameRate": 10,
            "SI.hStackManager.numSlices": 1,
        }
    }
    monkeypatch.setattr(
        tifffile.TiffFile, "scanimage_metadata", property(lambda _: metadata)
    )
    paths = []
    for part in range(2):
        path = tmp_path / f"acquisition_{part}.tif"
        with tifffile.TiffWriter(path) as writer:
            for frame in range(2):
                for channel in range(2):
                    writer.write(
                        np.full((4, 5), frame + channel, dtype=np.int16),
                        description=f"frameTimestamps_sec = {2 + part * 0.2 + frame * 0.1}\n",
                        metadata=None,
                    )
        paths.append(str(path))
    return paths, metadata


def test_scanimage_index_preserves_channels_timestamps_and_raw_pixels(recording):
    paths, _ = recording
    result = index_recording(paths)
    assert result["channels"] == [1, 2]
    assert len(result["frames"]["page"]) == 8
    assert [
        t
        for t, c in zip(
            result["frames"]["time_s"], result["frames"]["channel"], strict=True
        )
        if c == 2
    ] == pytest.approx([0, 0.1, 0.2, 0.3])
    assert result["duration_s"] == pytest.approx(0.4)
    assert tifffile.imread(paths[0]).dtype == np.int16


def test_scanimage_rejects_volume_and_reversed_files(recording):
    paths, meta = recording
    with pytest.raises(ValueError, match="increasing"):
        index_recording(paths[::-1])
    meta["FrameData"]["SI.hStackManager.numSlices"] = 2
    with pytest.raises(ValueError, match="single-plane"):
        index_recording(paths)


@pytest.mark.parametrize(
    "stack",
    [
        # As on the rig: a 25-slice stack is configured but stacks are disabled.
        {"SI.hStackManager.actualNumSlices": 1},
        {"SI.hStackManager.enable": False},
    ],
)
def test_scanimage_disabled_stack_records_one_plane(recording, stack):
    paths, meta = recording
    meta["FrameData"] |= {"SI.hStackManager.numSlices": 25, **stack}
    assert index_recording(paths)["channels"] == [1, 2]


def test_scanimage_rejects_an_active_stack(recording):
    paths, meta = recording
    meta["FrameData"] |= {
        "SI.hStackManager.numSlices": 25,
        "SI.hStackManager.actualNumSlices": 25,
    }
    with pytest.raises(ValueError, match="single-plane"):
        index_recording(paths)


def test_engine_connect_requires_exact_running_session(mocker):
    matlab = ModuleType("matlab")
    engine = ModuleType("matlab.engine")
    matlab.engine = engine
    engine.find_matlab = mocker.Mock(return_value=("OtherSession",))
    engine.connect_matlab = mocker.Mock()
    mocker.patch.dict(sys.modules, {"matlab": matlab, "matlab.engine": engine})
    with pytest.raises(ValueError, match="Named MATLAB session not found"):
        EngineAdapter().dispatch("connect", {"session": "NeuroSeqScanImage"})
    engine.connect_matlab.assert_not_called()


def test_engine_does_not_expose_arbitrary_eval():
    with pytest.raises(ValueError, match="Unknown"):
        EngineAdapter().dispatch("eval", {"code": "anything"})


def test_engine_indexes_recording_with_relocated_worker(recording):
    paths, _ = recording
    result = EngineAdapter().dispatch("index", {"paths": paths})
    assert result["channels"] == [1, 2]
    assert len(result["frames"]["page"]) == 8


def test_engine_connect_adds_existing_matlab_resources(mocker):
    matlab = ModuleType("matlab")
    engine = ModuleType("matlab.engine")
    matlab.engine = engine
    engine.find_matlab = mocker.Mock(return_value=("NeuroSeqScanImage",))
    session = mocker.Mock()
    session.neuroseq_scanimage.return_value = "{}"
    engine.connect_matlab = mocker.Mock(return_value=session)
    mocker.patch.dict(sys.modules, {"matlab": matlab, "matlab.engine": engine})
    EngineAdapter().dispatch("connect", {"session": "NeuroSeqScanImage"})
    resources = Path(session.addpath.call_args.args[0])
    assert (resources / "neuroseq_scanimage.m").is_file()
    assert (resources / "neuroseq_scanimage_event.m").is_file()


@pytest.fixture
def connection(qtbot, mocker):
    conn = ScanImageConnection()
    mocker.patch.object(conn.process, "write")
    yield conn
    conn.shutdown()
    with qtbot.waitSignal(conn.destroyed, timeout=WAIT_TIMEOUT_MS):
        conn.deleteLater()


def test_poll_is_bounded_and_abort_has_priority(connection):
    connection.connected = True
    connection._poll()
    connection._poll()
    assert connection.process.write.call_count == 1
    assert connection.command("abort")
    connection._reply({"id": 1, "result": {}})
    connection._poll()
    assert connection._pending[1] == "abort"


def test_stale_reply_cannot_complete_new_request(connection):
    connection._send("status")
    connection._reply({"id": 0, "result": {}})
    assert connection._pending == (1, "status")


def test_finalization_requires_matching_run_and_stable_files(
    connection, recording, mocker
):
    paths, _ = recording
    connection.run_id = "our-run"
    status = {
        "run": {"id": "other-run", "state": "completed"},
        "acq_state": "idle",
        "files": paths,
    }
    connection._send("status")
    connection._reply({"id": 1, "result": status})
    assert connection._pending is None
    status["run"]["id"] = "our-run"
    for request in (2, 3):
        connection._send("status")
        connection._reply({"id": request, "result": status})
    assert connection._pending[1] == "index"
    ready = mocker.Mock()
    finished = mocker.Mock()
    connection.recording_ready.connect(ready)
    connection.run_finished.connect(finished)
    connection._reply({"id": 4, "result": index_recording(paths)})
    assert ready.call_args.args[0]["run_id"] == "our-run"
    finished.assert_called_once_with("our-run")
    assert ready.call_args.args[0]["status"] == "completed"
    assert connection.run_id is None


@pytest.mark.parametrize(
    ("expected_frames", "status"), [(4, "completed"), (5, "aborted")]
)
def test_aborted_run_holding_every_requested_frame_is_completed(
    connection, recording, mocker, expected_frames, status
):
    """ScanImage can fire acqAbort at the end of a triggered Grab."""
    paths, _ = recording
    connection.run_id = "our-run"
    connection._expected_frames = expected_frames
    connection._last_status = {"run": {"id": "our-run", "state": "aborted"}}
    ready = mocker.Mock()
    connection.recording_ready.connect(ready)
    connection._send("index", {"paths": paths})
    connection._reply({"id": 1, "result": index_recording(paths)})
    assert ready.call_args.args[0]["status"] == status


def test_record_request_waits_for_poll_and_abort_cancels_it(connection):
    connection.connected = True
    connection._poll()
    assert connection.command("record", {"run_id": "queued"})
    assert connection.command("abort")
    connection._reply({"id": 1, "result": {}})
    connection._poll()
    assert connection.run_id is None
    assert connection._queued is None
    assert connection.process.write.call_count == 1


def test_acquisition_persistence_marks_unfinished_runs_interrupted(tmp_path, recording):
    from neuroseq.project import Project
    from neuroseq.save_load_options import SaveLoadOptions

    project = Project()
    index = index_recording(recording[0])
    index["run_id"] = "two"
    project.imaging_acquisitions = [
        {"run_id": "one", "status": "recording", "directory": "C:/recordings/one"},
        {"run_id": "two", "status": "completed", "recording": index},
    ]
    project.add_calcium_video([Path(p) for p in recording[0]], 10, run_id="two")
    path = tmp_path / "recording.nseq"
    project.save(path, SaveLoadOptions(True, True, False, False))
    loaded = Project(path)
    assert loaded.imaging_acquisitions[0]["status"] == "interrupted"
    assert loaded.imaging_acquisitions[0]["directory"] == "C:/recordings/one"
    assert loaded.calcium_scanimage_index == project.calcium_scanimage_index


def test_recording_stays_with_originating_project(qtbot, mocker, recording, tmp_path):
    from neuroseq.gui import Gui
    from neuroseq.project import Project

    window = Gui()
    qtbot.addWidget(window)
    gui = window.active_workspace
    original = gui.project
    mocker.patch.object(gui.scanimage_connection, "command", return_value=True)
    gui.acquisition_controller.start_scanimage_recording(
        {
            "run_id": "one",
            "name": "Test",
            "directory": str(tmp_path),
            "stem": "2026-10-22",
            "counter": 1,
            "frames": 4,
        }
    )
    assert original.imaging_acquisitions[0]["status"] == "requested"
    other = window._add_workspace(Project())
    result = index_recording(recording[0])
    result.update(run_id="one", status="completed")
    window.devices.scanimage.recording_ready.emit(result)
    assert "Test" in original.experiments
    assert not other.project.imaging_acquisitions
    assert original.calcium_video_paths
    original.dirty = False


def test_controls_trigger_timeout_and_foreign_acquisition(
    qtbot, connection, mocker, tmp_config_dir
):
    from neuroseq.device_control.scanimage_controls import ScanImageControls

    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    command = mocker.patch.object(connection, "command", return_value=True)
    panel._status({"acq_state": "grab", "run": {}, "can_abort": False})
    assert not panel.abort_button.isEnabled()
    assert not panel.record_button.isEnabled()
    connection.run_id = "run"
    mocker.patch(
        "neuroseq.device_control.scanimage_controls.time.monotonic", return_value=100
    )
    panel._status(
        {
            "acq_state": "grab",
            "run": {"id": "run", "state": "waiting_for_first_frame"},
            "can_abort": True,
        }
    )
    panel._waiting_since = 1
    panel._check_first_frame_timeout()
    command.assert_called_once_with("abort")
    assert "timeout" in panel.status_label.text()


def test_controls_remember_python_and_folder_across_sessions(
    qtbot, connection, mocker, tmp_config_dir, tmp_path
):
    from neuroseq.device_control.scanimage_controls import ScanImageControls

    python = tmp_path / "python.exe"
    python.write_text("")
    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    mocker.patch(
        "neuroseq.device_control.scanimage_controls.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(str(python), ""),
    )
    panel._browse_python()
    mocker.patch(
        "neuroseq.device_control.scanimage_controls.QtWidgets.QFileDialog.getExistingDirectory",
        return_value=str(tmp_path),
    )
    panel._browse_directory()

    # A new session reads them back from the config file as the defaults.
    reopened = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(reopened)
    assert reopened.python_edit.text() == str(python)
    assert reopened.directory_edit.text() == str(tmp_path)

    connect = mocker.patch.object(connection, "connect_session")
    other = tmp_path / "other-python.exe"
    other.write_text("")
    reopened.python_edit.setText(str(other))
    reopened.connect_button.click()
    connect.assert_called_once_with(str(other), "NeuroSeqScanImage")
    assert GuiSettings().scanimage_python_path == str(other)


def test_controls_start_at_channel_1_and_grab_300_fov_frames(
    qtbot, connection, tmp_config_dir
):
    from neuroseq.device_control.scanimage_controls import ScanImageControls

    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    assert panel.channel_spin.value() == 1
    assert panel.fov_frames_spin.value() == 300
    panel.channel_spin.setValue(2)

    reopened = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(reopened)
    assert reopened.channel_spin.value() == 1


def test_controls_messages_are_selectable(qtbot, connection, tmp_config_dir):
    from qtpy import QtCore

    from neuroseq.device_control.scanimage_controls import ScanImageControls

    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    for label in (panel.message_label, panel.status_label):
        assert (
            label.textInteractionFlags()
            & QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )


def test_record_failure_releases_run_and_timeout_stops_only_helper(connection, mocker):
    connection.connected = True
    connection.command("record", {"run_id": "run"})
    error = mocker.Mock()
    finished = mocker.Mock()
    connection.error.connect(error)
    connection.run_finished.connect(finished)
    connection._reply({"id": 1, "error": "unsupported volume"})
    assert connection.run_id is None
    finished.assert_called_once_with("run")
    error.assert_called_with("unsupported volume")
    kill = mocker.patch.object(connection.process, "kill")
    connection._timed_out()
    kill.assert_called_once()
    assert not connection.connected
    assert "unknown" in error.call_args.args[0]


def test_worker_writes_metadata_next_to_the_recording_before_starting(tmp_path, mocker):
    adapter = EngineAdapter()
    call = mocker.patch.object(adapter, "_call", return_value={})
    arguments = {"directory": str(tmp_path / "rec"), "stem": "2026-10-22", "counter": 3}
    adapter.dispatch("record", {**arguments, "manifest": {"run_id": "run"}})
    sidecar = tmp_path / "rec" / "2026-10-22_00003.json"
    # Human-readable, beside the TIFF ScanImage will write (#628).
    assert json.loads(sidecar.read_text()) == {"run_id": "run"}
    assert "\n  " in sidecar.read_text()
    assert "manifest" not in call.call_args.args[1]
    assert adapter.dispatch("status", {})["manifest_path"] == str(sidecar)
    with pytest.raises(FileExistsError):
        adapter.dispatch("record", {**arguments, "manifest": {}})
    assert call.call_count == 2


def test_worker_removes_metadata_when_scanimage_refuses(tmp_path, mocker):
    adapter = EngineAdapter()
    adapter.manifest_path = tmp_path / "earlier.json"
    mocker.patch.object(adapter, "_call", side_effect=RuntimeError("busy"))
    arguments = {"directory": str(tmp_path), "stem": "2026-10-22", "counter": 1}
    with pytest.raises(RuntimeError):
        adapter.dispatch("record", {**arguments, "manifest": {}})
    assert not (tmp_path / "2026-10-22_00001.json").exists()
    # A later status must not report the previous run's metadata.
    assert adapter.manifest_path is None


def test_reconnect_releases_untracked_run_without_claiming_completion(
    connection, mocker
):
    connection.run_id = "lost"
    states = mocker.Mock()
    finished = mocker.Mock()
    connection.status_changed.connect(states)
    connection.run_finished.connect(finished)
    connection._send("connect")
    connection._reply(
        {
            "id": 1,
            "result": {
                "acq_state": "idle",
                "run": {"id": "other", "state": "completed"},
            },
        }
    )
    assert connection.connected
    assert connection.run_id is None
    assert states.call_args_list[0].args[0]["run"]["state"] == "interrupted"
    finished.assert_called_once_with("lost")


def test_completed_count_mismatch_does_not_attach_recording(
    connection, recording, mocker
):
    connection.run_id = "run"
    connection._expected_frames = 100
    connection._last_status = {"run": {"id": "run", "state": "completed"}}
    connection._finalize_since = 0
    mocker.patch(
        "neuroseq.device_control.scanimage_connection.time.monotonic", return_value=100
    )
    ready = mocker.Mock()
    connection.recording_ready.connect(ready)
    connection._send("index")
    connection._reply({"id": 1, "result": index_recording(recording[0])})
    ready.assert_not_called()
    assert connection.run_id is None


def test_recording_completion_attaches_and_reopens_movie(
    qtbot, mocker, recording, tmp_path
):
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    mocker.patch.object(gui.scanimage_connection, "command", return_value=True)
    gui.acquisition_controller.start_scanimage_recording(
        {
            "run_id": "one",
            "name": "Test",
            "directory": str(tmp_path),
            "stem": "2026-10-22",
            "counter": 1,
            "frames": 4,
        }
    )
    gui.scanimage_connection.run_id = "one"
    gui.acquisition_controller.scanimage_status(
        {
            "run": {"id": "one", "state": "recording"},
            "matlab_version": "2023a",
            "settings": {"saved_channels": [1, 2]},
        }
    )
    record = gui.project.imaging_acquisitions[0]
    assert record["matlab_version"] == "2023a"
    gui.scanimage_connection.error.emit("temporary error")
    assert "error" not in record
    assert record["status"] == "recording"
    result = index_recording(recording[0])
    result.update(run_id="one", status="completed")
    gui.scanimage_connection.run_id = None
    gui.scanimage_controls.channel_spin.setValue(2)
    # A deleted group with the recording's name must not be resurrected over it.
    old_group = Experiment(name="Test", duration=1.0, timepoints=[], data=[])
    gui.project.add_experiment(old_group)
    gui.project.remove_experiments(["Test"])
    gui.acquisition_controller.scanimage_recording_ready(result)
    recorded = gui.project.experiments["Test"]
    assert recorded is not old_group
    assert gui.project.history.undo_label == "Add data group"
    assert gui.calcium_video_widget.recorded_channel == 2
    assert gui.project.calcium_scanimage_index == result
    assert len(gui.calcium_video_widget._frame_refs) == 4
    assert gui.cmb_video.count() == 1
    assert gui.cmb_video.currentText() == "Test"
    gui.project.history.undo()
    assert "Test" not in gui.project.experiments
    gui.project.history.undo()
    assert gui.project.experiments["Test"] is old_group
    gui._close_confirmed = True


def test_helper_process_connects_and_disconnects_named_session(qtbot, tmp_path):
    from qtpy import QtCore

    package = tmp_path / "matlab"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "engine.py").write_text("""
import json
def find_matlab():
    return ('Named',)
def connect_matlab(name):
    assert name == 'Named'
    print('engine diagnostic')
    return Engine()
class Engine:
    def addpath(self, *args, **kwargs):
        pass
    def neuroseq_scanimage(self, operation, payload, **kwargs):
        return json.dumps({'acq_state': 'idle', 'run': {'id': '', 'state': 'idle'}})
""")
    conn = ScanImageConnection()
    environment = QtCore.QProcessEnvironment.systemEnvironment()
    environment.insert("PYTHONPATH", str(tmp_path))
    conn.process.setProcessEnvironment(environment)
    try:
        conn.connect_session(sys.executable, "Named")
        qtbot.waitUntil(lambda: conn.connected, timeout=WAIT_TIMEOUT_MS)
        assert "engine diagnostic" in conn._stderr
        assert conn.command("disconnect")
        qtbot.waitUntil(lambda: not conn.connected, timeout=WAIT_TIMEOUT_MS)
        assert conn.process.state() == QtCore.QProcess.ProcessState.NotRunning
    finally:
        conn.shutdown()
        with qtbot.waitSignal(conn.destroyed, timeout=WAIT_TIMEOUT_MS):
            conn.deleteLater()


def test_standard_two_channel_run_has_compact_transport_and_metadata(
    tmp_path, recording
):
    from neuroseq.project import Project
    from neuroseq.save_load_options import SaveLoadOptions

    _, metadata = recording
    metadata["FrameData"]["SI.hRoiManager.scanFrameRate"] = 30
    path = tmp_path / "short.tif"
    pixels = np.zeros((4, 4), dtype=np.int16)
    with tifffile.TiffWriter(path) as writer:
        for frame in range(30):
            for _ in range(2):
                writer.write(
                    pixels,
                    metadata=None,
                    description=(
                        f"frameTimestamps_sec = {frame / 30}\nframeNumbers = {frame}\n"
                        "auxTrigger0 = []\n" + "x" * 1000
                    ),
                )
    index = index_recording([str(path)])
    assert set(index["frames"]) == {
        "file",
        "page",
        "channel",
        "time_s",
        "scanimage_time_s",
    }
    # Writing 54,000 real TIFF pages takes minutes under coverage tracing, so
    # extend the real index's columns to the standard 15 min, 30 Hz protocol.
    frames = 27000
    index["frames"] = {
        "file": [0] * (2 * frames),
        "page": list(range(2 * frames)),
        "channel": [1, 2] * frames,
        "time_s": [frame / 30 for frame in range(frames) for _ in range(2)],
        "scanimage_time_s": [frame / 30 for frame in range(frames) for _ in range(2)],
    }
    index["duration_s"] = frames / 30
    index.update(run_id="long", status="completed")
    assert len(json.dumps({"id": 1, "result": index}).encode()) < 5_000_000
    project = Project()
    project.imaging_acquisitions = [
        {"run_id": "long", "status": "completed", "recording": index}
    ]
    project.add_calcium_video([Path("long.tif")], 30, run_id="long")
    archive = tmp_path / "long.nseq"
    project.save(archive, SaveLoadOptions(True, True, False, False))
    with ZipFile(archive) as zf:
        raw = zf.read("project_metadata.json")
    assert len(raw) < 5_000_000
    assert json.loads(raw)["calcium_videos"][0]["run_id"] == "long"
    assert "index" not in json.loads(raw)["calcium_videos"][0]
    loaded = Project(archive)
    assert loaded.calcium_scanimage_index is loaded.imaging_acquisitions[0]["recording"]
    assert loaded.calcium_scanimage_index == index


def test_recorded_channel_uses_file_indices_through_symlink(qtbot, recording, tmp_path):
    from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget

    paths, _ = recording
    link = tmp_path / "linked"
    link.symlink_to(tmp_path, target_is_directory=True)
    linked_paths = [str(link / Path(p).name) for p in paths]
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths(
        [Path(p) for p in linked_paths],
        10,
        warn_if_missing_cellpose=False,
        scanimage_index=index_recording(linked_paths),
        channel=2,
    )
    assert widget.recorded_channel == 2
    assert len(widget._frame_refs) == 4
    for frame, expected in enumerate((1, 2, 1, 2)):
        np.testing.assert_array_equal(
            widget._read_frame(frame), np.full((4, 5), expected)
        )


@pytest.mark.parametrize("switch_project", [False, True])
def test_new_recording_preserves_imported_movie(
    qtbot, mocker, recording, switch_project, tmp_path
):
    from neuroseq.gui import Gui
    from neuroseq.project import Project

    window = Gui()
    qtbot.addWidget(window)
    gui = window.active_workspace
    project = gui.project
    imported = Path(recording[0][0])
    project.add_calcium_video([imported], 20, name="Imported")
    mocker.patch.object(gui.scanimage_connection, "command", return_value=True)
    gui.acquisition_controller.start_scanimage_recording(
        {
            "run_id": "new",
            "name": "New",
            "directory": str(tmp_path),
            "stem": "2026-10-22",
            "counter": 1,
            "frames": 4,
        }
    )
    if switch_project:
        window._add_workspace(Project())
    result = index_recording(recording[0])
    result.update(run_id="new", status="completed")
    window.devices.scanimage.recording_ready.emit(result)
    assert [video["name"] for video in project.calcium_videos] == ["Imported", "New"]
    assert project.calcium_videos[0]["paths"] == [str(imported)]
    assert project.calcium_videos[0]["fps"] == 20
    assert project.calcium_scanimage_index == result
    assert project.imaging_acquisitions[0]["recording"] == result
    if switch_project:
        window.project_tabs.setCurrentWidget(gui)
    assert [gui.cmb_video.itemText(i) for i in range(gui.cmb_video.count())] == [
        "Imported",
        "New",
    ]
    gui.cmb_video.activated.emit(0)
    assert project.calcium_video_paths == [imported]
    assert project.calcium_scanimage_index is None
    gui.scanimage_controls.channel_spin.setValue(2)
    gui.cmb_video.activated.emit(1)
    assert gui.calcium_video_widget.recorded_channel == 2
    gui.scanimage_controls.channel_spin.setValue(3)
    assert gui.calcium_video_widget.recorded_channel == 1
    # Only the display falls back; routines still measure the chosen channel.
    assert gui.scanimage_controls.channel_spin.value() == 3
    assert "was not saved" in gui.scanimage_controls.message_label.text()
    project.dirty = False


def test_transport_failure_is_run_scoped_but_ui_errors_are_not(connection, mocker):
    states = mocker.Mock()
    connection.status_changed.connect(states)
    connection.run_id = "owned"
    connection.error.emit("A command is already queued")
    states.assert_not_called()
    connection._timed_out()
    assert states.call_args.args[0]["run"]["id"] == "owned"
    assert states.call_args.args[0]["run"]["state"] == "unknown"
    assert "30 seconds" in states.call_args.args[0]["run"]["error"]
    assert connection.run_id == "owned"  # Reconnect can reconcile this run.
    finished = mocker.Mock()
    connection.run_finished.connect(finished)
    connection._end_run()
    finished.assert_called_once_with("owned")


def test_empty_imaging_experiment_is_safe_in_gui_and_import_selector(qtbot):
    from neuroseq.add_experiment_data_dialog import AddExperimentDataDialog
    from neuroseq.data import Experiment
    from neuroseq.data_manager_dialog import DataManagerDialog
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.project.experiments["Imaging"] = Experiment("Imaging", 10, [], [])
    gui.update_project(gui.project)
    for combo in (gui.cmb_trace, gui.cmb_spike, gui.cmb_synapse, gui.cmb_data):
        assert all("Imaging" not in combo.itemText(i) for i in range(combo.count()))
    manager = DataManagerDialog(gui.project)
    qtbot.addWidget(manager)
    row = next(
        i
        for i in range(manager.table.rowCount())
        if manager.table.item(i, 0).text() == "Imaging"
    )
    manager.table.selectRow(row)
    assert manager.table.item(row, 2).text() == "0"
    dialog = AddExperimentDataDialog(Experiment("Import", 1, [], []), gui.project)
    qtbot.addWidget(dialog)
    dialog.cmb_experiments.setCurrentText("Imaging")
    assert dialog.cmb_offset.count() == 0
    assert dialog.offset == 0
    gui._close_confirmed = True


def test_early_project_indexes_upgrade_without_descriptions(tmp_path, recording):
    from neuroseq.project import Project
    from neuroseq.save_load_options import SaveLoadOptions

    index = index_recording(recording[0])
    columns = index["frames"]
    legacy = {
        **index,
        "run_id": "old",
        "frames": [
            {
                "path": index["paths"][file],
                "description": "legacy metadata",
                **{
                    key: columns[key][i]
                    for key in ("page", "channel", "time_s", "scanimage_time_s")
                },
            }
            for i, file in enumerate(columns["file"])
        ],
    }
    project = Project()
    # Simulate the initial PR's archive format, including its duplicate index.
    archive = tmp_path / "old.nseq"
    project.save(archive, SaveLoadOptions(True, True, False, False))
    with ZipFile(archive) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    metadata = json.loads(entries["project_metadata.json"])
    metadata.update(
        imaging_acquisitions=[{"run_id": "old", "recording": legacy}],
        calcium_scanimage_index=legacy,
    )
    entries["project_metadata.json"] = json.dumps(metadata).encode()
    with ZipFile(archive, "w") as dest:
        for name, data in entries.items():
            dest.writestr(name, data)
    loaded = Project(archive)
    assert loaded.calcium_scanimage_index["frames"] == columns
    assert loaded.imaging_acquisitions[0]["recording"]["frames"] == columns
    saved = tmp_path / "new.nseq"
    loaded.save(saved, SaveLoadOptions(True, True, False, False))
    with ZipFile(saved) as zf:
        metadata = json.loads(zf.read("project_metadata.json"))
    assert metadata["calcium_videos"][0]["run_id"] == "old"
    assert "index" not in metadata["calcium_videos"][0]
    assert "calcium_scanimage_index" not in metadata
    assert "description" not in json.dumps(metadata["imaging_acquisitions"])


def test_image_acquisition_is_a_tab_of_the_stimulation_setup_pane(qtbot):
    from qtpy import QtWidgets

    from neuroseq.device_control.stimulation_setup_widget import SCANIMAGE_TAB_TITLE
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    # Not a window of its own (#651), and not above the Video tab's canvas,
    # which must keep mirroring the Cell segmentation tab.
    assert gui.scanimage_controls.window() is gui.window()
    assert gui.stimulation_setup.isAncestorOf(gui.scanimage_controls)
    assert not gui.calcium_video_widget.isAncestorOf(gui.scanimage_controls)
    toolbar = gui.btn_design_dynamic_input.parentWidget()
    assert isinstance(toolbar, QtWidgets.QToolBar)
    actions = toolbar.actions()
    widgets = [toolbar.widgetForAction(action) for action in actions]
    position = widgets.index(gui.btn_design_dynamic_input)
    assert actions[position + 1].isSeparator()
    assert widgets[position + 2 : position + 4] == [
        gui.btn_image_acquisition,
        gui.cmb_video,
    ]
    assert not gui.cmb_video.isEnabled()
    assert not gui.stimulation_setup.isVisibleTo(gui)
    gui.btn_image_acquisition.click()
    assert gui.stimulation_setup.isVisibleTo(gui)
    index = gui.stimulation_setup.currentIndex()
    assert gui.stimulation_setup.tabText(index) == SCANIMAGE_TAB_TITLE
    gui._close_confirmed = True


def test_video_tab_canvas_overlays_cell_segmentation_canvas(qtbot, recording):
    from qtpy import QtCore

    from neuroseq.project_workspace import ProjectWorkspace
    from tests.test_spatial_footprint_widget import _cellpose_result

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    footprint = gui.spatial_footprint_widget
    footprint.set_data(_cellpose_result())
    gui.project.add_calcium_video([Path(p) for p in recording[0]], 10)
    gui._refresh_calcium_video_widgets()

    def canvas_rect(canvas):
        top_left = canvas.mapTo(footprint.tabs, QtCore.QPoint(0, 0))
        return QtCore.QRect(top_left, canvas.size())

    footprint.tabs.setCurrentIndex(0)
    qtbot.waitUntil(
        lambda: footprint.overlay_widget.isVisible(), timeout=WAIT_TIMEOUT_MS
    )
    cell_rect = canvas_rect(footprint.overlay_widget)
    footprint.tabs.setCurrentIndex(1)
    qtbot.waitUntil(
        lambda: gui.calcium_video_widget.widget.isVisible(), timeout=WAIT_TIMEOUT_MS
    )
    assert canvas_rect(gui.calcium_video_widget.widget) == cell_rect
    gui._close_confirmed = True


def test_multiple_videos_are_selectable_and_persist(qtbot, tmp_path, recording):
    from neuroseq.project import Project
    from neuroseq.project_workspace import ProjectWorkspace
    from neuroseq.save_load_options import SaveLoadOptions

    first, second = (Path(p) for p in recording[0])
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.project.add_calcium_video([first], 10)
    gui.project.add_calcium_video([second], 20)
    gui._refresh_calcium_video_widgets()
    assert gui.cmb_video.count() == 2
    assert gui.cmb_video.currentIndex() == 1
    gui.cmb_video.activated.emit(0)
    assert gui.project.calcium_video_paths == [first]
    assert gui.calcium_video_widget._paths == [first]

    archive = tmp_path / "videos.nseq"
    gui.project.save(archive, SaveLoadOptions(True, True, False, False))
    loaded = Project(archive)
    assert [v["paths"] for v in loaded.calcium_videos] == [[str(first)], [str(second)]]
    assert loaded.calcium_video_paths == [first]
    assert loaded.calcium_video_fps == 10

    gui._on_calcium_video_remove_requested()
    assert gui.project.calcium_video_paths == [second]
    assert gui.cmb_video.count() == 1
    gui._close_confirmed = True


def test_selecting_a_video_with_missing_files_keeps_the_current_one(
    qtbot, recording, qmessagebox_calls
):
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    present = Path(recording[0][0])
    gui.project.add_calcium_video([Path("/missing/video.tif")], 10)
    gui.project.add_calcium_video([present], 10)
    gui._refresh_calcium_video_widgets()
    gui.cmb_video.activated.emit(0)
    assert gui.project.calcium_video_paths == [present]
    assert gui.cmb_video.currentIndex() == 1
    assert qmessagebox_calls
    gui._close_confirmed = True


def test_single_video_projects_from_main_load_as_a_one_item_list(tmp_path):
    from neuroseq.project import Project
    from neuroseq.save_load_options import SaveLoadOptions

    archive = tmp_path / "old.nseq"
    Project().save(archive, SaveLoadOptions(True, True, False, False))
    with ZipFile(archive) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    metadata = json.loads(entries["project_metadata.json"])
    metadata.update(calcium_video_paths=["/data/movie.tif"], calcium_video_fps=30)
    entries["project_metadata.json"] = json.dumps(metadata).encode()
    with ZipFile(archive, "w") as dest:
        for name, data in entries.items():
            dest.writestr(name, data)
    loaded = Project(archive)
    assert len(loaded.calcium_videos) == 1
    assert loaded.calcium_video_paths == [Path("/data/movie.tif")]
    assert loaded.calcium_video_fps == 30
    assert not loaded.dirty


@pytest.mark.parametrize("suffix", [".tif", ".gif"])
def test_exported_imaging_keeps_the_recording_orientation(qtbot, tmp_path, suffix):
    from PIL import Image

    from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget

    raw = np.zeros((20, 30), dtype=np.uint16)
    raw[:4, :6] = 1000  # bright block at the top-left of the recording
    movie = tmp_path / "movie.tif"
    tifffile.imwrite(movie, np.stack([raw, raw]))
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths([movie], 10, warn_if_missing_cellpose=False)
    export = tmp_path / f"export{suffix}"
    widget.export_labeled_video(export)
    if suffix == ".gif":
        with Image.open(export) as image:
            exported = np.asarray(image.convert("RGB"))
    else:
        exported = tifffile.imread(export)
    bright = exported[..., 0] > 127
    assert bright[:4, :6].all()
    assert not bright[4:].any()


def test_record_saves_the_channels_ticked_in_the_dialog(
    qtbot, connection, tmp_path, tmp_config_dir
):
    """Several channels can be saved in one recording; the checkboxes start
    as ScanImage's own selection."""
    from neuroseq.device_control.scanimage_controls import ScanImageControls

    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    requests: list[dict] = []
    panel.record_requested.connect(requests.append)
    panel.directory_edit.setText(str(tmp_path))

    assert panel.channel_spin.value() == 1
    panel._status({"acq_state": "idle", "channels_available": 4, "saved_channels": 2})
    assert [c.text() for c in panel.save_channel_checks] == ["1", "2", "3", "4"]
    assert panel.saved_channels() == [1]  # channel 1, whatever ScanImage saves
    panel._status(
        {"acq_state": "idle", "channels_available": 4, "saved_channels": [2, 3]}
    )
    assert panel.saved_channels() == [1]

    panel.save_channel_checks[2].setChecked(True)
    panel._status({"acq_state": "idle", "channels_available": 4, "saved_channels": 2})
    assert panel.saved_channels() == [1, 3]  # the user's choice is kept

    panel._record()
    assert requests[-1]["channels"] == [1, 3]

    for check in panel.save_channel_checks:
        check.setChecked(False)
    panel._record()
    assert len(requests) == 1
    assert "at least one channel" in panel.message_label.text()


def test_record_request_names_the_file_by_date_and_number(
    qtbot, connection, tmp_path, tmp_config_dir
):
    from neuroseq.device_control.scanimage_controls import ScanImageControls
    from neuroseq.device_control.scanimage_files import recording_stem

    panel = ScanImageControls(connection, GuiSettings())
    qtbot.addWidget(panel)
    panel.set_directory(str(tmp_path))
    stem = recording_stem()
    (tmp_path / f"{stem}_00001.tif").write_text("")
    panel._update_next_file()
    # A hidden tab does not list the folder; it catches up when shown.
    assert panel._next_file_stale
    panel.show()
    assert panel.next_file_label.text() == f"{stem}_00002.tif"
    requests: list[dict] = []
    panel.record_requested.connect(requests.append)
    panel.name_edit.setText("Slice 3 CA3")
    panel._record()
    request = requests[-1]
    # Directly in the folder (#628), not in a per-run subfolder.
    assert request["directory"] == str(tmp_path)
    assert (request["stem"], request["counter"]) == (stem, 2)
    assert request["name"] == "Slice 3 CA3"  # capitals kept
    panel.name_edit.clear()
    panel._record()
    assert requests[-1]["name"] == f"{stem}_00002"
    with pytest.raises(ValueError, match="recording folder"):
        panel.set_directory(str(tmp_path / "missing"))
        panel.recording_request(frames=1, external_trigger=True)


def test_fov_image_opens_cell_segmentation(qtbot, mocker, tmp_path):
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    segment = mocker.patch.object(gui, "btn_import_cellpose_data_clicked")
    image = tmp_path / "2026-10-23_00001_mean.tif"
    gui.acquisition_controller.fov_ready.emit(str(image))
    segment.assert_called_once_with(segment=image, auto_brightness=False)
    gui.scanimage_controls.fov_auto_brightness_check.setChecked(True)
    gui.acquisition_controller.fov_ready.emit(str(image))
    segment.assert_called_with(segment=image, auto_brightness=True)
    gui._close_confirmed = True


def test_routine_run_data_group_is_shown_on_its_first_block(qtbot, mocker):
    from neuroseq.project_workspace import ProjectWorkspace

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    select = mocker.patch.object(gui, "_select_experiment_data")
    recorded = gui.acquisition_controller.block_recorded
    recorded.emit("r1", {"experiment": "rheobase_2026-11-06_120000"})
    recorded.emit("r2", {"experiment": "rheobase_2026-11-06_120000"})
    recorded.emit("r3", {"experiment": "rheobase_2026-11-06_130000"})
    recorded.emit("r4", {})
    assert [c.args[0] for c in select.call_args_list] == [
        "rheobase_2026-11-06_120000",
        "rheobase_2026-11-06_130000",
    ]
    gui._close_confirmed = True
