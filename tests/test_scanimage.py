import base64
import sys
from types import ModuleType

import numpy as np
import pytest
import tifffile

from neuroseq.importer_scanimage import index_recording
from neuroseq.scanimage_connection import ScanImageConnection
from neuroseq.scanimage_engine_worker import EngineAdapter


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
    assert len(result["frames"]) == 8
    assert [
        f["time_s"] for f in result["frames"] if f["channel"] == 2
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


@pytest.fixture
def connection(qtbot, mocker):
    conn = ScanImageConnection()
    mocker.patch.object(conn.process, "write")
    yield conn
    conn.shutdown()
    conn.deleteLater()
    qtbot.wait(1)


def test_preview_protocol_decodes_matlab_column_order(connection, mocker):
    received = mocker.Mock()
    connection.frame_received.connect(received)
    pixels = np.arange(12, dtype="<f4").reshape(3, 4)
    connection._send("status")
    connection._reply(
        {
            "id": 1,
            "result": {
                "frame": {
                    "shape": [3, 4],
                    "dtype": "float32",
                    "data": base64.b64encode(pixels.tobytes(order="F")).decode(),
                }
            },
        }
    )
    np.testing.assert_array_equal(received.call_args.args[0]["pixels"], pixels)


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
    connection.recording_ready.connect(ready)
    connection._reply({"id": 4, "result": index_recording(paths)})
    assert ready.call_args.args[0]["run_id"] == "our-run"
    assert ready.call_args.args[0]["status"] == "completed"
    assert connection.run_id is None


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


def test_live_video_ignores_playback_slider_and_preserves_orientation(qtbot, recording):
    from pathlib import Path

    from neuroseq.calcium_video_widget import CalciumVideoWidget

    paths, _ = recording
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths(
        [Path(p) for p in paths],
        10,
        warn_if_missing_cellpose=False,
        scanimage_index=index_recording(paths),
    )
    assert len(widget._frame_refs) == 4
    widget.set_recorded_channel(2)
    widget.set_time_ms(250)
    assert widget.current_frame_index == 2
    widget.set_live(True)
    pixels = np.arange(20, dtype=np.float32).reshape(4, 5)
    widget.show_live_frame({"pixels": pixels})
    expected = widget._to_rgb(np.flipud(pixels))
    widget.set_time_ms(0)
    np.testing.assert_array_equal(widget.current_frame, expected)
    assert not widget.save_button.isEnabled()
    widget.set_live(False)
    assert widget.current_frame_index == 0


def test_acquisition_persistence_marks_unfinished_runs_interrupted(tmp_path):
    from neuroseq.project import Project
    from neuroseq.save_load_options import SaveLoadOptions

    project = Project()
    project.imaging_acquisitions = [
        {"run_id": "one", "status": "recording", "directory": "C:/recordings/one"}
    ]
    project.calcium_scanimage_index = {"channels": [2], "frames": [], "fps": 10}
    path = tmp_path / "recording.nseq"
    project.save(path, SaveLoadOptions(True, True, False, False))
    loaded = Project(path)
    assert loaded.imaging_acquisitions[0]["status"] == "interrupted"
    assert loaded.imaging_acquisitions[0]["directory"] == "C:/recordings/one"
    assert loaded.calcium_scanimage_index == project.calcium_scanimage_index


def test_recording_stays_with_originating_project(qtbot, mocker, recording):
    from neuroseq.gui import Gui
    from neuroseq.project import Project

    gui = Gui()
    qtbot.addWidget(gui)
    original = gui.project
    mocker.patch.object(gui.scanimage_connection, "command", return_value=True)
    gui._start_scanimage_recording(
        {"run_id": "one", "name": "Test", "directory": "/unused", "frames": 4}
    )
    assert original.imaging_acquisitions[0]["status"] == "requested"
    gui.update_project(Project())
    result = index_recording(recording[0])
    result.update(run_id="one", status="completed")
    gui._scanimage_recording_ready(result)
    assert "Test" in original.experiments
    assert not gui.project.imaging_acquisitions
    assert original.calcium_video_paths
    gui._close_confirmed = True
    gui.close()


def test_controls_trigger_timeout_and_foreign_acquisition(qtbot, connection, mocker):
    from neuroseq.scanimage_controls import ScanImageControls

    panel = ScanImageControls(connection)
    qtbot.addWidget(panel)
    command = mocker.patch.object(connection, "command", return_value=True)
    panel._status({"acq_state": "grab", "run": {}, "can_abort": False})
    assert not panel.abort_button.isEnabled()
    assert not panel.record_button.isEnabled()
    connection.run_id = "run"
    mocker.patch("neuroseq.scanimage_controls.time.monotonic", return_value=100)
    panel._status(
        {
            "acq_state": "grab",
            "run": {"id": "run", "state": "waiting_for_first_frame"},
            "can_abort": True,
        }
    )
    panel._waiting_since = 1
    panel._freshness()
    command.assert_called_once_with("abort")
    assert "timeout" in panel.status_label.text()


def test_protocol_rejects_corrupt_pixels(connection):
    connection._send("status")
    with pytest.raises(ValueError, match="payload"):
        connection._reply(
            {
                "id": 1,
                "result": {
                    "frame": {"shape": [2, 2], "dtype": "float32", "data": "AAAA"}
                },
            }
        )


def test_record_failure_releases_run_and_timeout_stops_only_helper(connection, mocker):
    connection.connected = True
    connection.command("record", {"run_id": "run"})
    error = mocker.Mock()
    connection.error.connect(error)
    connection._reply({"id": 1, "error": "unsupported volume"})
    assert connection.run_id is None
    error.assert_called_with("unsupported volume")
    kill = mocker.patch.object(connection.process, "kill")
    connection._timed_out()
    kill.assert_called_once()
    assert not connection.connected
    assert "unknown" in error.call_args.args[0]


def test_worker_writes_provenance_before_starting(tmp_path, mocker):
    adapter = EngineAdapter()
    call = mocker.patch.object(adapter, "_call", return_value={})
    directory = tmp_path / "run"
    adapter.dispatch(
        "record", {"directory": str(directory), "manifest": {"run_id": "run"}}
    )
    assert directory.with_suffix(".neuroseq.json").is_file()
    assert "manifest" not in call.call_args.args[1]
    with pytest.raises(FileExistsError):
        adapter.dispatch("record", {"directory": str(directory), "manifest": {}})
    assert call.call_count == 1


def test_reconnect_releases_untracked_run_without_claiming_completion(
    connection, mocker
):
    connection.run_id = "lost"
    states = mocker.Mock()
    connection.status_changed.connect(states)
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


def test_completed_count_mismatch_does_not_attach_recording(
    connection, recording, mocker
):
    connection.run_id = "run"
    connection._expected_frames = 100
    connection._last_status = {"run": {"id": "run", "state": "completed"}}
    connection._finalize_since = 0
    mocker.patch("neuroseq.scanimage_connection.time.monotonic", return_value=100)
    ready = mocker.Mock()
    connection.recording_ready.connect(ready)
    connection._send("index")
    connection._reply({"id": 1, "result": index_recording(recording[0])})
    ready.assert_not_called()
    assert connection.run_id is None


def test_recording_completion_attaches_and_reopens_movie(qtbot, mocker, recording):
    from neuroseq.gui import Gui

    gui = Gui()
    qtbot.addWidget(gui)
    mocker.patch.object(gui.scanimage_connection, "command", return_value=True)
    gui._start_scanimage_recording(
        {"run_id": "one", "name": "Test", "directory": "/unused", "frames": 4}
    )
    gui.scanimage_connection.run_id = "one"
    gui._scanimage_status(
        {
            "run": {"id": "one", "state": "recording"},
            "matlab_version": "2023a",
            "settings": {"saved_channels": [1, 2]},
        }
    )
    record = gui.project.imaging_acquisitions[0]
    assert record["matlab_version"] == "2023a"
    gui._scanimage_error("temporary error")
    assert record["error"] == "temporary error"
    result = index_recording(recording[0])
    result.update(run_id="one", status="completed")
    gui.scanimage_connection.run_id = None
    gui._scanimage_recording_ready(result)
    assert gui.project.calcium_scanimage_index == result
    assert len(gui.calcium_video_widget._frame_refs) == 4
    assert gui.scanimage_controls.recordings_combo.count() == 2
    gui.project.set_calcium_video([], None)
    gui._select_scanimage_recording("one")
    assert gui.project.calcium_video_paths
    assert gui.project.calcium_scanimage_index == result
    gui._close_confirmed = True
    gui.close()


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
        qtbot.waitUntil(lambda: conn.connected, timeout=10000)
        assert "engine diagnostic" in conn._stderr
        assert conn.command("disconnect")
        qtbot.waitUntil(lambda: not conn.connected, timeout=10000)
        assert conn.process.state() == QtCore.QProcess.ProcessState.NotRunning
    finally:
        conn.shutdown()
        conn.deleteLater()
        qtbot.wait(1)
