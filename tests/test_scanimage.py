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
