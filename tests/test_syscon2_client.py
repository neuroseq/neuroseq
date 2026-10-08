from __future__ import annotations

import sys

import pytest

from neuroseq.syscon2_client import LightsourceStatus
from neuroseq.syscon2_client import NamedPipeTransport
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_client import SysCon2ConnectionError
from neuroseq.syscon2_client import SysCon2UnavailableError
from neuroseq.syscon2_protocol import SysCon2Error
from neuroseq.syscon2_protocol import build_command_frame
from neuroseq.syscon2_protocol import encode_bool
from neuroseq.syscon2_protocol import encode_dword
from neuroseq.syscon2_protocol import encode_float
from neuroseq.syscon2_protocol import encode_string


class FakeTransport:
    """Queues one scripted response per ``send()``, mimicking the client's send-then-read cycle."""

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False
        self._buffer = b""

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        self._buffer += self._responses.pop(0)

    def recv(self, n: int) -> bytes:
        assert len(self._buffer) >= n, "not enough buffered response data"
        chunk, self._buffer = self._buffer[:n], self._buffer[n:]
        return chunk

    def close(self) -> None:
        self.closed = True


class BrokenTransport:
    """A transport whose ``recv`` fails, simulating a broken pipe mid-command."""

    def __init__(self) -> None:
        self.closed = False

    def send(self, data: bytes) -> None:
        pass

    def recv(self, n: int) -> bytes:
        raise SysCon2ConnectionError("SysCon2 pipe closed unexpectedly")

    def close(self) -> None:
        self.closed = True


def test_sci_get_instances() -> None:
    transport = FakeTransport([encode_string("Laser-1,Laser-2\r\n")])
    client = SysCon2Client(transport)

    result = client.sci.get_instances("Laser")

    assert result == ["Laser-1", "Laser-2"]
    assert transport.sent == [
        build_command_frame("sci", "GetInstances", encode_string("Laser"))
    ]


def test_sci_get_all_instances() -> None:
    text = "Laser:Laser-1\r\nSequence Manager:SeqMgr-1,SeqMgr-2\r\n"
    transport = FakeTransport([encode_string(text)])
    client = SysCon2Client(transport)

    result = client.sci.get_all_instances()

    assert result == {
        "Laser": ["Laser-1"],
        "Sequence Manager": ["SeqMgr-1", "SeqMgr-2"],
    }


def test_sci_get_profile_type() -> None:
    transport = FakeTransport([encode_string("Holo4D\r\n")])
    client = SysCon2Client(transport)

    assert client.sci.get_profile_type() == "Holo4D"


def test_laser_set_emission_ok() -> None:
    transport = FakeTransport([encode_string("OK")])
    client = SysCon2Client(transport)

    client.laser.set_emission("SN123", "Chameleon", True)

    assert transport.sent == [
        build_command_frame(
            "Laser",
            "SetEmission",
            encode_string("SN123"),
            encode_string("Chameleon"),
            encode_bool(True),
        )
    ]


def test_laser_set_emission_error_does_not_close_transport() -> None:
    transport = FakeTransport([encode_string("Error: device not found")])
    client = SysCon2Client(transport)

    with pytest.raises(SysCon2Error, match="device not found"):
        client.laser.set_emission("SN123", "Chameleon", True)

    # A well-formed "Error: ..." response is not a transport failure (manual §2.1 is about
    # pipe error *states*, not every rejected command) -- the connection stays usable.
    assert transport.closed is False


def test_laser_set_intensity_and_shutter() -> None:
    transport = FakeTransport([encode_string("OK"), encode_string("OK")])
    client = SysCon2Client(transport)

    client.laser.set_intensity("SN123", "Chameleon", 42.5)
    client.laser.set_shutter("SN123", False)

    assert transport.sent[0] == build_command_frame(
        "Laser",
        "SetIntensity",
        encode_string("SN123"),
        encode_string("Chameleon"),
        encode_float(42.5),
    )
    assert transport.sent[1] == build_command_frame(
        "Laser", "SetShutter", encode_string("SN123"), encode_bool(False)
    )


def test_laser_get_lightsources() -> None:
    text = "SN1:Chameleon\r\nSN2:Carbide,Topas4\r\n"
    transport = FakeTransport([encode_string(text)])
    client = SysCon2Client(transport)

    assert client.laser.get_lightsources() == {
        "SN1": ["Chameleon"],
        "SN2": ["Carbide", "Topas4"],
    }


def test_laser_get_lightsource_status() -> None:
    text = (
        "Connected:True\r\n"
        "Emission:False\r\n"
        "Shutter-open:True\r\n"
        "Intensity:12.5\r\n"
        "Mode:RMI-Analog\r\n"
    )
    transport = FakeTransport([encode_string(text)])
    client = SysCon2Client(transport)

    status = client.laser.get_lightsource_status("SN1", "Chameleon")

    assert status == LightsourceStatus(
        connected=True,
        emission=False,
        shutter_open=True,
        intensity=12.5,
        mode="RMI-Analog",
    )


def test_sequence_manager_round_trip() -> None:
    transport = FakeTransport(
        [
            encode_string("a.seq\r\nb.seq\r\n"),
            encode_string("OK"),
            encode_string("OK"),
            encode_string("OK"),
        ]
    )
    client = SysCon2Client(transport)

    assert client.sequence_manager.get_sequences() == ["a.seq", "b.seq"]
    client.sequence_manager.select_sequence("a.seq")
    client.sequence_manager.import_sequence("C:\\exports\\a.seq", "neuroseq")
    client.sequence_manager.delete_sequence("b.seq")

    assert transport.sent[1] == build_command_frame(
        "Sequence Manager", "SelectSequence", encode_string("a.seq")
    )
    assert transport.sent[2] == build_command_frame(
        "Sequence Manager",
        "ImportSequence",
        encode_string("C:\\exports\\a.seq"),
        encode_string("neuroseq"),
    )
    assert transport.sent[3] == build_command_frame(
        "Sequence Manager", "DeleteSequence", encode_string("b.seq")
    )


def test_holo4d_state_upload_run_stop() -> None:
    transport = FakeTransport(
        [
            encode_string("State:Idle\r\n"),
            encode_string("OK"),
            encode_string("OK"),
            encode_string("OK"),
        ]
    )
    client = SysCon2Client(transport)

    assert client.holo4d.get_state() == "Idle"
    client.holo4d.upload_sequence(5)
    client.holo4d.run_sequence("auto", "rising")
    client.holo4d.stop_sequence()

    assert transport.sent[1] == build_command_frame(
        "Holo4D", "UploadSequence", encode_dword(5)
    )
    assert transport.sent[2] == build_command_frame(
        "Holo4D", "RunSequence", encode_string("auto"), encode_string("rising")
    )
    assert transport.sent[3] == build_command_frame("Holo4D", "StopSequence")


def test_uga42_upload_and_run() -> None:
    transport = FakeTransport([encode_string("OK"), encode_string("OK")])
    client = SysCon2Client(transport)

    client.uga42.upload_sequence()
    client.uga42.run_sequence(0, "trigger1", "falling")

    assert transport.sent[0] == build_command_frame("UGA-42", "UploadSequence")
    assert transport.sent[1] == build_command_frame(
        "UGA-42",
        "RunSequence",
        encode_dword(0),
        encode_string("trigger1"),
        encode_string("falling"),
    )


def test_transport_failure_closes_client_and_propagates() -> None:
    transport = BrokenTransport()
    client = SysCon2Client(transport)

    with pytest.raises(SysCon2ConnectionError):
        client.holo4d.get_state()

    assert transport.closed is True


@pytest.mark.skipif(
    sys.platform == "win32", reason="asserts the non-Windows failure path"
)
def test_named_pipe_transport_unavailable_off_windows() -> None:
    with pytest.raises(SysCon2UnavailableError):
        NamedPipeTransport()


@pytest.fixture()
def pipe_api(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import neuroseq.syscon2_client as module

    api = SimpleNamespace(
        GENERIC_READ=1,
        GENERIC_WRITE=2,
        OPEN_EXISTING=3,
        FILE_FLAG_OVERLAPPED=4,
        CreateFile=Mock(return_value=123),
        CloseHandle=Mock(),
        WriteFile=Mock(),
        ReadFile=Mock(),
        WaitForSingleObject=Mock(),
    )
    monkeypatch.setitem(sys.modules, "_winapi", api)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    return api


def test_pipe_reads_fragments_and_writes_partial_data(pipe_api):
    from unittest.mock import Mock

    pipe = NamedPipeTransport()
    writes = [Mock(), Mock()]
    writes[0].GetOverlappedResult.return_value = (2, 0)
    writes[1].GetOverlappedResult.return_value = (3, 0)
    pipe_api.WriteFile.side_effect = [(op, 0) for op in writes]
    reads = [Mock(), Mock()]
    for op, data in zip(reads, [b"ab", b"cde"], strict=True):
        op.GetOverlappedResult.return_value = (len(data), 0)
        op.getbuffer.return_value = data
    pipe_api.ReadFile.side_effect = [(op, 0) for op in reads]
    pipe.send(b"12345")
    assert pipe_api.WriteFile.call_args_list[1].args == (123, b"345")
    assert pipe.recv(5) == b"abcde"
    pipe.close()
    pipe.close()
    pipe_api.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("direction", ["read", "write"])
def test_pipe_timeout_cancels_and_reaps_io_before_close(
    pipe_api, monkeypatch, direction
):
    from unittest.mock import Mock

    import neuroseq.syscon2_client as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    operation = Mock()
    operation.GetOverlappedResult.side_effect = lambda wait: (
        (0, 995) if wait else (0, 996)
    )
    pipe_api.ReadFile.return_value = (operation, 997)
    pipe_api.WriteFile.return_value = (operation, 997)
    pipe_api.WaitForSingleObject.side_effect = lambda *_: now.__setitem__(0, 2.0)
    pipe = NamedPipeTransport(timeout=1.0)
    with pytest.raises(SysCon2ConnectionError, match="timed out"):
        if direction == "read":
            pipe.recv(4)
        else:
            pipe.send(b"request")
    operation.cancel.assert_called_once()
    assert operation.GetOverlappedResult.call_args.args == (True,)
    pipe_api.CloseHandle.assert_not_called()
    pipe.close()


def test_pipe_cancellation_during_receive(pipe_api):
    import threading
    from unittest.mock import Mock

    cancel = threading.Event()
    operation = Mock()
    operation.GetOverlappedResult.return_value = (0, 996)
    pipe_api.ReadFile.return_value = (operation, 997)
    pipe_api.WaitForSingleObject.side_effect = lambda *_: cancel.set()
    pipe = NamedPipeTransport()
    pipe.set_cancel_event(cancel)
    with pytest.raises(SysCon2ConnectionError, match="cancelled"):
        pipe.recv(4)
    operation.cancel.assert_called_once()
    assert operation.GetOverlappedResult.call_args.args == (True,)
    pipe.close()


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows named pipes")
@pytest.mark.parametrize("direction", ["read", "write"])
def test_real_named_pipe_timeout(direction):
    import importlib
    import uuid

    api = importlib.import_module("_winapi")
    name = f"neuroseq_test_{uuid.uuid4().hex}"
    server = api.CreateNamedPipe(
        rf"\\.\pipe\{name}",
        api.PIPE_ACCESS_DUPLEX | api.FILE_FLAG_OVERLAPPED,
        0,
        1,
        1024,
        1024,
        0,
        0,
    )
    connection = api.ConnectNamedPipe(server, overlapped=True)
    pipe = None
    try:
        pipe = NamedPipeTransport(name, timeout=0.1)
        assert connection.GetOverlappedResult(True)[1] == 0
        with pytest.raises(SysCon2ConnectionError, match="timed out"):
            if direction == "read":
                pipe.recv(4)
            else:
                pipe.send(b"x" * 1_000_000)
    finally:
        if pipe is not None:
            pipe.close()
        connection.cancel()
        connection.GetOverlappedResult(True)
        api.CloseHandle(server)
