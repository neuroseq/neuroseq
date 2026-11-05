from __future__ import annotations

import struct
import sys
import threading

import pytest
from pytestqt.qtbot import QtBot

import neuroseq.device_control.syscon2_connection_manager as connection_manager_module
from neuroseq.device_control.syscon2_client import SysCon2ConnectionError
from neuroseq.device_control.syscon2_connection_manager import SysCon2ConnectionManager
from neuroseq.device_control.syscon2_protocol import encode_string
from tests.wait_timeout import WAIT_TIMEOUT_MS

# A string response whose declared length (DWord header) is honored, but whose
# payload byte isn't 7-bit ASCII - read_string() raises SysCon2ProtocolError
# for this (manual framing is otherwise intact, unlike SysCon2ConnectionError's
# transport-level failures), simulating a malformed/corrupted response rather
# than a truncated or dropped pipe.
_MALFORMED_RESPONSE = struct.pack("<I", 1) + b"\xff"


class FakeTransport:
    """Queues one scripted response per ``send()``, mimicking the client's send-then-read cycle."""

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False
        self._buffer = b""

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        if self._responses:
            self._buffer += self._responses.pop(0)

    def recv(self, n: int) -> bytes:
        assert len(self._buffer) >= n, "not enough buffered response data"
        chunk, self._buffer = self._buffer[:n], self._buffer[n:]
        return chunk

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def manager(qtbot: QtBot):
    # A very long poll interval keeps the background GetState timer from racing with (and
    # consuming scripted responses meant for) the commands each test submits explicitly.
    mgr = SysCon2ConnectionManager(poll_interval_ms=3_600_000)
    yield mgr
    mgr.shutdown()


def test_connect_runs_off_main_thread_and_emits_connected(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_thread_ids: list[int | None] = []

    def fake_named_pipe_transport() -> FakeTransport:
        seen_thread_ids.append(threading.current_thread().ident)
        return FakeTransport([])

    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", fake_named_pipe_transport
    )

    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    assert seen_thread_ids
    assert seen_thread_ids[0] != threading.main_thread().ident


def test_connect_failure_emits_disconnected_with_message(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_named_pipe_transport() -> FakeTransport:
        raise SysCon2ConnectionError("pipe missing")

    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", fake_named_pipe_transport
    )

    with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.connect_to_syscon2()

    assert "pipe missing" in blocker.args[0]


@pytest.mark.skipif(
    sys.platform == "win32", reason="asserts the non-Windows failure path"
)
def test_connect_reports_unavailable_off_windows(
    manager: SysCon2ConnectionManager, qtbot: QtBot
) -> None:
    with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.connect_to_syscon2()

    assert "Windows" in blocker.args[0]


def test_submit_runs_command_off_main_thread_and_emits_finished(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_manager_module,
        "NamedPipeTransport",
        lambda: FakeTransport([encode_string("State:Idle\r\n")]),
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    seen_thread_ids: list[int | None] = []

    def task(client):
        seen_thread_ids.append(threading.current_thread().ident)
        return client.holo4d.get_state()

    with qtbot.waitSignal(manager.command_finished, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.submit("get_state", task)

    assert blocker.args == ["get_state", "Idle"]
    assert seen_thread_ids[0] != threading.main_thread().ident


def test_submit_without_connection_emits_failed(
    manager: SysCon2ConnectionManager, qtbot: QtBot
) -> None:
    with qtbot.waitSignal(manager.command_failed, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.submit("noop", lambda client: None)

    assert blocker.args[0] == "noop"
    assert "not connected" in blocker.args[1].lower()


def test_submit_application_error_keeps_connection_open(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_manager_module,
        "NamedPipeTransport",
        lambda: FakeTransport([encode_string("Error: busy")]),
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    disconnected_seen: list[str] = []
    manager.disconnected.connect(disconnected_seen.append)

    with qtbot.waitSignal(manager.command_failed, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.submit("get_state", lambda client: client.holo4d.get_state())

    assert "busy" in blocker.args[1]
    assert disconnected_seen == []


def test_submit_connection_error_disconnects(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", lambda: FakeTransport([])
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    failed_seen: list[tuple[str, str]] = []
    manager.command_failed.connect(lambda name, msg: failed_seen.append((name, msg)))

    def failing(client):
        raise SysCon2ConnectionError("pipe broke")

    with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        manager.submit("run", failing)

    assert "pipe broke" in blocker.args[0]
    assert failed_seen == [("run", "pipe broke")]


def test_submit_malformed_response_disconnects(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a malformed (non-ASCII) response raises SysCon2ProtocolError,
    not SysCon2ConnectionError - but SysCon2Client._request() already closed the
    transport before re-raising, so the manager must disconnect for this case
    too, not just for SysCon2ConnectionError. Otherwise _client stays set and
    every later command/poll silently fails against the already-closed pipe
    with no reconnect prompt."""
    monkeypatch.setattr(
        connection_manager_module,
        "NamedPipeTransport",
        lambda: FakeTransport([_MALFORMED_RESPONSE]),
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    failed_seen: list[tuple[str, str]] = []
    manager.command_failed.connect(lambda name, msg: failed_seen.append((name, msg)))

    with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS):
        manager.submit("get_state", lambda client: client.holo4d.get_state())

    assert len(failed_seen) == 1
    assert failed_seen[0][0] == "get_state"


def test_polling_malformed_response_disconnects(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_manager_module,
        "NamedPipeTransport",
        lambda: FakeTransport([_MALFORMED_RESPONSE] * 10),
    )
    mgr = SysCon2ConnectionManager(poll_interval_ms=20)
    try:
        with qtbot.waitSignal(mgr.connected, timeout=WAIT_TIMEOUT_MS):
            mgr.connect_to_syscon2()
        with qtbot.waitSignal(mgr.disconnected, timeout=WAIT_TIMEOUT_MS):
            pass
    finally:
        mgr.shutdown()


def test_disconnect_closes_transport(
    manager: SysCon2ConnectionManager, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTransport([])
    monkeypatch.setattr(connection_manager_module, "NamedPipeTransport", lambda: fake)
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()

    with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS):
        manager.disconnect_from_syscon2()

    assert fake.closed is True


def test_polling_emits_state_changed(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        connection_manager_module,
        "NamedPipeTransport",
        lambda: FakeTransport([encode_string("State:Busy\r\n")] * 10),
    )
    mgr = SysCon2ConnectionManager(poll_interval_ms=20)
    states: list[dict] = []
    # Connect the collector before triggering connect() so an eager first poll tick can't
    # race ahead of a signal listener attached only after the fact.
    mgr.state_changed.connect(states.append)
    try:
        with qtbot.waitSignal(mgr.connected, timeout=WAIT_TIMEOUT_MS):
            mgr.connect_to_syscon2()
        qtbot.waitUntil(lambda: len(states) > 0, timeout=WAIT_TIMEOUT_MS)
        assert states[0] == {"Holo4D": "Busy"}
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("action", ["disconnect", "shutdown"])
def test_pending_io_is_cancelled_and_closed_on_worker(
    manager, qtbot, monkeypatch, action
):
    entered = threading.Event()
    closed_on = []

    class CancellableTransport(FakeTransport):
        def set_cancel_event(self, event):
            self.cancel_event = event

        def recv(self, n):
            entered.set()
            if not self.cancel_event.wait(WAIT_TIMEOUT_MS / 1000):
                raise AssertionError("Cancellation did not reach the transport")
            raise SysCon2ConnectionError("cancelled")

        def close(self):
            closed_on.append(threading.get_ident())
            super().close()

    transport = CancellableTransport([])
    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", lambda: transport
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()
    manager.submit("state", lambda client: client.holo4d.get_state())
    qtbot.waitUntil(entered.is_set, timeout=WAIT_TIMEOUT_MS)
    if action == "shutdown":
        manager.shutdown()
        assert not manager._thread.isRunning()
    else:
        with qtbot.waitSignal(manager.disconnected, timeout=WAIT_TIMEOUT_MS):
            manager.disconnect_from_syscon2()
    assert transport.closed
    assert closed_on and all(t != threading.get_ident() for t in closed_on)


def test_shutdown_closes_idle_connection(manager, qtbot, monkeypatch):
    transport = FakeTransport([])
    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", lambda: transport
    )
    with qtbot.waitSignal(manager.connected, timeout=WAIT_TIMEOUT_MS):
        manager.connect_to_syscon2()
    manager.shutdown()
    assert transport.closed
    assert not manager._thread.isRunning()
