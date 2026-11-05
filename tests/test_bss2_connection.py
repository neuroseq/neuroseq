from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.device_control import bss2_protocol as bp
from neuroseq.device_control.bss2_connection import STATE_CONNECTED
from neuroseq.device_control.bss2_connection import STATE_DISCONNECTED
from neuroseq.device_control.bss2_connection import Bss2Connection
from neuroseq.workspace_devices import WorkspaceDevices
from tests.wait_timeout import WAIT_TIMEOUT_MS

FAKE_HELPER = Path(__file__).with_name("bss2_fake_helper.py")
TOKEN = "eyJfake.secret-token-value.sig"


def fake_launcher(mode: str = "ok"):
    """Launch the fake helper with the environment the real launcher builds."""

    def launch(config: bp.LaunchConfig, token: str) -> bp.LaunchCommand:
        real = bp.helper_launch_command(config, token)
        assert not token or all(token not in part for part in real.arguments)
        return bp.LaunchCommand(
            sys.executable,
            ["-u", str(FAKE_HELPER)],
            {**real.environment, "FAKE_BSS2_MODE": mode},
        )

    return launch


def config() -> bp.LaunchConfig:
    return bp.LaunchConfig(
        container_image="/fake/image.sif",
        container_app="",
        host="quiggeldy.test",
        port=11717,
        user="tester",
    )


class Recorder:
    def __init__(self, connection: Bss2Connection) -> None:
        self.texts: list[str] = []
        self.finished: list[tuple[str, object]] = []
        self.failed: list[tuple[str, str]] = []
        self.progress: list[tuple[str, str]] = []
        self.disconnects: list[str] = []
        connection.status_changed.connect(self.texts.append)
        connection.log_message.connect(self.texts.append)
        connection.disconnected.connect(self.disconnects.append)
        connection.run_finished.connect(lambda *a: self.finished.append(a))
        connection.run_failed.connect(lambda *a: self.failed.append(a))
        connection.run_progress.connect(lambda *a: self.progress.append(a))


@pytest.fixture()
def make_connection(qtbot: QtBot):
    created: list[Bss2Connection] = []

    def make(mode: str = "ok", **kwargs) -> Bss2Connection:
        connection = Bss2Connection(launcher=fake_launcher(mode), **kwargs)
        created.append(connection)
        return connection

    yield make
    for connection in created:
        connection.shutdown()


def _connect(qtbot: QtBot, connection: Bss2Connection) -> str:
    with qtbot.waitSignal(connection.connected, timeout=WAIT_TIMEOUT_MS) as blocker:
        assert connection.connect_helper(config(), TOKEN, "expected-chip")
    return blocker.args[0]


def _run_arguments(n: int = 2) -> dict:
    settings = bp.Bss2Settings()
    plan = bp.plan_constant_current(np.ones((4, n)), 0.25, settings)
    return bp.build_run_arguments(plan, settings)


def test_connect_passes_environment_and_scrubs_token(qtbot, make_connection) -> None:
    connection = make_connection()
    recorder = Recorder(connection)
    identifier = _connect(qtbot, connection)
    assert identifier == "fakechip_quiggeldy.test_with-token"
    assert connection.state == STATE_CONNECTED and connection.is_connected
    assert "differs" in connection.status  # expected identifier did not match
    assert not connection.connect_helper(config(), TOKEN)  # already running
    qtbot.waitUntil(
        lambda: any("***" in t for t in recorder.texts), timeout=WAIT_TIMEOUT_MS
    )
    joined = "\n".join(recorder.texts)
    assert "Starting the BrainScaleS-2 container" in joined
    assert "activating environment esd" in joined  # stray stdout kept as log
    assert "secret-token-value" not in joined


def test_run_round_trip(qtbot, make_connection) -> None:
    connection = make_connection()
    recorder = Recorder(connection)
    _connect(qtbot, connection)
    with qtbot.waitSignal(connection.run_finished, timeout=WAIT_TIMEOUT_MS) as blocker:
        assert connection.run("r1", _run_arguments())
        assert connection.busy
        assert not connection.run("r2", _run_arguments())
    run_id, result = blocker.args
    assert run_id == "r1" and not connection.busy
    assert recorder.progress and recorder.progress[0][0] == "r1"
    readback = bp.decode_run_result(result, 1000.0)
    assert readback.spikes.spike_indices.tolist() == [0, 1]
    assert readback.duration_s == pytest.approx(1.0)
    assert readback.trace is not None


def test_run_requires_connection(make_connection) -> None:
    connection = make_connection()
    assert not connection.run("r", _run_arguments())
    connection.stop()  # nothing to stop
    connection.disconnect_helper()  # nothing to disconnect
    connection.shutdown()
    assert connection.state == STATE_DISCONNECTED


def test_run_error_keeps_connection(qtbot, make_connection) -> None:
    connection = make_connection("run_error")
    _connect(qtbot, connection)
    with qtbot.waitSignal(connection.run_failed, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.run("r1", _run_arguments())
    assert blocker.args[0] == "r1" and "fake chip failure" in blocker.args[1]
    assert connection.is_connected


def test_helper_crash_fails_run_and_disconnects(qtbot, make_connection) -> None:
    connection = make_connection("crash_on_run")
    recorder = Recorder(connection)
    _connect(qtbot, connection)
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.run("r1", _run_arguments())
    assert "exit code 5" in blocker.args[0]
    assert "segmentation fault" in blocker.args[0]
    assert recorder.failed and recorder.failed[0][0] == "r1"
    assert connection.state == STATE_DISCONNECTED


def test_connect_error_disconnects(qtbot, make_connection) -> None:
    connection = make_connection("connect_error")
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.connect_helper(config(), TOKEN)
    assert "quiggeldy refused the token" in blocker.args[0]
    assert connection.state == STATE_DISCONNECTED


def test_helper_exit_before_ready(qtbot, make_connection) -> None:
    connection = make_connection("exit")
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.connect_helper(config(), "")
    assert "exit code 3" in blocker.args[0]


def test_helper_fatal_error_is_in_the_status_line(qtbot, make_connection) -> None:
    connection = make_connection("fatal")
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.connect_helper(config(), "")
    first_line = blocker.args[0].splitlines()[0]
    assert "exit code 255" in first_line
    assert "FATAL:   could not open image" in first_line
    assert connection.status == first_line


def test_start_timeout(qtbot, make_connection) -> None:
    connection = make_connection("silent", start_timeout_ms=300)
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.connect_helper(config(), TOKEN)
    assert "container start-up" in blocker.args[0]


def test_run_timeout(qtbot, make_connection) -> None:
    connection = make_connection("slow_run", run_timeout_ms=300)
    recorder = Recorder(connection)
    _connect(qtbot, connection)
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS):
        connection.run("r1", _run_arguments())
    assert recorder.failed and "hardware run" in recorder.failed[0][1]


def test_stop_aborts_running_run(qtbot, make_connection) -> None:
    connection = make_connection("slow_run")
    recorder = Recorder(connection)
    _connect(qtbot, connection)
    connection.run("r1", _run_arguments())
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS):
        connection.stop()
    assert recorder.failed == [
        ("r1", "Stopped by the user; the helper was terminated.")
    ]


def test_disconnect_and_terminate_while_starting(qtbot, make_connection) -> None:
    connection = make_connection()
    _connect(qtbot, connection)
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        connection.disconnect_helper()
    assert blocker.args[0] == "Disconnected."

    slow = make_connection("silent")
    slow.connect_helper(config(), TOKEN)
    with qtbot.waitSignal(slow.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        slow.disconnect_helper()
    assert blocker.args[0] == "Disconnected."


def test_launch_failures(qtbot, make_connection) -> None:
    connection = Bss2Connection()  # real launcher: empty image is rejected
    with qtbot.waitSignal(connection.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        assert not connection.connect_helper(
            bp.LaunchConfig("", "", "h", 1, "u"), TOKEN
        )
    assert "image" in blocker.args[0]

    missing = Bss2Connection(
        launcher=lambda _c, _t: bp.LaunchCommand("/nonexistent/singularity", [], {})
    )
    with qtbot.waitSignal(missing.disconnected, timeout=WAIT_TIMEOUT_MS) as blocker:
        missing.connect_helper(config(), TOKEN)
    assert "Could not start" in blocker.args[0]


def test_unknown_and_malformed_messages_are_ignored(make_connection) -> None:
    connection = make_connection()
    connection._handle({"id": 99, "result": {}})
    connection._handle({"id": "x"})
    connection._read_stdout()
    connection._read_stderr()
    connection._process_finished()
    assert connection.state == STATE_DISCONNECTED


def test_sessions_route_results_to_their_workspace(qtbot: QtBot) -> None:
    devices = WorkspaceDevices()
    devices.bss2.launcher = fake_launcher()
    owner = QtCore.QObject()
    first = devices.bss2_session(owner)
    second = devices.bss2_session(owner)
    finished: dict[str, list] = {"first": [], "second": []}
    progress: list[str] = []
    first.run_finished.connect(lambda *a: finished["first"].append(a))
    second.run_finished.connect(lambda *a: finished["second"].append(a))
    first.run_progress.connect(progress.append)
    try:
        with qtbot.waitSignal(second.connected, timeout=WAIT_TIMEOUT_MS):
            assert first.connect_helper(config(), TOKEN)
        assert first.state == STATE_CONNECTED
        assert first.identifier == second.identifier
        with qtbot.waitSignal(first.run_finished, timeout=WAIT_TIMEOUT_MS):
            assert first.run("job", _run_arguments())
            assert first.owns_run and not second.owns_run
            assert second.busy
        assert finished["first"][0][0] == "job"
        assert finished["second"] == []
        assert progress

        # A session created later learns the current state.
        late = devices.bss2_session(owner)
        with qtbot.waitSignal(late.connected, timeout=WAIT_TIMEOUT_MS) as blocker:
            pass
        assert blocker.args[0] == first.identifier

        failures: list = []
        second.run_failed.connect(lambda *a: failures.append(a))
        second.shutdown()
        second.shutdown()
        assert not second.run("x", _run_arguments())
        assert not second.connect_helper(config(), TOKEN)
        second.stop()
        second.disconnect_helper()
        assert devices.bss2.is_connected
        with qtbot.waitSignal(first.run_failed, timeout=WAIT_TIMEOUT_MS):
            first.run("other", _run_arguments())
            first.stop()
        assert failures == []
    finally:
        devices.shutdown()
    assert devices.bss2.state == STATE_DISCONNECTED
