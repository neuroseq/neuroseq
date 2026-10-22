"""Application-wide transport to the BrainScaleS-2 container helper.

One helper process (``resources/bss2/neuroseq_bss2_helper.py`` inside the
EBRAINS container) is shared by all workspaces; :class:`Bss2Connection` starts
it with :class:`QtCore.QProcess`, speaks the JSON-lines protocol documented in
:mod:`neuroseq.device_control.bss2_protocol`, and reports through signals.
Per-workspace routing of run results lives in
:class:`neuroseq.workspace_devices.Bss2Session`.

The EBRAINS token is only placed in the helper's environment; anything the
helper writes to stderr or stray stdout is scrubbed before it is shown.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from qtpy import QtCore

from neuroseq.device_control.bss2_protocol import LaunchCommand
from neuroseq.device_control.bss2_protocol import LaunchConfig
from neuroseq.device_control.bss2_protocol import describe_launch
from neuroseq.device_control.bss2_protocol import helper_launch_command
from neuroseq.device_control.bss2_protocol import scrub_secrets
from neuroseq.device_control.bss2_protocol import summarize_message

STATE_DISCONNECTED = "disconnected"
STATE_STARTING = "starting"
STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"

#: The first spack environment activation alone takes ~90 s.
DEFAULT_START_TIMEOUT_MS = 300_000
DEFAULT_CONNECT_TIMEOUT_MS = 120_000
#: Calibration on first use plus the hardware queue can take many minutes.
DEFAULT_RUN_TIMEOUT_MS = 1_800_000
_MAX_LINE_BYTES = 256 * 1024 * 1024

Launcher = Callable[[LaunchConfig, str], LaunchCommand]


class Bss2Connection(QtCore.QObject):
    status_changed = QtCore.Signal(str)
    #: Emitted with the chip identifier once ``hxcomm`` answered.
    connected = QtCore.Signal(str)
    disconnected = QtCore.Signal(str)
    #: (request id, message) while a run is executing.
    run_progress = QtCore.Signal(str, str)
    run_finished = QtCore.Signal(str, object)
    run_failed = QtCore.Signal(str, str)
    #: Scrubbed helper diagnostics (stderr, stray stdout).
    log_message = QtCore.Signal(str)
    #: Scrubbed connection trace: launch command, protocol traffic, process
    #: lifecycle. For the debug log; never contains the token.
    debug_message = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        launcher: Launcher = helper_launch_command,
        start_timeout_ms: int = DEFAULT_START_TIMEOUT_MS,
        connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS,
        run_timeout_ms: int = DEFAULT_RUN_TIMEOUT_MS,
    ) -> None:
        super().__init__(parent)
        #: Builds program/argv/environment; tests substitute a fake helper.
        self.launcher = launcher
        self.start_timeout_ms = start_timeout_ms
        self.connect_timeout_ms = connect_timeout_ms
        self.run_timeout_ms = run_timeout_ms
        self.state = STATE_DISCONNECTED
        self.identifier: str | None = None
        self.status = "Disconnected"
        self.run_id: str | None = None
        self._process: QtCore.QProcess | None = None
        self._buffer = bytearray()
        self._stderr = ""
        self._secrets: list[str] = []
        self._serial = 0
        self._pending: dict[int, tuple[str, str | None]] = {}
        self._connect_arguments: dict[str, Any] = {}
        self._closing = False
        self._deadline = QtCore.QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._timed_out)
        self._waiting_for = ""

    # ---- public API ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.state == STATE_CONNECTED

    @property
    def busy(self) -> bool:
        return self.run_id is not None

    def connect_helper(
        self, config: LaunchConfig, token: str, expected_identifier: str = ""
    ) -> bool:
        """Start the helper in the container and open the quiggeldy connection."""
        if self.state != STATE_DISCONNECTED:
            return False
        try:
            command = self.launcher(config, token)
        except (OSError, ValueError) as exc:
            self._debug(f"Launch command could not be built: {exc}")
            self._set_status(f"Cannot start the helper: {exc}")
            self.disconnected.emit(str(exc))
            return False
        self._secrets = [token] if token else []
        for line in describe_launch(command):
            self._debug(line)
        self._buffer.clear()
        self._stderr = ""
        self._pending.clear()
        self._closing = False
        self._connect_arguments = {
            "host": config.host,
            "port": int(config.port),
            "user": config.user,
            "identifier": expected_identifier,
        }
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        for key, value in command.environment.items():
            environment.insert(key, value)
        process = QtCore.QProcess(self)
        process.setProcessEnvironment(environment)
        process.setProgram(command.program)
        process.setArguments(command.arguments)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        process.started.connect(
            lambda: self._debug(f"Helper process started (pid {process.processId()})")
        )
        self._process = process
        self.state = STATE_STARTING
        self._set_status(
            "Starting the BrainScaleS-2 container (activating its software "
            "environment can take ~90 s the first time)…"
        )
        self._arm("container start-up", self.start_timeout_ms)
        process.start()
        return True

    def run(self, request_id: str, arguments: dict[str, Any]) -> bool:
        """Submit a ``run``; results arrive as ``run_finished``/``run_failed``."""
        if not self.is_connected or self.busy:
            return False
        self.run_id = request_id
        self._send("run", arguments, request_id)
        self._arm("the hardware run", self.run_timeout_ms)
        self._set_status("Run submitted to BrainScaleS-2…")
        return True

    def stop(self) -> None:
        """Abort: a submitted hardware run cannot be recalled, so the helper is
        killed and the run reported as failed. Reconnect afterwards."""
        if self._process is None:
            return
        self._abort("Stopped by the user; the helper was terminated.")

    def disconnect_helper(self) -> None:
        """Ask the helper to exit; kill it if it does not within 5 s."""
        process = self._process
        if process is None:
            return
        self._closing = True
        if self.state in (STATE_CONNECTED, STATE_CONNECTING):
            self._send("shutdown", {}, None)
        else:
            process.terminate()
        self._set_status("Disconnecting…")
        QtCore.QTimer.singleShot(5000, lambda p=process: self._kill_if_running(p))

    def shutdown(self) -> None:
        """Synchronously stop the helper (application exit)."""
        process = self._process
        if process is None:
            return
        self._closing = True
        process.kill()
        process.waitForFinished(1000)
        # waitForFinished delivers ``finished`` in most cases; make sure.
        if self._process is process:
            self._process_finished()

    # ---- process I/O -----------------------------------------------------------

    def _send(
        self, command: str, arguments: dict[str, Any], run_id: str | None
    ) -> None:
        assert self._process is not None
        self._serial += 1
        self._pending[self._serial] = (command, run_id)
        message = {"id": self._serial, "command": command, "arguments": arguments}
        self._debug("→ " + summarize_message(message))
        self._process.write((json.dumps(message) + "\n").encode())

    def _read_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        self._buffer.extend(process.readAllStandardOutput().data())
        if len(self._buffer) > _MAX_LINE_BYTES:
            self._abort("The helper reply exceeds the size limit.")
            return
        while b"\n" in self._buffer:
            line, _, remaining = self._buffer.partition(b"\n")
            self._buffer = bytearray(remaining)
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
                if not isinstance(message, dict):
                    raise ValueError
            except ValueError:
                # Something in the container printed to stdout; keep it as a log.
                self.log_message.emit(scrub_secrets(text, self._secrets))
                continue
            self._handle(message)
            if self._process is not process:
                return  # the message handling ended the session

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        text = scrub_secrets(
            self._process.readAllStandardError().data().decode(errors="replace"),
            self._secrets,
        )
        self._stderr = (self._stderr + text)[-4000:]
        for line in text.splitlines():
            if line.strip():
                self.log_message.emit(line)

    def _handle(self, message: dict[str, Any]) -> None:
        self._debug("← " + summarize_message(message))
        if message.get("event") == "ready":
            self.state = STATE_CONNECTING
            host = self._connect_arguments.get("host") or "?"
            port = self._connect_arguments.get("port")
            self._set_status(
                f"Container environment ready; connecting to quiggeldy {host}:{port}…"
            )
            self._send("connect", self._connect_arguments, None)
            self._arm("the quiggeldy connection", self.connect_timeout_ms)
            return
        serial = message.get("id")
        if not isinstance(serial, int) or serial not in self._pending:
            return
        command, run_id = self._pending[serial]
        if message.get("event") == "progress":
            text = scrub_secrets(str(message.get("message", "")), self._secrets)
            self._set_status(text)
            if run_id is not None:
                self.run_progress.emit(run_id, text)
            return
        del self._pending[serial]
        error = message.get("error")
        if error is not None:
            error = scrub_secrets(str(error), self._secrets)
        if command == "connect":
            self._deadline.stop()
            if error is not None:
                self._abort(f"Connecting to BrainScaleS-2 failed: {error}")
                return
            result = message.get("result") or {}
            self.identifier = str(result.get("identifier", ""))
            self.state = STATE_CONNECTED
            note = (
                ""
                if result.get("matches_expected", True)
                else " (differs from the selected setup's identifier)"
            )
            self._set_status(f"Connected to {self.identifier}{note}")
            self.connected.emit(self.identifier)
        elif command == "run":
            self._deadline.stop()
            self.run_id = None
            assert run_id is not None
            if error is not None:
                self._set_status(f"Run failed: {error}")
                self.run_failed.emit(run_id, error)
            else:
                self._set_status(f"Run finished on {self.identifier}")
                self.run_finished.emit(run_id, message.get("result") or {})

    # ---- failure handling ------------------------------------------------------

    def _arm(self, waiting_for: str, timeout_ms: int) -> None:
        self._waiting_for = waiting_for
        self._deadline.start(timeout_ms)

    def _timed_out(self) -> None:
        seconds = self._deadline.interval() / 1000
        self._debug(f"Timed out after {seconds:g} s waiting for {self._waiting_for}")
        self._abort(f"No answer from {self._waiting_for} within {seconds:g} s.")

    def _process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if self._process is not None:
            self._debug(
                f"Process error {getattr(error, 'name', error)}: "
                f"{self._process.errorString()}"
            )
        if error == QtCore.QProcess.ProcessError.FailedToStart and self._process:
            self._abort(
                f"Could not start {self._process.program()}: "
                f"{self._process.errorString()}"
            )

    def _abort(self, reason: str) -> None:
        process = self._process
        if process is None:
            return
        self._closing = False
        self._end_session(reason)
        process.kill()
        process.waitForFinished(1000)

    def _kill_if_running(self, process: QtCore.QProcess) -> None:
        if process is self._process and (
            process.state() != QtCore.QProcess.ProcessState.NotRunning
        ):
            process.kill()

    def _process_finished(self, *_args: object) -> None:
        if self._process is None:
            return
        process = self._process
        code = process.exitCode()
        self._debug(
            f"Helper exited: code {code}, "
            f"{getattr(process.exitStatus(), 'name', process.exitStatus())}"
        )
        if self._closing:
            self._end_session("Disconnected.")
            return
        tail = self._stderr.strip().splitlines()[-5:]
        reason = f"The BrainScaleS-2 helper exited unexpectedly (exit code {code})"
        # The last stderr line (e.g. Singularity's FATAL message) is usually
        # the actual cause, so it goes into the status line itself.
        reason += f": {tail[-1].strip()}" if tail else "."
        if tail:
            reason += "\n" + "\n".join(tail)
        self._end_session(reason)

    def _end_session(self, reason: str) -> None:
        """Forget the process and report the outcome exactly once."""
        process, self._process = self._process, None
        self._deadline.stop()
        self._pending.clear()
        self._secrets = []
        run_id, self.run_id = self.run_id, None
        self.state = STATE_DISCONNECTED
        self.identifier = None
        if process is not None:
            process.deleteLater()
        if run_id is not None:
            self.run_failed.emit(run_id, reason)
        self._set_status(reason.splitlines()[0] if reason else "Disconnected")
        self.disconnected.emit(reason)

    def _debug(self, text: str) -> None:
        self.debug_message.emit(scrub_secrets(text, self._secrets))

    def _set_status(self, text: str) -> None:
        self.status = text
        self.status_changed.emit(text)
