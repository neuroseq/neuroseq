"""Asynchronous same-PC transport for the ScanImage adapter.

Only one request is in flight: preview frames cannot accumulate in Qt's event
queue. The JSON operation/reply boundary can also be implemented by a future
remote bridge without changing acquisition or display logic.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np
from qtpy import QtCore


class ScanImageConnection(QtCore.QObject):
    status_changed = QtCore.Signal(object)
    frame_received = QtCore.Signal(object)
    recording_ready = QtCore.Signal(object)
    error = QtCore.Signal(str)
    connected_changed = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._drain_stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._buffer = bytearray()
        self._serial = 0
        self._pending: tuple[int, str] | None = None
        self._abort_requested = False
        self.connected = False
        self.channel = 1
        self.run_id: str | None = None
        self._finalize_since: float | None = None
        self._file_signature: list | None = None
        self._last_status: dict = {}
        self._stderr = ""
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._poll)
        self.deadline = QtCore.QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(self._timed_out)

    def connect_session(self, python: str, session: str) -> None:
        if self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        self._buffer.clear()
        self._stderr = ""
        self._pending = None
        self._abort_requested = False
        self._last_status = {}
        self.process.setProgram(python)
        self.process.setArguments(
            ["-u", str(Path(__file__).with_name("scanimage_engine_worker.py"))]
        )
        self.process.start()
        # QProcess buffers writes while starting; no blocking wait in the GUI.
        self._send("connect", {"session": session})

    def _send(self, operation: str, arguments: dict | None = None) -> bool:
        if self._pending is not None:
            return False
        self._serial += 1
        self._pending = (self._serial, operation)
        request = {
            "id": self._serial,
            "operation": operation,
            "arguments": arguments or {},
        }
        self.process.write((json.dumps(request) + "\n").encode())
        self.deadline.start(30000)
        return True

    def command(self, operation: str, arguments: dict | None = None) -> bool:
        if not self.connected:
            self.error.emit("ScanImage is not connected.")
            return False
        if operation == "abort" and self._pending is not None:
            self._abort_requested = True
            return True
        if not self._send(operation, arguments):
            self.error.emit("ScanImage is responding to another request; try again.")
            return False
        if operation == "record":
            assert arguments is not None
            self.run_id = arguments["run_id"]
            self._finalize_since = None
            self._file_signature = None
        return True

    def _poll(self) -> None:
        if self.connected and self._pending is None:
            if self._abort_requested:
                self._abort_requested = False
                self._send("abort")
            else:
                self._send("status", {"channel": self.channel})

    def _read(self) -> None:
        self._buffer.extend(self.process.readAllStandardOutput().data())
        if len(self._buffer) > 64 * 1024 * 1024:
            self._fail("ScanImage reply exceeds the 64 MiB limit.")
            return
        while b"\n" in self._buffer:
            line, _, remaining = self._buffer.partition(b"\n")
            self._buffer = bytearray(remaining)
            try:
                self._reply(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                self._fail(f"Invalid ScanImage reply: {exc}")
                return

    def _reply(self, reply: dict) -> None:
        if self._pending is None or reply.get("id") != self._pending[0]:
            return
        _, operation = self._pending
        self._pending = None
        self.deadline.stop()
        if "error" in reply:
            if operation == "index" and self._finalize_since is not None:
                if time.monotonic() - self._finalize_since < 15:
                    return  # File may still be flushing. Retry through status polling.
                self.run_id = None
            self.error.emit(str(reply["error"]))
            if operation == "connect":
                self.shutdown()
            return
        result = reply["result"]
        if operation == "connect":
            self.connected = True
            self.connected_changed.emit(True)
            self.timer.start()
        if operation == "index":
            result["run_id"] = self.run_id
            result["status"] = self._last_status["run"]["state"]
            self.run_id = None
            self.recording_ready.emit(result)
            return
        if operation == "disconnect":
            self.shutdown()
            return
        self._last_status = result
        self.status_changed.emit(result)
        frame = result.pop("frame", None)
        if frame is not None:
            shape = tuple(int(n) for n in frame["shape"])
            if len(shape) != 2 or min(shape) < 1 or max(shape) > 16384:
                raise ValueError("Invalid frame dimensions")
            raw = base64.b64decode(frame["data"], validate=True)
            if frame["dtype"] != "float32" or len(raw) != shape[0] * shape[1] * 4:
                raise ValueError("Invalid frame payload")
            frame["pixels"] = np.frombuffer(raw, dtype="<f4").reshape(shape, order="F")
            self.frame_received.emit(frame)
        run = result.get("run", {})
        if (
            self.run_id
            and run.get("id") == self.run_id
            and run.get("state") in {"completed", "aborted", "failed"}
            and result.get("acq_state") == "idle"
        ):
            self._finalize(result)

    def _finalize(self, result: dict) -> None:
        if self._finalize_since is None:
            self._finalize_since = time.monotonic()
        files = result.get("files", [])
        if isinstance(files, str):
            files = [files]
        # File stats are cheap; TIFF indexing and validation run in the helper.
        try:
            signature = [
                (name, Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                for name in files
            ]
        except OSError:
            signature = []
        if signature and signature == self._file_signature:
            self._send("index", {"paths": files})
        elif time.monotonic() - self._finalize_since > 15:
            self.run_id = None
            self.error.emit(
                "Acquisition ended, but recording files could not be finalized. Keep the run directory for recovery."
            )
        self._file_signature = signature

    def _drain_stderr(self) -> None:
        self._stderr = (
            self._stderr
            + self.process.readAllStandardError().data().decode(errors="replace")
        )[-4000:]

    def _timed_out(self) -> None:
        self._fail(
            "ScanImage did not respond within 30 seconds. Acquisition state is unknown; check ScanImage before reconnecting."
        )

    def _process_error(self, _error) -> None:
        self._fail(f"ScanImage helper error: {self.process.errorString()}")

    def _fail(self, message: str) -> None:
        self.error.emit(message)
        self.shutdown()

    def _finished(self, *_args) -> None:
        unexpected = self.connected or self._pending is not None
        self.connected = False
        self._pending = None
        self.timer.stop()
        self.deadline.stop()
        self.connected_changed.emit(False)
        if unexpected:
            self.error.emit(
                "ScanImage helper exited. Acquisition state is unknown. " + self._stderr
            )

    def shutdown(self) -> None:
        """Stop transport only; never kill the microscope's MATLAB process."""
        self.connected = False
        self._pending = None
        self.timer.stop()
        self.deadline.stop()
        self.process.kill()
        self.process.waitForFinished(1000)
        self.connected_changed.emit(False)
