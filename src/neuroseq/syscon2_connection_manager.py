"""Qt-facing persistent connection to SysCon2, running on its own background thread.

:class:`SysCon2ConnectionManager` owns a :class:`~neuroseq.syscon2_client.SysCon2Client` on a
worker ``QThread`` and polls ``Holo4D::GetState`` roughly once a second. Callers on the GUI
thread use the public methods below; each emits a request signal that Qt marshals onto the
worker thread automatically (a signal/slot connection between objects living on different
threads is queued), so nothing here blocks the GUI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy import QtCore

from neuroseq.syscon2_client import NamedPipeTransport
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_client import SysCon2ConnectionError

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_MS = 1000


class SysCon2ConnectionManager(QtCore.QObject):
    """Signals (all delivered on the GUI thread):

    - ``connected()`` -- the pipe is open.
    - ``disconnected(str)`` -- not connected; the string is empty for a plain manual
      disconnect, or an error message if the connection dropped/failed to open.
    - ``state_changed(dict)`` -- latest polled plugin state, e.g. ``{"Holo4D": "Idle"}``.
    - ``command_finished(str, object)`` -- a :meth:`submit`'d command succeeded.
    - ``command_failed(str, str)`` -- a :meth:`submit`'d command raised; the connection stays
      open unless a ``disconnected`` signal follows too (a transport-level failure, per manual
      §2.1, closes the pipe and requires an explicit reconnect -- a well-formed application
      ``Error: ...`` response does not).
    """

    connected = QtCore.Signal()
    disconnected = QtCore.Signal(str)
    state_changed = QtCore.Signal(dict)
    command_finished = QtCore.Signal(str, object)
    command_failed = QtCore.Signal(str, str)

    _request_connect = QtCore.Signal()
    _request_disconnect = QtCore.Signal()
    _request_command = QtCore.Signal(str, object)

    def __init__(self, poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS) -> None:
        super().__init__()
        self._poll_interval_ms = poll_interval_ms
        self._client: SysCon2Client | None = None
        self._timer: QtCore.QTimer | None = None

        self._thread = QtCore.QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._on_thread_started)
        self._request_connect.connect(self._handle_connect)
        self._request_disconnect.connect(self._handle_disconnect)
        self._request_command.connect(self._handle_command)
        self._thread.start()

    # ---- public, GUI-thread API ----

    def connect_to_syscon2(self) -> None:
        self._request_connect.emit()

    def disconnect_from_syscon2(self) -> None:
        self._request_disconnect.emit()

    def submit(self, name: str, fn: Callable[[SysCon2Client], object]) -> None:
        """Run ``fn(client)`` on the worker thread; reports back via ``command_finished``/
        ``command_failed``. ``name`` is an arbitrary label echoed back for the caller to match
        the result to the request that made it.
        """
        self._request_command.emit(name, fn)

    def shutdown(self) -> None:
        """Disconnect and stop the worker thread. Call from the GUI's ``closeEvent``."""
        self._request_disconnect.emit()
        self._thread.quit()
        if not self._thread.wait(2000):
            logger.warning("SysCon2 connection manager thread did not stop in time.")

    # ---- worker-thread slots ----

    @QtCore.Slot()
    def _on_thread_started(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll_state)
        self._timer.start(self._poll_interval_ms)

    @QtCore.Slot()
    def _handle_connect(self) -> None:
        if self._client is not None:
            return
        try:
            transport = NamedPipeTransport()
        except SysCon2ConnectionError as exc:
            self.disconnected.emit(str(exc))
            return
        self._client = SysCon2Client(transport)
        self.connected.emit()

    @QtCore.Slot()
    def _handle_disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self.disconnected.emit("")

    @QtCore.Slot(str, object)
    def _handle_command(self, name: str, fn: Callable[[SysCon2Client], object]) -> None:
        if self._client is None:
            self.command_failed.emit(name, "Not connected to SysCon2")
            return
        try:
            result = fn(self._client)
        except SysCon2ConnectionError as exc:
            self._client = None
            self.command_failed.emit(name, str(exc))
            self.disconnected.emit(str(exc))
            return
        except Exception as exc:
            self.command_failed.emit(name, f"{type(exc).__name__}: {exc}")
            return
        self.command_finished.emit(name, result)

    @QtCore.Slot()
    def _poll_state(self) -> None:
        if self._client is None:
            return
        try:
            state = self._client.holo4d.get_state()
        except SysCon2ConnectionError as exc:
            self._client = None
            self.disconnected.emit(str(exc))
            return
        except Exception:
            # A transient error on the polling call isn't worth surfacing as a disconnect;
            # just skip this tick and try again on the next one.
            return
        self.state_changed.emit({"Holo4D": state})
