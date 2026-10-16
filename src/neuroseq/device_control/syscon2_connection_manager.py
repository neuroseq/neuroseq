"""Qt-facing persistent connection to SysCon2, running on its own background thread.

:class:`SysCon2ConnectionManager` owns a :class:`~neuroseq.device_control.syscon2_client.SysCon2Client` on a
worker ``QThread`` and polls ``Holo4D::GetState`` roughly once a second. Callers on the GUI
thread use the public methods below; each emits a request signal that Qt marshals onto the
worker thread automatically (a signal/slot connection between objects living on different
threads is queued), so nothing here blocks the GUI.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from qtpy import QtCore

from neuroseq.device_control.syscon2_client import NamedPipeTransport
from neuroseq.device_control.syscon2_client import SysCon2Client
from neuroseq.device_control.syscon2_client import SysCon2ConnectionError
from neuroseq.device_control.syscon2_protocol import SysCon2Error
from neuroseq.device_control.syscon2_protocol import SysCon2ProtocolError

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
    #: A state poll failed without dropping the connection; carries the reason.
    state_poll_failed = QtCore.Signal(str)

    _request_connect = QtCore.Signal()
    _request_disconnect = QtCore.Signal()
    _request_command = QtCore.Signal(str, object)

    def __init__(self, poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS) -> None:
        super().__init__()
        self._poll_interval_ms = poll_interval_ms
        self._cancel_io = threading.Event()
        self._client: SysCon2Client | None = None
        self._timer: QtCore.QTimer | None = None

        self._thread = QtCore.QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._on_thread_started)
        self._thread.finished.connect(self._on_thread_finished)
        self._request_connect.connect(self._handle_connect)
        self._request_disconnect.connect(self._handle_disconnect)
        self._request_command.connect(self._handle_command)
        self._thread.start()

    # ---- public, GUI-thread API ----

    def connect_to_syscon2(self) -> None:
        self._request_connect.emit()

    def disconnect_from_syscon2(self) -> None:
        self._cancel_io.set()
        self._request_disconnect.emit()

    def submit(self, name: str, fn: Callable[[SysCon2Client], object]) -> None:
        """Run ``fn(client)`` on the worker thread; reports back via ``command_finished``/
        ``command_failed``. ``name`` is an arbitrary label echoed back for the caller to match
        the result to the request that made it.
        """
        self._request_command.emit(name, fn)

    def shutdown(self) -> None:
        """Cancel pending I/O and stop the worker; close handles on the worker thread."""
        self._cancel_io.set()
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
    def _on_thread_finished(self) -> None:
        # ``finished`` is emitted from within the worker thread's own event loop just
        # before it stops, so this runs on the correct thread to stop the timer --
        # unlike relying on Python GC to destroy it later (typically from the GUI
        # thread), which trips Qt's "timers cannot be stopped from another thread".
        self._handle_disconnect()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    @QtCore.Slot()
    def _handle_connect(self) -> None:
        if self._client is not None:
            return
        self._cancel_io.clear()
        try:
            transport = NamedPipeTransport()
            set_cancel = getattr(transport, "set_cancel_event", None)
            if set_cancel is not None:
                set_cancel(self._cancel_io)
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
        except SysCon2Error as exc:
            # A well-formed "Error: ..." application response (manual §2.1) -
            # the pipe itself is still fine, so the connection stays up.
            self.command_failed.emit(name, str(exc))
            return
        except SysCon2ProtocolError as exc:
            # Malformed framing: SysCon2Client._request() already closed the
            # transport before re-raising, so the connection must follow suit
            # here too, or every later command/poll would silently keep
            # trying (and failing) against a dead pipe with no reconnect
            # prompt.
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
        except SysCon2Error as exc:
            # A transient application-level error on the polling call isn't
            # worth surfacing as a disconnect; skip this tick and try again on the
            # next one, but report it so a persistently failing poll is visible.
            logger.warning("Holo4D state poll failed: %s", exc)
            self.state_poll_failed.emit(str(exc))
            return
        except SysCon2ProtocolError as exc:
            # See _handle_command's matching except clause - the transport is
            # already closed, so the connection must be torn down too.
            self._client = None
            self.disconnected.emit(str(exc))
            return
        except Exception as exc:
            logger.warning("Holo4D state poll failed: %r", exc)
            self.state_poll_failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.state_changed.emit({"Holo4D": state})
