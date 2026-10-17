"""Application connections with separately routed command replies per workspace."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from qtpy import QtCore

from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.device_control.syscon2_client import SysCon2Client
from neuroseq.device_control.syscon2_connection_manager import SysCon2ConnectionManager


class SysCon2Session(QtCore.QObject):
    """Share the transport, delivering command replies only to their requester."""

    connected = QtCore.Signal()
    disconnected = QtCore.Signal(str)
    state_changed = QtCore.Signal(dict)
    state_poll_failed = QtCore.Signal(str)
    command_finished = QtCore.Signal(str, object)
    command_failed = QtCore.Signal(str, str)

    def __init__(self, devices: WorkspaceDevices, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._devices = devices
        self._prefix = f"{uuid4().hex}:"
        self._closed = False
        self._connections = [
            (devices.syscon.connected, self.connected.emit),
            (devices.syscon.disconnected, self.disconnected.emit),
            (devices.syscon.state_changed, self.state_changed.emit),
            (devices.syscon.state_poll_failed, self.state_poll_failed.emit),
            (devices.syscon.command_finished, self._finished),
            (devices.syscon.command_failed, self._failed),
        ]
        for signal, slot in self._connections:
            signal.connect(slot)
        # The panel connects its slots after construction.
        QtCore.QTimer.singleShot(0, self._replay_connection_state)

    def _replay_connection_state(self) -> None:
        if not self._closed and self._devices.syscon_connected:
            self.connected.emit()
            if self._devices.syscon_state is not None:
                self.state_changed.emit(self._devices.syscon_state)

    def connect_to_syscon2(self) -> None:
        self._devices.syscon.connect_to_syscon2()

    def disconnect_from_syscon2(self) -> None:
        self._devices.syscon.disconnect_from_syscon2()

    def submit(self, name: str, fn: Callable[[SysCon2Client], object]) -> None:
        if not self._closed:
            self._devices.syscon.submit(self._prefix + name, fn)

    def _finished(self, name: str, result: object) -> None:
        if not self._closed and name.startswith(self._prefix):
            self.command_finished.emit(name.removeprefix(self._prefix), result)

    def _failed(self, name: str, message: str) -> None:
        if not self._closed and name.startswith(self._prefix):
            self.command_failed.emit(name.removeprefix(self._prefix), message)

    def shutdown(self) -> None:
        """Detach this workspace without disconnecting the physical device."""
        if self._closed:
            return
        self._closed = True
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections.clear()


class WorkspaceDevices(QtCore.QObject):
    """Own physical connections for the lifetime of the application."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.scanimage = ScanImageConnection(self)
        self.syscon = SysCon2ConnectionManager()
        self.syscon_connected = False
        self.syscon_state: dict | None = None
        self.syscon.connected.connect(self._connected)
        self.syscon.disconnected.connect(self._disconnected)
        self.syscon.state_changed.connect(self._state_changed)

    def _connected(self) -> None:
        self.syscon_connected = True

    def _disconnected(self, _message: str) -> None:
        self.syscon_connected = False
        self.syscon_state = None

    def _state_changed(self, state: dict) -> None:
        self.syscon_state = state

    def syscon_session(self, parent: QtCore.QObject) -> SysCon2Session:
        return SysCon2Session(self, parent)

    def shutdown(self) -> None:
        self.scanimage.shutdown()
        self.syscon.shutdown()
