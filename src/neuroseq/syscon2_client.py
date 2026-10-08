"""Typed client for the SysCon2 SCI-Protocol (RAPP "SysCon2 Control Interface Manual" v1.2).

:class:`SysCon2Client` talks to an injected :class:`Transport` (``send``/``recv``/``close``) so
it can be driven by a fake in-memory stream in tests. The production transport,
:class:`NamedPipeTransport`, opens the local Windows named pipe directly -- SysCon2 must be
running on the same machine as neuroseq for now (see the project plan for the deferred
network-bridge transport).
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import Protocol

from neuroseq.syscon2_protocol import Reader
from neuroseq.syscon2_protocol import build_command_frame
from neuroseq.syscon2_protocol import encode_bool
from neuroseq.syscon2_protocol import encode_dword
from neuroseq.syscon2_protocol import encode_float
from neuroseq.syscon2_protocol import encode_string
from neuroseq.syscon2_protocol import parse_standard_response
from neuroseq.syscon2_protocol import parse_string_response
from neuroseq.syscon2_protocol import read_string

PIPE_NAME = "SysCon2ControlInterface"


class SysCon2ConnectionError(Exception):
    """The named pipe could not be opened, or broke while a command was in flight."""


class SysCon2UnavailableError(SysCon2ConnectionError):
    """SysCon2 control was attempted on a non-Windows platform."""


class Transport(Protocol):
    def send(self, data: bytes) -> None: ...

    def recv(self, n: int) -> bytes: ...

    def close(self) -> None: ...


class NamedPipeTransport:
    """Local Windows pipe with cancellable overlapped I/O and a command deadline.

    Uses CPython's Windows API wrapper, as multiprocessing.connection does. Pending
    operations are cancelled and reaped on the worker before releasing their buffers.
    The 30-second default allows slow device commands (SCI manual section 2.1);
    disconnect and shutdown can still cancel immediately.
    """

    _api: Any
    _cancel_event: threading.Event
    _timeout: float
    _deadline: float
    _handle: int | None

    def __init__(self, pipe_name: str = PIPE_NAME, timeout: float = 30.0) -> None:
        if sys.platform != "win32":
            raise SysCon2UnavailableError("SysCon2 control requires Windows")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._api = importlib.import_module("_winapi")
        self._cancel_event = threading.Event()
        self._timeout = timeout
        self._deadline = time.monotonic() + timeout
        self._handle: int | None = None
        try:
            self._handle = self._api.CreateFile(
                rf"\\.\pipe\{pipe_name}",
                self._api.GENERIC_READ | self._api.GENERIC_WRITE,
                0,
                0,
                self._api.OPEN_EXISTING,
                self._api.FILE_FLAG_OVERLAPPED,
                0,
            )
        except OSError as exc:
            raise SysCon2ConnectionError(
                f"Could not open the SysCon2 named pipe {pipe_name!r} -- is SysCon2 running?"
            ) from exc

    def set_cancel_event(self, event: threading.Event) -> None:
        self._cancel_event = event

    def _check_deadline(self) -> None:
        if self._cancel_event.is_set():
            raise SysCon2ConnectionError("SysCon2 operation cancelled")
        if time.monotonic() >= self._deadline:
            raise SysCon2ConnectionError(
                "SysCon2 command timed out; reconnect explicitly"
            )

    def _complete(self, operation: Any) -> int:
        try:
            while True:
                self._check_deadline()
                size, error = operation.GetOverlappedResult(False)
                if error == 0:
                    return int(size)
                # ERROR_IO_INCOMPLETE is not exported by CPython's _winapi.
                if error != 996:
                    raise SysCon2ConnectionError(f"SysCon2 pipe I/O failed ({error})")
                self._api.WaitForSingleObject(operation.event, 50)
        except BaseException:
            # Cancellation alone does not release the kernel's buffer references.
            operation.cancel()
            with contextlib.suppress(OSError):
                operation.GetOverlappedResult(True)
            raise

    def send(self, data: bytes) -> None:
        self._deadline = time.monotonic() + self._timeout
        try:
            offset = 0
            while offset < len(data):
                self._check_deadline()
                operation, _ = self._api.WriteFile(
                    self._handle, data[offset:], overlapped=True
                )
                written = self._complete(operation)
                if not written:
                    raise SysCon2ConnectionError("SysCon2 pipe closed unexpectedly")
                offset += written
        except OSError as exc:
            raise SysCon2ConnectionError(f"Failed writing to SysCon2: {exc}") from exc

    def recv(self, n: int) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        try:
            while remaining > 0:
                self._check_deadline()
                operation, _ = self._api.ReadFile(
                    self._handle, min(remaining, 65536), overlapped=True
                )
                size = self._complete(operation)
                if not size:
                    raise SysCon2ConnectionError("SysCon2 pipe closed unexpectedly")
                chunks.append(operation.getbuffer())
                remaining -= size
        except OSError as exc:
            raise SysCon2ConnectionError(f"Failed reading from SysCon2: {exc}") from exc
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._api.CloseHandle(self._handle)
            self._handle = None


def _split_lines(text: str) -> list[str]:
    return [line for line in text.split("\r\n") if line]


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.strip("\r\n").split(",") if item.strip()]


def _parse_key_value_lines(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in _split_lines(text):
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _parse_key_to_csv_map(text: str) -> dict[str, list[str]]:
    """Parse the "``key:item-1,item-2,...\\r\\n`` per line" shape shared by GetAllInstances and
    GetLightsources responses.
    """
    result: dict[str, list[str]] = {}
    for line in _split_lines(text):
        key, _, rest = line.partition(":")
        result[key.strip()] = _split_csv(rest)
    return result


def _parse_bool_field(value: str) -> bool:
    return value.strip().lower() == "true"


@dataclass
class LightsourceStatus:
    connected: bool
    emission: bool
    shutter_open: bool
    intensity: float
    mode: str

    @classmethod
    def from_response_text(cls, text: str) -> LightsourceStatus:
        fields = _parse_key_value_lines(text)
        return cls(
            connected=_parse_bool_field(fields["Connected"]),
            emission=_parse_bool_field(fields["Emission"]),
            shutter_open=_parse_bool_field(fields["Shutter-open"]),
            intensity=float(fields["Intensity"]),
            mode=fields["Mode"],
        )


class SysCon2Client:
    """Sends SCI-Protocol commands over an injected :class:`Transport` and decodes responses.

    Any transport/framing failure closes the connection immediately (manual §2.1: "disconnect
    the pipe on every kind of error ... to flush the server from leftover data"). A well-formed
    application-level ``Error: ...`` response does not -- that's parsed after the transport call
    already succeeded, so the pipe itself is still fine.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.sci = SciCommands(self)
        self.laser = LaserCommands(self)
        self.sequence_manager = SequenceManagerCommands(self)
        self.uga42 = UGA42Commands(self)
        self.holo4d = Holo4DCommands(self)

    def close(self) -> None:
        self._transport.close()

    def _request(self, instance: str, command: str, *params: bytes) -> str:
        frame = build_command_frame(instance, command, *params)
        try:
            self._transport.send(frame)
            return read_string(self._reader())
        except Exception:
            self.close()
            raise

    def _reader(self) -> Reader:
        return self._transport.recv

    def call_standard(self, instance: str, command: str, *params: bytes) -> None:
        parse_standard_response(self._request(instance, command, *params))

    def call_string(self, instance: str, command: str, *params: bytes) -> str:
        return parse_string_response(self._request(instance, command, *params))


class SciCommands:
    """General instance-discovery commands (manual §3.1), always addressed as ``sci``."""

    _INSTANCE = "sci"

    def __init__(self, client: SysCon2Client) -> None:
        self._client = client

    def get_instances(self, plugin_id: str) -> list[str]:
        text = self._client.call_string(
            self._INSTANCE, "GetInstances", encode_string(plugin_id)
        )
        return _split_csv(text)

    def get_all_instances(self) -> dict[str, list[str]]:
        text = self._client.call_string(self._INSTANCE, "GetAllInstances")
        return _parse_key_to_csv_map(text)

    def get_profile_type(self) -> str:
        text = self._client.call_string(self._INSTANCE, "GetProfileType")
        return text.strip("\r\n")


class LaserCommands:
    """Laser plugin commands (manual §3.2)."""

    def __init__(self, client: SysCon2Client, instance: str = "Laser") -> None:
        self._client = client
        self._instance = instance

    def set_emission(self, device_sn: str, description: str, on: bool) -> None:
        self._client.call_standard(
            self._instance,
            "SetEmission",
            encode_string(device_sn),
            encode_string(description),
            encode_bool(on),
        )

    def set_shutter(self, device_sn: str, open_: bool) -> None:
        self._client.call_standard(
            self._instance, "SetShutter", encode_string(device_sn), encode_bool(open_)
        )

    def set_intensity(self, device_sn: str, description: str, intensity: float) -> None:
        self._client.call_standard(
            self._instance,
            "SetIntensity",
            encode_string(device_sn),
            encode_string(description),
            encode_float(intensity),
        )

    def get_lightsources(self) -> dict[str, list[str]]:
        text = self._client.call_string(self._instance, "GetLightsources")
        return _parse_key_to_csv_map(text)

    def get_lightsource_status(
        self, device_sn: str, description: str
    ) -> LightsourceStatus:
        text = self._client.call_string(
            self._instance,
            "GetLightsourceStatus",
            encode_string(device_sn),
            encode_string(description),
        )
        return LightsourceStatus.from_response_text(text)


class SequenceManagerCommands:
    """Sequence manager plugin commands (manual §3.3).

    These operate on ``.seq`` files already on disk (e.g. from ``exporter_syscon2``) rather than
    raw sequence data -- ``ImportSequence`` takes a filepath, not a byte payload.
    The default instance name is specified in the Holo4D profile table (manual section 4).
    """

    def __init__(
        self, client: SysCon2Client, instance: str = "Sequence Manager"
    ) -> None:
        self._client = client
        self._instance = instance

    def get_sequences(self) -> list[str]:
        text = self._client.call_string(self._instance, "GetSequences")
        return _split_lines(text)

    def select_sequence(self, filepath: str) -> None:
        self._client.call_standard(
            self._instance, "SelectSequence", encode_string(filepath)
        )

    def import_sequence(self, filepath: str, target_directory: str) -> None:
        self._client.call_standard(
            self._instance,
            "ImportSequence",
            encode_string(filepath),
            encode_string(target_directory),
        )

    def delete_sequence(self, filepath: str) -> None:
        self._client.call_standard(
            self._instance, "DeleteSequence", encode_string(filepath)
        )


def _parse_state(text: str) -> str:
    return _parse_key_value_lines(text).get("State", "").strip()


class UGA42Commands:
    """UGA-42 plugin commands (manual §3.4). Not used by the Holo4D-profile rig this lab runs,
    kept for full protocol coverage / other profiles.
    """

    def __init__(self, client: SysCon2Client, instance: str = "UGA-42") -> None:
        self._client = client
        self._instance = instance

    def get_state(self) -> str:
        return _parse_state(self._client.call_string(self._instance, "GetState"))

    def upload_sequence(self) -> None:
        self._client.call_standard(self._instance, "UploadSequence")

    def run_sequence(
        self, runs: int, start_condition: str, trigger_behavior: str
    ) -> None:
        # Manual doesn't pin down "Runs"'s wire width beyond "integer"; DWord matches the other
        # plain-integer command parameters documented elsewhere. Verify against the real rig.
        self._client.call_standard(
            self._instance,
            "RunSequence",
            encode_dword(runs),
            encode_string(start_condition),
            encode_string(trigger_behavior),
        )

    def stop_sequence(self) -> None:
        self._client.call_standard(self._instance, "StopSequence")


class Holo4DCommands:
    """Holo4D plugin commands (manual §3.5) -- this lab's rig profile."""

    def __init__(self, client: SysCon2Client, instance: str = "Holo4D") -> None:
        self._client = client
        self._instance = instance

    def get_state(self) -> str:
        return _parse_state(self._client.call_string(self._instance, "GetState"))

    def upload_sequence(self, runs: int) -> None:
        # Manual section 3.5 calls Runs an "integer" without specifying its width;
        # use the same DWord representation as UGA-42 pending rig verification.
        self._client.call_standard(self._instance, "UploadSequence", encode_dword(runs))

    def run_sequence(self, start_condition: str, trigger_behavior: str) -> None:
        self._client.call_standard(
            self._instance,
            "RunSequence",
            encode_string(start_condition),
            encode_string(trigger_behavior),
        )

    def stop_sequence(self) -> None:
        self._client.call_standard(self._instance, "StopSequence")
