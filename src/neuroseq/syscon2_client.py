"""Typed client for the SysCon2 SCI-Protocol (RAPP "SysCon2 Control Interface Manual" v1.2).

:class:`SysCon2Client` talks to an injected :class:`Transport` (``send``/``recv``/``close``) so
it can be driven by a fake in-memory stream in tests. The production transport,
:class:`NamedPipeTransport`, opens the local Windows named pipe directly -- SysCon2 must be
running on the same machine as neuroseq for now (see the project plan for the deferred
network-bridge transport).
"""

from __future__ import annotations

import contextlib
import sys
import typing
from dataclasses import dataclass
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
    """Opens ``\\\\.\\pipe\\<pipe_name>`` directly -- Windows treats a named pipe as a regular
    file path, so plain ``open()``/``read()``/``write()`` works without ``pywin32``/``ctypes``.
    """

    def __init__(self, pipe_name: str = PIPE_NAME) -> None:
        if sys.platform != "win32":
            raise SysCon2UnavailableError("SysCon2 control requires Windows")
        self._file: typing.BinaryIO
        try:
            self._file = open(f"\\\\.\\pipe\\{pipe_name}", "r+b", buffering=0)  # noqa: SIM115
        except OSError as exc:
            raise SysCon2ConnectionError(
                f"Could not open the SysCon2 named pipe {pipe_name!r} -- is SysCon2 running?"
            ) from exc

    def send(self, data: bytes) -> None:
        try:
            self._file.write(data)
        except OSError as exc:
            raise SysCon2ConnectionError(
                f"Failed writing to the SysCon2 pipe: {exc}"
            ) from exc

    def recv(self, n: int) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            try:
                chunk = self._file.read(remaining)
            except OSError as exc:
                raise SysCon2ConnectionError(
                    f"Failed reading from the SysCon2 pipe: {exc}"
                ) from exc
            if not chunk:
                raise SysCon2ConnectionError("SysCon2 pipe closed unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._file.close()


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
