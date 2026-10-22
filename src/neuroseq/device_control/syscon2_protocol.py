"""Codec for the SysCon2 SCI-Protocol (RAPP "SysCon2 Control Interface Manual" v1.2).

Pure encode/decode logic, no I/O: values are packed to ``bytes`` for sending, and read via an
injected :data:`Reader` callable (``recv(n) -> bytes``) so the same code works against a real
transport or an in-memory buffer in tests. All multi-byte fields are little-endian (manual §1.1).
"""

from __future__ import annotations

import struct
from collections.abc import Callable

Reader = Callable[[int], bytes]

_ERROR_PREFIX = "Error: "


class SysCon2ProtocolError(Exception):
    """Malformed SCI-Protocol framing: a truncated read, non-ASCII text, a multi-line command."""


class SysCon2Error(SysCon2ProtocolError):
    """SysCon2 returned an ``Error: ...`` response (or a nonzero Data response status)."""


def bytes_reader(data: bytes) -> Reader:
    """Wrap a fixed buffer as a :data:`Reader`, for tests and for decoding a Structures body."""
    pos = 0

    def _read(n: int) -> bytes:
        nonlocal pos
        chunk = data[pos : pos + n]
        if len(chunk) != n:
            raise SysCon2ProtocolError(f"expected {n} bytes, got {len(chunk)}")
        pos += n
        return chunk

    return _read


def _ascii_bytes(text: str) -> bytes:
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SysCon2ProtocolError(f"not 7-bit ASCII: {text!r}") from exc


def _ascii_str(data: bytes) -> str:
    try:
        return data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SysCon2ProtocolError(f"not 7-bit ASCII: {data!r}") from exc


# Byte: a single (unsigned) byte of data.
def encode_byte(value: int) -> bytes:
    return struct.pack("<B", value)


def read_byte(reader: Reader) -> int:
    return struct.unpack("<B", reader(1))[0]


# Bool: 1 byte, 0 -> False, anything else -> True.
def encode_bool(value: bool) -> bytes:
    return struct.pack("<B", 1 if value else 0)


def read_bool(reader: Reader) -> bool:
    return read_byte(reader) != 0


# Word / DWord / Qword: 2/4/8-byte integers, signedness depends on the command's context.
def encode_word(value: int, *, signed: bool = False) -> bytes:
    return struct.pack("<h" if signed else "<H", value)


def read_word(reader: Reader, *, signed: bool = False) -> int:
    return struct.unpack("<h" if signed else "<H", reader(2))[0]


def encode_dword(value: int, *, signed: bool = False) -> bytes:
    return struct.pack("<i" if signed else "<I", value)


def read_dword(reader: Reader, *, signed: bool = False) -> int:
    return struct.unpack("<i" if signed else "<I", reader(4))[0]


def encode_qword(value: int, *, signed: bool = False) -> bytes:
    return struct.pack("<q" if signed else "<Q", value)


def read_qword(reader: Reader, *, signed: bool = False) -> int:
    return struct.unpack("<q" if signed else "<Q", reader(8))[0]


# Float / Double: IEEE 754.
def encode_float(value: float) -> bytes:
    return struct.pack("<f", value)


def read_float(reader: Reader) -> float:
    return struct.unpack("<f", reader(4))[0]


def encode_double(value: float) -> bytes:
    return struct.pack("<d", value)


def read_double(reader: Reader) -> float:
    return struct.unpack("<d", reader(8))[0]


# Raw: an 8-byte (Qword) length header followed by that many raw data bytes.
def encode_raw(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + data


def read_raw(reader: Reader) -> bytes:
    length = read_qword(reader)
    return reader(length)


# String: a 4-byte (DWord) length header (character count) followed by 7-bit ASCII bytes.
# Multi-line strings embed "\r\n" as ordinary content; that's not special-cased here.
def encode_string(text: str) -> bytes:
    payload = _ascii_bytes(text)
    return struct.pack("<I", len(payload)) + payload


def read_string(reader: Reader) -> str:
    length = read_dword(reader)
    return _ascii_str(reader(length))


# Command: like String but with no length header; single line only, terminated by "\r\n".
def encode_command(name: str) -> bytes:
    if "\r" in name or "\n" in name:
        raise SysCon2ProtocolError(f"command name must be a single line: {name!r}")
    return _ascii_bytes(name) + b"\r\n"


# Structures: a 4-byte (DWord) header giving the total body length, then the (caller-assembled,
# possibly nested) field bytes.
def encode_structures(body: bytes) -> bytes:
    return struct.pack("<I", len(body)) + body


def read_structures(reader: Reader) -> bytes:
    length = read_dword(reader)
    return reader(length)


def build_command_frame(instance: str, command: str, *params: bytes) -> bytes:
    """Build a full SCI-Protocol request: ``instance::command\\r\\n`` followed by encoded params.

    Per manual §2.3, capitalization of the instance/command names is not significant.
    """
    return encode_command(f"{instance}::{command}") + b"".join(params)


def parse_standard_response(text: str) -> None:
    """Validate a Standard response, raising :class:`SysCon2Error` on ``Error: ...`` (manual §1.2)."""
    if text.startswith(_ERROR_PREFIX):
        raise SysCon2Error(text[len(_ERROR_PREFIX) :])


def parse_string_response(text: str) -> str:
    """Return a String response's payload, raising :class:`SysCon2Error` on ``Error: ...``."""
    if text.startswith(_ERROR_PREFIX):
        raise SysCon2Error(text[len(_ERROR_PREFIX) :])
    return text


def read_data_response_status(reader: Reader) -> None:
    """Read a Data response's leading status byte, raising :class:`SysCon2Error` if nonzero.

    On success, the remaining bytes are whatever data type the specific command documents; the
    caller reads them separately with the ordinary ``read_*`` functions on the same reader.
    """
    status = read_byte(reader)
    if status != 0:
        raise SysCon2Error(f"status={status}")


def normalized_holo4d_state(state: str) -> str:
    """``Holo4D::GetState`` reply as a lower-case state id (``State: Idle`` -> ``idle``)."""
    return state.lower().removeprefix("state:").strip()


def holo4d_ready_to_run(state: str) -> bool:
    """Whether an uploaded sequence can be run: the hardware reports a state and is
    neither uploading, busy nor disconnected. Unknown states count as ready rather
    than leaving a caller stuck on a state name it does not know."""
    state = normalized_holo4d_state(state)
    return (
        bool(state)
        and state not in {"busy", "disconnected"}
        and not state.startswith("upload")
    )
