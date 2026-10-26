from __future__ import annotations

import pytest

from neuroseq.device_control.syscon2_protocol import SysCon2Error
from neuroseq.device_control.syscon2_protocol import SysCon2ProtocolError
from neuroseq.device_control.syscon2_protocol import build_command_frame
from neuroseq.device_control.syscon2_protocol import bytes_reader
from neuroseq.device_control.syscon2_protocol import encode_bool
from neuroseq.device_control.syscon2_protocol import encode_byte
from neuroseq.device_control.syscon2_protocol import encode_command
from neuroseq.device_control.syscon2_protocol import encode_double
from neuroseq.device_control.syscon2_protocol import encode_dword
from neuroseq.device_control.syscon2_protocol import encode_float
from neuroseq.device_control.syscon2_protocol import encode_qword
from neuroseq.device_control.syscon2_protocol import encode_raw
from neuroseq.device_control.syscon2_protocol import encode_string
from neuroseq.device_control.syscon2_protocol import encode_structures
from neuroseq.device_control.syscon2_protocol import encode_word
from neuroseq.device_control.syscon2_protocol import holo4d_ready_to_run
from neuroseq.device_control.syscon2_protocol import normalized_holo4d_state
from neuroseq.device_control.syscon2_protocol import parse_standard_response
from neuroseq.device_control.syscon2_protocol import parse_string_response
from neuroseq.device_control.syscon2_protocol import read_bool
from neuroseq.device_control.syscon2_protocol import read_byte
from neuroseq.device_control.syscon2_protocol import read_data_response_status
from neuroseq.device_control.syscon2_protocol import read_double
from neuroseq.device_control.syscon2_protocol import read_dword
from neuroseq.device_control.syscon2_protocol import read_float
from neuroseq.device_control.syscon2_protocol import read_qword
from neuroseq.device_control.syscon2_protocol import read_raw
from neuroseq.device_control.syscon2_protocol import read_string
from neuroseq.device_control.syscon2_protocol import read_structures
from neuroseq.device_control.syscon2_protocol import read_word

# Golden vectors transcribed verbatim from the manual's §1.1.1 "Examples".
_BYTE_EXAMPLE = (bytes([0x74]), 116)
_QWORD_EXAMPLE = (bytes([0xA1, 0x30, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]), 77985)
_DOUBLE_EXAMPLE = (bytes([0xEC, 0x51, 0xB8, 0x1E, 0x85, 0x6B, 0x36, 0x40]), 22.42)
_RAW_EXAMPLE = (
    bytes([0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    + bytes([0x54, 0xF1, 0x33, 0xCC, 0xBE, 0xEF, 0xFA, 0xCE]),
    bytes([0x54, 0xF1, 0x33, 0xCC, 0xBE, 0xEF, 0xFA, 0xCE]),
)
_STRING_EXAMPLE = (bytes([0x05, 0x00, 0x00, 0x00]) + b"Hello", "Hello")
_STRUCT_EXAMPLE_WIRE = (
    bytes([0x0C, 0x00, 0x00, 0x00])
    + bytes([0x11, 0x00, 0x12, 0x00])
    + bytes([0x21, 0x00, 0x22, 0x00])
    + bytes([0x31, 0x00, 0x32, 0x00])
)
_STRUCT_EXAMPLE_VALUES = [(17, 18), (33, 34), (49, 50)]


def test_byte_example() -> None:
    wire, value = _BYTE_EXAMPLE
    assert read_byte(bytes_reader(wire)) == value
    assert encode_byte(value) == wire


def test_qword_example() -> None:
    wire, value = _QWORD_EXAMPLE
    assert read_qword(bytes_reader(wire)) == value
    assert encode_qword(value) == wire


def test_double_example() -> None:
    wire, value = _DOUBLE_EXAMPLE
    assert read_double(bytes_reader(wire)) == pytest.approx(value)
    assert encode_double(value) == wire


def test_raw_example() -> None:
    wire, data = _RAW_EXAMPLE
    assert read_raw(bytes_reader(wire)) == data
    assert encode_raw(data) == wire


def test_string_example() -> None:
    wire, text = _STRING_EXAMPLE
    assert read_string(bytes_reader(wire)) == text
    assert encode_string(text) == wire


def test_struct_example() -> None:
    body = read_structures(bytes_reader(_STRUCT_EXAMPLE_WIRE))
    reader = bytes_reader(body)
    decoded = [(read_word(reader), read_word(reader)) for _ in range(3)]
    assert decoded == _STRUCT_EXAMPLE_VALUES

    encoded_body = b"".join(
        encode_word(a) + encode_word(b) for a, b in _STRUCT_EXAMPLE_VALUES
    )
    assert encode_structures(encoded_body) == _STRUCT_EXAMPLE_WIRE


def test_bool_round_trip() -> None:
    assert read_bool(bytes_reader(encode_bool(True))) is True
    assert read_bool(bytes_reader(encode_bool(False))) is False
    # Spec: any nonzero byte is True, not just 0x01.
    assert read_bool(bytes_reader(bytes([0x2A]))) is True


def test_signed_word_dword_qword_round_trip() -> None:
    assert read_word(bytes_reader(encode_word(-1, signed=True)), signed=True) == -1
    assert read_dword(bytes_reader(encode_dword(-1, signed=True)), signed=True) == -1
    assert read_qword(bytes_reader(encode_qword(-1, signed=True)), signed=True) == -1


def test_float_round_trip() -> None:
    wire = encode_float(1.5)
    assert read_float(bytes_reader(wire)) == pytest.approx(1.5)


def test_string_rejects_non_ascii() -> None:
    with pytest.raises(SysCon2ProtocolError):
        encode_string("café")


def test_command_single_line_terminated() -> None:
    assert encode_command("sci::GetAllInstances") == b"sci::GetAllInstances\r\n"


def test_command_rejects_multiple_lines() -> None:
    with pytest.raises(SysCon2ProtocolError):
        encode_command("sci::GetAllInstances\r\nsci::GetProfileType")


def test_build_command_frame_concatenates_params() -> None:
    frame = build_command_frame("sci", "GetInstances", encode_string("Laser"))
    assert frame == b"sci::GetInstances\r\n" + encode_string("Laser")


def test_parse_standard_response_ok() -> None:
    parse_standard_response("OK")  # must not raise


def test_parse_standard_response_error() -> None:
    with pytest.raises(SysCon2Error, match="device not found"):
        parse_standard_response("Error: device not found")


def test_parse_string_response_ok() -> None:
    assert parse_string_response("Laser-1,Laser-2\r\n") == "Laser-1,Laser-2\r\n"


def test_parse_string_response_error() -> None:
    with pytest.raises(SysCon2Error, match="bad plugin id"):
        parse_string_response("Error: bad plugin id")


def test_read_data_response_status_ok() -> None:
    read_data_response_status(bytes_reader(bytes([0x00])))  # must not raise


def test_read_data_response_status_error() -> None:
    with pytest.raises(SysCon2Error, match="status=1"):
        read_data_response_status(bytes_reader(bytes([0x01])))


def test_bytes_reader_raises_on_truncated_data() -> None:
    with pytest.raises(SysCon2ProtocolError):
        read_dword(bytes_reader(bytes([0x01, 0x02])))


def test_holo4d_state_rules_shared_by_panel_and_routine_host() -> None:
    assert normalized_holo4d_state("State: Idle ") == "idle"
    assert holo4d_ready_to_run("State:Idle")
    assert holo4d_ready_to_run("Ready")  # unknown states do not block forever
    for state in ("", "Busy", "Disconnected", "Uploading", "upload pending"):
        assert not holo4d_ready_to_run(state)


def test_strict_readiness_accepts_only_idle() -> None:
    assert holo4d_ready_to_run("State: Idle", strict=True)
    for state in ("Error", "Ready", "", "Busy", "Uploading"):
        assert not holo4d_ready_to_run(state, strict=True)
