from __future__ import annotations

import pytest

from neuroseq.power_lut import DEFAULT_POWER_LUTS
from neuroseq.power_lut import PowerLut
from neuroseq.power_lut import parse_power_lut_text
from neuroseq.power_lut import power_lut_mw_to_percent
from neuroseq.power_lut import resolve_power_lut

_REFERENCE_LUT_TEXT = """F2-EC-FD-4A-EB-0E-8C-29-3A-0D-22-7A-94-1D-D1-94
[_]
Description=1100nm 1 spot
Unit=mW
Enabled=False
TuplesCount=9
Tuple0=0;0
Tuple1=5;0
Tuple2=10;2.1
Tuple3=20;8.4
Tuple4=40;21.9
Tuple5=60;36
Tuple6=80;49
Tuple7=90;49
Tuple8=100;49
[/_]
"""


def test_parse_power_lut_text_parses_reference_example() -> None:
    lut = parse_power_lut_text(_REFERENCE_LUT_TEXT)

    assert lut.description == "1100nm 1 spot"
    assert len(lut.tuples) == 9
    assert lut.tuples[-1] == (100.0, 49.0)


def test_power_lut_mw_to_percent_interpolates_linearly() -> None:
    lut = PowerLut(description="linear", tuples=((0.0, 0.0), (50.0, 25.0)))

    percent, saturated = power_lut_mw_to_percent(lut, 12.5)

    assert percent == pytest.approx(25.0)
    assert saturated is False


def test_power_lut_mw_to_percent_saturates_at_first_max_percent() -> None:
    lut = parse_power_lut_text(_REFERENCE_LUT_TEXT)

    percent, saturated = power_lut_mw_to_percent(lut, 60.0)

    assert percent == pytest.approx(80.0)
    assert saturated is True


def test_power_lut_mw_to_percent_zero_target_returns_zero() -> None:
    lut = parse_power_lut_text(_REFERENCE_LUT_TEXT)

    percent, saturated = power_lut_mw_to_percent(lut, 0.0)

    assert percent == 0.0
    assert saturated is False


def test_power_lut_serializable_roundtrip() -> None:
    lut = PowerLut(description="roundtrip", tuples=((0.0, 0.0), (100.0, 50.0)))

    restored = PowerLut.from_serializable(lut.to_serializable())

    assert restored == lut


def test_default_power_luts_includes_1100nm() -> None:
    lut = DEFAULT_POWER_LUTS[1100]

    assert lut.description == "1100nm 1 spot"
    assert len(lut.tuples) == 9
    assert lut.tuples[-1] == (100.0, 49.0)


def test_resolve_power_lut_prefers_project_override_over_default() -> None:
    override = PowerLut(description="override", tuples=((0.0, 0.0), (100.0, 10.0)))

    lut, is_override = resolve_power_lut({1100: override}, 1100)

    assert lut == override
    assert is_override is True


def test_resolve_power_lut_falls_back_to_default_when_project_has_none() -> None:
    lut, is_override = resolve_power_lut({}, 1100)

    assert lut == DEFAULT_POWER_LUTS[1100]
    assert is_override is False


def test_resolve_power_lut_returns_none_for_unknown_wavelength() -> None:
    lut, is_override = resolve_power_lut({}, 940)

    assert lut is None
    assert is_override is False
