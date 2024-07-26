from __future__ import annotations

from typing import Any

import brian2 as b2
import pytest
from formozov_project import utils


def test_quantity_parser() -> None:
    assert utils.parse_string_to_quantity("3") == 3

    assert utils.parse_string_to_quantity("3 * ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("3 ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("1/(10) * ms") == 1 / 10 * b2.ms

    assert utils.parse_string_to_quantity("35") == 35

    assert utils.parse_string_to_quantity("[3, 10]") == [3, 10]

    assert utils.parse_string_to_quantity("x*y") == "x*y"

    with pytest.raises(RuntimeError, match="Error when interpreting unit string"):
        utils.parse_string_to_quantity("1/(10) * Mx")


def test_update_nested_dict() -> None:
    d: dict[Any, Any] = {
        "h": {
            "l": 5,
            3: "hello",
            "m": 66,
        },
        "r": {"d": {"x": 12, "y": "z"}, "e": "test"},
        "a": 3.14,
        "b": 42,
    }

    utils.update_nested_dict(
        d["h"],
        {
            "l": 55,
        },
    )

    assert d["h"]["l"] == 55
    # rest is as was
    assert d["r"] == {"d": {"x": 12, "y": "z"}, "e": "test"}

    utils.update_nested_dict(
        d,
        {
            "r": {
                55: "other",
            }
        },
    )

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }

    assert d["r"] == {55: "other", "d": {"x": 12, "y": "z"}, "e": "test"}

    utils.update_nested_dict(d, {"a": 6.28})

    assert d["a"] == 6.28
    assert d["r"] == {55: "other", "d": {"x": 12, "y": "z"}, "e": "test"}

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }


def test_shuffle_index_small_n() -> None:
    assert utils.shuffle_index(0, 1) == (0, 1)
    assert utils.shuffle_index(0, 2) == (0, 2)
    assert utils.shuffle_index(1, 2) == (1, 2)


@pytest.mark.parametrize("n", [3, 7, 20, 100, 2000, 2001])
def test_shuffle_index(n: int) -> None:
    ordered_indices = list(range(n))
    shuffled_indices = [utils.shuffle_index(i, n)[0] for i in ordered_indices]
    assert len(shuffled_indices) == n
    assert shuffled_indices != ordered_indices
    assert sorted(shuffled_indices) == ordered_indices


def test_get_downsample_factor_valid() -> None:
    assert utils.get_downsample_factor(100, 1) == 1
    assert utils.get_downsample_factor(1024 * 1024 - 1, 4, bits_per_float=32) == 1
    assert utils.get_downsample_factor(1024 * 1024, 4, bits_per_float=32) == 2
    assert utils.get_downsample_factor(1024 * 1024 - 1, 8) == 1
    assert utils.get_downsample_factor(1024 * 1024, 8) == 2
    assert utils.get_downsample_factor(2 * 1024 * 1024, 8) == 3
    assert utils.get_downsample_factor(1024 * 1024 * 1024, 8) == 1025


def test_get_downsample_factor_invalid() -> None:
    assert utils.get_downsample_factor(-5, 8) == 1
    assert utils.get_downsample_factor(500, 0) == 1
    assert utils.get_downsample_factor(100, -2) == 1
    assert utils.get_downsample_factor(-5, -2) == 1
