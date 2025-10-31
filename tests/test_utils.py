from __future__ import annotations

from typing import Any

import brian2 as b2
import pytest
from brian2 import numpy_ as np

from neuroseq import utils


@pytest.mark.parametrize(
    "quantity",
    [
        3 * b2.amp,
        0.0002 * b2.ms,
        5e-12 * b2.ms,
        -2e-3 * b2.ms,
        -3 * b2.ms,
        1 / 10 * b2.ms,
        1.5 * 1e-5 * b2.ms,
        7 * b2.us,
        21.3 * b2.mS,
    ],
)
def test_quantity_parser_round_trip_quantities(quantity: b2.Quantity) -> None:
    string = str(quantity.in_best_unit(python_code=True))
    parsed_quantity = utils.parse_string_to_quantity(string)
    assert isinstance(parsed_quantity, b2.Quantity)
    assert parsed_quantity.has_same_dimensions(quantity)
    assert np.allclose(np.asarray(parsed_quantity), np.asarray(quantity))


def test_quantity_parser() -> None:
    assert utils.parse_string_to_quantity("3") == 3

    assert utils.parse_string_to_quantity("3 * ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("+3 * ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("-3 * ms") == -3 * b2.ms

    assert utils.parse_string_to_quantity("3 ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("3e-1 ms") == 0.3 * b2.ms

    assert utils.parse_string_to_quantity("3. ms") == 3 * b2.ms

    assert utils.parse_string_to_quantity("3.   ms  ") == 3 * b2.ms

    assert utils.parse_string_to_quantity("1/(10) * ms") == 1 / 10 * b2.ms

    assert utils.parse_string_to_quantity("35") == 35

    assert utils.parse_string_to_quantity("[3, 10]") == [3, 10]

    assert utils.parse_string_to_quantity("x*y") == "x*y"

    with pytest.raises(RuntimeError, match="Error when interpreting unit string"):
        utils.parse_string_to_quantity("1/(10) * Mx")

    assert utils.parse_string_to_quantity("euler") == "euler"

    assert utils.parse_string_to_quantity("exponential_euler") == "exponential_euler"

    assert utils.parse_string_to_quantity("exact") == "exact"

    assert utils.parse_string_to_quantity("rk2") == "rk2"

    assert utils.parse_string_to_quantity("rk4") == "rk4"

    assert utils.parse_string_to_quantity("heun") == "heun"

    assert utils.parse_string_to_quantity("milstein") == "milstein"

    assert utils.parse_string_to_quantity("2e10 ms") == 2e10 * b2.ms

    assert utils.parse_string_to_quantity("2e-5 ms") == 2e-5 * b2.ms


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


@pytest.mark.parametrize("n", [3, 7, 20, 100, 2000, 2001])
def test_cmap_transform_neuron_indices(n: int) -> None:
    ordered_indices = np.arange(n)
    shuffled_indices = utils.cmap_transform_neuron_indices(n)
    assert shuffled_indices.shape == (n,)
    assert not np.allclose(shuffled_indices, ordered_indices)
    sorted_shuffled_indices = np.sort(shuffled_indices)
    assert np.allclose(sorted_shuffled_indices, ordered_indices)
    # shuffle an array of indices, check that mapping is the same as above for each index
    arr = utils.cmap_transform_neuron_indices(n, [2, 0, 1, 0])
    assert list(arr) == [
        shuffled_indices[2],
        shuffled_indices[0],
        shuffled_indices[1],
        shuffled_indices[0],
    ]


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
