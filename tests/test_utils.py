from __future__ import annotations

from typing import Any

import pytest
from brian2 import numpy_ as np

from neuroseq import utils


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
def test_neuron_indices_to_colors(n: int) -> None:
    reference_colors = utils.neuron_indices_to_colors(np.arange(2048))
    ordered_indices = np.arange(n)
    colors = utils.neuron_indices_to_colors(ordered_indices)
    # returns n colors as RGBA floats with values in [0,1]
    assert colors.shape == (n, 4)
    assert np.max(colors) <= 1.0
    assert np.min(colors) >= 0.0
    # same index always gets the same color regardless of number of indices
    for i in range(n):
        np.testing.assert_array_equal(colors[i], reference_colors[i])


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
