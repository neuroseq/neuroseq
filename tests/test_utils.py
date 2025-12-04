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
