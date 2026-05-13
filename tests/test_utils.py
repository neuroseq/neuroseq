from __future__ import annotations

from typing import Any

import pytest
from brian2 import numpy_ as np

from neuroseq.data import SynapseData
from neuroseq.utils import append_data
from neuroseq.utils import get_downsample_factor
from neuroseq.utils import neuron_indices_to_colors
from neuroseq.utils import update_nested_dict


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

    update_nested_dict(
        d["h"],
        {
            "l": 55,
        },
    )

    assert d["h"]["l"] == 55
    # rest is as was
    assert d["r"] == {"d": {"x": 12, "y": "z"}, "e": "test"}

    update_nested_dict(
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

    update_nested_dict(d, {"a": 6.28})

    assert d["a"] == 6.28
    assert d["r"] == {55: "other", "d": {"x": 12, "y": "z"}, "e": "test"}

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }


@pytest.mark.parametrize("n", [3, 7, 20, 100, 2000, 2001])
def test_neuron_indices_to_colors(n: int) -> None:
    reference_colors = neuron_indices_to_colors(np.arange(2048))
    ordered_indices = np.arange(n)
    colors = neuron_indices_to_colors(ordered_indices)
    # returns n colors as RGBA floats with values in [0,1]
    assert colors.shape == (n, 4)
    assert np.max(colors) <= 1.0
    assert np.min(colors) >= 0.0
    # same index always gets the same color regardless of number of indices
    for i in range(n):
        np.testing.assert_array_equal(colors[i], reference_colors[i])


def test_get_downsample_factor_valid() -> None:
    assert get_downsample_factor(100, 1) == 1
    assert get_downsample_factor(1024 * 1024 - 1, 4, bits_per_float=32) == 1
    assert get_downsample_factor(1024 * 1024, 4, bits_per_float=32) == 2
    assert get_downsample_factor(1024 * 1024 - 1, 8) == 1
    assert get_downsample_factor(1024 * 1024, 8) == 2
    assert get_downsample_factor(2 * 1024 * 1024, 8) == 3
    assert get_downsample_factor(1024 * 1024 * 1024, 8) == 1025


def test_get_downsample_factor_invalid() -> None:
    assert get_downsample_factor(-5, 8) == 1
    assert get_downsample_factor(500, 0) == 1
    assert get_downsample_factor(100, -2) == 1
    assert get_downsample_factor(-5, -2) == 1


def test_append_data_synapse_concatenates_along_time_axis() -> None:
    """Regression: appending two non-overlapping SynapseData chunks must
    concatenate along the timestamp axis (axis 2), not along the neuron axis.

    The earlier bug used `existing.values[0:idx0]` where ``idx0`` is a
    timestamp index — that slices axis 0 (n_neurons) and gives a shape that
    can't concatenate with the new data along axis 2.
    """
    n_neurons = 54
    n_snaps = 20

    existing_values = np.arange(
        n_neurons * n_neurons * n_snaps, dtype=np.float32
    ).reshape(n_neurons, n_neurons, n_snaps)
    new_values = (
        np.arange(n_neurons * n_neurons * n_snaps, dtype=np.float32) + 1.0e6
    ).reshape(n_neurons, n_neurons, n_snaps)

    existing = SynapseData(
        name="dendritic_weights",
        values=existing_values.copy(),
        timestamps=np.linspace(0.0, 0.95, n_snaps, dtype=np.float32),
    )
    new = SynapseData(
        name="dendritic_weights",
        values=new_values.copy(),
        # new chunk's timestamps are per-run-local; append_data offsets them.
        timestamps=np.linspace(0.0, 0.95, n_snaps, dtype=np.float32),
    )

    append_data(
        existing,
        new,
        data_duration_seconds=1.0,
        time_offset_seconds=1.0,
    )

    # Concatenated along axis 2: 20 + 20 = 40 timestamps, neuron axes preserved.
    assert existing.values.shape == (n_neurons, n_neurons, 2 * n_snaps)
    assert existing.timestamps.shape == (2 * n_snaps,)

    # First half is the original data, untouched.
    np.testing.assert_array_equal(existing.values[:, :, :n_snaps], existing_values)
    # Second half is the new data, also untouched.
    np.testing.assert_array_equal(existing.values[:, :, n_snaps:], new_values)

    # Timestamps: original 0..0.95, then new offset by 1.0 -> 1.0..1.95.
    assert existing.timestamps[0] == pytest.approx(0.0)
    assert existing.timestamps[n_snaps - 1] == pytest.approx(0.95)
    assert existing.timestamps[n_snaps] == pytest.approx(1.0)
    assert existing.timestamps[-1] == pytest.approx(1.95)


def test_append_data_synapse_overwrites_overlap() -> None:
    """Overlapping SynapseData chunks: existing samples inside the overlap
    window are dropped, replaced by the new chunk, and the tail is kept.
    """
    n_neurons = 3
    n_snaps = 10

    existing = SynapseData(
        name="w",
        values=np.zeros((n_neurons, n_neurons, n_snaps), dtype=np.float32),
        timestamps=np.linspace(0.0, 0.9, n_snaps, dtype=np.float32),
    )
    # New chunk is 4 samples long, lands at t=[0.4, 0.5, 0.6, 0.7] after offset.
    new = SynapseData(
        name="w",
        values=np.ones((n_neurons, n_neurons, 4), dtype=np.float32),
        timestamps=np.linspace(0.0, 0.3, 4, dtype=np.float32),
    )
    append_data(
        existing,
        new,
        data_duration_seconds=0.4,
        time_offset_seconds=0.4,
    )

    # Before overlap (indices 0..3 at t=0.0..0.3): kept, all zeros.
    assert np.all(existing.values[:, :, :4] == 0.0)
    # New data slot: 4 ones.
    assert np.all(existing.values[:, :, 4:8] == 1.0)
    # After overlap (originally indices 8..9 at t=0.8, 0.9): kept, all zeros.
    assert np.all(existing.values[:, :, 8:] == 0.0)
    assert existing.values.shape[2] == existing.timestamps.shape[0]
