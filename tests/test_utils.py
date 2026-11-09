from __future__ import annotations

from typing import Any

import pytest
from brian2 import numpy_ as np

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.utils import append_data
from neuroseq.utils import detect_period
from neuroseq.utils import get_downsample_factor
from neuroseq.utils import neuron_indices_to_colors
from neuroseq.utils import parse_number
from neuroseq.utils import safe_concat
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


def test_detect_period_returns_period_for_repeat() -> None:
    values = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), 4)
    assert detect_period(values) == 3


def test_detect_period_returns_none_for_non_repeat() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    assert detect_period(values) is None


def test_detect_period_returns_none_for_short_array() -> None:
    values = np.array([1.0], dtype=np.float32)
    assert detect_period(values) is None


def test_detect_period_handles_2d_arrays() -> None:
    block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values = np.tile(block, (3, 1))
    assert detect_period(values) == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", 1.0),
        ("1.5", 1.5),
        ("-3", -3.0),
        ("  2.0  ", 2.0),
        ("1e-12", 1e-12),
        ("2.5e3", 2500.0),
        ("10^-12", 1e-12),
        ("10^(-12)", 1e-12),
        ("10^12", 1e12),
        ("10^(12)", 1e12),
        ("2*10^-12", 2e-12),
        ("2.5*10^(-12)", 2.5e-12),
        ("3 x 10^6", 3e6),
        ("4×10^-3", 4e-3),
        ("1.5*10^3", 1500.0),
    ],
)
def test_parse_number_accepts_common_formats(text: str, expected: float) -> None:
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["abc", "10^", "^-12", "10^^2", "10*-12", "", "10^(-12"],
)
def test_parse_number_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_number(text)


def test_wait_cursor_is_restored_after_an_error(qtbot) -> None:
    """An error in a decorated function left the wait cursor on for good."""
    from qtpy import QtWidgets

    from neuroseq.utils import with_wait_cursor

    @with_wait_cursor
    def _fails() -> None:
        assert QtWidgets.QApplication.overrideCursor() is not None
        raise KeyError("Simulation :: P")

    with pytest.raises(KeyError):
        _fails()
    assert QtWidgets.QApplication.overrideCursor() is None


def test_append_trace_data_past_end_keeps_time_gap() -> None:
    existing = TraceData("t", np.ones((2, 10)), np.array([0, 1]), "mV", 0.1)  # 0-1 s
    new = TraceData("t", 2 * np.ones((2, 10)), np.array([0, 1]), "mV", 0.1)

    append_data(existing, new, 1.0, 2.0)  # place at 2-3 s

    assert existing.traces.shape == (2, 30)
    np.testing.assert_array_equal(existing.traces[:, :10], 1.0)
    np.testing.assert_array_equal(existing.traces[:, 10:20], 0.0)
    np.testing.assert_array_equal(existing.traces[:, 20:], 2.0)


def test_append_spike_data_keeps_raw_cellpose_ids_aligned() -> None:
    existing = SpikeData(
        "s", np.array([0, 1]), np.array([0.5, 0.2]), raw_cellpose_ids=np.array([5, 6])
    )
    new = SpikeData("s", np.array([2]), np.array([0.3]), raw_cellpose_ids=np.array([7]))

    append_data(existing, new, 1.0, 0.0)

    np.testing.assert_array_equal(existing.spike_indices, [1, 2, 0])
    np.testing.assert_array_equal(existing.raw_cellpose_ids, [6, 7, 5])


def test_append_spike_data_overlap_keeps_raw_cellpose_ids_aligned() -> None:
    existing = SpikeData(
        "s",
        np.array([0, 0, 1]),
        np.array([0.1, 1.5, 0.2]),
        raw_cellpose_ids=np.array([5, 5, 6]),
    )
    new = SpikeData("s", np.array([0]), np.array([0.4]), raw_cellpose_ids=np.array([9]))

    append_data(existing, new, 1.0, 1.0)  # replaces neuron 0 within 1-2 s

    np.testing.assert_array_equal(existing.spike_indices, [0, 1, 0])
    np.testing.assert_array_equal(existing.raw_cellpose_ids, [5, 6, 9])


def test_append_spike_data_drops_raw_cellpose_ids_when_one_side_lacks_them() -> None:
    existing = SpikeData(
        "s", np.array([0]), np.array([0.1]), raw_cellpose_ids=np.array([5])
    )
    new = SpikeData("s", np.array([1]), np.array([0.3]))

    append_data(existing, new, 1.0, 0.0)

    assert existing.spike_indices.size == 2
    assert existing.raw_cellpose_ids is None


def test_append_trace_data_extends_raw_cellpose_ids_for_new_rows() -> None:
    existing = TraceData(
        "t", np.ones((1, 10)), np.array([0]), "mV", 0.1, raw_cellpose_ids=np.array([4])
    )
    new = TraceData(
        "t",
        np.ones((2, 10)),
        np.array([0, 3]),
        "mV",
        0.1,
        raw_cellpose_ids=np.array([4, 8]),
    )

    append_data(existing, new, 1.0, 1.0)

    np.testing.assert_array_equal(existing.indices, [0, 3])
    np.testing.assert_array_equal(existing.raw_cellpose_ids, [4, 8])


def test_append_trace_data_disjoint_keeps_raw_cellpose_ids_aligned() -> None:
    existing = TraceData(
        "t", np.ones((1, 10)), np.array([0]), "mV", 0.1, raw_cellpose_ids=np.array([4])
    )
    new = TraceData(
        "t", np.ones((1, 10)), np.array([1]), "mV", 0.1, raw_cellpose_ids=np.array([9])
    )
    append_data(existing, new, 1.0, 0.0)
    np.testing.assert_array_equal(existing.raw_cellpose_ids, [4, 9])

    # rows without raw ids cannot be aligned any more
    append_data(
        existing, TraceData("t", np.ones((1, 10)), np.array([2]), "mV", 0.1), 1.0, 0.0
    )
    assert existing.traces.shape == (3, 10)
    assert existing.raw_cellpose_ids is None


def test_append_empty_spike_data() -> None:
    existing = SpikeData(
        "s", np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    )
    append_data(
        existing,
        SpikeData("s", np.array([], dtype=np.float64), np.array([], dtype=np.float64)),
        1.0,
        1.0,
    )

    assert existing.spike_indices.size == 0
    assert existing.spike_times.size == 0


def test_safe_concat_all_empty_returns_empty_array() -> None:
    result = safe_concat([np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)])
    assert result.size == 0
    assert result.dtype == np.float32
    mixed = safe_concat([np.empty((0, 3)), np.empty(0)], axis=1)
    assert mixed.size == 0
    assert safe_concat([]).size == 0


def test_append_data_trace_offset_rounds_to_nearest_sample() -> None:
    # 31 * (1/30) / (1/30) is 30.999..., which truncation put one sample early.
    dt = 1 / 30
    n = 31
    existing = TraceData("t", np.ones((1, n), np.float32), np.array([0]), "", dt)
    new = TraceData("t", 2 * np.ones((1, 10), np.float32), np.array([0]), "", dt)
    append_data(existing, new, 10 * dt, n * dt)
    assert existing.traces.shape == (1, n + 10)
    np.testing.assert_array_equal(existing.traces[0, :n], 1)
    np.testing.assert_array_equal(existing.traces[0, n:], 2)
