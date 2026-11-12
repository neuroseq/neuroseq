from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import TraceData
from neuroseq.visualization_support.trace_comparison import ARRAY_COLORS
from neuroseq.visualization_support.trace_comparison import array_color
from neuroseq.visualization_support.trace_comparison import array_labels
from neuroseq.visualization_support.trace_comparison import build_trace_comparison
from neuroseq.visualization_support.trace_comparison import format_neuron_ids
from neuroseq.visualization_support.trace_comparison import parse_neuron_ids
from neuroseq.visualization_support.trace_comparison import time_axis_ms


def _trace(name: str, ids, n_samples: int, dt: float, units: str = "dF/F"):
    ids = np.asarray(ids)
    traces = np.arange(len(ids) * n_samples, dtype=np.float32).reshape(
        len(ids), n_samples
    )
    return TraceData(name, traces, ids, units, dt)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("3", [3]),
        ("0-3, 7", [0, 1, 2, 3, 7]),
        ("5-3", [3, 4, 5]),
        ("1..2; 9:10 4", [1, 2, 4, 9, 10]),
        (" 2 , 2 ", [2]),
    ],
)
def test_parse_neuron_ids(text: str, expected: list[int]) -> None:
    assert parse_neuron_ids(text) == expected


@pytest.mark.parametrize("text", ["a", "1-", "-3", "1-2-3", "1.5"])
def test_parse_neuron_ids_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="not a neuron id or range"):
        parse_neuron_ids(text)


def test_format_neuron_ids_round_trips() -> None:
    assert format_neuron_ids([7, 0, 1, 2, 3, 9, 10]) == "0-3, 7, 9-10"
    assert format_neuron_ids([]) == ""
    assert parse_neuron_ids(format_neuron_ids([4, 5, 8])) == [4, 5, 8]


def test_array_colors_are_distinct_and_cycle() -> None:
    colors = [array_color(i) for i in range(len(ARRAY_COLORS))]
    assert len(set(colors)) == len(ARRAY_COLORS)
    assert array_color(len(ARRAY_COLORS)) == array_color(0)


def test_array_labels_use_experiment_names_for_repeats() -> None:
    assert array_labels(["Rep 1 :: dF/F", "Rep 2 :: dF/F"]) == ["Rep 1", "Rep 2"]
    # Different data names, a single array or simulation data keep full names.
    assert array_labels(["A :: x", "B :: y"]) == ["A :: x", "B :: y"]
    assert array_labels(["A :: x"]) == ["A :: x"]
    assert array_labels(["Simulation :: n / v", "A :: n / v"]) == [
        "Simulation :: n / v",
        "A :: n / v",
    ]


def test_build_comparison_two_arrays_three_neurons() -> None:
    a = _trace("dF/F", [0, 1, 2], 10, 0.1)
    b = _trace("dF/F", [0, 1, 2], 10, 0.1)
    comparison = build_trace_comparison(
        [("Rep 1 :: dF/F", a), ("Rep 2 :: dF/F", b)], [2, 0, 1]
    )
    assert [row.neuron_id for row in comparison.rows] == [2, 0, 1]
    assert comparison.legend == [
        ("Rep 1", ARRAY_COLORS[0]),
        ("Rep 2", ARRAY_COLORS[1]),
    ]
    first = comparison.rows[0]
    assert [s.label for s in first.series] == ["Rep 1", "Rep 2"]
    np.testing.assert_array_equal(first.series[0].values, a.traces[2])
    assert first.series[1].color == ARRAY_COLORS[1]
    assert comparison.missing == {}
    assert comparison.missing_note() == ""
    assert comparison.units == "dF/F"
    assert comparison.max_time_ms == pytest.approx(900.0)


def test_build_comparison_uses_each_arrays_own_time_axis() -> None:
    fast = _trace("x", [0], 30, 1 / 30)
    slow = _trace("y", [0], 5, 0.2, units="mV")
    comparison = build_trace_comparison([("A :: x", fast), ("B :: y", slow)], [0])
    fast_series, slow_series = comparison.rows[0].series
    np.testing.assert_allclose(fast_series.time_ms, time_axis_ms(fast))
    assert fast_series.time_ms[1] == pytest.approx(1000 / 30)
    np.testing.assert_allclose(slow_series.time_ms, [0, 200, 400, 600, 800])
    assert comparison.max_time_ms == pytest.approx(29 * 1000 / 30)
    # Arrays disagree on units, so no common unit is claimed.
    assert comparison.units == ""


def test_build_comparison_skips_missing_neurons_with_note() -> None:
    a = _trace("t", [0, 1, 5], 4, 0.1)
    b = _trace("t", [0, 1], 4, 0.1)
    comparison = build_trace_comparison([("R1 :: t", a), ("R2 :: t", b)], [0, 5, 7, 0])
    # 7 is in no array, so it gets no row; duplicates collapse.
    assert [row.neuron_id for row in comparison.rows] == [0, 5]
    assert [s.label for s in comparison.rows[1].series] == ["R1"]
    assert comparison.missing == {"R1": [7], "R2": [5, 7]}
    note = comparison.missing_note()
    assert "R1: 7" in note
    assert "R2: 5, 7" in note


def test_build_comparison_empty() -> None:
    comparison = build_trace_comparison([], [1, 2])
    assert comparison.rows == []
    assert comparison.max_time_ms == 0.0
    assert comparison.units == ""
