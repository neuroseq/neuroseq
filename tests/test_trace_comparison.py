from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import TraceData
from neuroseq.visualization_support.trace_comparison import ARRAY_COLORS
from neuroseq.visualization_support.trace_comparison import TIME_AXIS
from neuroseq.visualization_support.trace_comparison import VALUE_AXIS
from neuroseq.visualization_support.trace_comparison import AxisLinkGroup
from neuroseq.visualization_support.trace_comparison import array_color
from neuroseq.visualization_support.trace_comparison import array_labels
from neuroseq.visualization_support.trace_comparison import build_trace_comparison
from neuroseq.visualization_support.trace_comparison import build_trace_concatenation
from neuroseq.visualization_support.trace_comparison import cascade_offsets
from neuroseq.visualization_support.trace_comparison import concatenation_data_name
from neuroseq.visualization_support.trace_comparison import concatenation_group_name
from neuroseq.visualization_support.trace_comparison import format_neuron_ids
from neuroseq.visualization_support.trace_comparison import group_name_problem
from neuroseq.visualization_support.trace_comparison import parse_neuron_ids
from neuroseq.visualization_support.trace_comparison import shared_dt
from neuroseq.visualization_support.trace_comparison import time_axis_ms
from neuroseq.visualization_support.trace_comparison import unique_name


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


# -- Concatenation -----------------------------------------------------------


def test_concatenation_offsets_follow_each_arrays_duration() -> None:
    a = _trace("t", [0, 1], 10, 0.1)  # 1000 ms
    b = _trace("t", [0, 1], 5, 0.2)  # 1000 ms on its own dt
    c = _trace("t", [0, 1], 3, 0.1)
    joined = build_trace_concatenation(
        [("R1 :: t", a), ("R2 :: t", b), ("R3 :: t", c)], [1, 0]
    )
    starts = [segment.start_ms for segment in joined.segments]
    assert starts == pytest.approx([0.0, 1000.0, 2000.0])
    assert [segment.label for segment in joined.segments] == ["R1", "R2", "R3"]
    assert [segment.color for segment in joined.segments] == list(ARRAY_COLORS[:3])
    np.testing.assert_allclose(
        joined.segments[1].time_ms, [1000, 1200, 1400, 1600, 1800]
    )
    assert joined.total_ms == pytest.approx(2300.0)
    assert joined.boundaries_ms == pytest.approx([1000.0, 2000.0])
    assert [row.neuron_id for row in joined.rows] == [1, 0]
    np.testing.assert_array_equal(joined.rows[0].values[1], b.traces[1])
    assert not joined.same_dt
    with pytest.raises(ValueError, match="same sampling interval"):
        joined.to_trace_data("x")


def test_concatenation_gap_is_snapped_to_samples_for_equal_dt() -> None:
    a = _trace("t", [0], 4, 0.1)
    b = _trace("t", [0], 4, 0.1)
    joined = build_trace_concatenation([("A :: t", a), ("B :: t", b)], [0], 240.0)
    assert joined.gap_ms == pytest.approx(200.0)
    assert joined.segments[1].start_ms == pytest.approx(600.0)
    assert joined.boundaries_ms == pytest.approx([500.0])
    data = joined.to_trace_data("t")
    assert data.traces.shape == (1, 10)
    assert np.isnan(data.traces[0, 4:6]).all()
    np.testing.assert_array_equal(data.traces[0, 6:], b.traces[0])

    # Different dt: the gap is kept as given.
    c = _trace("t", [0], 4, 0.3)
    joined = build_trace_concatenation([("A :: t", a), ("C :: t", c)], [0], 240.0)
    assert joined.gap_ms == pytest.approx(240.0)
    assert joined.segments[1].start_ms == pytest.approx(640.0)


def test_concatenation_nan_fills_missing_neurons() -> None:
    a = _trace("t", [0, 1, 5], 3, 0.1)
    b = _trace("t", [0, 1], 2, 0.1)
    joined = build_trace_concatenation([("R1 :: t", a), ("R2 :: t", b)], [5, 0, 9])
    assert [row.neuron_id for row in joined.rows] == [5, 0]
    assert np.isnan(joined.rows[0].values[1]).all()
    assert joined.rows[0].values[1].shape == (2,)
    assert joined.missing == {"R1": [9], "R2": [5, 9]}
    assert "left empty there" in joined.missing_note()
    assert joined.same_dt
    assert joined.units == "dF/F"

    data = joined.to_trace_data("dF/F")
    np.testing.assert_array_equal(data.indices, [0, 5])  # sorted by id
    assert data.dt == pytest.approx(0.1)
    assert data.units == "dF/F"
    np.testing.assert_array_equal(data.traces[0], np.r_[a.traces[0], b.traces[0]])
    np.testing.assert_array_equal(data.traces[1, :3], a.traces[2])
    assert np.isnan(data.traces[1, 3:]).all()


def test_concatenation_empty_and_no_rows() -> None:
    assert build_trace_concatenation([], [0]).total_ms == 0.0
    joined = build_trace_concatenation([("A :: t", _trace("t", [0], 3, 0.1))], [4])
    assert joined.rows == []
    assert joined.to_trace_data("t").traces.shape == (0, 3)


def test_shared_dt() -> None:
    assert shared_dt([]) is None
    assert shared_dt([0.1, 0.1]) == pytest.approx(0.1)
    assert shared_dt([1 / 30, 1 / 30]) == pytest.approx(1 / 30)
    assert shared_dt([0.1, 0.2]) is None


def test_concatenation_names() -> None:
    names = ["Rep 1 :: dF/F", "Rep 2 :: dF/F"]
    assert concatenation_group_name(names) == "Concatenated: Rep 1 + Rep 2"
    assert (
        concatenation_group_name(names, ["Concatenated: Rep 1 + Rep 2"])
        == "Concatenated: Rep 1 + Rep 2 (2)"
    )
    many = [f"Rep {i} :: dF/F" for i in range(1, 7)]
    assert (
        concatenation_group_name(many) == "Concatenated: Rep 1 + … + Rep 6 (6 arrays)"
    )
    # Full names never carry the group separator into the new group name.
    assert (
        concatenation_group_name(["A :: x", "B :: y"]) == "Concatenated: A / x + B / y"
    )
    assert concatenation_group_name([]) == "Concatenated"
    assert concatenation_data_name(names) == "dF/F"
    assert concatenation_data_name(["A :: x", "B :: y"]) == "traces"
    assert unique_name("a", ["a", "a (2)"]) == "a (3)"


def test_group_name_problem() -> None:
    assert group_name_problem("New", ["Old"]) == ""
    assert "Enter a name" in group_name_problem("  ", [])
    assert "must not contain" in group_name_problem("A :: b", [])
    assert "already exists" in group_name_problem("Old", ["Old"])
    assert "already exists" in group_name_problem("Simulation", [])


# -- Axis linking ------------------------------------------------------------


def test_axis_link_group_propagates_only_between_locked_members() -> None:
    group: AxisLinkGroup[object] = AxisLinkGroup()
    a, b, c = object(), object(), object()
    group.add(a, lock_time=True, lock_value=False)
    group.add(b, lock_time=True, lock_value=True)
    group.add(c, lock_time=False, lock_value=True)
    assert group.members == [a, b, c]

    reached: list[object] = []
    assert group.propagate(a, TIME_AXIS, reached.append) == [b]
    assert reached == [b]
    assert group.targets(a, VALUE_AXIS) == []  # a's value axis is unlocked
    assert group.targets(b, VALUE_AXIS) == [c]
    assert group.reference(a, TIME_AXIS) is b

    group.set_locked(b, TIME_AXIS, False)
    assert group.targets(a, TIME_AXIS) == []
    assert group.reference(a, TIME_AXIS) is None
    group.set_locked(object(), TIME_AXIS, True)  # unknown member: ignored

    group.remove(c)
    group.remove(c)
    assert group.targets(b, VALUE_AXIS) == []
    assert not group.is_locked(c, VALUE_AXIS)


def test_axis_link_group_does_not_echo() -> None:
    group: AxisLinkGroup[object] = AxisLinkGroup()
    a, b = object(), object()
    group.add(a, lock_time=True, lock_value=False)
    group.add(b, lock_time=True, lock_value=False)
    calls: list[object] = []

    def apply(target: object) -> None:
        calls.append(target)
        # The target reports its own range change: it must not bounce back.
        assert group.propagate(target, TIME_AXIS, calls.append) == []

    assert group.propagate(a, TIME_AXIS, apply) == [b]
    assert calls == [b]


def test_cascade_offsets() -> None:
    assert cascade_offsets(3, step=10) == [(0, 0), (10, 10), (20, 20)]
    assert cascade_offsets(0) == []
