import pytest

from neuroseq.stimulation.dynamic_input.stdp_window import classify_window
from neuroseq.stimulation.dynamic_input.stdp_window import delta_t_sweep
from neuroseq.stimulation.dynamic_input.stdp_window import interleave_by_magnitude
from neuroseq.stimulation.dynamic_input.stdp_window import percent_change


def test_delta_t_sweep_symmetric_grid_excludes_zero() -> None:
    assert delta_t_sweep(-100.0, 100.0, 5) == [-100.0, -50.0, 50.0, 100.0]


def test_delta_t_sweep_includes_zero_when_requested() -> None:
    assert delta_t_sweep(-100.0, 100.0, 5, include_zero=True) == [
        -100.0,
        -50.0,
        0.0,
        50.0,
        100.0,
    ]


def test_delta_t_sweep_sorted_and_counted() -> None:
    values = delta_t_sweep(-30.0, 90.0, 4)

    assert values == sorted(values)
    assert len(values) == 3
    assert values == pytest.approx([-30.0, 50.0, 90.0])
    assert delta_t_sweep(-10.0, 10.0, 0) == []
    assert delta_t_sweep(-10.0, 10.0, 1) == [0.0]


def test_percent_change_and_zero_baseline() -> None:
    assert percent_change(2.0, 3.0, 2.0) == pytest.approx(50.0)
    assert percent_change(3.0, 2.0, 2.0) == pytest.approx(-50.0)
    assert percent_change(3.0, 4.0, 0.0) == 0.0


def test_classify_window_sign() -> None:
    assert classify_window(20.0) == "LTP"
    assert classify_window(-20.0) == "LTD"
    assert classify_window(0.0) == "none"


def test_interleave_by_magnitude_alternates_sign() -> None:
    assert interleave_by_magnitude([-100.0, -50.0, 50.0, 100.0]) == [
        -50.0,
        50.0,
        -100.0,
        100.0,
    ]
