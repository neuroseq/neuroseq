import pytest

from neuroseq.visualization_support.time_format import format_time_ms
from neuroseq.visualization_support.time_format import time_decimals_for_span
from neuroseq.visualization_support.time_format import time_unit_for_span


@pytest.mark.parametrize(
    ("span_ms", "unit"),
    [(0.5, "ms"), (1999.0, "ms"), (2000.0, "s"), (900_000.0, "s"), (-5000.0, "s")],
)
def test_time_unit_for_span(span_ms: float, unit: str) -> None:
    assert time_unit_for_span(span_ms) == unit


@pytest.mark.parametrize(
    ("span_ms", "decimals"),
    [
        (900_000.0, 0),  # 15 min recording: whole seconds
        (100_000.0, 0),
        (20_000.0, 1),
        (2_000.0, 2),
        (1_000.0, 0),  # in ms from here on
        (50.0, 1),
        (5.0, 2),
        (0.5, 3),
        (0.001, 3),  # capped
        (0.0, 3),
        (float("nan"), 3),
    ],
)
def test_time_decimals_for_span(span_ms: float, decimals: int) -> None:
    assert time_decimals_for_span(span_ms) == decimals


@pytest.mark.parametrize(
    ("time_ms", "span_ms", "expected"),
    [
        # Zoomed out on a long recording: no more "123456ms".
        (123_456.0, 900_000.0, "123 s"),
        (123_456.0, 20_000.0, "123.5 s"),
        (1_234.5678, 1_000.0, "1235 ms"),
        # Zoomed in on single laser pulses: sub-millisecond detail.
        (1_234.5678, 5.0, "1234.57 ms"),
        (16.6, 0.5, "16.600 ms"),
        (-0.0001, 5.0, "0.00 ms"),
    ],
)
def test_format_time_ms(time_ms: float, span_ms: float, expected: str) -> None:
    assert format_time_ms(time_ms, span_ms) == expected


@pytest.mark.parametrize(
    ("time_ms", "span_ms", "expected"),
    [
        (500.0, 2_500.0, "0.5 s"),
        (250.0, 2_500.0, "0.25 s"),
        (4_000.0, 20_000.0, "4 s"),
        (0.0, 20_000.0, "0 s"),
        (120.0, 500.0, "120 ms"),
    ],
)
def test_format_time_ms_strip_zeros(
    time_ms: float, span_ms: float, expected: str
) -> None:
    assert format_time_ms(time_ms, span_ms, strip_zeros=True) == expected
