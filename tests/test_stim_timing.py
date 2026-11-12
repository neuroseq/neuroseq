from __future__ import annotations

import numpy as np
import pytest

from neuroseq.stimulation.stim_timing import ceil_sample_position
from neuroseq.stimulation.stim_timing import floor_sample_position
from neuroseq.stimulation.stim_timing import format_milliseconds
from neuroseq.stimulation.stim_timing import milliseconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import padded_length_seconds
from neuroseq.stimulation.stim_timing import remap_stim_values_to_dt
from neuroseq.stimulation.stim_timing import resample_zero_order_hold
from neuroseq.stimulation.stim_timing import round_sample_position
from neuroseq.stimulation.stim_timing import samples_to_milliseconds
from neuroseq.stimulation.stim_timing import samples_to_seconds
from neuroseq.stimulation.stim_timing import seconds_to_samples_ceil
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import snap_milliseconds_to_dt
from neuroseq.stimulation.stim_timing import stim_dt_matches
from neuroseq.stimulation.stim_timing import validate_stim_dt_seconds
from neuroseq.stimulation.stim_timing import zero_order_hold_sample_count
from neuroseq.stimulation.stim_timing import zero_order_hold_source_indices


@pytest.mark.parametrize("dt_seconds", [0.001, 0.0005208125, 2])
def test_validate_stim_dt_seconds_accepts_positive_finite_values(
    dt_seconds: float | int,
) -> None:
    assert validate_stim_dt_seconds(dt_seconds) == pytest.approx(float(dt_seconds))


@pytest.mark.parametrize(
    "dt_seconds",
    [0.0, -0.001, float("nan"), float("inf"), -float("inf")],
)
def test_validate_stim_dt_seconds_rejects_non_positive_or_non_finite_values(
    dt_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        validate_stim_dt_seconds(dt_seconds)


def test_stim_dt_matches_uses_small_absolute_tolerance() -> None:
    assert stim_dt_matches(0.001, 0.001 + 5e-13)
    assert not stim_dt_matches(0.001, 0.001 + 2e-12)


def test_sample_duration_conversions() -> None:
    assert samples_to_seconds(4, 0.001) == pytest.approx(0.004)
    assert samples_to_milliseconds(4, 0.001) == pytest.approx(4.0)


def test_seconds_to_samples_helpers_round_as_expected() -> None:
    assert seconds_to_samples_ceil(0.0011, 0.001) == 2
    assert seconds_to_samples_nearest(0.00149, 0.001) == 1
    assert seconds_to_samples_nearest(0.0015, 0.001) == 2
    assert seconds_to_samples_nearest(-1.0, 0.001) == 0


def test_milliseconds_to_samples_nearest_supports_non_standard_dt() -> None:
    dt_seconds = 0.0005208125

    assert milliseconds_to_samples_nearest(0.5208125, dt_seconds) == 1
    assert milliseconds_to_samples_nearest(1.0, dt_seconds) == 2


def test_snap_milliseconds_to_dt_rounds_and_reports_effective() -> None:
    assert snap_milliseconds_to_dt(15.0, 0.001) == (15, 15.0)
    assert snap_milliseconds_to_dt(15.25, 0.001) == (15, 15.0)
    assert snap_milliseconds_to_dt(15.5, 0.001) == (16, 16.0)
    assert snap_milliseconds_to_dt(15.6, 0.001) == (16, 16.0)
    assert snap_milliseconds_to_dt(0.2, 0.001) == (1, 1.0)


def test_format_milliseconds_preserves_precision_without_trailing_zeros() -> None:
    assert format_milliseconds(1.0) == "1"
    assert format_milliseconds(1.2300000) == "1.23"
    assert format_milliseconds(0.5208125) == "0.5208125"


def test_remap_stim_values_to_dt_returns_equal_copy_when_dt_matches() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    remapped = remap_stim_values_to_dt(
        values, old_dt_seconds=0.001, new_dt_seconds=0.001
    )

    assert np.array_equal(remapped, values)
    assert remapped is not values
    assert remapped.dtype == values.dtype


def test_remap_stim_values_to_dt_remaps_2d_to_coarser_grid() -> None:
    values = np.arange(10, dtype=np.float32).reshape(5, 2)

    remapped = remap_stim_values_to_dt(
        values, old_dt_seconds=0.001, new_dt_seconds=0.002
    )

    expected = values[[0, 2, 4], :]
    assert np.array_equal(remapped, expected)
    assert remapped.dtype == values.dtype


def test_remap_stim_values_to_dt_remaps_2d_to_finer_grid() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

    remapped = remap_stim_values_to_dt(
        values, old_dt_seconds=0.002, new_dt_seconds=0.001
    )

    expected = np.array(
        [
            [1.0, 10.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(remapped, expected)


def test_remap_stim_values_to_dt_preserves_empty_array_shape() -> None:
    values = np.zeros((0, 3), dtype=np.float32)

    remapped = remap_stim_values_to_dt(
        values, old_dt_seconds=0.001, new_dt_seconds=0.002
    )

    assert remapped.shape == (0, 3)
    assert remapped.dtype == values.dtype


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 3, 4), dtype=np.float32),
    ],
)
def test_remap_stim_values_to_dt_rejects_non_2d_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        remap_stim_values_to_dt(values, old_dt_seconds=0.001, new_dt_seconds=0.002)


# (old_dt, new_dt) as exact decimal strings, so the expectation can use
# rational arithmetic while the code under test sees the float values.
_RESAMPLE_GRIDS = [
    ("0.0001", "0.001"),  # 100 us -> 1 ms
    ("0.001", "0.0001"),
    ("0.00002", "0.001"),  # 20 us (50 kHz) -> 1 ms
    ("0.001", "0.0005208125"),  # 1 ms <-> SysCon2 tick
    ("0.0005208125", "0.001"),
    ("0.001", "0.0005"),
    ("0.0005", "0.001"),
    ("0.000033", "0.001"),
]


@pytest.mark.parametrize(("old_dt", "new_dt"), _RESAMPLE_GRIDS)
@pytest.mark.parametrize("n_old", [1, 7, 950, 1000, 8333, 20_000])
def test_zero_order_hold_matches_exact_rational_arithmetic(
    old_dt: str, new_dt: str, n_old: int
) -> None:
    from fractions import Fraction

    old, new = Fraction(old_dt), Fraction(new_dt)
    expected_count = -((-n_old * old) // new)  # exact ceil
    assert zero_order_hold_sample_count(n_old, float(old), float(new)) == (
        expected_count
    )
    expected_indices = [(k * new) // old for k in range(expected_count)]
    indices = zero_order_hold_source_indices(expected_count, float(old), float(new))
    assert indices.tolist() == expected_indices
    values = np.arange(n_old, dtype=np.float32)[:, np.newaxis]
    resampled = resample_zero_order_hold(values, float(old), float(new))
    assert resampled[:, 0].tolist() == [min(i, n_old - 1) for i in expected_indices]


@pytest.mark.parametrize(
    ("old_dt", "new_dt", "target_index", "source_index"),
    [
        (0.0001, 0.001, 11, 110),  # floor(109.99999999999999) used to give 109
        (0.001, 0.0005208125, 16_000, 8333),
        (0.0005208125, 0.001, 8333, 16_000),
    ],
)
def test_zero_order_hold_source_index_float_boundaries(
    old_dt: float, new_dt: float, target_index: int, source_index: int
) -> None:
    indices = zero_order_hold_source_indices(target_index + 1, old_dt, new_dt)
    assert indices[target_index] == source_index


def test_zero_order_hold_sample_count_float_boundary() -> None:
    # 950 * 20e-6 / 1e-3 == 19.000000000000004, which ceil() used to turn into 20.
    assert zero_order_hold_sample_count(950, 0.00002, 0.001) == 19
    assert seconds_to_samples_ceil(0.007, 0.001) == 7
    assert seconds_to_samples_nearest(0.0025, 0.001) == 3


def test_sample_position_rounding_helpers_tolerate_float_error() -> None:
    positions = np.array([109.99999999999999, 2.0000000000000004, 2.4999999999999996])
    assert floor_sample_position(positions).tolist() == [110, 2, 2]
    assert ceil_sample_position(positions).tolist() == [110, 2, 3]
    assert round_sample_position(positions).tolist() == [110, 2, 3]
    assert ceil_sample_position(2.1) == 3
    assert floor_sample_position(2.9) == 2


def test_resample_zero_order_hold_returns_new_arrays() -> None:
    values = np.ones((3, 2), dtype=np.float32)
    assert resample_zero_order_hold(values, 0.001, 0.001) is not values
    assert resample_zero_order_hold(np.zeros((0, 2)), 0.001, 0.002).shape == (0, 2)
    with pytest.raises(ValueError, match="2-dimensional"):
        remap_stim_values_to_dt(np.ones(3), 0.001, 0.002)


@pytest.mark.parametrize(
    ("text", "to_end_time", "expected"),
    [
        ("2", False, 3.0),
        ("0.5", False, 1.5),
        ("4", True, 4.0),
        ("1.0006", True, 1.0006),
    ],
)
def test_padded_length_seconds(text, to_end_time, expected) -> None:
    assert padded_length_seconds(
        1.0, text, to_end_time=to_end_time, dt_seconds=0.001
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "to_end_time", "message"),
    [
        ("abc", False, "Could not parse"),
        ("nan", True, "Could not parse"),
        ("0", False, "must be positive"),
        ("-1", False, "must be positive"),
        ("0.5", True, "not after the current end"),
        ("1.0004", True, "not after the current end"),
    ],
)
def test_padded_length_seconds_rejects(text, to_end_time, message) -> None:
    with pytest.raises(ValueError, match=message):
        padded_length_seconds(1.0, text, to_end_time=to_end_time, dt_seconds=0.001)
