from __future__ import annotations

import numpy as np
import pytest

from neuroseq.stim_timing import format_milliseconds
from neuroseq.stim_timing import milliseconds_to_samples_nearest
from neuroseq.stim_timing import remap_stim_values_to_dt
from neuroseq.stim_timing import samples_to_milliseconds
from neuroseq.stim_timing import samples_to_seconds
from neuroseq.stim_timing import seconds_to_samples_ceil
from neuroseq.stim_timing import seconds_to_samples_nearest
from neuroseq.stim_timing import snap_milliseconds_to_dt
from neuroseq.stim_timing import stim_dt_matches
from neuroseq.stim_timing import validate_stim_dt_seconds


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
