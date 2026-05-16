from __future__ import annotations

import brian2.numpy_ as np
import pytest

from neuroseq.example_projects import _build_fuer_elise_opsin_stims
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_timing import DEFAULT_STIM_DT_SECONDS


def _make_stim(name: str, values: np.ndarray, dt: float = 0.001) -> StimArray:
    return StimArray(
        name=name,
        stim_type=StimType.Excitation,
        values=values.astype(np.float32),
        units="mV",
        dt=dt,
    )


def test_suggested_time_for_fully_filled_stim() -> None:
    project = Project()
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    project.add_stim_array(_make_stim("stim", values))

    assert project.suggested_simulation_time_seconds() == pytest.approx(0.1)


def test_suggested_time_includes_constant_trailing_tail() -> None:
    """Trailing zeros (e.g. from a manual Pad) ARE included in the suggested
    time so that Pad+Run is equivalent to a built-in 'ring-out' workflow:
    the simulator reads the entire stim, including the trailing zeros."""
    project = Project()
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    values[50:] = 0.0
    project.add_stim_array(_make_stim("stim", values))

    assert project.suggested_simulation_time_seconds() == pytest.approx(0.1)


def test_suggested_time_full_length_when_all_rows_identical() -> None:
    project = Project()
    values = np.zeros((100, 3), dtype=np.float32)
    project.add_stim_array(_make_stim("stim", values))

    assert project.suggested_simulation_time_seconds() == pytest.approx(0.1)


def test_suggested_time_takes_max_across_arrays() -> None:
    project = Project()
    short_values = np.arange(240, dtype=np.float32).reshape(80, 3)
    long_values = np.arange(360, dtype=np.float32).reshape(120, 3)
    project.add_stim_array(_make_stim("short", short_values))
    project.add_stim_array(_make_stim("long", long_values))

    # Longest stim wins: 120 samples × 0.001 dt = 0.12 s.
    assert project.suggested_simulation_time_seconds() == pytest.approx(0.12)


def test_suggested_time_no_stim_arrays_defaults_to_one_second() -> None:
    project = Project()

    assert project.suggested_simulation_time_seconds() == pytest.approx(1.0)


def test_suggested_time_no_stim_arrays_uses_positive_max_time_secs() -> None:
    project = Project()
    project.max_time_secs = 2.5

    assert project.suggested_simulation_time_seconds() == pytest.approx(2.5)


def test_suggested_time_for_fuer_elise_default_is_one_point_nine_two_seconds() -> None:
    project = Project()
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims()
    project.add_stim_array(
        _make_stim(
            "opsin_exc",
            stim_exc,
            dt=DEFAULT_STIM_DT_SECONDS,
        )
    )
    project.add_stim_array(
        _make_stim(
            "opsin_inh",
            stim_inh,
            dt=DEFAULT_STIM_DT_SECONDS,
        )
    )

    assert project.suggested_simulation_time_seconds() == pytest.approx(1.92)


def test_suggested_time_includes_single_channel_uniform_tail() -> None:
    """Single-channel uniform trailing zeros are not trimmed either —
    same Pad+Run equivalence as the multi-channel case above."""
    project = Project()
    values = np.zeros((100, 1), dtype=np.float32)
    values[:30] = 1.0
    project.add_stim_array(_make_stim("stim", values))

    assert project.suggested_simulation_time_seconds() == pytest.approx(0.1)
