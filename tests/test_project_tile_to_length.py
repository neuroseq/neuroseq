from __future__ import annotations

import brian2.numpy_ as np
import pytest

from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def _make_stim(name: str, values: np.ndarray, dt: float = 0.001) -> StimArray:
    return StimArray(
        name=name,
        stim_type=StimType.Excitation,
        values=values.astype(np.float32),
        units="mV",
        dt=dt,
    )


def test_tile_to_length_extends_short_stim() -> None:
    project = Project()
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    project.add_stim_array(_make_stim("stim", values))

    project.tile_stim_arrays_to_length(0.25)

    tiled = project.stim_arrays["stim"].values
    assert tiled.shape == (250, 3)
    assert np.allclose(tiled[:100], values)
    assert np.allclose(tiled[100:200], values)
    assert np.allclose(tiled[200:250], values[:50])


def test_tile_to_length_is_noop_when_already_long_enough() -> None:
    project = Project()
    values = np.arange(600, dtype=np.float32).reshape(200, 3)
    project.add_stim_array(_make_stim("stim", values.copy()))

    project.tile_stim_arrays_to_length(0.1)

    tiled = project.stim_arrays["stim"].values
    assert tiled.shape == (200, 3)
    assert np.allclose(tiled, values)


def test_tile_to_length_handles_exact_multiple() -> None:
    project = Project()
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    project.add_stim_array(_make_stim("stim", values))

    project.tile_stim_arrays_to_length(0.2)

    tiled = project.stim_arrays["stim"].values
    assert tiled.shape == (200, 3)
    assert np.allclose(tiled, np.tile(values, (2, 1)))


def test_tile_to_length_applies_to_all_stims() -> None:
    project = Project()
    project.add_stim_array(_make_stim("a", np.ones((100, 3), dtype=np.float32)))
    project.add_stim_array(_make_stim("b", np.ones((100, 2), dtype=np.float32) * 2.0))

    project.tile_stim_arrays_to_length(0.25)

    assert project.stim_arrays["a"].values.shape == (250, 3)
    assert project.stim_arrays["b"].values.shape == (250, 2)


def test_tile_to_length_rejects_negative_target() -> None:
    project = Project()
    project.add_stim_array(_make_stim("stim", np.zeros((100, 3), dtype=np.float32)))

    with pytest.raises(ValueError, match="target_seconds must be >= 0"):
        project.tile_stim_arrays_to_length(-1.0)


def test_tile_to_length_skips_empty_arrays() -> None:
    project = Project()
    project.add_stim_array(_make_stim("stim", np.zeros((0, 3), dtype=np.float32)))

    project.tile_stim_arrays_to_length(0.25)

    assert project.stim_arrays["stim"].values.shape == (0, 3)
