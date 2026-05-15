"""Smoke tests for the ELISE example project."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neuroseq.example_projects import _build_fuer_elise_pattern
from neuroseq.example_projects import example_projects


def test_elise_example_project_registered():
    assert "Network" in example_projects
    assert "ELISE Sequence Learning Network" in example_projects["Network"]


def test_elise_example_project_builds():
    factory = example_projects["Network"]["ELISE Sequence Learning Network"]
    project = factory()

    assert project.project_name == "ELISE Sequence Learning Network"
    assert project.project_section == "Network"
    assert project.external_simulate_path == "neuroseq:elise"
    assert callable(project.external_simulate_function)
    assert "u_inp" in project.stim_arrays

    stim = project.stim_arrays["u_inp"]
    # 2 s pattern with 13 pitch neurons (Kriener et al., 2024 Fig. 2)
    assert stim.values.shape == (2000, 13)
    assert stim.dt == 0.001

    # First event (0..10 ms) activates pitch 0 (the opening E5 of Für Elise).
    first_event = stim.values[0:10, :]
    assert np.allclose(first_event[:, 0], -50.0)
    assert np.allclose(first_event[:, 1:], -70.0)


@pytest.mark.skipif(
    importlib.util.find_spec("elise") is None,
    reason="ELISE simulation test requires the private 'elise' package.",
)
def test_elise_example_simulate_runs():
    """End-to-end: build the example, then call simulate via the project."""
    factory = example_projects["Network"]["ELISE Sequence Learning Network"]
    project = factory()
    stim = project.stim_arrays["u_inp"]

    stims_dict = {"u_inp": {"array": stim.values, "dt": stim.dt}}
    flat_params = project.external_simulate_params_flat()
    # Smaller latent population for test speed.
    flat_params["num_lat"] = 8

    exp = project.external_simulate_function(
        stims_dict, simulation_time=0.1, params=flat_params
    )

    by_name = {d.name: d for d in exp.data}
    assert set(by_name) == {
        "somatic_potential_u",
        "dendritic_potential_v",
        "firing_rate_r",
        "dendritic_weights",
    }
    n_total = 13 + 8  # n_vis (13 Für Elise pitches) + num_lat
    n_steps_expected = int(round(0.1 * 1000.0 / flat_params["dt_ms"]))
    assert by_name["somatic_potential_u"].traces.shape == (n_total, n_steps_expected)


def test_fuer_elise_pattern_cycles_correctly():
    """Two full 240-ms cycles of the 24-event intro should be identical."""
    pattern = _build_fuer_elise_pattern(
        tfinal_ms=480, event_ms=10, rest_mv=-70.0, active_mv=-50.0
    )
    assert pattern.shape == (480, 13)
    np.testing.assert_array_equal(pattern[0:240, :], pattern[240:480, :])


def test_fuer_elise_pattern_first_events():
    """Check the first few events match the upstream score
    (`fuer_elise_original.txt`): pitches 0, 1, 0, 1, 0, 4, 2, 3, then a chord."""
    pattern = _build_fuer_elise_pattern(
        tfinal_ms=300, event_ms=10, rest_mv=-70.0, active_mv=-50.0
    )

    expected_events: list[tuple[int, ...]] = [
        (0,),
        (1,),
        (0,),
        (1,),
        (0,),
        (4,),
        (2,),
        (3,),
        (5, 11),  # first chord
    ]
    for event_idx, active_pitches in enumerate(expected_events):
        slot = pattern[event_idx * 10 : (event_idx + 1) * 10, :]
        for pitch in range(13):
            expected_v = -50.0 if pitch in active_pitches else -70.0
            assert np.allclose(slot[:, pitch], expected_v), (
                f"event {event_idx} pitch {pitch}: "
                f"expected {expected_v}, got {slot[0, pitch]}"
            )


def test_fuer_elise_pattern_default_is_two_seconds():
    """Calling with no args returns a 2-second, 13-pitch stim."""
    pattern = _build_fuer_elise_pattern()
    assert pattern.shape == (2000, 13)
    # 8 full 240-ms cycles + 80 ms of trailing rest at the end.
    trailing = pattern[1920:2000, :]
    assert np.allclose(trailing, -70.0)
