"""Smoke tests for the ELISE example project."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neuroseq.example_projects import _FUER_ELISE_PHI_EXC_ACTIVE
from neuroseq.example_projects import _FUER_ELISE_PHI_EXC_REST
from neuroseq.example_projects import _FUER_ELISE_PHI_INH_ACTIVE
from neuroseq.example_projects import _FUER_ELISE_PHI_INH_REST
from neuroseq.example_projects import _build_fuer_elise_opsin_stims
from neuroseq.example_projects import example_projects
from neuroseq.utils import detect_period


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
    assert "opsin_exc" in project.stim_arrays
    assert "opsin_inh" in project.stim_arrays

    stim_exc = project.stim_arrays["opsin_exc"]
    stim_inh = project.stim_arrays["opsin_inh"]
    assert stim_exc.values.shape == (1920, 13)
    assert stim_inh.values.shape == (1920, 13)
    assert stim_exc.dt == 0.001
    assert stim_inh.dt == 0.001

    first_exc = stim_exc.values[0:10, :]
    first_inh = stim_inh.values[0:10, :]
    assert np.allclose(first_exc[:, 0], _FUER_ELISE_PHI_EXC_ACTIVE)
    assert np.allclose(first_exc[:, 1:], _FUER_ELISE_PHI_EXC_REST)
    assert np.allclose(first_inh[:, 0], _FUER_ELISE_PHI_INH_ACTIVE)
    assert np.allclose(first_inh[:, 1:], _FUER_ELISE_PHI_INH_REST)


@pytest.mark.skipif(
    importlib.util.find_spec("elise") is None,
    reason="ELISE simulation test requires the private 'elise' package.",
)
def test_elise_example_simulate_runs():
    """End-to-end: build the example, then call simulate via the project."""
    factory = example_projects["Network"]["ELISE Sequence Learning Network"]
    project = factory()
    stim_exc = project.stim_arrays["opsin_exc"]
    stim_inh = project.stim_arrays["opsin_inh"]

    stims_dict = {
        "opsin_exc": {"array": stim_exc.values, "dt": stim_exc.dt},
        "opsin_inh": {"array": stim_inh.values, "dt": stim_inh.dt},
    }
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


def test_fuer_elise_opsin_stims_cycle_correctly():
    """Two full 240-ms cycles of the 24-event intro should be identical."""
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims(tfinal_ms=480, event_ms=10)
    assert stim_exc.shape == (480, 13)
    assert stim_inh.shape == (480, 13)
    np.testing.assert_array_equal(stim_exc[0:240, :], stim_exc[240:480, :])
    np.testing.assert_array_equal(stim_inh[0:240, :], stim_inh[240:480, :])


def test_fuer_elise_opsin_stims_first_events():
    """Check the first few events match the upstream score
    (`fuer_elise_original.txt`): pitches 0, 1, 0, 1, 0, 4, 2, 3, then a chord."""
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims(tfinal_ms=300, event_ms=10)

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
        slot_exc = stim_exc[event_idx * 10 : (event_idx + 1) * 10, :]
        slot_inh = stim_inh[event_idx * 10 : (event_idx + 1) * 10, :]
        for pitch in range(13):
            expected_exc = (
                _FUER_ELISE_PHI_EXC_ACTIVE
                if pitch in active_pitches
                else _FUER_ELISE_PHI_EXC_REST
            )
            expected_inh = (
                _FUER_ELISE_PHI_INH_ACTIVE
                if pitch in active_pitches
                else _FUER_ELISE_PHI_INH_REST
            )
            assert np.allclose(slot_exc[:, pitch], expected_exc), (
                f"event {event_idx} pitch {pitch}: "
                f"expected {expected_exc}, got {slot_exc[0, pitch]}"
            )
            assert np.allclose(slot_inh[:, pitch], expected_inh), (
                f"event {event_idx} pitch {pitch}: "
                f"expected {expected_inh}, got {slot_inh[0, pitch]}"
            )


def test_fuer_elise_opsin_stims_default_are_eight_full_cycles():
    """Default (no args) is 8 × 240 ms cycles, no trailing rest."""
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims()
    assert stim_exc.shape == (1920, 13)
    assert stim_inh.shape == (1920, 13)
    cycle_exc = stim_exc[:240]
    cycle_inh = stim_inh[:240]
    for c in range(1, 8):
        np.testing.assert_array_equal(stim_exc[c * 240 : (c + 1) * 240], cycle_exc)
        np.testing.assert_array_equal(stim_inh[c * 240 : (c + 1) * 240], cycle_inh)


def test_fuer_elise_opsin_stims_default_are_detected_as_periodic():
    """Both opsin stims pulse on/off together at each event slot
    (lights-off rest, phi_m active), so the 240-ms cycle is detectable
    on either array."""
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims()
    assert detect_period(stim_exc) == 240
    assert detect_period(stim_inh) == 240
