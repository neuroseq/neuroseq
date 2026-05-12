"""Smoke tests for the ELISE example project."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neuroseq.example_projects import _build_elise_sequence_pattern
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
    assert stim.values.shape == (5000, 4)
    assert stim.dt == 0.001

    # Sanity check: at t=0..50ms only neuron 0 should be at -55 mV.
    first_pulse = stim.values[0:50, :]
    assert np.allclose(first_pulse[:, 0], -55.0)
    assert np.allclose(first_pulse[:, 1:], -70.0)


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
    n_total = 4 + 8  # n_vis + num_lat
    n_steps_expected = int(round(0.1 * 1000.0 / flat_params["dt_ms"]))
    assert by_name["somatic_potential_u"].traces.shape == (n_total, n_steps_expected)


def test_sequence_pattern_cycles_correctly():
    pattern = _build_elise_sequence_pattern(
        tfinal_ms=400, n_vis=4, slot_ms=50, rest_mv=-70.0, active_mv=-55.0
    )
    assert pattern.shape == (400, 4)
    # Cycle is 200 ms (4 * 50). Verify second cycle replicates the first.
    assert np.allclose(pattern[0:200, :], pattern[200:400, :])
    # Each slot is fully active for its own neuron and resting for others.
    for k in range(4):
        slot = pattern[k * 50 : (k + 1) * 50, :]
        assert np.allclose(slot[:, k], -55.0)
        for j in range(4):
            if j != k:
                assert np.allclose(slot[:, j], -70.0)
