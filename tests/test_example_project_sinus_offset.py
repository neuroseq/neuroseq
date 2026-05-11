"""Smoke tests for the sinus-offset external simulation example project."""

from __future__ import annotations

import numpy as np

from neuroseq.data import TraceData
from neuroseq.example_projects import _build_sinus_offset_pattern
from neuroseq.example_projects import example_projects


def test_sinus_offset_example_project_registered() -> None:
    assert "Network" in example_projects
    assert "Sinus Offset External Simulator" in example_projects["Network"]


def test_sinus_offset_example_project_builds() -> None:
    factory = example_projects["Network"]["Sinus Offset External Simulator"]
    project = factory()

    assert project.project_name == "Sinus Offset External Simulator"
    assert project.project_section == "Network"
    assert project.external_simulate_path == "neuroseq:sinus_offset"
    assert callable(project.external_simulate_function)
    assert "offset" in project.stim_arrays

    stim = project.stim_arrays["offset"]
    assert stim.values.shape == (4000, 5)
    assert stim.dt == 0.001


def test_sinus_offset_example_simulate_runs() -> None:
    factory = example_projects["Network"]["Sinus Offset External Simulator"]
    project = factory()
    stim = project.stim_arrays["offset"]
    assert project.external_simulate_function is not None

    stims_dict = {"offset": {"array": stim.values, "dt": stim.dt}}
    exp = project.external_simulate_function(
        stims_dict,
        simulation_time=0.2,
        params=project.external_simulate_params_flat(),
    )

    by_name = {d.name: d for d in exp.data}
    assert set(by_name) == {"sinusoid_membrane_potential", "stimulus_offset"}
    membrane = by_name["sinusoid_membrane_potential"]
    assert isinstance(membrane, TraceData)
    assert membrane.traces.shape[0] == 5


def test_sinus_offset_pattern_cycles() -> None:
    pattern = _build_sinus_offset_pattern(
        tfinal_ms=1000,
        n_units=5,
        slot_ms=100,
        active_offset_mv=3.0,
    )
    assert pattern.shape == (1000, 5)

    # One full cycle is 500 ms and should repeat.
    assert np.allclose(pattern[0:500, :], pattern[500:1000, :])

    for k in range(5):
        slot = pattern[k * 100 : (k + 1) * 100, :]
        assert np.allclose(slot[:, k], 3.0)
        for j in range(5):
            if j != k:
                assert np.allclose(slot[:, j], 0.0)
