"""Smoke tests for the PyNN AdEx external-simulation example project.

Building the project must work without ``pyNN`` (the backend is imported lazily
only when the simulation runs), so only the simulate test skips when ``pyNN`` is
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_projects import _build_pynn_stim_pattern
from neuroseq.example_projects import example_projects


def test_pynn_demo_example_project_registered() -> None:
    assert "Network" in example_projects
    assert "PyNN AdEx Neuromorphic Network" in example_projects["Network"]


def test_pynn_demo_example_project_builds_without_pynn() -> None:
    factory = example_projects["Network"]["PyNN AdEx Neuromorphic Network"]
    project = factory()

    assert project.project_name == "PyNN AdEx Neuromorphic Network"
    assert project.project_section == "Network"
    assert project.external_simulate_path == "neuroseq:pynn_demo"
    assert callable(project.external_simulate_function)
    assert project.external_simulate_params["backend"]["default"] == "brian2"

    assert "opsin" in project.stim_arrays
    stim = project.stim_arrays["opsin"]
    assert stim.values.shape == (3200, 16)  # 2 sweeps x 16 neurons x 100 ms
    assert stim.dt == 0.001


def test_pynn_demo_example_simulate_runs() -> None:
    pytest.importorskip("pyNN.brian2", reason="pyNN not installed")

    factory = example_projects["Network"]["PyNN AdEx Neuromorphic Network"]
    project = factory()
    stim = project.stim_arrays["opsin"]
    assert project.external_simulate_function is not None

    stims_dict = {"opsin": {"array": stim.values, "dt": stim.dt}}
    exp = project.external_simulate_function(
        stims_dict,
        simulation_time=0.3,
        params=project.external_simulate_params_flat(),
    )

    by_name = {d.name: d for d in exp.data}
    assert set(by_name) == {"pynn_spikes", "pynn_membrane_potential"}
    assert isinstance(by_name["pynn_spikes"], SpikeData)
    traces = by_name["pynn_membrane_potential"]
    assert isinstance(traces, TraceData)
    assert traces.traces.shape[0] == 16
    # The rolling stimulation must evoke spikes.
    assert by_name["pynn_spikes"].spike_times.size > 0


def test_pynn_stim_pattern_sweeps_and_repeats() -> None:
    pattern = _build_pynn_stim_pattern(
        tfinal_ms=800, n_neurons=4, slot_ms=100, active=1.0
    )
    assert pattern.shape == (800, 4)

    # One full sweep is 4 x 100 = 400 ms and should repeat.
    assert np.allclose(pattern[0:400, :], pattern[400:800, :])

    # In each 100 ms slot exactly one neuron is active.
    for k in range(4):
        slot = pattern[k * 100 : (k + 1) * 100, :]
        assert np.allclose(slot[:, k], 1.0)
        for j in range(4):
            if j != k:
                assert np.allclose(slot[:, j], 0.0)
