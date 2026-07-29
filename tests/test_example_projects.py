from __future__ import annotations

import pathlib

import brian2 as b2
import brian2.numpy_ as np
import pytest

import neuroseq.example_projects as example_projects
from neuroseq.example_projects import default_project
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions

project_generators = [
    pytest.param(project_generator, id=f"{section}/{name}")
    for section, section_projects in example_projects.example_projects.items()
    for name, project_generator in section_projects.items()
]


@pytest.mark.parametrize("project_generator", project_generators)
def test_example_projects(project_generator, tmp_path: pathlib.Path) -> None:
    project = project_generator()
    # do a short simulation
    project.run_simulation(0.001)
    # check we can save and re-load the model
    save_options = SaveLoadOptions(True, True, True, True)
    project.save(tmp_path / "model.nseq", save_options)
    model2 = Project(tmp_path / "model.nseq")
    # check saved model matches original
    assert list(model2.stim_arrays.keys()) == list(project.stim_arrays.keys())
    assert list(model2.b2_model.parameters.keys()) == list(
        project.b2_model.parameters.keys()
    )
    assert list(model2.b2_model.neurongroups.keys()) == list(
        project.b2_model.neurongroups.keys()
    )
    assert list(model2.b2_model.synapses.keys()) == list(
        project.b2_model.synapses.keys()
    )
    assert list(model2.b2_model.monitors.keys()) == list(
        project.b2_model.monitors.keys()
    )
    assert list(model2.b2_model.initials.keys()) == list(
        project.b2_model.initials.keys()
    )
    assert list(model2.b2_model.functions.keys()) == list(
        project.b2_model.functions.keys()
    )


def test_default_project_uses_10_neurons_and_sequential_stimulus() -> None:
    project = default_project()

    stimulus = project.stim_arrays["stimulus"]
    assert stimulus.values.shape == (1180, 10)
    assert project.b2_model.neurongroups["P"]["neuronnumber"] == 10

    pulse_n = int(round(0.100 / stimulus.dt))
    gap_n = int(round(0.018 / stimulus.dt))
    for i in range(stimulus.values.shape[1]):
        t0 = i * (pulse_n + gap_n)
        assert np.all(stimulus.values[t0 : t0 + pulse_n, i] == 1.0)
        if t0 > 0:
            assert np.all(stimulus.values[t0 - gap_n : t0, i] == 0.0)
        for j in range(stimulus.values.shape[1]):
            if j != i:
                assert np.all(stimulus.values[t0 : t0 + pulse_n, j] == 0.0)


def test_hh_stdp_example_is_two_neuron_and_potentiates() -> None:
    project = example_projects.hh_neuron_with_plasticity()
    model = project.b2_model

    # 2-neuron layout: neuron 0 excitatory/pre (Pe), neuron 1 inhibitory/post (Pi)
    assert model.neurongroups["P"]["neuronnumber"] == 2
    for name, sa in project.stim_arrays.items():
        assert sa.values.shape[1] == 2, name

    model.assemble_network()
    assert model.network["Pe"].N == 1
    assert model.network["Pi"].N == 1
    # the plastic Ce synapse wires the single 0 -> 1 pair
    assert len(model.network["Ce"]) == 1

    # the pre-before-post pairing protocol potentiates the plastic weight
    project.run_simulation(project.suggested_simulation_time_seconds())
    we = np.asarray(model.network["synapse"].we).ravel() / float(b2.nS)
    assert we[-1] > we[0]
