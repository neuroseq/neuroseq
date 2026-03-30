from __future__ import annotations

import pathlib

import pytest

import neuroseq.example_projects as example_projects
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
