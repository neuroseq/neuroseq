from __future__ import annotations

import inspect
import pathlib

import pytest

import neuroseq.example_projects as example_projects
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions


def _is_project_generator(obj) -> bool:
    return (
        inspect.isfunction(obj)
        and obj.__module__ == example_projects.__name__
        and obj.__name__ != "register_example_project"
    )


project_generators = [
    f for _, f in inspect.getmembers(example_projects, _is_project_generator)
]


@pytest.mark.parametrize(
    "project_generator",
    project_generators,
    ids=lambda f: f.__name__,
)
def test_example_projects(project_generator, tmp_path: pathlib.Path) -> None:
    project = project_generator()
    # do a short simulation
    project.run_simulation(0.01)
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
