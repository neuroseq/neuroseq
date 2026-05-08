"""Unit tests for `Project` external-simulation state and persistence."""

from __future__ import annotations

import textwrap

from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions


def _full_options() -> SaveLoadOptions:
    return SaveLoadOptions(True, True, True, True)


def test_default_state_is_empty():
    project = Project()
    assert project.external_simulate_path is None
    assert project.external_simulate_params == {}
    assert project.external_simulate_function is None


def test_load_neuroseq_sentinel_resolves_to_callable():
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.load_external_simulator()
    assert callable(project.external_simulate_function)
    assert (
        project.external_simulate_function.__module__ == "neuroseq.example_models.elise"
    )


def test_load_filesystem_path(tmp_path):
    sim_path = tmp_path / "fake_simulate.py"
    sim_path.write_text(
        textwrap.dedent(
            """
            def simulate(stims_dict, simulation_time, params=None):
                return ("called", simulation_time, params)
            """
        )
    )
    project = Project()
    project.external_simulate_path = str(sim_path)
    project.load_external_simulator()
    assert project.external_simulate_function is not None
    result = project.external_simulate_function({}, 1.0, params={"x": 5})
    assert result == ("called", 1.0, {"x": 5})


def test_load_clears_function_when_path_is_none():
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.load_external_simulator()
    assert project.external_simulate_function is not None

    project.external_simulate_path = None
    project.load_external_simulator()
    assert project.external_simulate_function is None


def test_params_flat_strips_metadata():
    project = Project()
    project.external_simulate_params = {
        "alpha": {
            "value": 1.5,
            "default": 1.0,
            "type": "float",
            "group": "G",
            "description": "...",
        },
        "beta": {
            "value": 7,
            "default": 0,
            "type": "int",
            "group": "G",
            "description": "...",
        },
    }
    flat = project.external_simulate_params_flat()
    assert flat == {"alpha": 1.5, "beta": 7}


def test_metadata_roundtrip_preserves_external_simulate_state(tmp_path):
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = {
        "num_lat": {
            "value": 12,
            "default": 50,
            "type": "int",
            "group": "Network",
            "description": "Latent neurons.",
        },
    }

    out = tmp_path / "roundtrip.nseq"
    project.save(out, _full_options())

    reloaded = Project(out)
    assert reloaded.external_simulate_path == "neuroseq:elise"
    assert reloaded.external_simulate_params == project.external_simulate_params
    # function is reconstructed transparently from the path
    assert callable(reloaded.external_simulate_function)


def test_metadata_roundtrip_omits_external_when_unset(tmp_path):
    project = Project()
    out = tmp_path / "roundtrip_empty.nseq"
    project.save(out, _full_options())

    reloaded = Project(out)
    assert reloaded.external_simulate_path is None
    assert reloaded.external_simulate_params == {}
    assert reloaded.external_simulate_function is None


def test_metadata_load_failure_does_not_break_load(tmp_path, monkeypatch):
    project = Project()
    project.external_simulate_path = "neuroseq:not_a_real_module"
    out = tmp_path / "roundtrip_bad.nseq"
    project.save(out, _full_options())

    # Reloading should warn but not raise; function ends up None.
    reloaded = Project(out)
    assert reloaded.external_simulate_path == "neuroseq:not_a_real_module"
    assert reloaded.external_simulate_function is None
