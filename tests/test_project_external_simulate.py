"""Unit tests for `Project` external-simulation state and persistence."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

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
    project.load_NN()
    assert project.external_simulator is not None
    assert callable(project.external_simulate_function)
    assert (
        project.external_simulator.__class__.__module__
        == "neuroseq.example_models.elise"
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
    project.load_NN()
    assert project.external_simulate_function is not None
    result = project.external_simulate_function({}, 1.0, params={"x": 5})
    assert result == ("called", 1.0, {"x": 5})


def test_load_clears_function_when_path_is_none():
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.load_NN()
    assert project.external_simulate_function is not None

    project.external_simulate_path = None
    project.load_NN()
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


def test_load_alias_elise_simulate_maps_to_elise(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    class _Sim:
        def simulate(self, *args, **kwargs):
            return ("ok", args, kwargs)

    simulator = _Sim()

    def _fake_import(name: str):
        calls.append(name)
        return SimpleNamespace(get_simulator=lambda: simulator)

    monkeypatch.setattr("neuroseq.project.importlib.import_module", _fake_import)

    project = Project()
    project.external_simulate_path = "neuroseq:elise_simulate"
    project.load_NN()

    assert calls == ["neuroseq.example_models.elise"]
    assert project.external_simulator is simulator
    assert project.external_simulate_function is not None
    assert project.external_simulate_function.__self__ is simulator
    assert project.external_simulate_function.__func__ is _Sim.simulate


def test_load_neuroseq_module_with_NN_class(monkeypatch: pytest.MonkeyPatch):
    class _Backend:
        def simulate(self, *args, **kwargs):
            return ("class-backend", args, kwargs)

    monkeypatch.setattr(
        "neuroseq.project.importlib.import_module",
        lambda name: SimpleNamespace(NN=_Backend),
    )

    project = Project()
    project.external_simulate_path = "neuroseq:dummy"
    project.load_NN()

    assert isinstance(project.external_simulator, _Backend)
    assert project.external_simulate_function is not None
    assert project.external_simulate_function.__self__ is project.external_simulator
    assert project.external_simulate_function.__func__ is _Backend.simulate


def test_load_neuroseq_module_with_simulate_fallback(monkeypatch: pytest.MonkeyPatch):
    def _simulate(*args, **kwargs):
        return ("module-sim", args, kwargs)

    monkeypatch.setattr(
        "neuroseq.project.importlib.import_module",
        lambda name: SimpleNamespace(simulate=_simulate),
    )

    project = Project()
    project.external_simulate_path = "neuroseq:dummy"
    project.load_NN()

    assert project.external_simulator is None
    assert project.external_simulate_function is _simulate


def test_load_neuroseq_module_without_entrypoint_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "neuroseq.project.importlib.import_module",
        lambda name: SimpleNamespace(),
    )

    project = Project()
    project.external_simulate_path = "neuroseq:dummy"
    with pytest.raises(AttributeError, match="has no simulator entry point"):
        project.load_NN()


def test_load_external_simulator_wrapper_calls_load_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Project()
    called: list[str] = []
    monkeypatch.setattr(project, "load_NN", lambda: called.append("load_NN"))

    project.load_external_simulator()
    assert called == ["load_NN"]
