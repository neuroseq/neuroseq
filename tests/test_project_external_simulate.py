"""Unit tests for `Project` external-simulation state and persistence."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq.example_models.elise import _matching_potential_to_opsins
from neuroseq.example_models.elise import _migrate_legacy_elise_stim
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


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


@pytest.mark.parametrize("has_migration", [False, True])
def test_archive_load_supports_optional_backend_migration(
    tmp_path, monkeypatch: pytest.MonkeyPatch, has_migration: bool
) -> None:
    project = Project()
    project.external_simulate_path = "neuroseq:custom_backend"
    project.add_stim_array(
        StimArray("input", StimType.External, np.ones((2, 1)), "1", 0.001)
    )
    archive = tmp_path / "custom_backend.nseq"
    project.save(archive, _full_options())
    migrated: list[Project] = []

    def migrate_project(loaded: Project) -> None:
        # The backend sees the restored stimulus data and archive version.
        assert loaded._loaded_schema_version == 2
        np.testing.assert_array_equal(
            loaded.stim_arrays["input"].values, np.ones((2, 1))
        )
        loaded.stim_arrays["input"].units = "mV"
        migrated.append(loaded)

    backend = SimpleNamespace(simulate=lambda *args, **kwargs: None)
    if has_migration:
        backend.migrate_project = migrate_project
    monkeypatch.setattr(
        "neuroseq.project.importlib.import_module",
        lambda name: SimpleNamespace(get_simulator=lambda: backend),
    )

    loaded = Project(archive)

    assert callable(loaded.external_simulate_function)
    assert migrated == ([loaded] if has_migration else [])
    assert loaded.stim_arrays["input"].units == ("mV" if has_migration else "1")
    assert not loaded.dirty


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
    result = project.external_simulate_function({}, 1.0, params={"x": 5})
    assert result == ("ok", ({}, 1.0), {"params": {"x": 5}})


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
    result = project.external_simulate_function({}, 1.0, params={"x": 5})
    assert result == ("class-backend", ({}, 1.0), {"params": {"x": 5}})


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


def test_legacy_u_inp_stim_migrated_to_opsin_pair_on_load() -> None:
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    u_inp = np.array([[-70.0, -50.0], [-60.0, -65.0]], dtype=np.float32)
    project.add_stim_array(
        StimArray(
            "u_inp",
            StimType.External,
            u_inp,
            "mV",
            0.001,
        )
    )

    _migrate_legacy_elise_stim(project)

    assert "u_inp" not in project.stim_arrays
    assert "opsin_exc" in project.stim_arrays
    assert "opsin_inh" in project.stim_arrays
    stim_exc = project.stim_arrays["opsin_exc"]
    stim_inh = project.stim_arrays["opsin_inh"]
    assert stim_exc.values.shape == u_inp.shape
    assert stim_inh.values.shape == u_inp.shape
    assert stim_exc.dt == pytest.approx(0.001)
    assert stim_inh.dt == pytest.approx(0.001)

    expected = _matching_potential_to_opsins(u_inp.astype(np.float64), 0.0, -75.0)
    assert expected is not None
    expected_exc, expected_inh = expected
    np.testing.assert_allclose(stim_exc.values, expected_exc)
    np.testing.assert_allclose(stim_inh.values, expected_inh)


def test_legacy_u_inp_archive_migrated_on_reload(tmp_path) -> None:
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.add_stim_array(
        StimArray(
            "u_inp",
            StimType.External,
            np.array([[-70.0], [-50.0]], dtype=np.float32),
            "mV",
            0.001,
        )
    )

    out = tmp_path / "legacy_elise.nseq"
    project.save(out, _full_options())

    reloaded = Project(out)
    assert "u_inp" not in reloaded.stim_arrays
    assert "opsin_exc" in reloaded.stim_arrays
    assert "opsin_inh" in reloaded.stim_arrays


def test_migration_is_idempotent() -> None:
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.add_stim_array(
        StimArray(
            "u_inp",
            StimType.External,
            np.array([[-70.0], [-50.0]], dtype=np.float32),
            "mV",
            0.001,
        )
    )

    _migrate_legacy_elise_stim(project)
    first_exc = np.array(project.stim_arrays["opsin_exc"].values, copy=True)
    first_inh = np.array(project.stim_arrays["opsin_inh"].values, copy=True)

    _migrate_legacy_elise_stim(project)

    np.testing.assert_array_equal(project.stim_arrays["opsin_exc"].values, first_exc)
    np.testing.assert_array_equal(project.stim_arrays["opsin_inh"].values, first_inh)


def test_migration_only_for_elise_external_path() -> None:
    project = Project()
    project.external_simulate_path = "neuroseq:sinus_offset"
    project.add_stim_array(
        StimArray(
            "u_inp",
            StimType.External,
            np.array([[-70.0], [-50.0]], dtype=np.float32),
            "mV",
            0.001,
        )
    )

    _migrate_legacy_elise_stim(project)

    assert "u_inp" in project.stim_arrays
    assert "opsin_exc" not in project.stim_arrays
    assert "opsin_inh" not in project.stim_arrays
