from __future__ import annotations

import builtins
import sys
import types
from typing import Any
from typing import cast

import numpy as np
import pytest

import neuroseq.example_models.elise as elise_module
from neuroseq.data import Experiment
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.example_models.elise import DEFAULT_PARAMS as ELISE_DEFAULT_PARAMS
from neuroseq.example_models.elise import Elise
from neuroseq.example_models.elise import _bool_param as elise_bool_param
from neuroseq.example_models.elise import _build_namespaces
from neuroseq.example_models.elise import _flat_defaults as elise_flat_defaults
from neuroseq.example_models.elise import _float_param as elise_float_param
from neuroseq.example_models.elise import _int_param as elise_int_param
from neuroseq.example_models.NN import NN
from neuroseq.example_models.sinus_offset import SinusOffset
from neuroseq.example_models.sinus_offset import _float_param as sinus_float_param
from neuroseq.example_models.sinus_offset import _stim_as_unit_time_series


class _DummyBackend(NN):
    def __init__(self) -> None:
        super().__init__(
            {
                "alpha": {
                    "value": 1,
                    "default": 1,
                    "type": "int",
                    "group": "G",
                    "description": ".",
                }
            }
        )
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self._simulated_time_s = 0.0

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params=None,
        progress=None,
    ) -> Experiment:
        self._simulated_time_s += simulation_time
        if progress is not None:
            progress(1.0)
        return Experiment("dummy", simulation_time, [], [])


def test_nn_base_helpers_and_not_implemented() -> None:
    backend = _DummyBackend()

    schema_copy = backend.default_params_schema()
    flat_copy = backend.flat_default_params()
    assert schema_copy["alpha"]["default"] == 1
    assert flat_copy == {"alpha": 1}

    schema_copy["alpha"]["default"] = 99
    flat_copy["alpha"] = 99
    assert backend.default_params_schema()["alpha"]["default"] == 1
    assert backend.flat_default_params() == {"alpha": 1}

    backend.update_params({"alpha": 2})
    assert backend.reset_calls == 1
    assert backend.flat_default_params() == {"alpha": 1}

    backend.update_params({"alpha": 2})
    assert backend.reset_calls == 1

    base = NN(schema_copy)
    with pytest.raises(NotImplementedError, match="NN.reset"):
        base.reset()
    with pytest.raises(NotImplementedError, match="NN.simulate"):
        base.simulate({}, 0.1)


def test_elise_param_helpers_and_namespace_builder() -> None:
    defaults = elise_flat_defaults()
    assert defaults["num_lat"] == 50
    assert elise_int_param({"k": True}, "k") == 1
    assert elise_int_param({"k": 4.9}, "k") == 4
    assert elise_float_param({"k": True}, "k") == 1.0
    assert elise_float_param({"k": "1.25"}, "k") == 1.25
    assert elise_bool_param({"k": "yes"}, "k") is True
    assert elise_bool_param({"k": 0}, "k") is False

    with pytest.raises(TypeError):
        elise_int_param(cast(Any, {"k": object()}), "k")
    with pytest.raises(TypeError):
        elise_float_param(cast(Any, {"k": object()}), "k")
    with pytest.raises(TypeError):
        elise_bool_param(cast(Any, {"k": object()}), "k")

    network_params, neuron_params, weight_params = _build_namespaces(
        defaults, num_vis=4
    )
    assert network_params.num_vis == 4
    assert network_params.num_lat == 50
    assert neuron_params.E_l == -70.0
    assert weight_params.W_vis_vis == (-0.5, 0.5)
    assert weight_params.d_int == 25


def _install_fake_elise(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    created: dict[str, Any] = {}

    class FakeSimpleUpdater:
        def __init__(self, eta: float) -> None:
            self.eta = eta
            etas = cast(list[float], created.setdefault("etas", []))
            etas.append(eta)

    class FakeBuffer:
        pass

    class FakeDendriticWeights:
        def __init__(self, weight_params, rng_w, rng_d) -> None:
            self.weight_params = weight_params
            self.rng_w = rng_w
            self.rng_d = rng_d
            created["dendritic_weights_ctor"] = self

    class FakeSomaticWeights:
        def __init__(self, weight_params, rng_w, rng_d) -> None:
            self.weight_params = weight_params
            self.rng_w = rng_w
            self.rng_d = rng_d
            created["somatic_weights_ctor"] = self

    class FakeNetwork:
        def __init__(
            self,
            network_params,
            neuron_params,
            dendritic_weights,
            somatic_weights,
            buffer_type,
        ) -> None:
            self.network_params = network_params
            self.neuron_params = neuron_params
            self.dendritic_weights_ctor = dendritic_weights
            self.somatic_weights_ctor = somatic_weights
            self.buffer_type = buffer_type
            self.num_all = network_params.num_vis + network_params.num_lat
            self.u = np.full(self.num_all, neuron_params.E_l, dtype=np.float32)
            self.v = np.full(self.num_all, neuron_params.E_l, dtype=np.float32)
            self.r = np.zeros(self.num_all, dtype=np.float32)
            self.dendritic_weights = np.zeros(
                (self.num_all, self.num_all), dtype=np.float32
            )
            self.calls: list[tuple[np.ndarray | None, bool]] = []
            self.prepared: tuple[float, FakeSimpleUpdater, FakeSimpleUpdater] | None = (
                None
            )
            created["network"] = self

        def prepare_for_simulation(
            self,
            dt_ms: float,
            opt_vis: FakeSimpleUpdater,
            opt_lat: FakeSimpleUpdater,
        ) -> None:
            self.prepared = (dt_ms, opt_vis, opt_lat)

        def __call__(self, u_inp, learn: bool = True) -> None:
            if u_inp is None:
                self.calls.append((None, learn))
                self.u[:] = 100.0 + len(self.calls)
            else:
                arr = np.array(u_inp, copy=True)
                self.calls.append((arr, learn))
                self.u[: len(arr)] = arr
                self.u[len(arr) :] = 100.0 + len(self.calls)
            self.v[:] = self.u + 1.0
            self.r[:] = np.clip(self.u / 100.0, 0.0, 1.0)
            if learn:
                self.dendritic_weights += 1.0

    elise_pkg = types.ModuleType("elise")
    model_mod: Any = types.ModuleType("elise.model")
    optimizer_mod: Any = types.ModuleType("elise.optimizer")
    rate_buffer_mod: Any = types.ModuleType("elise.rate_buffer")
    weights_mod: Any = types.ModuleType("elise.weights")

    model_mod.Network = FakeNetwork
    optimizer_mod.SimpleUpdater = FakeSimpleUpdater
    rate_buffer_mod.Buffer = FakeBuffer
    weights_mod.DendriticWeights = FakeDendriticWeights
    weights_mod.SomaticWeights = FakeSomaticWeights

    monkeypatch.setitem(sys.modules, "elise", elise_pkg)
    monkeypatch.setitem(sys.modules, "elise.model", model_mod)
    monkeypatch.setitem(sys.modules, "elise.optimizer", optimizer_mod)
    monkeypatch.setitem(sys.modules, "elise.rate_buffer", rate_buffer_mod)
    monkeypatch.setitem(sys.modules, "elise.weights", weights_mod)
    return created


def test_elise_import_initialize_and_cached_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake_elise(monkeypatch)
    sim = Elise()

    imported = sim._import_elise_types()
    imported_cached = sim._import_elise_types()
    assert imported is imported_cached

    sim._initialize_network(num_vis=3)
    network = cast(Any, created["network"])
    assert sim._network is not None
    assert network.prepared is not None
    dt_ms, opt_vis, opt_lat = network.prepared
    expected_dt_ms = cast(float, ELISE_DEFAULT_PARAMS["dt_ms"]["default"])
    expected_eta_vis = cast(float, ELISE_DEFAULT_PARAMS["eta_vis"]["default"])
    expected_eta_lat = cast(float, ELISE_DEFAULT_PARAMS["eta_lat"]["default"])
    assert dt_ms == expected_dt_ms
    assert opt_vis.eta == pytest.approx(expected_eta_vis / expected_dt_ms)
    assert opt_lat.eta == pytest.approx(expected_eta_lat / expected_dt_ms)


def test_elise_missing_dependency_error_is_explanatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "elise.model":
            raise ModuleNotFoundError("No module named 'elise'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sim = Elise()

    with pytest.raises(
        ImportError,
        match="requires the separate private 'elise' package",
    ):
        sim._import_elise_types()


def test_elise_simulate_with_fake_backend_and_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_elise(monkeypatch)
    sim = Elise()

    with pytest.raises(
        ValueError, match="must contain both 'opsin_exc' and 'opsin_inh'"
    ):
        sim.simulate({}, 0.1)
    with pytest.raises(ValueError, match="must have the same shape"):
        sim.simulate(
            {
                "opsin_exc": {"array": np.zeros((1, 1), dtype=np.float32), "dt": 0.1},
                "opsin_inh": {"array": np.zeros((1, 2), dtype=np.float32), "dt": 0.1},
            },
            0.1,
        )
    with pytest.raises(ValueError, match="must be 2D"):
        sim.simulate(
            {
                "opsin_exc": {"array": np.zeros(0), "dt": 0.1},
                "opsin_inh": {"array": np.zeros(0), "dt": 0.1},
            },
            0.1,
        )

    params = {
        "num_lat": 2,
        "dt_ms": 100.0,
        "weight_snapshot_interval_ms": 100.0,
        "learn": True,
        # Identity g0 so the monkeypatched kinetic step (O := phi)
        # makes the matching-potential math equivalent to the pre-
        # kinetic model.
        "g0_exc": 1.0,
        "g0_inh": 1.0,
    }

    def _identity_step(open_fraction, D, phi, k_a_max, phi_m, k_d, k_r, dt):
        open_fraction[:] = phi

    monkeypatch.setattr("neuroseq.example_models.elise._step_kinetics", _identity_step)

    u_tgt = np.array(
        [[-10.0, -20.0], [-30.0, -40.0], [-50.0, -60.0], [-25.0, -35.0]],
        dtype=np.float32,
    )
    g_exc = ((-75.0 - u_tgt) / -75.0).astype(np.float32)
    g_inh = ((u_tgt - 0.0) / -75.0).astype(np.float32)
    stims_dict = {
        "opsin_exc": {"array": g_exc, "dt": 0.1},
        "opsin_inh": {"array": g_inh, "dt": 0.1},
    }

    first = sim.simulate(stims_dict, 0.2, params=params)
    second = sim.simulate(stims_dict, 0.2, params=params)

    first_u = cast(
        TraceData,
        next(d for d in first.data if d.name == "somatic_potential_u"),
    )
    first_w = cast(
        SynapseData,
        next(d for d in first.data if d.name == "dendritic_weights"),
    )
    second_u = cast(
        TraceData,
        next(d for d in second.data if d.name == "somatic_potential_u"),
    )
    assert first_u.traces.shape == (4, 2)
    assert first_w.timestamps.tolist() == pytest.approx([0.0, 0.1])
    assert np.allclose(first_u.traces[:2, 0], [-10.0, -20.0])
    assert np.allclose(second_u.traces[:2, 0], [-50.0, -60.0])
    assert sim.simulated_time_seconds == pytest.approx(0.4)

    sim.reset()
    assert sim.simulated_time_seconds == 0.0

    original = elise_module._SIMULATOR
    try:
        dummy = _DummyBackend()
        monkeypatch.setattr(elise_module, "_SIMULATOR", dummy)
        exp = elise_module.simulate(stims_dict, 0.05, params=None)
        assert exp.name == "dummy"
        assert elise_module.get_simulator() is dummy
    finally:
        monkeypatch.setattr(elise_module, "_SIMULATOR", original)


def test_elise_simulate_raises_if_initialization_did_not_set_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = Elise()
    monkeypatch.setattr(sim, "_initialize_network", lambda num_vis: None)
    stims_dict = {
        "opsin_exc": {"array": np.ones((2, 1), dtype=np.float32), "dt": 0.1},
        "opsin_inh": {"array": np.ones((2, 1), dtype=np.float32), "dt": 0.1},
    }
    with pytest.raises(RuntimeError, match="failed to initialize"):
        sim.simulate(stims_dict, 0.1, params={"dt_ms": 100.0})


def test_sinus_offset_helper_branches_and_validation() -> None:
    assert sinus_float_param({"x": True}, "x") == 1.0
    assert sinus_float_param({"x": 2}, "x") == 2.0
    assert sinus_float_param({"x": "2.5"}, "x") == 2.5
    with pytest.raises(TypeError):
        sinus_float_param(cast(Any, {"x": object()}), "x")

    sample_indices = np.array([0, 1], dtype=np.int64)
    one_dim = _stim_as_unit_time_series(np.array([1.0, 2.0]), sample_indices, 3)
    assert np.allclose(one_dim, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        _stim_as_unit_time_series(np.zeros((0, 1), dtype=np.float32), sample_indices, 3)

    exact = _stim_as_unit_time_series(
        np.arange(10, dtype=np.float32).reshape(2, 5), sample_indices, 5
    )
    assert exact.shape == (2, 5)

    truncated = _stim_as_unit_time_series(
        np.arange(12, dtype=np.float32).reshape(2, 6), sample_indices, 5
    )
    assert truncated.shape == (2, 5)
    assert np.allclose(
        truncated,
        np.arange(12, dtype=np.float32).reshape(2, 6)[:, :5],
    )

    padded = _stim_as_unit_time_series(
        np.arange(6, dtype=np.float32).reshape(2, 3), sample_indices, 5
    )
    assert padded.shape == (2, 5)
    assert np.allclose(padded[:, :3], np.arange(6, dtype=np.float32).reshape(2, 3))
    assert np.allclose(padded[:, 3:], 0.0)

    sim = SinusOffset()
    with pytest.raises(ValueError, match="dt_ms must be > 0"):
        sim.simulate({}, 0.1, params={"dt_ms": 0.0})
    with pytest.raises(ValueError, match="simulation_time too short"):
        sim.simulate({}, 0.0001, params={"dt_ms": 1.0})
    with pytest.raises(ValueError, match="Stim dt must be > 0"):
        sim.simulate(
            {"offset": {"array": np.zeros((2, 1), dtype=np.float32), "dt": 0.0}},
            0.01,
            params={"dt_ms": 1.0},
        )
