from __future__ import annotations

import importlib.util
from typing import Any
from typing import cast

import numpy as np
import pytest

from neuroseq.data import TraceData
from neuroseq.example_models.elise import DEFAULT_PARAMS
from neuroseq.example_models.elise import Elise
from neuroseq.example_models.elise import get_simulator

ELISE_AVAILABLE = importlib.util.find_spec("elise") is not None


def test_clamp_visible_in_default_params() -> None:
    assert "clamp_visible" in DEFAULT_PARAMS
    entry = DEFAULT_PARAMS["clamp_visible"]
    assert entry["value"] is True
    assert entry["default"] is True
    assert entry["type"] == "bool"
    assert entry["group"] == "Simulation"


def test_phase_params_includes_clamp_visible_and_learn() -> None:
    assert "clamp_visible" in Elise._PHASE_PARAMS
    assert "learn" in Elise._PHASE_PARAMS


def test_update_params_toggling_phase_does_not_reset() -> None:
    sim = Elise()
    sim._network = cast(Any, object())
    sim._signature = cast(Any, ("sentinel",))
    sim._simulated_time_s = 1.234

    sim.update_params({"clamp_visible": False})

    assert sim._network is not None
    assert sim._signature == ("sentinel",)
    assert sim._simulated_time_s == 1.234
    assert sim._params["clamp_visible"] is False


def test_update_params_non_phase_change_resets() -> None:
    sim = Elise()
    sim._network = cast(Any, object())
    sim._signature = cast(Any, ("sentinel",))
    sim._simulated_time_s = 1.234

    sim.update_params({"num_lat": 4})

    assert sim._network is None
    assert sim._signature is None
    assert sim._simulated_time_s == 0.0
    assert sim._params["num_lat"] == 4


def test_update_params_mixed_change_resets() -> None:
    sim = Elise()
    sim._network = cast(Any, object())
    sim._signature = cast(Any, ("sentinel",))

    sim.update_params({"clamp_visible": False, "num_lat": 4})

    assert sim._network is None
    assert sim._params["num_lat"] == 4
    assert sim._params["clamp_visible"] is False


def test_simulate_clamp_visible_false_passes_none_and_preserves_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSimpleUpdater:
        def __init__(self, eta: float) -> None:
            self.eta = eta

    class FakeBuffer:
        pass

    class FakeWeights:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeNetwork:
        def __init__(
            self,
            network_params: Any,
            neuron_params: Any,
            dendritic_weights: Any,
            somatic_weights: Any,
            buffer_type: Any,
        ) -> None:
            self.num_all = int(network_params.num_vis + network_params.num_lat)
            self.num_vis = int(network_params.num_vis)
            self.u = np.full(self.num_all, neuron_params.E_l, dtype=np.float32)
            self.v = np.full(self.num_all, neuron_params.E_l, dtype=np.float32)
            self.r = np.zeros(self.num_all, dtype=np.float32)
            self.dendritic_weights = np.zeros(
                (self.num_all, self.num_all), dtype=np.float32
            )
            self.calls: list[tuple[np.ndarray | None, bool]] = []

        def prepare_for_simulation(
            self,
            dt_ms: float,
            opt_vis: FakeSimpleUpdater,
            opt_lat: FakeSimpleUpdater,
        ) -> None:
            self.prepared = (dt_ms, opt_vis, opt_lat)

        def __call__(self, u_inp: np.ndarray | None, learn: bool = True) -> None:
            if u_inp is None:
                self.calls.append((None, learn))
            else:
                arr = np.array(u_inp, copy=True)
                self.calls.append((arr, learn))
                self.u[: len(arr)] = arr
            self.v[:] = self.u + 1.0
            self.r[:] = np.clip((self.u + 70.0) / 20.0, 0.0, 1.0)
            if learn:
                self.dendritic_weights += 1.0

    sim = Elise()
    sim.reset()
    monkeypatch.setattr(
        sim,
        "_import_elise_types",
        lambda: (
            FakeNetwork,
            FakeSimpleUpdater,
            FakeBuffer,
            FakeWeights,
            FakeWeights,
        ),
    )

    stims_dict = {
        "u_inp": {
            "array": np.full((4, 2), -50.0, dtype=np.float32),
            "dt": 0.1,
        }
    }
    params = {
        "num_lat": 2,
        "dt_ms": 100.0,
        "weight_snapshot_interval_ms": 100.0,
        "learn": False,
        "clamp_visible": True,
    }

    sim.simulate(stims_dict, 0.2, params=params)
    first_network = cast(Any, sim._network)
    assert first_network.calls
    assert first_network.calls[0][0] is not None

    sim.simulate(
        stims_dict,
        0.2,
        params={**params, "clamp_visible": False},
    )
    second_network = cast(Any, sim._network)

    assert second_network is first_network
    assert sim.simulated_time_seconds == pytest.approx(0.4)
    assert all(call[0] is None for call in second_network.calls[2:])


@pytest.mark.skipif(
    not ELISE_AVAILABLE,
    reason="ELISE clamp-visible integration tests require the private 'elise' package.",
)
def test_clamp_visible_true_passes_stim_vector_to_network() -> None:
    sim = get_simulator()
    sim.reset()

    num_vis = 5
    duration_s = 0.1
    stim = np.full((100, num_vis), -50.0, dtype=np.float32)
    stims_dict = {"u_inp": {"array": stim, "dt": 0.001}}

    exp = sim.simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"num_lat": 4, "clamp_visible": True, "learn": False},
    )

    u_data = cast(
        TraceData,
        next(d for d in exp.data if d.name == "somatic_potential_u"),
    )
    final_visible = u_data.traces[:num_vis, -1]
    assert np.any(np.abs(final_visible + 70.0) > 1.0)


@pytest.mark.skipif(
    not ELISE_AVAILABLE,
    reason="ELISE clamp-visible integration tests require the private 'elise' package.",
)
def test_clamp_visible_false_keeps_visible_near_rest() -> None:
    sim = get_simulator()
    sim.reset()

    num_vis = 5
    duration_s = 0.1
    stim = np.full((100, num_vis), -50.0, dtype=np.float32)
    stims_dict = {"u_inp": {"array": stim, "dt": 0.001}}

    exp = sim.simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"num_lat": 4, "clamp_visible": False, "learn": False},
    )

    u_data = cast(
        TraceData,
        next(d for d in exp.data if d.name == "somatic_potential_u"),
    )
    final_visible = u_data.traces[:num_vis, -1]
    assert np.all(np.abs(final_visible + 70.0) < 5.0)


@pytest.mark.skipif(
    not ELISE_AVAILABLE,
    reason="ELISE clamp-visible integration tests require the private 'elise' package.",
)
def test_clamp_visible_toggle_preserves_network_across_runs() -> None:
    sim = get_simulator()
    sim.reset()

    stims_dict = {
        "u_inp": {"array": np.full((50, 5), -50.0, dtype=np.float32), "dt": 0.001}
    }
    base_params = {"num_lat": 4, "learn": False}

    sim.simulate(
        stims_dict,
        simulation_time=0.05,
        params={**base_params, "clamp_visible": True},
    )
    first_network_id = id(sim._network)

    sim.simulate(
        stims_dict,
        simulation_time=0.05,
        params={**base_params, "clamp_visible": False},
    )

    assert id(sim._network) == first_network_id
    assert sim.simulated_time_seconds == pytest.approx(0.1)
