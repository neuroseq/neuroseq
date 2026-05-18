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


def test_opsin_exc_inh_not_in_default_params_under_simulation_group() -> None:
    assert "opsin_exc" not in DEFAULT_PARAMS
    assert "opsin_inh" not in DEFAULT_PARAMS
    assert "clamp_visible" not in DEFAULT_PARAMS


def test_phase_params_no_longer_contains_clamp_visible() -> None:
    assert "clamp_visible" not in Elise._PHASE_PARAMS
    assert "learn" in Elise._PHASE_PARAMS


def test_clamp_visible_in_legacy_archive_is_dropped() -> None:
    sim = Elise()
    sim.update_params({"clamp_visible": False, "num_lat": 4})

    assert "clamp_visible" not in sim._params
    assert sim._params["num_lat"] == 4


def test_simulate_both_stims_zero_passes_none_to_network(
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
            self.calls.append(
                (None if u_inp is None else np.array(u_inp, copy=True), learn)
            )

    sim = Elise()
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

    zero = np.zeros((4, 2), dtype=np.float32)
    sim.simulate(
        {
            "opsin_exc": {"array": zero, "dt": 0.1},
            "opsin_inh": {"array": zero, "dt": 0.1},
        },
        0.2,
        params={"num_lat": 2, "dt_ms": 100.0, "learn": False},
    )

    network = cast(Any, sim._network)
    assert network.calls
    assert all(call[0] is None for call in network.calls)


def test_simulate_mixed_stim_drives_per_pitch_u_tgt(
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
            self.calls.append(
                (None if u_inp is None else np.array(u_inp, copy=True), learn)
            )

    sim = Elise()
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

    g_exc = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    g_inh = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    sim.simulate(
        {
            "opsin_exc": {"array": g_exc, "dt": 0.1},
            "opsin_inh": {"array": g_inh, "dt": 0.1},
        },
        0.1,
        params={"num_lat": 1, "dt_ms": 100.0, "learn": False},
    )

    network = cast(Any, sim._network)
    driven = cast(np.ndarray, network.calls[0][0])
    assert driven is not None
    assert driven[0] == pytest.approx(-50.0)
    assert driven[1] == pytest.approx(-70.0)
    assert driven[2] == pytest.approx(-70.0)


def test_simulate_preserves_network_across_releases(
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
            self.calls.append(
                (None if u_inp is None else np.array(u_inp, copy=True), learn)
            )

    sim = Elise()
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

    active_exc = np.ones((2, 2), dtype=np.float32)
    active_inh = np.full((2, 2), 2.0, dtype=np.float32)
    zero = np.zeros((2, 2), dtype=np.float32)
    params = {"num_lat": 2, "dt_ms": 100.0, "learn": False}

    sim.simulate(
        {
            "opsin_exc": {"array": active_exc, "dt": 0.1},
            "opsin_inh": {"array": active_inh, "dt": 0.1},
        },
        0.2,
        params=params,
    )
    first_network = cast(Any, sim._network)

    sim.simulate(
        {
            "opsin_exc": {"array": zero, "dt": 0.1},
            "opsin_inh": {"array": zero, "dt": 0.1},
        },
        0.2,
        params=params,
    )

    second_network = cast(Any, sim._network)
    assert second_network is first_network
    assert sim.simulated_time_seconds == pytest.approx(0.4)
    assert all(call[0] is None for call in second_network.calls[2:])


@pytest.mark.skipif(
    not ELISE_AVAILABLE,
    reason="ELISE opsin integration tests require the private 'elise' package.",
)
def test_nonzero_opsins_drive_visible_neurons() -> None:
    sim = get_simulator()
    sim.reset()

    num_vis = 5
    duration_s = 0.1
    stims_dict = {
        "opsin_exc": {"array": np.ones((100, num_vis), dtype=np.float32), "dt": 0.001},
        "opsin_inh": {
            "array": np.full((100, num_vis), 2.0, dtype=np.float32),
            "dt": 0.001,
        },
    }

    exp = sim.simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"num_lat": 4, "learn": False},
    )

    u_data = cast(
        TraceData,
        next(d for d in exp.data if d.name == "somatic_potential_u"),
    )
    final_visible = u_data.traces[:num_vis, -1]
    assert np.any(np.abs(final_visible + 70.0) > 1.0)


@pytest.mark.skipif(
    not ELISE_AVAILABLE,
    reason="ELISE opsin integration tests require the private 'elise' package.",
)
def test_all_zero_opsins_release_visible_neurons() -> None:
    sim = get_simulator()
    sim.reset()

    num_vis = 5
    duration_s = 0.1
    zero = np.zeros((100, num_vis), dtype=np.float32)
    stims_dict = {
        "opsin_exc": {"array": zero, "dt": 0.001},
        "opsin_inh": {"array": zero, "dt": 0.001},
    }

    exp = sim.simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"num_lat": 4, "learn": False},
    )

    u_data = cast(
        TraceData,
        next(d for d in exp.data if d.name == "somatic_potential_u"),
    )
    final_visible = u_data.traces[:num_vis, -1]
    assert np.all(np.abs(final_visible + 70.0) < 5.0)
