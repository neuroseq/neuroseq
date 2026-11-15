"""Smoke tests for the PyNN AdEx external-simulation example project.

Building the project must work without ``pyNN`` (the backend is imported lazily
only when the simulation runs), so only the simulate test skips when ``pyNN`` is
absent.
"""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control.bss2_protocol import channels_to_neurons
from neuroseq.example_models import pynn_demo


def test_pynn_demo_example_project_registered() -> None:
    from neuroseq.example_projects import example_projects

    assert "Network" in example_projects
    assert "PyNN AdEx Neuromorphic Network" in example_projects["Network"]


def test_pynn_demo_example_project_builds_without_pynn() -> None:
    from neuroseq.example_projects import example_projects

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
    from neuroseq.example_projects import example_projects

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
    from neuroseq.example_projects import _build_pynn_stim_pattern

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


def test_pynn_helpers_and_stim_to_spike_times() -> None:
    assert pynn_demo._float({"x": "1.25"}, "x") == pytest.approx(1.25)

    sampled = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(channels_to_neurons(sampled, 2), sampled)
    assert np.array_equal(
        channels_to_neurons(sampled[:, :1], 3),
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
    )
    assert np.array_equal(
        channels_to_neurons(sampled, 1),
        np.array([[1.0]], dtype=np.float32),
    )
    assert np.array_equal(
        channels_to_neurons(sampled, 3),
        np.array([[1.0, 2.0, 0.0]], dtype=np.float32),
    )

    simulator = pynn_demo.PynnDemo()
    spikes = simulator._stim_to_spike_times(
        stims_dict={"opsin": {"array": np.array([0.0, 1.0, 0.0, 2.0]), "dt": 0.1}},
        simulation_time=0.4,
        n_neurons=2,
        threshold=0.5,
        event_ms=100.0,
    )
    assert len(spikes) == 2
    assert np.array_equal(spikes[0], np.array([150.0, 350.0]))
    assert np.array_equal(spikes[1], np.array([150.0, 350.0]))

    with pytest.raises(ValueError, match="Input dt must be > 0"):
        simulator._stim_to_spike_times(
            stims_dict={"opsin": {"array": np.array([1.0]), "dt": 0.0}},
            simulation_time=0.1,
            n_neurons=1,
            threshold=0.0,
            event_ms=100.0,
        )


def test_make_backend_import_errors_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"pyNN.brian2", "pynn_brainscales.brainscales2"}:
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="needs pyNN installed"):
        pynn_demo._make_backend("brian2")
    with pytest.raises(ImportError, match="needs the BrainScaleS-2 software stack"):
        pynn_demo._make_backend("brainscales2")
    with pytest.raises(ValueError, match="unknown backend"):
        pynn_demo._make_backend("other")


def test_make_backend_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_brian2 = types.SimpleNamespace(
        prefs=types.SimpleNamespace(codegen=types.SimpleNamespace(target="cython"))
    )

    class FakeB2Sim:
        def __init__(self) -> None:
            self.population_calls: list[tuple[int, object, str]] = []

        def EIF_cond_exp_isfa_ista(self, **kwargs):
            return ("celltype", kwargs)

        def Population(self, n: int, celltype: object, label: str):
            self.population_calls.append((n, celltype, label))
            return {"n": n, "celltype": celltype, "label": label}

    fake_b2_sim = FakeB2Sim()
    monkeypatch.setitem(sys.modules, "brian2", fake_brian2)
    monkeypatch.setitem(sys.modules, "pyNN", types.ModuleType("pyNN"))
    monkeypatch.setitem(sys.modules, "pyNN.brian2", fake_b2_sim)

    sim, make_cells, weights_are_int = pynn_demo._make_backend("brian2")
    assert sim is fake_b2_sim
    assert weights_are_int is False
    assert fake_brian2.prefs.codegen.target == "numpy"
    assert make_cells(3)["label"] == "exc"

    class FakeBss2Cells:
        def CalibHXNeuronCoba(self, **kwargs):
            return ("bss2-celltype", kwargs)

    class FakeBss2Sim(FakeB2Sim):
        def __init__(self) -> None:
            super().__init__()
            self.cells = FakeBss2Cells()

    fake_bss2_sim = FakeBss2Sim()
    monkeypatch.setitem(
        sys.modules, "pynn_brainscales", types.ModuleType("pynn_brainscales")
    )
    monkeypatch.setitem(sys.modules, "pynn_brainscales.brainscales2", fake_bss2_sim)

    sim, make_cells, weights_are_int = pynn_demo._make_backend("brainscales2")
    assert sim is fake_bss2_sim
    assert weights_are_int is True
    assert make_cells(2)["label"] == "exc"


class _FakeQuantity:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class _FakeTrain:
    def __init__(self, values):
        self._values = values

    def rescale(self, unit: str) -> _FakeQuantity:
        assert unit == "s"
        return _FakeQuantity(self._values)


class _FakeSamplingPeriod:
    def __init__(self, value: float):
        self._value = value

    def rescale(self, unit: str) -> _FakeQuantity:
        assert unit == "s"
        return _FakeQuantity(self._value)


class _FakeSignal:
    def __init__(self, magnitude, dt_s: float):
        self.magnitude = magnitude
        self.sampling_period = _FakeSamplingPeriod(dt_s)


class _FakeSegment:
    def __init__(self, analogsignals):
        self.spiketrains = [_FakeTrain([0.1, 0.2]), _FakeTrain([0.3])]
        self.analogsignals = analogsignals


def test_segment_to_neuroseq_converts_spikes_and_optional_traces() -> None:
    with_trace = pynn_demo._segment_to_neuroseq(
        _FakeSegment([_FakeSignal([[1.0, 2.0], [3.0, 4.0]], 0.002)]),
        n_neurons=2,
        backend="brian2",
    )
    assert isinstance(with_trace[0], SpikeData)
    assert np.array_equal(
        with_trace[0].spike_indices, np.array([0, 0, 1], dtype=np.uint32)
    )
    assert np.allclose(
        with_trace[0].spike_times, np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    assert isinstance(with_trace[1], TraceData)
    assert with_trace[1].units == "mV"
    assert with_trace[1].dt == pytest.approx(0.002)
    assert with_trace[1].traces.shape == (2, 2)

    without_trace = pynn_demo._segment_to_neuroseq(
        _FakeSegment([]),
        n_neurons=2,
        backend="brainscales2",
    )
    assert len(without_trace) == 1
    assert isinstance(without_trace[0], SpikeData)


def test_pynn_demo_simulate_and_module_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_parameters = types.ModuleType("pyNN.parameters")
    monkeypatch.setattr(
        fake_parameters,
        "Sequence",
        lambda values: ("sequence", tuple(values)),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "pyNN", types.ModuleType("pyNN"))
    monkeypatch.setitem(sys.modules, "pyNN.parameters", fake_parameters)

    segment = _FakeSegment([_FakeSignal([[1.0, 2.0], [3.0, 4.0]], 0.001)])

    class FakeCells:
        def __init__(self) -> None:
            self.recorded: list[list[str]] = []

        def record(self, variables: list[str]) -> None:
            self.recorded.append(variables)

        def get_data(self):
            return types.SimpleNamespace(segments=[segment])

    class FakeSim:
        def __init__(self) -> None:
            self.setup_calls = 0
            self.end_calls = 0
            self.projections: list[dict[str, object]] = []
            self.ran_ms: float | None = None

        def setup(self) -> None:
            self.setup_calls += 1

        def end(self) -> None:
            self.end_calls += 1

        def Population(self, n: int, celltype: object, label: str):
            return {"n": n, "celltype": celltype, "label": label}

        def SpikeSourceArray(self, *, spike_times):
            return {"spike_times": spike_times}

        def OneToOneConnector(self) -> str:
            return "one-to-one"

        def StaticSynapse(self, *, weight, delay):
            return {"weight": weight, "delay": delay}

        def Projection(self, pre, post, connector, synapse, receptor_type: str) -> None:
            self.projections.append(
                {
                    "pre": pre,
                    "post": post,
                    "connector": connector,
                    "synapse": synapse,
                    "receptor_type": receptor_type,
                }
            )

        def FromListConnector(self, pairs):
            return list(pairs)

        def run(self, duration_ms: float) -> None:
            self.ran_ms = duration_ms

    fake_sim = FakeSim()
    fake_cells = FakeCells()

    def _fake_make_backend(backend: str):
        assert backend == "brainscales2"

        def make_cells(n: int):
            assert n == 2
            return fake_cells

        return fake_sim, make_cells, True

    monkeypatch.setattr(pynn_demo, "_make_backend", _fake_make_backend)

    simulator = pynn_demo.PynnDemo()
    progress_values: list[float] = []
    stims_dict = {
        "opsin": {
            "array": np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32),
            "dt": 0.1,
        }
    }
    experiment = simulator.simulate(
        stims_dict,
        simulation_time=0.3,
        params={
            "backend": "brainscales2",
            "n_neurons": 2,
            "p_connect": 1.0,
            "drive_weight": 1.6,
            "recurrent_weight": 2.2,
            "delay_ms": 4.0,
            "stim_threshold": 0.5,
            "stim_event_ms": 100.0,
            "seed": 7,
        },
        progress=progress_values.append,
    )

    assert progress_values == [0.0, 1.0]
    assert fake_sim.setup_calls == 1
    assert fake_sim.end_calls == 1
    assert fake_sim.ran_ms == pytest.approx(300.0)
    assert fake_cells.recorded == [["spikes", "v"]]
    assert len(fake_sim.projections) == 2
    assert fake_sim.projections[0]["synapse"] == {"weight": 2, "delay": 4.0}
    assert fake_sim.projections[1]["synapse"] == {"weight": 2, "delay": 6.0}
    assert experiment.name == "PyNN AdEx network (brainscales2, 2 neurons)"
    assert experiment.duration == pytest.approx(0.3)
    assert isinstance(experiment.data[0], SpikeData)
    assert isinstance(experiment.data[1], TraceData)
    assert simulator._simulated_time_s == pytest.approx(0.3)

    monkeypatch.setattr(pynn_demo, "_SIMULATOR", simulator)
    wrapped_experiment = pynn_demo.simulate(
        stims_dict,
        simulation_time=0.1,
        params={
            "backend": "brainscales2",
            "n_neurons": 2,
            "p_connect": 0.0,
            "drive_weight": 1.0,
            "recurrent_weight": 2.0,
            "delay_ms": 1.0,
            "stim_threshold": 0.5,
            "stim_event_ms": 100.0,
            "seed": 7,
        },
    )
    assert wrapped_experiment.duration == pytest.approx(0.1)
    assert pynn_demo.get_simulator() is simulator
    simulator.reset()
    assert simulator._simulated_time_s == 0.0


def test_pynn_demo_simulate_rejects_invalid_args() -> None:
    simulator = pynn_demo.PynnDemo()

    with pytest.raises(ValueError, match="n_neurons must be > 0"):
        simulator.simulate({}, simulation_time=0.1, params={"n_neurons": 0})
    with pytest.raises(ValueError, match="simulation_time must be > 0"):
        simulator.simulate({}, simulation_time=0.0, params={"n_neurons": 1})
