"""Smoke tests for the PyNN AdEx external-simulation example.

The tests exercise the Brian2 PyNN backend, which needs ``pyNN`` installed.
``pyNN`` is not a declared NeuroSeq dependency, so the whole module is skipped
when it (or the Brian2 backend) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyNN.brian2", reason="pyNN not installed")

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_models.pynn_demo import DEFAULT_PARAMS
from neuroseq.example_models.pynn_demo import PynnDemo
from neuroseq.example_models.pynn_demo import simulate


def _drive_all_neurons(n_neurons: int, duration_s: float, stim_dt_s: float):
    """A constant, above-threshold stim for every neuron."""
    n_stim = int(duration_s / stim_dt_s)
    stim = np.ones((n_stim, n_neurons), dtype=np.float32)
    return {"opsin": {"array": stim, "dt": stim_dt_s}}


def test_backend_default_is_brian2() -> None:
    assert DEFAULT_PARAMS["backend"]["default"] == "brian2"


def test_pynn_demo_runs_and_returns_spikes_and_traces() -> None:
    n = 8
    duration_s = 0.2
    stims_dict = _drive_all_neurons(n, duration_s, stim_dt_s=0.01)

    exp = simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"n_neurons": n, "drive_weight": 0.06, "seed": 1},
    )

    assert exp.duration == duration_s
    by_name = {d.name: d for d in exp.data}
    assert set(by_name) == {"pynn_spikes", "pynn_membrane_potential"}

    spikes = by_name["pynn_spikes"]
    traces = by_name["pynn_membrane_potential"]
    assert isinstance(spikes, SpikeData)
    assert isinstance(traces, TraceData)

    # Voltage traces: one row per neuron, dt positive, biological mV units.
    assert traces.traces.shape[0] == n
    assert traces.traces.shape[1] > 0
    assert traces.dt > 0
    assert traces.units == "mV"

    # A supra-threshold drive to every neuron must evoke at least one spike,
    # and every reported spike index is a valid neuron.
    assert spikes.spike_times.size > 0
    assert spikes.spike_indices.max() < n
    assert np.all(spikes.spike_times >= 0)
    assert np.all(spikes.spike_times <= duration_s + 1e-3)


def test_pynn_demo_silent_without_stim() -> None:
    """No stimulus, no external drive -> no spikes (network is quiescent)."""
    n = 6
    duration_s = 0.1
    zero = np.zeros((int(duration_s / 0.01), n), dtype=np.float32)
    stims_dict = {"opsin": {"array": zero, "dt": 0.01}}

    exp = simulate(stims_dict, simulation_time=duration_s, params={"n_neurons": n})
    spikes = next(d for d in exp.data if isinstance(d, SpikeData))
    assert spikes.spike_times.size == 0


def test_pynn_demo_emits_progress_ticks() -> None:
    n = 4
    duration_s = 0.05
    stims_dict = _drive_all_neurons(n, duration_s, stim_dt_s=0.01)

    fractions: list[float] = []
    simulate(
        stims_dict,
        simulation_time=duration_s,
        params={"n_neurons": n},
        progress=fractions.append,
    )
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0


def test_unknown_backend_raises() -> None:
    sim = PynnDemo()
    with pytest.raises(ValueError, match="unknown backend"):
        sim.simulate(
            {"opsin": {"array": np.ones((10, 4), dtype=np.float32), "dt": 0.01}},
            simulation_time=0.1,
            params={"backend": "nonsense", "n_neurons": 4},
        )
