"""PyNN external-simulation example: a small AdEx spiking network.

This backend runs an adaptive-exponential integrate-and-fire (AdEx) network
through `PyNN <https://neuralensemble.org/PyNN/>`_, the simulator-independent
network API. The AdEx model is the neuron model the BrainScaleS-2 neuromorphic
chip implements physically (Pehle et al. 2022, Eq. 1-2), so the same network
description can, in principle, run on hardware by switching the ``backend``
parameter.

Two backends are supported:

* ``"brian2"`` (default) -- pure software, uses PyNN's Brian2 backend. Brian2 is
  already a NeuroSeq dependency, so this runs everywhere with only ``pyNN``
  added. The neuron uses the Brette & Gerstner (2005) AdEx defaults in
  biological units (mV / ms).
* ``"brainscales2"`` -- the BrainScaleS-2 accelerated analog hardware, via
  ``pynn-brainscales``. Requires the BSS-2 software stack and a chip allocation
  (Calix calibration runs on-chip), so it is not runnable in software. The
  neuron uses ``CalibHXNeuronCoba`` with hardware-unit parameters (dimensionless
  voltages, microsecond time constants).

Only :func:`_make_backend` is backend-specific; the network construction, the
stimulus-to-drive mapping and the output conversion are shared.

``pyNN`` is intentionally *not* a declared NeuroSeq dependency. Importing this
module is cheap; the backend is only imported when :meth:`PynnDemo.simulate`
runs, and a missing backend raises a clear :class:`ImportError`.

pip install "PyNN==0.12.3"
uv run --with "PyNN==0.13" neuroseq

"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control.bss2_protocol import BSS2_NEURON_PARAMETERS
from neuroseq.device_control.bss2_protocol import stim_to_spike_times
from neuroseq.example_models.NN import NN
from neuroseq.example_models.NN import ParamValue
from neuroseq.example_models.NN import ProgressCallback

FlatParams = dict[str, ParamValue]

# Brette & Gerstner (2005) AdEx defaults, as shipped by pyNN for
# EIF_cond_exp_isfa_ista (biological units: nF, ms, mV, nS, nA).
_BRIAN2_NEURON = dict(
    cm=0.281,
    tau_m=9.3667,
    tau_refrac=0.1,
    v_rest=-70.6,
    v_reset=-70.6,
    delta_T=2.0,
    v_thresh=-50.4,
    v_spike=-40.0,
    a=4.0,
    b=0.0805,
    tau_w=144.0,
    e_rev_E=0.0,
    e_rev_I=-80.0,
    tau_syn_E=5.0,
    tau_syn_I=5.0,
)

DEFAULT_PARAMS: dict[str, dict[str, ParamValue | str]] = {
    "backend": {
        "value": "brian2",
        "default": "brian2",
        "type": "str",
        "group": "Backend",
        "description": "PyNN backend: 'brian2' (software, default) or "
        "'brainscales2' (BrainScaleS-2 hardware; needs the BSS-2 stack).",
    },
    "n_neurons": {
        "value": 16,
        "default": 16,
        "type": "int",
        "group": "Network",
        "description": "Number of AdEx neurons in the recurrent population.",
    },
    "p_connect": {
        "value": 0.1,
        "default": 0.1,
        "type": "float",
        "group": "Network",
        "description": "Recurrent excitatory connection probability.",
    },
    "drive_weight": {
        "value": 0.04,
        "default": 0.04,
        "type": "float",
        "group": "Network",
        "description": "Input->neuron synaptic weight. Units depend on backend: "
        "brian2 conductance in microsiemens; brainscales2 6-bit integer [0, 63].",
    },
    "recurrent_weight": {
        "value": 0.02,
        "default": 0.02,
        "type": "float",
        "group": "Network",
        "description": "Recurrent excitatory synaptic weight (see drive_weight "
        "for units).",
    },
    "delay_ms": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "Network",
        "description": "Synaptic delay in milliseconds (recurrent uses 1.5x).",
    },
    "stim_threshold": {
        "value": 0.0,
        "default": 0.0,
        "type": "float",
        "group": "Input",
        "description": "An input spike is emitted where the summed input "
        "value for a neuron exceeds this threshold.",
    },
    "stim_event_ms": {
        "value": 20.0,
        "default": 20.0,
        "type": "float",
        "group": "Input",
        "description": "Input grid: spacing in ms at which the input array "
        "is sampled for evoked drive spikes.",
    },
    "seed": {
        "value": 1234,
        "default": 1234,
        "type": "int",
        "group": "Simulation",
        "description": "Seed for the recurrent random connectivity.",
    },
}


def _float(params: Mapping[str, ParamValue], key: str) -> float:
    return float(params[key])  # type: ignore[arg-type]


def _make_backend(backend: str):
    """Return ``(sim_module, make_cells(n), weights_are_int)`` for a backend."""
    if backend == "brian2":
        try:
            import brian2
            import pyNN.brian2 as sim
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "The 'brian2' PyNN backend needs pyNN installed: `pip install pyNN`."
            ) from exc

        # Avoid C++/Cython codegen (portable, matches NeuroSeq's test setup).
        brian2.prefs.codegen.target = "numpy"

        def make_cells(n: int):
            return sim.Population(
                n, sim.EIF_cond_exp_isfa_ista(**_BRIAN2_NEURON), label="exc"
            )

        return sim, make_cells, False

    if backend == "brainscales2":
        try:
            import pynn_brainscales.brainscales2 as sim
        except ImportError as exc:
            raise ImportError(
                "The 'brainscales2' backend needs the BrainScaleS-2 software "
                "stack (pynn-brainscales) and a chip allocation; it cannot run "
                "in software. Use backend='brian2' for a local run."
            ) from exc

        def make_cells(n: int):
            return sim.Population(
                n, sim.cells.CalibHXNeuronCoba(**BSS2_NEURON_PARAMETERS), label="exc"
            )

        # BrainScaleS-2 synapse weights are 6-bit integers in [0, 63].
        return sim, make_cells, True

    raise ValueError(
        f"unknown backend {backend!r}; expected 'brian2' or 'brainscales2'"
    )


class PynnDemo(NN):
    """Small AdEx spiking network driven by NeuroSeq stimulus arrays.

    Unlike :class:`~neuroseq.example_models.sinus_offset.SinusOffset`, this
    backend is *stateless* between runs: PyNN's ``setup()``/``end()`` lifecycle
    rebuilds the network each call, so the network always restarts at t=0.
    """

    def __init__(self) -> None:
        super().__init__(DEFAULT_PARAMS)

    def reset(self) -> None:
        self._simulated_time_s = 0.0

    def _stim_to_spike_times(
        self,
        stims_dict: dict,
        simulation_time: float,
        n_neurons: int,
        threshold: float,
        event_ms: float,
    ) -> list[np.ndarray]:
        """Convert stim arrays into one evoked-spike train per neuron (ms).

        See :func:`neuroseq.device_control.bss2_protocol.stim_to_spike_times`,
        shared with BrainScaleS-2 stimulus delivery.
        """
        return stim_to_spike_times(
            ((info["array"], info["dt"]) for info in stims_dict.values()),
            simulation_time,
            n_neurons,
            threshold=threshold,
            event_ms=event_ms,
        )

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, ParamValue] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Experiment:
        if progress is not None:
            progress(0.0)
        self.update_params(params)
        p = self._params

        backend = str(p["backend"])
        n = int(p["n_neurons"])
        if n <= 0:
            raise ValueError("n_neurons must be > 0")
        if simulation_time <= 0:
            raise ValueError("simulation_time must be > 0")

        sim, make_cells, weights_are_int = _make_backend(backend)

        def _weight(key: str) -> float | int:
            w = _float(p, key)
            return int(round(w)) if weights_are_int else w

        delay_ms = _float(p, "delay_ms")
        spike_times = self._stim_to_spike_times(
            stims_dict,
            simulation_time,
            n,
            threshold=_float(p, "stim_threshold"),
            event_ms=_float(p, "stim_event_ms"),
        )

        sim.setup()
        try:
            from pyNN.parameters import Sequence

            cells = make_cells(n)
            drive = sim.Population(
                n,
                sim.SpikeSourceArray(spike_times=[Sequence(t) for t in spike_times]),
                label="drive",
            )
            sim.Projection(
                drive,
                cells,
                sim.OneToOneConnector(),
                sim.StaticSynapse(weight=_weight("drive_weight"), delay=delay_ms),
                receptor_type="excitatory",
            )
            # Draw recurrent connectivity ourselves (deterministic, backend
            # independent) so we can skip an empty projection: Brian2 refuses to
            # run a Synapses object that ended up with zero connections.
            rng = np.random.default_rng(int(p["seed"]))
            mask = rng.random((n, n)) < _float(p, "p_connect")
            np.fill_diagonal(mask, False)
            pre, post = np.nonzero(mask)
            if pre.size:
                sim.Projection(
                    cells,
                    cells,
                    sim.FromListConnector(
                        list(zip(pre.tolist(), post.tolist(), strict=True))
                    ),
                    sim.StaticSynapse(
                        weight=_weight("recurrent_weight"), delay=1.5 * delay_ms
                    ),
                    receptor_type="excitatory",
                )

            cells.record(["spikes", "v"])
            sim.run(simulation_time * 1000.0)  # PyNN works in ms

            segment = cells.get_data().segments[0]
            data = _segment_to_neuroseq(segment, n, backend)
        finally:
            sim.end()

        self._simulated_time_s += simulation_time
        experiment = Experiment(
            name=f"PyNN AdEx network ({backend}, {n} neurons)",
            duration=simulation_time,
            timepoints=[],
            data=data,
        )
        if progress is not None:
            progress(1.0)
        return experiment


def _segment_to_neuroseq(segment, n_neurons: int, backend: str) -> list[Data]:
    """Convert a Neo segment (spikes + membrane voltage) to NeuroSeq data."""
    spike_indices: list[int] = []
    spike_times: list[float] = []
    for i, train in enumerate(segment.spiketrains):
        times_s = np.asarray(train.rescale("s").magnitude, dtype=np.float32)
        spike_indices.extend([i] * times_s.size)
        spike_times.extend(times_s.tolist())

    data: list[Data] = [
        SpikeData(
            name="pynn_spikes",
            spike_indices=np.asarray(spike_indices, dtype=np.uint32),
            spike_times=np.asarray(spike_times, dtype=np.float32),
        )
    ]

    if segment.analogsignals:
        vsig = segment.analogsignals[0]
        traces = np.asarray(vsig.magnitude, dtype=np.float32).T  # (neurons, time)
        dt_s = float(vsig.sampling_period.rescale("s").magnitude)
        data.append(
            TraceData(
                name="pynn_membrane_potential",
                traces=traces,
                indices=np.arange(traces.shape[0], dtype=np.uint32),
                units="mV" if backend == "brian2" else "a.u.",
                dt=dt_s,
            )
        )
    return data


def simulate(
    stims_dict: dict,
    simulation_time: float,
    params: Mapping[str, ParamValue] | None = None,
    progress: ProgressCallback | None = None,
) -> Experiment:
    return _SIMULATOR.simulate(
        stims_dict, simulation_time, params=params, progress=progress
    )


def get_simulator() -> PynnDemo:
    return _SIMULATOR


_SIMULATOR = PynnDemo()
