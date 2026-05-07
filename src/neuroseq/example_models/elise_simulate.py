"""ELISE network model exposed as a neuroseq external simulate function.

Wraps the rate-coded ELISE network from the vendored ``_elise`` subpackage
(Kriener et al., 2024 — see ``_elise/__init__.py`` for upstream attribution)
behind neuroseq's ``simulate(stims_dict, simulation_time, params=None) -> Experiment``
contract.

ELISE only ships ``SimpleUpdater`` as an optimizer; that is what we use here
(matches the upstream ``scripts/train.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SynapseData
from neuroseq.data import TraceData

ParamValue = int | float | bool | str
FlatParams = dict[str, ParamValue]

# Editable parameter schema. The neuroseq GUI consumes this via
# ``project.external_simulate_params`` to populate the parameter editor and
# passes the flat ``{name: value}`` dict back through ``simulate(params=...)``.
DEFAULT_PARAMS: dict[str, dict[str, ParamValue | str]] = {
    # Network
    "num_lat": {
        "value": 50,
        "default": 50,
        "type": "int",
        "group": "Network",
        "description": "Number of latent (recurrent) neurons.",
    },
    # Neuron
    "C_v": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "Neuron",
        "description": "Dendritic membrane capacitance.",
    },
    "C_u": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "Neuron",
        "description": "Somatic membrane capacitance.",
    },
    "E_l": {
        "value": -70.0,
        "default": -70.0,
        "type": "float",
        "group": "Neuron",
        "description": "Leak reversal potential (mV).",
    },
    "E_exc": {
        "value": 0.0,
        "default": 0.0,
        "type": "float",
        "group": "Neuron",
        "description": "Excitatory reversal potential (mV).",
    },
    "E_inh": {
        "value": -75.0,
        "default": -75.0,
        "type": "float",
        "group": "Neuron",
        "description": "Inhibitory reversal potential (mV).",
    },
    "g_l": {
        "value": 0.1,
        "default": 0.1,
        "type": "float",
        "group": "Neuron",
        "description": "Leak conductance.",
    },
    "g_den": {
        "value": 2.0,
        "default": 2.0,
        "type": "float",
        "group": "Neuron",
        "description": "Dendritic-somatic coupling conductance.",
    },
    "g_exc_0": {
        "value": 0.3,
        "default": 0.3,
        "type": "float",
        "group": "Neuron",
        "description": "Baseline excitatory conductance.",
    },
    "g_inh_0": {
        "value": 6.0,
        "default": 6.0,
        "type": "float",
        "group": "Neuron",
        "description": "Baseline inhibitory conductance.",
    },
    "a": {
        "value": 0.3,
        "default": 0.3,
        "type": "float",
        "group": "Neuron",
        "description": "Sigmoid slope for phi(u) = 1 / (1 + exp(a*(b - u))).",
    },
    "b": {
        "value": -58.0,
        "default": -58.0,
        "type": "float",
        "group": "Neuron",
        "description": "Sigmoid threshold (mV); phi=0.5 at u=b.",
    },
    "lam": {
        "value": 0.7,
        "default": 0.7,
        "type": "float",
        "group": "Neuron",
        "description": "Visible-input dendritic/somatic coupling weight, [0, 1).",
    },
    # Weights
    "weight_init_min": {
        "value": -0.5,
        "default": -0.5,
        "type": "float",
        "group": "Weights",
        "description": "Lower bound of the uniform initial dendritic weight range.",
    },
    "weight_init_max": {
        "value": 0.5,
        "default": 0.5,
        "type": "float",
        "group": "Weights",
        "description": "Upper bound of the uniform initial dendritic weight range.",
    },
    "d_den_min": {
        "value": 5,
        "default": 5,
        "type": "int",
        "group": "Weights",
        "description": "Minimum dendritic delay (ms).",
    },
    "d_den_max": {
        "value": 25,
        "default": 25,
        "type": "int",
        "group": "Weights",
        "description": "Maximum dendritic delay (ms).",
    },
    "d_som_min": {
        "value": 5,
        "default": 5,
        "type": "int",
        "group": "Weights",
        "description": "Minimum somatic delay (ms).",
    },
    "d_som_max": {
        "value": 25,
        "default": 25,
        "type": "int",
        "group": "Weights",
        "description": "Maximum somatic delay (ms).",
    },
    "d_int": {
        "value": 25,
        "default": 25,
        "type": "int",
        "group": "Weights",
        "description": "Fixed lateral inhibition delay (ms).",
    },
    "p": {
        "value": 0.2,
        "default": 0.2,
        "type": "float",
        "group": "Weights",
        "description": "Per-synapse connection probability for somatic weights.",
    },
    "q": {
        "value": 0.0,
        "default": 0.0,
        "type": "float",
        "group": "Weights",
        "description": "Probability that a target neuron accepts an incoming connection.",
    },
    "p0": {
        "value": 0.04,
        "default": 0.04,
        "type": "float",
        "group": "Weights",
        "description": "Baseline probability for the first connection attempt.",
    },
    # Simulation
    "dt_ms": {
        "value": 0.1,
        "default": 0.1,
        "type": "float",
        "group": "Simulation",
        "description": "Integration time step (ms).",
    },
    "eta_vis": {
        "value": 0.001,
        "default": 0.001,
        "type": "float",
        "group": "Simulation",
        "description": "Learning rate for visible (output) weights.",
    },
    "eta_lat": {
        "value": 0.01,
        "default": 0.01,
        "type": "float",
        "group": "Simulation",
        "description": "Learning rate for latent weights.",
    },
    "learn": {
        "value": True,
        "default": True,
        "type": "bool",
        "group": "Simulation",
        "description": "Enable plasticity during the simulation.",
    },
    "weight_snapshot_interval_ms": {
        "value": 50.0,
        "default": 50.0,
        "type": "float",
        "group": "Simulation",
        "description": "Interval between dendritic weight snapshots (ms).",
    },
    # RNG
    "seed": {
        "value": 2,
        "default": 2,
        "type": "int",
        "group": "RNG",
        "description": "Master seed for weight and delay sampling.",
    },
}


def _flat_defaults() -> FlatParams:
    return {k: v["default"] for k, v in DEFAULT_PARAMS.items()}


def _int_param(params: Mapping[str, ParamValue], key: str) -> int:
    value = params[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    raise TypeError(f"Parameter '{key}' must be int-compatible, got {type(value)}")


def _float_param(params: Mapping[str, ParamValue], key: str) -> float:
    value = params[key]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Parameter '{key}' must be float-compatible, got {type(value)}")


def _bool_param(params: Mapping[str, ParamValue], key: str) -> bool:
    value = params[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    raise TypeError(f"Parameter '{key}' must be bool-compatible, got {type(value)}")


def _build_namespaces(
    params: Mapping[str, ParamValue], num_vis: int
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    network_params = SimpleNamespace(num_vis=num_vis, num_lat=_int_param(params, "num_lat"))
    neuron_params = SimpleNamespace(
        C_v=_float_param(params, "C_v"),
        C_u=_float_param(params, "C_u"),
        E_l=_float_param(params, "E_l"),
        E_exc=_float_param(params, "E_exc"),
        E_inh=_float_param(params, "E_inh"),
        g_l=_float_param(params, "g_l"),
        g_den=_float_param(params, "g_den"),
        g_exc_0=_float_param(params, "g_exc_0"),
        g_inh_0=_float_param(params, "g_inh_0"),
        a=_float_param(params, "a"),
        b=_float_param(params, "b"),
        lam=_float_param(params, "lam"),
    )
    w_range = (
        _float_param(params, "weight_init_min"),
        _float_param(params, "weight_init_max"),
    )
    weight_params = SimpleNamespace(
        W_vis_vis=w_range,
        W_vis_lat=w_range,
        W_lat_vis=w_range,
        W_lat_lat=w_range,
        d_den=(_int_param(params, "d_den_min"), _int_param(params, "d_den_max")),
        d_som=(_int_param(params, "d_som_min"), _int_param(params, "d_som_max")),
        d_int=_int_param(params, "d_int"),
        p=_float_param(params, "p"),
        q=_float_param(params, "q"),
        p0=_float_param(params, "p0"),
    )
    return network_params, neuron_params, weight_params


def simulate(
    stims_dict: dict, simulation_time: float, params: Mapping[str, ParamValue] | None = None
) -> Experiment:
    """Run ELISE for ``simulation_time`` seconds, driven by ``stims_dict["u_inp"]``.

    Args:
        stims_dict: ``{stim_name: {"array": ndarray[T_stim, num_vis], "dt": seconds}}``.
            Must contain an entry named ``"u_inp"`` providing the visible-layer
            membrane-potential drive (mV) over time.
        simulation_time: Total simulation duration in seconds.
        params: Optional flat parameter overrides (see ``DEFAULT_PARAMS`` for keys).

    Returns:
        An ``Experiment`` containing somatic potential, dendritic potential,
        firing rate traces, and dendritic weight snapshots.
    """
    # Imported here so DEFAULT_PARAMS can be inspected without numba installed.
    # The vendored ``_elise.model`` calls ``np.seterr(all="raise")`` at module
    # load, which would leak into the rest of the process and break unrelated
    # numpy code that tolerates over/underflow. Save and restore the global
    # errstate around the import so the change is scoped to ELISE-internal
    # computation.
    _prev_seterr = np.seterr()
    from neuroseq.example_models._elise import Buffer
    from neuroseq.example_models._elise import DendriticWeights
    from neuroseq.example_models._elise import Network
    from neuroseq.example_models._elise import SimpleUpdater
    from neuroseq.example_models._elise import SomaticWeights

    np.seterr(**_prev_seterr)

    p = _flat_defaults()
    if params:
        p.update(params)

    if "u_inp" not in stims_dict:
        raise ValueError(
            "stims_dict must contain a 'u_inp' entry (visible-neuron drive). "
            f"Got: {list(stims_dict)}"
        )
    u_inp_full = np.asarray(stims_dict["u_inp"]["array"], dtype=np.float64)
    if u_inp_full.ndim != 2 or u_inp_full.shape[0] == 0:
        raise ValueError(
            f"'u_inp' array must be 2D with shape [T, num_vis]; got {u_inp_full.shape}"
        )
    stim_dt_s = float(stims_dict["u_inp"]["dt"])
    num_vis = int(u_inp_full.shape[1])

    network_params, neuron_params, weight_params = _build_namespaces(p, num_vis)

    # Spawn child RNGs (mirrors sequence-learning/scripts/train.py:setup_network).
    child_seeds = np.random.SeedSequence(_int_param(p, "seed")).spawn(4)
    rng_d_den, rng_d_som, rng_w_den, rng_w_som = (
        np.random.default_rng(s) for s in child_seeds
    )

    dendritic_weights = DendriticWeights(
        weight_params, rng_w=rng_w_den, rng_d=rng_d_den
    )
    somatic_weights = SomaticWeights(weight_params, rng_w=rng_w_som, rng_d=rng_d_som)

    network = Network(
        network_params, neuron_params, dendritic_weights, somatic_weights, Buffer
    )

    dt_ms = _float_param(p, "dt_ms")
    # ELISE scales eta by 1/dt internally — see scripts/simulator.py:33-34.
    network.prepare_for_simulation(
        dt_ms,
        SimpleUpdater(_float_param(p, "eta_vis") / dt_ms),
        SimpleUpdater(_float_param(p, "eta_lat") / dt_ms),
    )

    n_steps = int(np.round(simulation_time * 1000.0 / dt_ms))
    n_total = network.num_all
    learn = _bool_param(p, "learn")

    u_trace = np.empty((n_total, n_steps), dtype=np.float32)
    v_trace = np.empty((n_total, n_steps), dtype=np.float32)
    r_trace = np.empty((n_total, n_steps), dtype=np.float32)

    snap_interval_ms = _float_param(p, "weight_snapshot_interval_ms")
    snap_every_n_steps = max(1, int(np.round(snap_interval_ms / dt_ms)))
    snap_indices = list(range(0, n_steps, snap_every_n_steps))
    weight_snaps = np.empty((n_total, n_total, len(snap_indices)), dtype=np.float32)
    snap_timestamps = np.empty(len(snap_indices), dtype=np.float32)
    snap_cursor = 0

    stim_T = u_inp_full.shape[0]

    for step in range(n_steps):
        t_s = step * dt_ms / 1000.0
        stim_idx = min(int(t_s / stim_dt_s), stim_T - 1)
        u_inp_step = u_inp_full[stim_idx]

        network(u_inp=u_inp_step, learn=learn)

        u_trace[:, step] = network.u
        v_trace[:, step] = network.v
        r_trace[:, step] = network.r

        if snap_cursor < len(snap_indices) and step == snap_indices[snap_cursor]:
            weight_snaps[:, :, snap_cursor] = network.dendritic_weights
            snap_timestamps[snap_cursor] = t_s
            snap_cursor += 1

    indices = np.arange(n_total, dtype=np.uint32)
    dt_s = dt_ms / 1000.0

    data: list[Data] = [
        TraceData(
            name="somatic_potential_u",
            traces=u_trace,
            indices=indices,
            units="mV",
            dt=dt_s,
        ),
        TraceData(
            name="dendritic_potential_v",
            traces=v_trace,
            indices=indices,
            units="mV",
            dt=dt_s,
        ),
        TraceData(
            name="firing_rate_r",
            traces=r_trace,
            indices=indices,
            units="1",
            dt=dt_s,
        ),
        SynapseData(
            name="dendritic_weights",
            values=weight_snaps,
            timestamps=snap_timestamps,
        ),
    ]
    return Experiment(
        name="ELISE simulation",
        duration=simulation_time,
        timepoints=[],
        data=data,
    )
