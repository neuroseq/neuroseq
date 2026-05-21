"""ELISE network model exposed through a stateful simulator class.

Wraps the rate-coded ELISE network from the ``elise`` package (sequence-learning
by Kriener et al., 2024) behind neuroseq's
``simulate(stims_dict, simulation_time, params=None) -> Experiment`` contract.

The user-facing stim API uses two virtual opsins, ``opsin_exc`` and
``opsin_inh``. Their per-pitch ratio determines the matching potential;
their absolute magnitudes do not. Overall teaching strength remains
controlled by ``lam``.

The module-level ``simulate`` function delegates to a singleton ``NN``
instance so existing project and GUI integration remains unchanged while
preserving simulation state across repeated runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.example_models.NN import NN
from neuroseq.example_models.NN import ParamValue
from neuroseq.example_models.NN import ProgressCallback

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


class Elise(NN):
    """Stateful ELISE simulator backend.

    Responsibilities:
    1) expose editable parameter metadata for the GUI,
    2) initialize from default parameters,
    3) retain network state and cumulative simulated time,
    4) continue simulation across repeated runs,
    5) return each run as an ``Experiment``,
    6) consume two opsin-activation stim arrays (``opsin_exc``,
       ``opsin_inh``) and derive the upstream ``u_inp`` from their
       ratio per pitch per step. Release is "both opsins at zero
       everywhere" - no separate parameter required.
    """

    # Parameters whose toggling between runs should NOT discard the
    # network state - phase controls rather than trajectory controls.
    _PHASE_PARAMS: frozenset[str] = frozenset({"learn"})

    def __init__(self) -> None:
        super().__init__(DEFAULT_PARAMS)
        self._network = None
        self._signature: tuple[int, tuple[tuple[str, ParamValue], ...]] | None = None
        self._elise_types: tuple[type, type, type, type, type] | None = None

    def reset(self) -> None:
        self._network = None
        self._simulated_time_s = 0.0
        self._signature = None

    def update_params(self, params: Mapping[str, ParamValue] | None) -> None:
        merged = self._flat_defaults()
        if params:
            # Drop unknown keys (legacy archives may still carry
            # ``clamp_visible``); they're no longer meaningful.
            known = set(self._default_params_schema.keys())
            merged.update({k: v for k, v in params.items() if k in known})
        old_traj = {
            k: v for k, v in self._params.items() if k not in self._PHASE_PARAMS
        }
        new_traj = {k: v for k, v in merged.items() if k not in self._PHASE_PARAMS}
        if old_traj != new_traj:
            self._params = merged
            self.reset()
        else:
            self._params = merged

    def _import_elise_types(self) -> tuple[type, type, type, type, type]:
        if self._elise_types is not None:
            return self._elise_types
        # Imported here so DEFAULT_PARAMS can be inspected without elise installed.
        # `elise.model` calls ``np.seterr(all="raise")`` at module load, which would
        # leak into the rest of the process and break unrelated numpy code that
        # tolerates over/underflow. Save and restore the global errstate around the
        # import so the change is scoped to ELISE-internal computation.
        _prev_seterr = np.seterr()
        from elise.model import Network
        from elise.optimizer import SimpleUpdater
        from elise.rate_buffer import Buffer
        from elise.weights import DendriticWeights
        from elise.weights import SomaticWeights

        np.seterr(**_prev_seterr)
        self._elise_types = (Network, SimpleUpdater, Buffer, DendriticWeights, SomaticWeights)
        return self._elise_types

    def _initialize_network(self, num_vis: int) -> None:
        (
            Network,
            SimpleUpdater,
            Buffer,
            DendriticWeights,
            SomaticWeights,
        ) = self._import_elise_types()
        network_params, neuron_params, weight_params = _build_namespaces(self._params, num_vis)

        # Spawn child RNGs (mirrors sequence-learning/scripts/train.py:setup_network).
        child_seeds = np.random.SeedSequence(_int_param(self._params, "seed")).spawn(4)
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

        dt_ms = _float_param(self._params, "dt_ms")
        # ELISE scales eta by 1/dt internally — see scripts/simulator.py:33-34.
        network.prepare_for_simulation(
            dt_ms,
            SimpleUpdater(_float_param(self._params, "eta_vis") / dt_ms),
            SimpleUpdater(_float_param(self._params, "eta_lat") / dt_ms),
        )
        self._network = network

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, ParamValue] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Experiment:
        if "opsin_exc" not in stims_dict or "opsin_inh" not in stims_dict:
            raise ValueError(
                "stims_dict must contain both 'opsin_exc' and 'opsin_inh' "
                "entries (excitatory and inhibitory virtual opsin "
                f"activations). Got: {list(stims_dict)}"
            )
        g_exc_full = np.asarray(stims_dict["opsin_exc"]["array"], dtype=np.float64)
        g_inh_full = np.asarray(stims_dict["opsin_inh"]["array"], dtype=np.float64)
        if g_exc_full.shape != g_inh_full.shape:
            raise ValueError(
                f"'opsin_exc' and 'opsin_inh' must have the same shape; "
                f"got {g_exc_full.shape} vs {g_inh_full.shape}"
            )
        if g_exc_full.ndim != 2 or g_exc_full.shape[0] == 0:
            raise ValueError(
                f"opsin stim arrays must be 2D with shape [T, num_vis]; "
                f"got {g_exc_full.shape}"
            )
        stim_dt_s = float(stims_dict["opsin_exc"]["dt"])
        num_vis = int(g_exc_full.shape[1])

        self.update_params(params)
        signature = (
            num_vis,
            tuple(
                sorted(
                    (k, v) for k, v in self._params.items() if k not in self._PHASE_PARAMS
                )
            ),
        )
        if self._network is None or signature != self._signature:
            self._initialize_network(num_vis)
            self._signature = signature
            self._simulated_time_s = 0.0

        if self._network is None:
            raise RuntimeError("ELISE network failed to initialize")

        dt_ms = _float_param(self._params, "dt_ms")
        n_steps = int(np.round(simulation_time * 1000.0 / dt_ms))
        n_total = self._network.num_all
        learn = _bool_param(self._params, "learn")
        E_exc = _float_param(self._params, "E_exc")
        E_inh = _float_param(self._params, "E_inh")
        E_l = _float_param(self._params, "E_l")

        u_trace = np.empty((n_total, n_steps), dtype=np.float32)
        v_trace = np.empty((n_total, n_steps), dtype=np.float32)
        r_trace = np.empty((n_total, n_steps), dtype=np.float32)

        snap_interval_ms = _float_param(self._params, "weight_snapshot_interval_ms")
        snap_every_n_steps = max(1, int(np.round(snap_interval_ms / dt_ms)))
        snap_indices = list(range(0, n_steps, snap_every_n_steps))
        weight_snaps = np.empty((n_total, n_total, len(snap_indices)), dtype=np.float32)
        snap_timestamps = np.empty(len(snap_indices), dtype=np.float32)
        snap_cursor = 0

        stim_T = g_exc_full.shape[0]
        t0_s = self._simulated_time_s

        # Throttle progress callbacks so they refresh roughly every 5 % of
        # the simulation without dominating the loop's runtime.
        progress_stride = max(1, n_steps // 20) if progress is not None else 0
        if progress is not None and n_steps > 0:
            progress(0.0)

        for step in range(n_steps):
            t_s = step * dt_ms / 1000.0
            t_abs_s = t0_s + t_s
            stim_idx = min(int(t_abs_s / stim_dt_s), stim_T - 1)
            g_exc_k = g_exc_full[stim_idx]
            g_inh_k = g_inh_full[stim_idx]
            sum_k = g_exc_k + g_inh_k
            if not np.any(sum_k > 0):
                u_inp_step = None
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    u_tgt = (g_exc_k * E_exc + g_inh_k * E_inh) / sum_k
                u_tgt = np.where(sum_k > 0, u_tgt, E_l)
                u_inp_step = u_tgt.astype(np.float64)

            self._network(u_inp=u_inp_step, learn=learn)

            u_trace[:, step] = self._network.u
            v_trace[:, step] = self._network.v
            r_trace[:, step] = self._network.r

            if snap_cursor < len(snap_indices) and step == snap_indices[snap_cursor]:
                weight_snaps[:, :, snap_cursor] = self._network.dendritic_weights
                # Keep timestamps local to this run so appending can offset cleanly.
                snap_timestamps[snap_cursor] = t_s
                snap_cursor += 1

            if progress_stride and (step + 1) % progress_stride == 0:
                progress((step + 1) / n_steps)

        if progress is not None and n_steps > 0:
            progress(1.0)

        self._simulated_time_s += simulation_time

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


def simulate(
    stims_dict: dict,
    simulation_time: float,
    params: Mapping[str, ParamValue] | None = None,
    progress: ProgressCallback | None = None,
) -> Experiment:
    """Compatibility wrapper using the singleton ``Elise`` instance."""
    return _SIMULATOR.simulate(
        stims_dict, simulation_time, params=params, progress=progress
    )


def get_simulator() -> Elise:
    return _SIMULATOR



_SIMULATOR = Elise()
