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

import sys
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
from neuroseq.replay_metrics import compute_performance_against_target

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
    # Opsin kinetics (excitatory) — 3-state collapsed PyRho model.
    "k_a_max_exc": {
        "value": 1000.0,
        "default": 1000.0,
        "type": "float",
        "group": "Opsin (excitatory)",
        "description": "Saturating activation rate (1/s).",
    },
    "phi_m_exc": {
        "value": 5e17,
        "default": 5e17,
        "type": "float",
        "group": "Opsin (excitatory)",
        "description": "Hill half-activation photon flux (1/mm^2/s).",
    },
    "k_d_exc": {
        "value": 50.0,
        "default": 50.0,
        "type": "float",
        "group": "Opsin (excitatory)",
        "description": "Desensitization rate O→D (1/s).",
    },
    "k_r_exc": {
        "value": 5.0,
        "default": 5.0,
        "type": "float",
        "group": "Opsin (excitatory)",
        "description": "Recovery rate D→C (1/s).",
    },
    "g0_exc": {
        "value": 0.3690150955780965,
        "default": 0.3690150955780965,
        "type": "float",
        "group": "Opsin (excitatory)",
        "description": "Peak conductance scaling (dimensionless).",
    },
    # Opsin kinetics (inhibitory).
    "k_a_max_inh": {
        "value": 800.0,
        "default": 800.0,
        "type": "float",
        "group": "Opsin (inhibitory)",
        "description": "Saturating activation rate (1/s).",
    },
    "phi_m_inh": {
        "value": 5e17,
        "default": 5e17,
        "type": "float",
        "group": "Opsin (inhibitory)",
        "description": "Hill half-activation photon flux (1/mm^2/s).",
    },
    "k_d_inh": {
        "value": 10.0,
        "default": 10.0,
        "type": "float",
        "group": "Opsin (inhibitory)",
        "description": "Desensitization rate O→D (1/s).",
    },
    "k_r_inh": {
        "value": 5.0,
        "default": 5.0,
        "type": "float",
        "group": "Opsin (inhibitory)",
        "description": "Recovery rate D→C (1/s).",
    },
    "g0_inh": {
        "value": 0.6309849044219035,
        "default": 0.6309849044219035,
        "type": "float",
        "group": "Opsin (inhibitory)",
        "description": "Peak conductance scaling (dimensionless).",
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


def _step_kinetics(
    O: np.ndarray,  # noqa: E741 — "O" is the open-state fraction (matches the 3-state PyRho spec)
    D: np.ndarray,
    phi: np.ndarray,
    k_a_max: float,
    phi_m: float,
    k_d: float,
    k_r: float,
    dt: float,
) -> None:
    """Forward-Euler step on the 3-state opsin kinetics.

    Mutates ``O`` (open) and ``D`` (desensitized) in place. ``C = 1 - O - D``
    (closed) is reconstructed each step. Hill activation with exponent 1;
    ``phi = 0`` yields ``k_a = 0`` cleanly. ``dt`` is in seconds; rates in s^-1.
    """
    k_a = k_a_max * phi / (phi + phi_m)
    C = 1.0 - O - D
    dO = (k_a * C - k_d * O) * dt
    dD = (k_d * O - k_r * D) * dt
    O += dO  # noqa: E741
    D += dD


def _matching_potential(
    g_exc: np.ndarray,
    g_inh: np.ndarray,
    E_exc: float,
    E_inh: float,
    E_l: float,
) -> np.ndarray:
    """Conductance-based matching potential ``u_tgt`` (same as :meth:`Elise.simulate`).

    Where no opsin conductance is present, the teacher falls through to the
    leak potential ``E_l`` (i.e. ``u_inp = None`` for that pitch).
    """
    sum_g = g_exc + g_inh
    with np.errstate(invalid="ignore", divide="ignore"):
        u_tgt = (g_exc * E_exc + g_inh * E_inh) / sum_g
    return np.where(sum_g > 0, u_tgt, E_l)


def _opsin_u_tgt_cycle(
    flux_exc_cycle: np.ndarray,
    flux_inh_cycle: np.ndarray,
    *,
    kin_exc: tuple[float, float, float, float],
    kin_inh: tuple[float, float, float, float],
    g0_exc: float,
    g0_inh: float,
    E_exc: float,
    E_inh: float,
    E_l: float,
    dt_s: float,
    warmup_cycles: int = 3,
) -> np.ndarray:
    """Steady-state opsin teacher voltage over one cycle.

    Runs the 3-state kinetics on the (tiled, one-cycle) photon-flux arrays for
    ``warmup_cycles`` cycles so the slow desensitization/recovery tails reach
    their periodic steady state, then returns the last cycle's matching
    potential ``u_tgt(t)``. This is the voltage the opsin teacher *actually*
    nudges the soma toward, so ``φ(u_tgt)`` is the paper-consistent target for
    ``teacher_mode="opsin"`` (the delivered teacher, not the crisp binary).
    """
    nv, T = flux_exc_cycle.shape
    O_exc = np.zeros(nv)
    D_exc = np.zeros(nv)
    O_inh = np.zeros(nv)
    D_inh = np.zeros(nv)
    u_tgt = np.empty((nv, T), dtype=np.float64)
    for _ in range(max(1, warmup_cycles)):
        for t in range(T):
            _step_kinetics(O_exc, D_exc, flux_exc_cycle[:, t], *kin_exc, dt_s)
            _step_kinetics(O_inh, D_inh, flux_inh_cycle[:, t], *kin_inh, dt_s)
            u_tgt[:, t] = _matching_potential(
                g0_exc * O_exc, g0_inh * O_inh, E_exc, E_inh, E_l
            )
    return u_tgt


def _detect_pattern_cycle(stim: np.ndarray) -> int:
    """Return the smallest period (in samples) at which ``stim`` repeats.

    A period ``T`` is accepted only if **every** whole cycle of length
    ``T`` matches the first one, i.e. ``stim[k·T : (k+1)·T] == stim[:T]``
    for all ``k = 1 .. n // T - 1``. Falls back to ``stim.shape[0]`` if
    no shorter period is found.
    """
    n = int(stim.shape[0])
    for T in range(2, n // 2 + 1):
        n_full = n // T
        if n_full < 2:
            break
        first = stim[:T]
        if all(
            np.array_equal(stim[k * T : (k + 1) * T], first) for k in range(1, n_full)
        ):
            return T
    return n


def build_target_rate_pattern(
    opsin_exc: np.ndarray,
    *,
    E_l: float = -70.0,
    a: float = 0.3,
    b: float = -58.0,
    u_active_offset: float = 20.0,
    cycle_length: int | None = None,
) -> tuple[np.ndarray, int]:
    """Construct one cycle of the paper-style target firing rate.

    Implements the target ``y`` of Kriener et al. (2024), Sec. 5.3.3 +
    Eq. 6. A pitch is "active" at a stim sample whenever ``opsin_exc > 0``;
    the corresponding output neuron is being teacher-driven toward
    ``u_tgt = E_l + u_active_offset`` (default −50 mV), and toward ``E_l``
    otherwise. The asymptotic firing rate the soma would produce at that
    voltage is ``y = 1 / (1 + exp(a · (b − u_tgt)))``.

    Parameters
    ----------
    opsin_exc : np.ndarray, shape ``(T_total, num_vis)``
        Excitatory opsin photon-flux stim array. Only the binary
        on/off footprint per sample is used; absolute fluxes are ignored.
    E_l, a, b, u_active_offset : float
        Neuron parameters (must match the running simulation).
    cycle_length : int, optional
        Pattern period in stim samples. Autodetected if ``None``.

    Returns
    -------
    target_rate : np.ndarray, shape ``(num_vis, cycle_length)``
        Target firing rate in [0, 1], one full cycle.
    cycle_length : int
        Detected (or supplied) cycle length, in stim samples.
    """
    if opsin_exc.ndim != 2:
        raise ValueError("opsin_exc must be 2-D (T_total, num_vis)")

    pattern = (np.asarray(opsin_exc) > 0).astype(np.float64)
    if cycle_length is None:
        # Trim trailing all-zero rows (ELiSe ring-out is encoded that way)
        # before detecting the period, so the silence tail doesn't force
        # the detector to fall back to the full stim length.
        active_mask = np.any(pattern > 0, axis=1)
        active_idx = np.flatnonzero(active_mask)
        if active_idx.size == 0:
            raise ValueError(
                "opsin_exc has no active samples; cannot infer a target pattern"
            )
        active_len = int(active_idx[-1]) + 1
        cycle_length = _detect_pattern_cycle(pattern[:active_len])
    if cycle_length < 2 or cycle_length > pattern.shape[0]:
        raise ValueError(
            f"cycle_length must be in [2, {pattern.shape[0]}]; got {cycle_length}"
        )

    one_cycle = pattern[:cycle_length, :].T  # (num_vis, cycle_length)
    u_tgt = E_l + u_active_offset * one_cycle
    target_rate = 1.0 / (1.0 + np.exp(a * (b - u_tgt)))
    return target_rate, int(cycle_length)


def _build_namespaces(
    params: Mapping[str, ParamValue], num_vis: int
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    network_params = SimpleNamespace(
        num_vis=num_vis, num_lat=_int_param(params, "num_lat")
    )
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
        self._O_exc: np.ndarray | None = None
        self._D_exc: np.ndarray | None = None
        self._O_inh: np.ndarray | None = None
        self._D_inh: np.ndarray | None = None

    def reset(self) -> None:
        self._network = None
        self._simulated_time_s = 0.0
        self._signature = None
        self._O_exc = None
        self._D_exc = None
        self._O_inh = None
        self._D_inh = None

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
        try:
            from elise.model import Network
            from elise.optimizer import SimpleUpdater
            from elise.rate_buffer import Buffer
            from elise.weights import DendriticWeights
            from elise.weights import SomaticWeights
        except ModuleNotFoundError as exc:
            np.seterr(**_prev_seterr)
            if (
                exc.name and (exc.name == "elise" or exc.name.startswith("elise."))
            ) or "No module named 'elise'" in str(exc):
                raise ImportError(
                    "The Elise example model requires the separate private "
                    f"'elise' package in the active Python environment "
                    f"({sys.executable}). Install/configure that package there "
                    "before running this example."
                ) from exc
            raise

        np.seterr(**_prev_seterr)
        self._elise_types = (
            Network,
            SimpleUpdater,
            Buffer,
            DendriticWeights,
            SomaticWeights,
        )
        return self._elise_types

    def _initialize_network(self, num_vis: int) -> None:
        (
            Network,
            SimpleUpdater,
            Buffer,
            DendriticWeights,
            SomaticWeights,
        ) = self._import_elise_types()
        network_params, neuron_params, weight_params = _build_namespaces(
            self._params, num_vis
        )

        # Spawn child RNGs (mirrors sequence-learning/scripts/train.py:setup_network).
        child_seeds = np.random.SeedSequence(_int_param(self._params, "seed")).spawn(4)
        rng_d_den, rng_d_som, rng_w_den, rng_w_som = (
            np.random.default_rng(s) for s in child_seeds
        )

        dendritic_weights = DendriticWeights(
            weight_params, rng_w=rng_w_den, rng_d=rng_d_den
        )
        somatic_weights = SomaticWeights(
            weight_params, rng_w=rng_w_som, rng_d=rng_d_som
        )

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
        # 3-state opsin kinetics: O (open) and D (desensitized) per pitch.
        # C = 1 - O - D is reconstructed each step. All start fully closed.
        self._O_exc = np.zeros(num_vis, dtype=np.float64)
        self._D_exc = np.zeros(num_vis, dtype=np.float64)
        self._O_inh = np.zeros(num_vis, dtype=np.float64)
        self._D_inh = np.zeros(num_vis, dtype=np.float64)

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
                    (k, v)
                    for k, v in self._params.items()
                    if k not in self._PHASE_PARAMS
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
        dt_s = dt_ms / 1000.0
        n_steps = int(np.round(simulation_time * 1000.0 / dt_ms))
        n_total = self._network.num_all
        learn = _bool_param(self._params, "learn")
        E_exc = _float_param(self._params, "E_exc")
        E_inh = _float_param(self._params, "E_inh")
        E_l = _float_param(self._params, "E_l")
        # Opsin kinetic parameters — read once per run.
        # NOTE: g_exc_full and g_inh_full now carry photon-flux density
        # (mm^-2 s^-1), not fractional conductance. The kinetic step
        # below converts flux → open fraction → conductance.
        k_a_max_exc = _float_param(self._params, "k_a_max_exc")
        phi_m_exc = _float_param(self._params, "phi_m_exc")
        k_d_exc = _float_param(self._params, "k_d_exc")
        k_r_exc = _float_param(self._params, "k_r_exc")
        g0_exc = _float_param(self._params, "g0_exc")
        k_a_max_inh = _float_param(self._params, "k_a_max_inh")
        phi_m_inh = _float_param(self._params, "phi_m_inh")
        k_d_inh = _float_param(self._params, "k_d_inh")
        k_r_inh = _float_param(self._params, "k_r_inh")
        g0_inh = _float_param(self._params, "g0_inh")
        assert self._O_exc is not None
        assert self._D_exc is not None
        assert self._O_inh is not None
        assert self._D_inh is not None

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

        # Throttle progress callbacks so they refresh ~0.1 % of the run.
        # Frequent enough that the GUI can pump events smoothly during long
        # training, sparse enough that the callback's own cost is < 1 %.
        progress_stride = max(1, n_steps // 1000) if progress is not None else 0
        if progress is not None and n_steps > 0:
            progress(0.0)

        for step in range(n_steps):
            t_s = step * dt_ms / 1000.0
            t_abs_s = t0_s + t_s
            stim_idx = min(int(t_abs_s / stim_dt_s), stim_T - 1)
            phi_exc = g_exc_full[stim_idx]
            phi_inh = g_inh_full[stim_idx]

            # Forward-Euler kinetic step in place on singleton state arrays.
            _step_kinetics(
                self._O_exc,
                self._D_exc,
                phi_exc,
                k_a_max_exc,
                phi_m_exc,
                k_d_exc,
                k_r_exc,
                dt_s,
            )
            _step_kinetics(
                self._O_inh,
                self._D_inh,
                phi_inh,
                k_a_max_inh,
                phi_m_inh,
                k_d_inh,
                k_r_inh,
                dt_s,
            )

            g_exc_k = g0_exc * self._O_exc
            g_inh_k = g0_inh * self._O_inh
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

    def run_protocol(
        self,
        opsin_exc: np.ndarray,
        opsin_inh: np.ndarray,
        stim_dt_s: float,
        *,
        base_params: Mapping[str, ParamValue] | None = None,
        teacher_mode: str = "direct",
        training_epochs: int = 10,
        training_cycles: int = 25,
        validation_cycles: int = 2,
        replay_epochs: int = 50,
        replay_cycles: int = 2,
        replay_learning: bool = True,
        u_active_offset: float = 20.0,
        cycle_length: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Run the paper's train / validation / replay protocol (Fig. 2).

        Reproduces Kriener et al. (2024), Sec. 5.5–5.6 inside neuroseq by
        stepping the ELiSe network directly (not through :meth:`simulate`),
        so learning, teacher release, and scoring all follow the paper:

        * **Training** — the teaching pattern is presented for
          ``training_cycles`` with plasticity on. The soma is nudged toward the
          target voltage ``u_tgt`` (see ``teacher_mode``).
        * **Validation** — after each training epoch the teacher is released
          (``u_inp = None``) with plasticity off; the free replay over
          ``validation_cycles`` is scored against ``y = φ(u_tgt)`` (Eq. 6).
          This is the Fig. 2b/c *training* curve.
        * **Replay** — after training, ``replay_epochs`` teacher-off epochs are
          scored the same way (``learn=replay_learning``). This is the Fig. 2b/c
          *replay* curve.

        Network state (voltages, rates, weights) carries across every phase —
        there is no reset between cycles, matching the paper. The metric is the
        shift-aligned MSE / Pearson correlation
        (:func:`neuroseq.replay_metrics.compute_performance_against_target`).

        Parameters
        ----------
        opsin_exc, opsin_inh : np.ndarray, shape ``(T_total, num_vis)``
            The project's excitatory / inhibitory opsin stim arrays. Only one
            teaching cycle is used (autodetected, or ``cycle_length``); the rest
            is ignored. For ``teacher_mode="direct"`` only the binary on/off
            footprint of ``opsin_exc`` matters.
        stim_dt_s : float
            Stim sampling interval (seconds). Must be an integer multiple of the
            simulation trace dt (``dt_ms / 1000``).
        base_params : mapping, optional
            Flat ``{name: value}`` parameter overrides (the project's editable
            ELiSe params). The network is (re)initialised fresh from these.
        teacher_mode : {"direct", "opsin"}
            ``"direct"`` (default, paper-faithful): nudge the soma straight to
            the crisp target voltage ``u_tgt = E_l + u_active_offset·pattern``,
            exactly as ``scripts/train.py``. Reproduces Fig. 2 with the crisp
            teaching pattern. ``"opsin"``: derive ``u_tgt`` from the 3-state
            opsin kinetics (as :meth:`simulate` does) — biologically realistic
            for the holographic-stimulation story. The kinetic ramp and
            desensitization tail smear the delivered teacher relative to the
            crisp binary, so in this mode the network is scored against that
            *delivered* teacher (``y = φ`` of the steady-state opsin ``u_tgt``,
            not the crisp binary). It therefore still reaches low replay error —
            learning to replay what the opsin stimulation actually delivers.
        u_active_offset : float
            mV above ``E_l`` for an active pitch's target (paper maps 1 → E_l+20).

        Returns
        -------
        dict
            ``val_mse``, ``val_corr`` (one entry per training epoch),
            ``rep_mse``, ``rep_corr`` (one entry per replay epoch), plus
            ``target_rate`` (resampled), ``cycle_trace_samples``, ``trace_dt_s``,
            ``num_vis``, ``final_replay_rate`` and ``config``.
        """
        if teacher_mode not in ("direct", "opsin"):
            raise ValueError(
                f"teacher_mode must be 'direct' or 'opsin', got {teacher_mode!r}"
            )
        opsin_exc = np.asarray(opsin_exc, dtype=np.float64)
        opsin_inh = np.asarray(opsin_inh, dtype=np.float64)
        if opsin_exc.shape != opsin_inh.shape:
            raise ValueError("opsin_exc and opsin_inh must have the same shape")
        if opsin_exc.ndim != 2 or opsin_exc.shape[0] == 0:
            raise ValueError("opsin arrays must be 2-D (T_total, num_vis)")

        training_epochs = max(1, int(training_epochs))
        training_cycles = max(1, int(training_cycles))
        # compute_performance_against_target needs at least two whole cycles.
        validation_cycles = max(2, int(validation_cycles))
        replay_epochs = max(1, int(replay_epochs))
        replay_cycles = max(2, int(replay_cycles))

        # Fresh network initialised from the requested parameters.
        self.reset()
        self.update_params(base_params)
        base = dict(self._params)

        E_l = _float_param(base, "E_l")
        a = _float_param(base, "a")
        b = _float_param(base, "b")
        trace_dt_s = _float_param(base, "dt_ms") / 1000.0

        target_rate_stim, cyc_samples = build_target_rate_pattern(
            opsin_exc,
            E_l=E_l,
            a=a,
            b=b,
            u_active_offset=u_active_offset,
            cycle_length=cycle_length,
        )
        upsample = int(round(stim_dt_s / trace_dt_s))
        if upsample < 1 or abs(stim_dt_s - upsample * trace_dt_s) > 1e-9:
            raise ValueError(
                f"stim dt ({stim_dt_s * 1000:.4g} ms) must be an integer "
                f"multiple of trace dt ({trace_dt_s * 1000:.4g} ms)"
            )
        num_vis = int(opsin_exc.shape[1])
        T = int(cyc_samples * upsample)

        # Per-trace-step teacher drive, one cycle at trace dt.
        pattern_cycle = (opsin_exc[:cyc_samples, :] > 0).T.astype(np.float64)
        pattern_cycle = np.repeat(pattern_cycle, upsample, axis=1)  # (num_vis, T)
        u_tgt_cycle = E_l + u_active_offset * pattern_cycle  # crisp direct teacher
        flux_exc_cycle = np.repeat(opsin_exc[:cyc_samples, :].T, upsample, axis=1)
        flux_inh_cycle = np.repeat(opsin_inh[:cyc_samples, :].T, upsample, axis=1)

        # Opsin drive parameters (only used for teacher_mode="opsin").
        E_exc = _float_param(base, "E_exc")
        E_inh = _float_param(base, "E_inh")
        kin_exc = (
            _float_param(base, "k_a_max_exc"),
            _float_param(base, "phi_m_exc"),
            _float_param(base, "k_d_exc"),
            _float_param(base, "k_r_exc"),
        )
        kin_inh = (
            _float_param(base, "k_a_max_inh"),
            _float_param(base, "phi_m_inh"),
            _float_param(base, "k_d_inh"),
            _float_param(base, "k_r_inh"),
        )
        g0_exc = _float_param(base, "g0_exc")
        g0_inh = _float_param(base, "g0_inh")

        # The metric target is φ(u_tgt) for the teacher that is *actually*
        # delivered (paper Eq. 6). For the crisp direct teacher that is the
        # binary pattern; for the opsin teacher it is the smeared steady-state
        # kinetic drive, so the network is scored against what it was taught.
        if teacher_mode == "direct":
            target_rate = np.repeat(target_rate_stim, upsample, axis=1)
        else:
            u_steady = _opsin_u_tgt_cycle(
                flux_exc_cycle,
                flux_inh_cycle,
                kin_exc=kin_exc,
                kin_inh=kin_inh,
                g0_exc=g0_exc,
                g0_inh=g0_inh,
                E_exc=E_exc,
                E_inh=E_inh,
                E_l=E_l,
                dt_s=trace_dt_s,
            )
            target_rate = 1.0 / (1.0 + np.exp(a * (b - u_steady)))

        self._initialize_network(num_vis)
        net = self._network
        assert net is not None

        def _teacher_u(t_in_cycle: int) -> np.ndarray:
            if teacher_mode == "direct":
                return u_tgt_cycle[:, t_in_cycle]
            # opsin: advance the network's kinetic state, then matching potential.
            _step_kinetics(
                self._O_exc,
                self._D_exc,
                flux_exc_cycle[:, t_in_cycle],
                *kin_exc,
                trace_dt_s,
            )
            _step_kinetics(
                self._O_inh,
                self._D_inh,
                flux_inh_cycle[:, t_in_cycle],
                *kin_inh,
                trace_dt_s,
            )
            return _matching_potential(
                g0_exc * self._O_exc, g0_inh * self._O_inh, E_exc, E_inh, E_l
            )

        total_steps = (
            training_epochs * (training_cycles + validation_cycles)
            + replay_epochs * replay_cycles
        ) * T
        counter = {"n": 0}

        def _tick() -> None:
            counter["n"] += 1
            if progress is not None and counter["n"] % 2000 == 0:
                progress(counter["n"] / total_steps)

        def _train(n_cycles: int) -> None:
            for k in range(n_cycles * T):
                net(u_inp=_teacher_u(k % T), learn=True)
                _tick()

        def _free_run(n_cycles: int, learn: bool) -> np.ndarray:
            rate = np.empty((num_vis, n_cycles * T), dtype=np.float64)
            for k in range(n_cycles * T):
                net(u_inp=None, learn=learn)
                rate[:, k] = net.r[:num_vis]
                _tick()
            return rate

        def _score(rate: np.ndarray) -> tuple[float, float]:
            mses, corrs, _ = compute_performance_against_target(rate, target_rate)
            return float(np.mean(mses)), float(np.nanmean(corrs))

        if progress is not None:
            progress(0.0)

        val_mse = np.empty(training_epochs, dtype=np.float64)
        val_corr = np.empty(training_epochs, dtype=np.float64)
        for e in range(training_epochs):
            _train(training_cycles)
            val_mse[e], val_corr[e] = _score(_free_run(validation_cycles, False))

        rep_mse = np.empty(replay_epochs, dtype=np.float64)
        rep_corr = np.empty(replay_epochs, dtype=np.float64)
        last_rate: np.ndarray | None = None
        for e in range(replay_epochs):
            last_rate = _free_run(replay_cycles, replay_learning)
            rep_mse[e], rep_corr[e] = _score(last_rate)

        # Record a teacher-off replay over the full stim timeline as a standard
        # Experiment, so the trained activity lines up with the opsin stim
        # arrays in the GUI (u/v/r + weight snapshots, exactly like a normal
        # simulate() run). Weights are frozen (learn=False) so it captures the
        # trained attractor without replay-time drift.
        experiment = self._record_replay_experiment(opsin_exc, stim_dt_s, base)

        if progress is not None:
            progress(1.0)

        return {
            "val_mse": val_mse,
            "val_corr": val_corr,
            "rep_mse": rep_mse,
            "rep_corr": rep_corr,
            "target_rate": target_rate,
            "cycle_trace_samples": T,
            "trace_dt_s": trace_dt_s,
            "num_vis": num_vis,
            "final_replay_rate": last_rate,
            "experiment": experiment,
            "config": {
                "teacher_mode": teacher_mode,
                "training_epochs": training_epochs,
                "training_cycles": training_cycles,
                "validation_cycles": validation_cycles,
                "replay_epochs": replay_epochs,
                "replay_cycles": replay_cycles,
                "replay_learning": bool(replay_learning),
            },
        }

    def _record_replay_experiment(
        self,
        opsin_exc: np.ndarray,
        stim_dt_s: float,
        base: Mapping[str, ParamValue],
    ) -> Experiment:
        """Record a teacher-off replay of the trained network as an Experiment.

        Drives the (already trained) network with zeroed opsin stims for the
        full stim timeline with plasticity off, via :meth:`simulate`, and
        returns the standard Experiment (``somatic_potential_u`` /
        ``dendritic_potential_v`` / ``firing_rate_r`` / ``dendritic_weights``)
        so the GUI can line the learned activity up with the opsin stim arrays.
        """
        zeros = np.zeros_like(opsin_exc)
        rec_stims = {
            "opsin_exc": {"array": zeros, "dt": stim_dt_s},
            "opsin_inh": {"array": zeros, "dt": stim_dt_s},
        }
        rec_params = dict(base)
        rec_params["learn"] = False
        # Restart the stim-index origin so the recording covers [0, duration).
        self._simulated_time_s = 0.0
        experiment = self.simulate(
            rec_stims, opsin_exc.shape[0] * stim_dt_s, params=rec_params
        )
        experiment.name = "ELiSe replay (teacher off, trained)"
        return experiment


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


def run_protocol(
    opsin_exc: np.ndarray,
    opsin_inh: np.ndarray,
    stim_dt_s: float,
    **kwargs: object,
) -> dict:
    """Run the train/validation/replay protocol on the singleton simulator.

    Thin wrapper around :meth:`Elise.run_protocol`; see it for the contract.
    """
    return _SIMULATOR.run_protocol(opsin_exc, opsin_inh, stim_dt_s, **kwargs)  # type: ignore[arg-type]


def get_simulator() -> Elise:
    return _SIMULATOR


_SIMULATOR = Elise()
