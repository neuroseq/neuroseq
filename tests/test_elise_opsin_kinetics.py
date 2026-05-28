"""Unit and integration tests for the 3-state opsin kinetics layer in ELISE.

Covers the standalone kinetic step (dark decay, saturation, paired-pulse
depression, closed-form steady state), the inverse-Hill calibration helper,
the Für-Elise steady-state u_tgt regression, reset/state-persistence
contracts, and signature-driven cache invalidation.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroseq.example_models.elise import DEFAULT_PARAMS
from neuroseq.example_models.elise import Elise
from neuroseq.example_models.elise import _step_kinetics
from neuroseq.example_projects import _ELISE_G0_EXC
from neuroseq.example_projects import _ELISE_G0_INH
from neuroseq.example_projects import _ELISE_KIN_EXC
from neuroseq.example_projects import _ELISE_KIN_INH
from neuroseq.example_projects import _FUER_ELISE_PHI_EXC_ACTIVE
from neuroseq.example_projects import _FUER_ELISE_PHI_EXC_REST
from neuroseq.example_projects import _FUER_ELISE_PHI_INH_ACTIVE
from neuroseq.example_projects import _FUER_ELISE_PHI_INH_REST
from neuroseq.example_projects import _calibrate_phi_for_target_open_fraction
from neuroseq.example_projects import elise_project

# ---------------------------------------------------------------------------
# _step_kinetics — standalone tests (no ELISE wrapper)
# ---------------------------------------------------------------------------


def test_step_kinetics_dark_returns_O_to_zero():
    """phi=0 with O initialised to 0.5: O decays to ~0 within 2 s; D peaks then decays."""
    O = np.array([0.5], dtype=np.float64)  # noqa: E741
    D = np.zeros(1, dtype=np.float64)
    phi = np.zeros(1, dtype=np.float64)
    dt = 1e-4
    D_peak = 0.0
    for _ in range(int(2.0 / dt)):
        _step_kinetics(O, D, phi, k_a_max=1000.0, phi_m=5e17, k_d=50.0, k_r=5.0, dt=dt)
        if D[0] > D_peak:
            D_peak = D[0]
    assert O[0] < 1e-6
    assert D_peak > 0.1
    assert D[0] < D_peak * 0.5


def test_step_kinetics_reaches_steady_state_at_saturating_light():
    """At φ = 100·phi_m, O converges to the closed-form O*_sat within 1e-3."""
    O = np.zeros(1)  # noqa: E741
    D = np.zeros(1)
    phi_m, k_a_max, k_d, k_r = 5e17, 1000.0, 50.0, 5.0
    phi = np.array([100.0 * phi_m])
    dt = 1e-4
    for _ in range(int(0.5 / dt)):
        _step_kinetics(O, D, phi, k_a_max, phi_m, k_d, k_r, dt)
    k_a = k_a_max * phi[0] / (phi[0] + phi_m)
    O_star = (k_a * k_r) / (k_a * k_r + k_a * k_d + k_d * k_r)
    assert abs(O[0] - O_star) < 1e-3


def test_step_kinetics_paired_pulse_depression():
    """Two 10-ms pulses, 200 ms ISI: second peak is 30–80% of first."""
    O = np.zeros(1)  # noqa: E741
    D = np.zeros(1)
    phi_sat = np.array([5e19])
    phi_dark = np.zeros(1)
    dt = 1e-4
    peak1 = 0.0
    for _ in range(int(0.01 / dt)):
        _step_kinetics(O, D, phi_sat, 1000.0, 5e17, 50.0, 5.0, dt)
        if O[0] > peak1:
            peak1 = O[0]
    for _ in range(int(0.2 / dt)):
        _step_kinetics(O, D, phi_dark, 1000.0, 5e17, 50.0, 5.0, dt)
    peak2 = 0.0
    for _ in range(int(0.01 / dt)):
        _step_kinetics(O, D, phi_sat, 1000.0, 5e17, 50.0, 5.0, dt)
        if O[0] > peak2:
            peak2 = O[0]
    ratio = peak2 / peak1
    assert 0.3 <= ratio <= 0.8, f"PPR ratio {ratio} out of bounds"


def test_steady_state_open_fraction_matches_closed_form_grid():
    """Forward-integrated O* matches the analytic formula across a φ grid."""
    k_a_max, phi_m, k_d, k_r = 1000.0, 5e17, 50.0, 5.0
    dt = 1e-4
    for phi_scalar in [0.0, 1e15, 5e17, 1e18, 1e20]:
        O = np.zeros(1)  # noqa: E741
        D = np.zeros(1)
        phi = np.array([phi_scalar])
        for _ in range(int(1.0 / dt)):
            _step_kinetics(O, D, phi, k_a_max, phi_m, k_d, k_r, dt)
        k_a = k_a_max * phi_scalar / (phi_scalar + phi_m)
        O_star = 0.0 if k_a == 0 else (k_a * k_r) / (k_a * k_r + k_a * k_d + k_d * k_r)
        assert abs(O[0] - O_star) < 1e-4, (
            f"phi={phi_scalar:.1e}: integrated {O[0]:.6f} vs O*={O_star:.6f}"
        )


def test_step_kinetics_vectorised_across_neurons():
    """The same step on n=5 independent neurons matches the scalar path."""
    O = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # noqa: E741
    D = np.zeros(5)
    phi = np.array([0.0, 1e17, 5e17, 1e18, 1e20])
    O_scalar = O.copy()
    D_scalar = D.copy()
    dt = 1e-4
    _step_kinetics(O, D, phi, 1000.0, 5e17, 50.0, 5.0, dt)
    for k in range(5):
        Ok = O_scalar[k : k + 1].copy()
        Dk = D_scalar[k : k + 1].copy()
        _step_kinetics(Ok, Dk, phi[k : k + 1], 1000.0, 5e17, 50.0, 5.0, dt)
        assert np.isclose(O[k], Ok[0])
        assert np.isclose(D[k], Dk[0])


# ---------------------------------------------------------------------------
# Inverse Hill calibration
# ---------------------------------------------------------------------------


def test_calibrate_phi_inverse_hill_roundtrip():
    k_a_max, phi_m, k_d, k_r = 1000.0, 5e17, 50.0, 5.0
    for O_target in [0.001, 0.01, 0.05, 0.08, 0.09]:
        phi = _calibrate_phi_for_target_open_fraction(
            O_target, k_a_max, phi_m, k_d, k_r
        )
        k_a = k_a_max * phi / (phi + phi_m)
        O_star = (k_a * k_r) / (k_a * k_r + k_a * k_d + k_d * k_r)
        assert abs(O_star - O_target) < 1e-9


def test_calibrate_phi_zero_target_returns_zero():
    assert _calibrate_phi_for_target_open_fraction(0.0, 1000.0, 5e17, 50.0, 5.0) == 0.0


def test_calibrate_phi_raises_above_physical_maximum():
    # k_d=50, k_r=5: denom = k_r*(1-O) - k_d*O > 0 only when O < k_r/(k_d+k_r) = 5/55 ≈ 0.09
    with pytest.raises(ValueError):
        _calibrate_phi_for_target_open_fraction(
            0.5, k_a_max=1000.0, phi_m=5e17, k_d=50.0, k_r=5.0
        )


def test_calibrate_phi_raises_above_k_a_max_saturation():
    # Inhibitory params (k_d=10, k_r=5) have a denom cutoff at O = 1/3.
    # An O_target just under that cutoff should require k_a > k_a_max → raise.
    with pytest.raises(ValueError):
        _calibrate_phi_for_target_open_fraction(
            0.3325, k_a_max=800.0, phi_m=5e17, k_d=10.0, k_r=5.0
        )


# ---------------------------------------------------------------------------
# Für-Elise per-pulse u_tgt regression (transient calibration)
# ---------------------------------------------------------------------------


def _pulse_peak_u_tgt(
    phi_exc: float,
    phi_inh: float,
    pulse_ms: int = 10,
) -> float:
    """Integrate the kinetic step at the given fluxes for `pulse_ms`,
    starting from O = D = 0, and return the matching potential at
    pulse peak."""
    O_exc = np.zeros(1)  # noqa: E741
    D_exc = np.zeros(1)
    O_inh = np.zeros(1)  # noqa: E741
    D_inh = np.zeros(1)
    dt = 1e-4
    phi_e_arr = np.array([phi_exc])
    phi_i_arr = np.array([phi_inh])
    O_e_peak = O_i_peak = 0.0
    for _ in range(int(round(pulse_ms * 1e-3 / dt))):
        _step_kinetics(O_exc, D_exc, phi_e_arr, dt=dt, **_ELISE_KIN_EXC)
        _step_kinetics(O_inh, D_inh, phi_i_arr, dt=dt, **_ELISE_KIN_INH)
        if O_exc[0] > O_e_peak:
            O_e_peak = float(O_exc[0])
        if O_inh[0] > O_i_peak:
            O_i_peak = float(O_inh[0])
    g_exc = _ELISE_G0_EXC * O_e_peak
    g_inh = _ELISE_G0_INH * O_i_peak
    return (g_exc * 0.0 + g_inh * (-75.0)) / (g_exc + g_inh)


def test_fuer_elise_active_10ms_pulse_peak_u_tgt_is_minus_50_mv():
    """The active state is a 10 ms square pulse at phi_m on both opsins.
    The transient calibration of g0 places the matching potential at peak
    exactly at the −50 mV teaching target."""
    u = _pulse_peak_u_tgt(_FUER_ELISE_PHI_EXC_ACTIVE, _FUER_ELISE_PHI_INH_ACTIVE)
    assert abs(u - (-50.0)) < 0.5, f"active peak u_tgt={u} mV"


def test_fuer_elise_rest_is_lights_off():
    """Rest pitches receive zero flux on both opsins. The simulator's
    `np.where(sum_k > 0, u_tgt, E_l)` branch covers inactive pitches
    during another pitch's active event; when ALL pitches are off the
    outer guard yields `u_inp_step = None`."""
    assert _FUER_ELISE_PHI_EXC_REST == 0.0
    assert _FUER_ELISE_PHI_INH_REST == 0.0


# ---------------------------------------------------------------------------
# ELISE wrapper integration: reset, state continuity, signature invalidation
# ---------------------------------------------------------------------------


def _elise_stims_dict(project):
    stim_exc = project.stim_arrays["opsin_exc"]
    stim_inh = project.stim_arrays["opsin_inh"]
    return {
        "opsin_exc": {"array": stim_exc.values, "dt": stim_exc.dt},
        "opsin_inh": {"array": stim_inh.values, "dt": stim_inh.dt},
    }


def test_kinetic_state_allocated_after_first_simulate():
    project = elise_project()
    sim = Elise()
    sim.simulate(_elise_stims_dict(project), simulation_time=0.005)
    assert sim._O_exc is not None and sim._O_exc.shape == (13,)
    assert sim._D_exc is not None and sim._D_exc.shape == (13,)
    assert sim._O_inh is not None and sim._O_inh.shape == (13,)
    assert sim._D_inh is not None and sim._D_inh.shape == (13,)
    # Saturating active flux on pitch 0 (event 0) → O_exc[0] > 0.
    assert sim._O_exc[0] > 0


def test_kinetic_reset_zeros_state_arrays():
    project = elise_project()
    sim = Elise()
    sim.simulate(_elise_stims_dict(project), simulation_time=0.01)
    assert sim._O_exc is not None and sim._O_exc.any()
    sim.reset()
    assert sim._O_exc is None
    assert sim._D_exc is None
    assert sim._O_inh is None
    assert sim._D_inh is None


def test_kinetic_state_persists_across_runs():
    """Two 50 ms runs should match one 100 ms run on (O, D)."""
    project = elise_project()
    sd = _elise_stims_dict(project)
    sim_a = Elise()
    sim_a.simulate(sd, simulation_time=0.05)
    sim_a.simulate(sd, simulation_time=0.05)
    sim_b = Elise()
    sim_b.simulate(sd, simulation_time=0.10)
    np.testing.assert_allclose(sim_a._O_exc, sim_b._O_exc, atol=1e-10)
    np.testing.assert_allclose(sim_a._D_exc, sim_b._D_exc, atol=1e-10)
    np.testing.assert_allclose(sim_a._O_inh, sim_b._O_inh, atol=1e-10)
    np.testing.assert_allclose(sim_a._D_inh, sim_b._D_inh, atol=1e-10)


def test_kinetic_param_change_invalidates_state():
    """Changing k_d_exc between runs triggers reset() via the trajectory-param
    partition in update_params, so the state arrays get re-grown from zero."""
    project = elise_project()
    sd = _elise_stims_dict(project)
    sim = Elise()
    sim.simulate(sd, simulation_time=0.05)
    old_O = sim._O_exc.copy()
    params = {k: v["default"] for k, v in DEFAULT_PARAMS.items()}
    params["k_d_exc"] = 80.0
    sim.simulate(sd, simulation_time=0.05, params=params)
    assert not np.allclose(sim._O_exc, old_O), (
        "State should have been reset and re-grown after a kinetic param change"
    )
