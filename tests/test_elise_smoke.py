"""Standalone smoke test for the ELISE external simulate wrapper.

Run with: uv run pytest tests/test_elise_smoke.py -s
Or: uv run python tests/test_elise_smoke.py
"""

import numpy as np


def test_elise_runs():
    from neuroseq.example_models.elise import DEFAULT_PARAMS
    from neuroseq.example_models.elise import simulate

    num_vis = 5
    duration_s = 0.2
    stim_dt_s = 0.001
    n_stim = int(duration_s / stim_dt_s)

    t = np.arange(n_stim) * stim_dt_s
    base = -70.0 + 10.0 * np.sin(2 * np.pi * 5.0 * t)
    stim = np.repeat(base[:, None], num_vis, axis=1).astype(np.float32)
    stims_dict = {"u_inp": {"array": stim, "dt": stim_dt_s}}

    params = {"num_lat": 8, "weight_snapshot_interval_ms": 20.0}
    exp = simulate(stims_dict, simulation_time=duration_s, params=params)

    expected_steps = int(
        round(duration_s * 1000.0 / DEFAULT_PARAMS["dt_ms"]["default"])
    )
    n_total = num_vis + 8

    by_name = {d.name: d for d in exp.data}
    u_data = by_name["somatic_potential_u"]
    v_data = by_name["dendritic_potential_v"]
    r_data = by_name["firing_rate_r"]
    w_data = by_name["dendritic_weights"]

    assert u_data.traces.shape == (n_total, expected_steps), u_data.traces.shape
    assert v_data.traces.shape == (n_total, expected_steps)
    assert r_data.traces.shape == (n_total, expected_steps)
    assert np.isfinite(u_data.traces).all()
    assert np.isfinite(v_data.traces).all()
    assert ((r_data.traces >= 0.0) & (r_data.traces <= 1.0)).all()

    assert w_data.values.shape[:2] == (n_total, n_total), w_data.values.shape
    assert w_data.values.shape[2] == w_data.timestamps.shape[0]
    assert w_data.values.shape[2] >= 1

    print(
        f"PASS: u {u_data.traces.shape}, "
        f"r in [{r_data.traces.min():.3f}, {r_data.traces.max():.3f}], "
        f"weight snaps {w_data.values.shape}"
    )


def test_elise_simulator_continues_state_across_runs():
    from neuroseq.example_models.elise import Simulator

    num_vis = 4
    duration_s = 0.01
    stim_dt_s = 0.001
    n_stim = int(duration_s / stim_dt_s)

    stim = np.full((n_stim, num_vis), -60.0, dtype=np.float32)
    stims_dict = {"u_inp": {"array": stim, "dt": stim_dt_s}}
    params = {
        "num_lat": 6,
        "learn": False,
        "weight_snapshot_interval_ms": 5.0,
    }

    sim = Simulator()
    first = sim.simulate(stims_dict, simulation_time=duration_s, params=params)
    second = sim.simulate(stims_dict, simulation_time=duration_s, params=params)

    assert np.isclose(sim.simulated_time_seconds, 2 * duration_s)

    first_u = next(d for d in first.data if d.name == "somatic_potential_u")
    second_u = next(d for d in second.data if d.name == "somatic_potential_u")

    # Compare against a fresh simulator's first run to confirm continuation.
    fresh = Simulator()
    fresh_first = fresh.simulate(stims_dict, simulation_time=duration_s, params=params)
    fresh_u = next(d for d in fresh_first.data if d.name == "somatic_potential_u")

    assert not np.allclose(second_u.traces[:, 0], fresh_u.traces[:, 0])
    # Sanity check: both runs still produce the same output shape.
    assert first_u.traces.shape == second_u.traces.shape == fresh_u.traces.shape


if __name__ == "__main__":
    test_elise_runs()
