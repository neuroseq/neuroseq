from __future__ import annotations

import numpy as np

from neuroseq.data import TraceData


def test_sinus_offset_runs_and_has_expected_shape() -> None:
    from neuroseq.example_models.sinus_offset import DEFAULT_PARAMS
    from neuroseq.example_models.sinus_offset import simulate

    duration_s = 0.2
    stim_dt_s = 0.01
    n_stim = int(duration_s / stim_dt_s)

    # Constant offset (+2 mV) applied to all 5 units.
    stim = np.full((n_stim, 1), 2.0, dtype=np.float32)
    stims_dict = {"offset": {"array": stim, "dt": stim_dt_s}}

    params = {
        "baseline_mv": -50.0,
        "amplitude_mv": 0.0,
        "frequency_hz": 1.0,
        "stim_gain": 1.5,
    }
    exp = simulate(stims_dict, simulation_time=duration_s, params=params)

    dt_s = float(DEFAULT_PARAMS["dt_ms"]["default"]) / 1000.0
    expected_steps = int(round(duration_s / dt_s))

    by_name = {d.name: d for d in exp.data}
    membrane = by_name["sinusoid_membrane_potential"]
    stim_offset = by_name["stimulus_offset"]
    assert isinstance(membrane, TraceData)
    assert isinstance(stim_offset, TraceData)

    assert membrane.traces.shape == (5, expected_steps)
    assert stim_offset.traces.shape == (5, expected_steps)

    expected_value = -50.0 + 1.5 * 2.0
    assert np.allclose(membrane.traces, expected_value)


def test_sinus_offset_continues_phase_between_runs() -> None:
    from neuroseq.example_models.sinus_offset import SinusOffset

    sim = SinusOffset()

    duration_s = 0.3
    stim_dt_s = 0.01
    n_stim = int(duration_s / stim_dt_s)
    zero_stim = np.zeros((n_stim, 5), dtype=np.float32)
    stims_dict = {"offset": {"array": zero_stim, "dt": stim_dt_s}}

    params = {
        "baseline_mv": 0.0,
        "amplitude_mv": 1.0,
        "frequency_hz": 2.0,
        "phase_step_rad": 0.0,
        "stim_gain": 1.0,
    }

    first = sim.simulate(stims_dict, simulation_time=duration_s, params=params)
    second = sim.simulate(stims_dict, simulation_time=duration_s, params=params)

    first_mem = next(d for d in first.data if d.name == "sinusoid_membrane_potential")
    second_mem = next(d for d in second.data if d.name == "sinusoid_membrane_potential")
    assert isinstance(first_mem, TraceData)
    assert isinstance(second_mem, TraceData)

    # First sample of run 2 should equal sin(2*pi*f*duration), not sin(0).
    expected_second_start = np.sin(2.0 * np.pi * 2.0 * duration_s)
    assert np.allclose(second_mem.traces[:, 0], expected_second_start, atol=1e-5)
    assert np.allclose(first_mem.traces[:, 0], 0.0, atol=1e-7)
    assert np.isclose(sim.simulated_time_seconds, 2.0 * duration_s)
