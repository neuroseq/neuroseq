import pytest

from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import EPSPMeasurement
from neuroseq.stimulation.dynamic_input.stdp_synaptic_tuning_dialog import (
    StdpSynapticTuningBackend,
)
from neuroseq.stimulation.dynamic_input.stdp_synaptic_tuning_dialog import (
    StdpSynapticTuningDialog,
)
from neuroseq.stimulation.dynamic_input.stdp_synaptic_tuning_dialog import (
    StdpSynapticTuningRunConfig,
)
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowBackend
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowDialog
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowRunConfig


class Host:
    def __init__(self):
        self.requests = []
        self.starts = 0

    def request_simulation(self, seconds):
        self.requests.append(seconds)

    def append_block(self, block):
        start = self.starts
        self.starts += len(block)
        return start


def test_tuning_backend_converges_to_target_without_qt():
    """Drives StdpSynapticTuningBackend's real closed loop (probe -> pairing
    -> probe -> record_measurement) with no Qt involved, mirroring the
    non-Qt portion of StdpSynapticTuningDialog._advance_domain exactly:
    while running, a "pairing" phase means re-probe, anything else means a
    probe just finished and needs measuring.
    """
    host = Host()
    backend = StdpSynapticTuningBackend(host)
    finished = []
    backend.on_finished = finished.append
    config = StdpSynapticTuningRunConfig(
        target=120.0,
        tolerance=1.0,
        max_iterations=20,
        run_all_iterations=False,
        pairing_strategy=StdpSynapticTuningDialog.FIXED_PAIRING_STRATEGY,
        pairing_interval_ms=30.0,
        initial_pairing_time_ms=30.0,
        pairing_increment_ms=1.0,
        max_pairing_magnitude_ms=1000.0,
        presynaptic_id=0,
        postsynaptic_id=1,
        amplitude=5.0,
        pulse_ms=5.0,
        stim_dt=0.001,
        n_neurons=2,
    )
    backend.begin_protocol(config)  # kicks off the first (baseline) probe
    # 2.0 mV is the baseline (100%); 2.4 mV is exactly the 120% target.
    amplitudes = iter([2.0, 2.4])
    while backend.running:
        if backend.phase == "pairing":
            backend.probe()
        else:
            backend.record_measurement(next(amplitudes), "mV")
    assert backend.baseline_epsp == 2.0
    assert backend.epsp_percentages == pytest.approx([100.0, 120.0])
    assert backend.pairing_times_ms == pytest.approx([30.0])
    assert finished == ["EPSP is within tolerance; target achieved."]


def test_window_backend_sweeps_all_points_without_qt():
    """Drives StdpWindowBackend's real Δt-sweep closed loop (baseline probe
    -> pairing -> probe -> measurement, repeated per point) with no Qt
    involved, mirroring the non-Qt portion of
    StdpWindowDialog._advance_domain exactly.
    """
    host = Host()
    backend = StdpWindowBackend(host)
    finished = []
    backend.on_finished = finished.append
    config = StdpWindowRunConfig(
        dt_min_ms=-10.0,
        dt_max_ms=10.0,
        dt_points=3,
        include_zero=True,
        pairings_per_point=1,
        readout=StdpWindowDialog.EPSP_READOUT,
        sweep_mode=StdpWindowDialog.RESET_MODE,
        presynaptic_id=0,
        postsynaptic_id=1,
        amplitude=5.0,
        pulse_ms=5.0,
        recovery_ms=150.0,
        response_ms=50.0,
        stim_dt=0.001,
        n_neurons=2,
    )
    delta_ts = [-10.0, 0.0, 10.0]
    backend.begin_sweep(config, delta_ts, weight_baseline=None)  # -> baseline probe
    # Baseline EPSP is 2.0 mV; the three points come back at 2.2/2.0/2.4 mV,
    # i.e. +10%/0%/+20% relative to baseline (reset mode always measures
    # against the fixed baseline, not the running value). A delta_t of
    # exactly 0.0 ms is snapped up to one sample (1.0 ms at this stim_dt) by
    # snap_milliseconds_to_dt's minimum_samples=1 floor, so pre/post pulses
    # are never placed on the identical sample - pre-existing behavior, not
    # something this migration changed.
    post_probe_amplitudes = iter([2.2, 2.0, 2.4])
    while backend.running:
        if backend.phase == "pair":
            if backend.should_continue_pairing():
                backend.apply_next_pairing()
            else:
                backend.begin_probe_phase()
        elif backend.phase == "baseline_probe":
            backend.record_baseline_probe(2.0, "mV")
        elif backend.phase == "probe":
            amplitude = next(post_probe_amplitudes)
            backend.handle_probe_measurement(
                EPSPMeasurement(
                    amplitude=amplitude,
                    pulse_amplitudes=(amplitude,) * 5,
                    units="mV",
                )
            )
    assert backend.baseline_epsp == 2.0
    expected = [(-10.0, 10.0), (1.0, 0.0), (10.0, 20.0)]
    assert len(backend.results) == len(expected)
    for (actual_dt, actual_percent), (expected_dt, expected_percent) in zip(
        backend.results, expected, strict=True
    ):
        assert actual_dt == expected_dt
        assert actual_percent == pytest.approx(expected_percent)
    assert finished == ["STDP window sweep finished; measured 3/3 points."]


def test_window_backend_handles_undetectable_probe_without_qt():
    """A probe that could not be measured (NoEpspDetected on the dialog
    side, surfaced here as `measurement=None`) is a skipped point, not a
    fatal error - the sweep must continue."""
    host = Host()
    backend = StdpWindowBackend(host)
    config = StdpWindowRunConfig(
        dt_min_ms=0.0,
        dt_max_ms=10.0,
        dt_points=2,
        include_zero=False,
        pairings_per_point=1,
        readout=StdpWindowDialog.EPSP_READOUT,
        sweep_mode=StdpWindowDialog.CONTINUOUS_MODE,
        presynaptic_id=0,
        postsynaptic_id=1,
        amplitude=5.0,
        pulse_ms=5.0,
        recovery_ms=150.0,
        response_ms=50.0,
        stim_dt=0.001,
        n_neurons=2,
    )
    delta_ts = [5.0, 10.0]
    backend.begin_sweep(config, delta_ts, weight_baseline=None)
    while backend.running:
        if backend.phase == "pair":
            if backend.should_continue_pairing():
                backend.apply_next_pairing()
            else:
                backend.begin_probe_phase()
        elif backend.phase == "baseline_probe":
            backend.record_baseline_probe(2.0, "mV")
        elif backend.phase == "probe":
            # First post-baseline probe is undetectable; the second succeeds.
            if len(backend.results) == 0:
                backend.handle_probe_measurement(None)
            else:
                backend.handle_probe_measurement(
                    EPSPMeasurement(
                        amplitude=2.2, pulse_amplitudes=(2.2,) * 5, units="mV"
                    )
                )
    assert len(backend.results) == 2
    delta_t, percent = backend.results[0]
    assert delta_t == 5.0
    assert percent != percent  # NaN: the skipped point
    second_delta_t, second_percent = backend.results[1]
    assert second_delta_t == 10.0
    assert second_percent == pytest.approx(10.0)


def test_window_backend_weight_readout_without_qt():
    """Weight-mode sweep: no probe pulses at all, the synapse weight is
    read directly after each pairing."""
    host = Host()
    backend = StdpWindowBackend(host)
    finished = []
    backend.on_finished = finished.append
    config = StdpWindowRunConfig(
        dt_min_ms=-10.0,
        dt_max_ms=10.0,
        dt_points=2,
        include_zero=False,
        pairings_per_point=1,
        readout=StdpWindowDialog.WEIGHT_READOUT,
        sweep_mode=StdpWindowDialog.RESET_MODE,
        presynaptic_id=0,
        postsynaptic_id=1,
        amplitude=5.0,
        pulse_ms=5.0,
        recovery_ms=150.0,
        response_ms=50.0,
        stim_dt=0.001,
        n_neurons=2,
    )
    delta_ts = [-10.0, 10.0]
    # Weight mode: begin_sweep is handed the already-read baseline weight
    # directly, so it skips straight to the first pairing (no probe phase).
    backend.begin_sweep(config, delta_ts, weight_baseline=1.0)
    weights = iter([1.1, 0.9])
    while backend.running:
        if backend.should_continue_pairing():
            backend.apply_next_pairing()
        else:
            backend.record_weight_point(next(weights))
    expected = [(-10.0, 10.0), (10.0, -10.0)]
    assert len(backend.results) == len(expected)
    for (actual_dt, actual_percent), (expected_dt, expected_percent) in zip(
        backend.results, expected, strict=True
    ):
        assert actual_dt == expected_dt
        assert actual_percent == pytest.approx(expected_percent)
    assert finished == ["STDP window sweep finished; measured 2/2 points."]
