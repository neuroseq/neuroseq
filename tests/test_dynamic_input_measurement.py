from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.dynamic_input_measurement import NoEpspDetected
from neuroseq.dynamic_input_measurement import measure_calcium_response
from neuroseq.dynamic_input_measurement import measure_calcium_spike_count
from neuroseq.dynamic_input_measurement import measure_epsp
from neuroseq.dynamic_input_measurement import measure_spike_count


def test_measure_spike_count_uses_requested_neuron_and_window() -> None:
    values = np.full((1, 20), -70.0, dtype=np.float32)
    values[0, 5:8] = 20.0
    trace = TraceData("trace", values, np.array([4]), "mV", 0.001)
    assert measure_spike_count(trace, 4, 0.004, 5.0, -20.0) == 1


def test_measurement_rejects_missing_epsp() -> None:
    trace = TraceData(
        "trace", np.full((1, 100), -70.0, dtype=np.float32), np.array([2]), "mV", 0.001
    )
    with pytest.raises(NoEpspDetected):
        measure_epsp(trace, 2, [0.02], baseline_ms=5.0, response_ms=10.0)


def test_measurement_rejects_trace_that_ends_before_probe() -> None:
    trace = TraceData(
        "trace", np.full((1, 10), -70.0, dtype=np.float32), np.array([2]), "mV", 0.001
    )
    with pytest.raises(ValueError, match="ends at .* before the probe"):
        measure_epsp(trace, 2, [0.02], baseline_ms=5.0, response_ms=10.0)


def test_measure_calcium_spike_count_filters_neuron_and_window() -> None:
    spikes = SpikeData("s", np.array([1, 1, 2, 1]), np.array([0.10, 0.50, 0.12, 0.30]))
    assert measure_calcium_spike_count(spikes, 1, 0.05, 300.0) == 2
    assert measure_calcium_spike_count(spikes, 2, 0.05, 300.0) == 1
    assert measure_calcium_spike_count(spikes, 3, 0.0, 1000.0) == 0


def test_measure_calcium_response_is_peak_minus_prepulse_baseline() -> None:
    dt = 0.1
    values = np.ones((1, 40), dtype=np.float32)
    values[0, 10:12] = 3.0  # response after a pulse at t = 1.0 s
    values[0, 20:22] = 5.0  # response after a pulse at t = 2.0 s
    trace = TraceData("c", values, np.array([7]), "dF/F", dt)

    result = measure_calcium_response(trace, 7, [1.0, 2.0], 300.0, 300.0)

    assert result.pulse_amplitudes == pytest.approx((2.0, 4.0))
    assert result.amplitude == pytest.approx(3.0)
    assert result.units == "dF/F"


def test_measure_calcium_response_rejects_unknown_neuron_and_short_trace() -> None:
    trace = TraceData("c", np.ones((1, 10), dtype=np.float32), np.array([0]), "", 0.1)
    with pytest.raises(ValueError, match="not recorded"):
        measure_calcium_response(trace, 5, [0.2], 100.0, 100.0)
    with pytest.raises(ValueError, match="ends at"):
        measure_calcium_response(trace, 0, [5.0], 100.0, 100.0)
