from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import TraceData
from neuroseq.dynamic_input_measurement import NoEpspDetected
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
