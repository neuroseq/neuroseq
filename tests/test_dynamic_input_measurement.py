from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import NoEpspDetected
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    calcium_rise_fraction,
)
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    measure_calcium_spike_count,
)
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import measure_epsp
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    measure_spike_count,
)


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


def test_measure_spike_count_handles_simulated_volt_traces_labelled_mv() -> None:
    values = np.full((1, 20), -0.07, dtype=np.float32)
    values[0, 5:8] = 0.02
    trace = TraceData("trace", values, np.array([4]), "mV", 0.001)
    assert measure_spike_count(trace, 4, 0.004, 5.0, -20.0) == 1


def test_raw_fluorescence_rise_is_relative_to_its_baseline() -> None:
    trace = TraceData(
        "F", np.array([[100.0] * 10 + [130.0] * 5]), np.array([0]), "fluorescence", 0.1
    )
    assert calcium_rise_fraction(trace, 0, 1.0, 500.0, 300.0) == pytest.approx(0.3)


def test_spike_in_the_onset_frame_counts_only_with_frame_tolerance() -> None:
    frame = 1 / 30
    # Timed at the start of frame 15, which contains an onset at 0.51 s.
    spikes = SpikeData("s", np.array([0, 0]), np.array([14 * frame, 15 * frame]))
    assert measure_calcium_spike_count(spikes, 0, 0.51, 100.0) == 0
    assert measure_calcium_spike_count(spikes, 0, 0.51, 100.0, frame) == 1


def test_dff_rise_is_the_change_itself() -> None:
    trace = TraceData(
        "dF/F", np.array([[0.0] * 10 + [0.4] * 5]), np.array([3]), "dF/F", 0.1
    )
    assert calcium_rise_fraction(trace, 3, 1.0, 500.0, 300.0) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="not recorded"):
        calcium_rise_fraction(trace, 4, 1.0, 500.0, 300.0)
