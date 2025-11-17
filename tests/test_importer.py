from __future__ import annotations

import numpy as np
import pytest

from neuroseq.importer import compute_normalized_suite2p_spikes
from neuroseq.importer import compute_normalized_suite2p_traces
from neuroseq.importer import prepare_suite2p_spikes
from neuroseq.importer import prepare_suite2p_traces


def test_prepare_suite2p_traces_produces_widget_payload(suite2p_import_data_dir):
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    fneu_data = np.load(suite2p_import_data_dir / "Fneu.npy")

    data = prepare_suite2p_traces(
        f_data,
        fneu_data,
        max_neurons=3,
        sampling_rate=30.0,
    )

    assert "trace" in data.name.lower()
    assert data.traces.shape == (3, 900)
    assert data.dt == pytest.approx(1 / 30.0)
    expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
    np.testing.assert_allclose(data.traces, expected_traces[:3, :])


def test_prepare_suite2p_traces_works_without_fneu(suite2p_import_data_dir):
    f_data = np.load(suite2p_import_data_dir / "F.npy")
    data = prepare_suite2p_traces(
        f_data,
        None,
        max_neurons=2,
        sampling_rate=15.0,
    )

    assert "trace" in data.name.lower()
    assert data.traces.shape[0] == 2
    assert data.dt == pytest.approx(1 / 15.0)
    expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, None)
    np.testing.assert_allclose(data.traces, expected_traces[:2, :])


def test_prepare_suite2p_spikes_respects_threshold_and_limits_neurons(
    suite2p_import_data_dir,
):
    spike_data = np.load(suite2p_import_data_dir / "spks.npy")
    dt = 1 / 30.0

    threshold = 0.2
    data = prepare_suite2p_spikes(
        spike_data,
        dt=dt,
        threshold=threshold,
        max_neurons=4,
    )

    assert "spike" in data.name.lower()
    assert len(data.spike_indices) == len(data.spike_times)

    time_indices = np.rint(data.spike_times / dt).astype(int)
    normalized_spikes, _, _ = compute_normalized_suite2p_spikes(spike_data)
    trimmed_data = normalized_spikes[:4, :]
    values = trimmed_data[data.spike_indices, time_indices]
    assert np.all(values >= threshold)

    high_threshold_data = prepare_suite2p_spikes(
        spike_data,
        dt=dt,
        threshold=1.1,
        max_neurons=4,
    )
    assert len(high_threshold_data.spike_indices) == 0
