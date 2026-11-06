import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.export.exporter_numpy import save_matrix
from neuroseq.export.exporter_numpy import spike_matrix
from neuroseq.export.exporter_numpy import trace_matrix


def test_trace_matrix_is_neurons_by_samples_ordered_by_id() -> None:
    traces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix = trace_matrix(TraceData("t", traces, np.array([7, 3]), "mV", 0.5))
    np.testing.assert_array_equal(matrix.values, traces[::-1])
    np.testing.assert_array_equal(matrix.neuron_ids, [3, 7])
    np.testing.assert_allclose(matrix.times, [0.0, 0.5, 1.0])
    assert matrix.units == "mV"


def test_spike_matrix_counts_spikes_per_neuron_and_bin() -> None:
    spikes = SpikeData("s", np.array([0, 2, 2, 2]), np.array([0.0, 0.1, 0.15, 0.3]))
    matrix = spike_matrix(spikes, 0.1)
    # Row i is neuron i; neuron 1 never spikes but keeps its row.
    np.testing.assert_array_equal(
        matrix.values, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 1]]
    )
    np.testing.assert_array_equal(matrix.neuron_ids, [0, 1, 2])
    np.testing.assert_allclose(matrix.times, [0.0, 0.1, 0.2, 0.3])
    assert matrix.units == "spikes"


def test_spike_matrix_covers_the_duration_and_the_given_neurons() -> None:
    spikes = SpikeData("s", np.array([1, 9]), np.array([0.05, 0.05]))
    matrix = spike_matrix(spikes, 0.1, duration_s=1.0, neuron_ids=np.arange(4))
    assert matrix.values.shape == (4, 10)
    assert matrix.values.sum() == 1  # neuron 9 is not exported
    assert matrix.values[1, 0] == 1
    empty = spike_matrix(SpikeData("s", np.array([]), np.array([])), 0.1, 0.25)
    assert empty.values.shape == (0, 3)
    with pytest.raises(ValueError):
        spike_matrix(spikes, 0.0)


def test_save_matrix_as_npy_or_npz(tmp_path) -> None:
    matrix = trace_matrix(
        TraceData("t", np.ones((2, 3)), np.array([0, 1]), "dF/F", 1 / 30)
    )
    save_matrix(matrix, tmp_path / "a.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), matrix.values)
    save_matrix(matrix, tmp_path / "a.npz")
    with np.load(tmp_path / "a.npz") as archive:
        np.testing.assert_array_equal(archive["values"], matrix.values)
        np.testing.assert_array_equal(archive["neuron_ids"], [0, 1])
        np.testing.assert_allclose(archive["times"], matrix.times)
        assert str(archive["units"]) == "dF/F"
    with pytest.raises(ValueError):
        save_matrix(matrix, tmp_path / "a.csv")
