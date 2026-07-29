from __future__ import annotations

import zipfile
from pathlib import Path

import brian2 as b2
import brian2.numpy_ as np
import pytest
from brian2.tests.utils import assert_allclose

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_projects import default_project
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.power_lut import PowerLut
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def _make_experiment(
    name="test_experiment", n_neurons=12, n_timestamps=37, dt=0.1
) -> Experiment:
    trace_data = TraceData(
        name="test_trace",
        traces=np.random.rand(n_neurons, n_timestamps).astype(np.float32),
        indices=np.arange(0, n_neurons, dtype=np.uint32),
        units="mV",
        dt=dt,
    )
    spike_data = SpikeData(
        name="test_spike",
        spike_indices=np.array([1, 4, 5]),
        spike_times=np.array([0.1, 0.6, 0.99]),
    )
    return Experiment(name, n_timestamps * dt, [], [trace_data, spike_data])


def test_add_experimental_data_overwrite_raises_error() -> None:
    project = Project()
    project.add_experiment(_make_experiment())

    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 37)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (3,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (3,)
    assert project.max_time_secs == pytest.approx(3.7, abs=0.01)
    assert project.max_neurons == 12

    with pytest.raises(ValueError):
        project.add_experiment(_make_experiment())

    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 37)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (3,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (3,)
    assert project.max_time_secs == pytest.approx(3.7, abs=0.01)


def test_update_limits_normalizes_quantity_values_to_float() -> None:
    project = Project()
    project.experiments["legacy"] = Experiment("legacy", 2.0 * b2.second, [], [])
    project.stim_arrays["legacy_stim"] = StimArray(
        "legacy_stim",
        StimType.Excitation,
        np.zeros((8, 3), dtype=np.float32),
        "nS",
        0.5 * b2.second,
    )
    project.update_limits()

    assert isinstance(project.max_time_secs, float)
    assert project.max_time_secs == pytest.approx(4.0)
    assert project.max_neurons == 3


def test_append_experimental_data() -> None:
    project = Project()
    project.add_experiment(_make_experiment())

    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 37)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (3,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (3,)
    assert project.max_time_secs == pytest.approx(3.7, abs=0.01)

    # append to end
    project.append_experimental_data(
        "test_experiment", 3.7, _make_experiment().data, 3.7
    )
    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 74)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (6,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (6,)
    assert project.max_time_secs == pytest.approx(7.4, abs=0.01)

    # append to end
    project.append_experimental_data(
        "test_experiment", 3.7, _make_experiment().data, 7.4
    )
    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 111)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (9,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (9,)
    assert project.max_time_secs == pytest.approx(11.1, abs=0.01)

    # append at t=0: overwrites existing data with the same data -> no change in shapes of data arrays
    project.append_experimental_data(
        "test_experiment", 3.7, _make_experiment().data, 0.0
    )
    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 111)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (9,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (9,)
    assert project.max_time_secs == pytest.approx(11.1, abs=0.01)

    # overwrite from first timepoint at 3.7 with zeros for the trace data
    new_data = _make_experiment(
        "test_experiment", n_neurons=12, n_timestamps=37, dt=0.1
    )
    assert isinstance(new_data.data[0], TraceData)
    new_data.data[0].traces[:] = 0.0
    project.append_experimental_data(
        "test_experiment", new_data.duration, new_data.data, time_offset_seconds=3.7
    )
    # the overall duration and shape of data should be unchanged
    assert len(project.experiments) == 1
    assert "test_experiment" in project.experiments
    assert project.experiments["test_experiment"].duration == pytest.approx(
        11.1, abs=0.01
    )
    assert isinstance(project.experiments["test_experiment"].data[0], TraceData)
    assert project.experiments["test_experiment"].data[0].traces.shape == (12, 111)
    assert isinstance(project.experiments["test_experiment"].data[1], SpikeData)
    assert project.experiments["test_experiment"].data[1].spike_indices.shape == (9,)
    assert project.experiments["test_experiment"].data[1].spike_times.shape == (9,)
    assert project.max_time_secs == pytest.approx(11.1, abs=0.01)
    assert np.allclose(
        project.experiments["test_experiment"].timepoints, [0.0, 3.7, 7.4, 11.1]
    )

    # but there should now be zeros in the trace data from 3.7s to 7.4s
    start_idx = int(3.7 / 0.1)
    end_idx = int(7.4 / 0.1)
    assert np.allclose(
        project.experiments["test_experiment"].data[0].traces[:, start_idx:end_idx],
        0.0,
    )


def test_append_experimental_data_disjoint_spikes_are_additive() -> None:
    project = Project()
    existing_ids = np.array([0, 1], dtype=np.uint32)
    totally_disjoint_ids = np.array([4, 77], dtype=np.uint32)

    existing = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(existing.data[0], TraceData)
    assert isinstance(existing.data[1], SpikeData)
    existing.data[0].indices = existing_ids
    existing.data[1].spike_indices = existing_ids
    existing.data[1].spike_times = np.array([0.1, 0.6], dtype=np.float32)
    project.add_experiment(existing)

    disjoint = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(disjoint.data[0], TraceData)
    assert isinstance(disjoint.data[1], SpikeData)
    disjoint.data[0].indices = totally_disjoint_ids
    disjoint.data[1].spike_indices = totally_disjoint_ids
    disjoint.data[1].spike_times = np.array([0.2, 0.4], dtype=np.float32)

    project.append_experimental_data(
        "test_experiment", disjoint.duration, disjoint.data, 0.0
    )

    trace = project.experiments["test_experiment"].data[0]
    spikes = project.experiments["test_experiment"].data[1]
    assert isinstance(trace, TraceData)
    assert isinstance(spikes, SpikeData)
    assert trace.traces.shape[0] == 4
    assert set(map(int, trace.indices)) == {0, 1, 4, 77}
    assert len(spikes.spike_indices) == 4
    assert set(map(int, spikes.spike_indices)) == {0, 1, 4, 77}


def test_add_experimental_data_append_invalid() -> None:
    project = Project()
    project.add_experiment(_make_experiment(n_neurons=6, dt=0.2))
    issue = project.issues_with_appending_experimental_data(
        "test_experiment",
        _make_experiment(n_neurons=7, dt=0.2).data,
    )
    assert issue == ""
    issue = project.issues_with_appending_experimental_data(
        "test_experiment",
        _make_experiment(n_neurons=6, dt=0.15).data,
    )
    assert "dt" in issue


# split trace and spike assertions for partial overlap behavior.
def test_append_experimental_data_partial_overlap_traces_replaces_and_adds() -> None:
    project = Project()
    existing_ids = np.array([0, 1], dtype=np.uint32)
    partially_overlapping_ids = np.array([0, 10], dtype=np.uint32)

    existing = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(existing.data[0], TraceData)
    assert isinstance(existing.data[1], SpikeData)
    existing.data[0].indices = existing_ids
    existing.data[0].traces = np.array([[1.0] * 10, [2.0] * 10], dtype=np.float32)
    existing.data[1].spike_indices = np.array([0, 1, 1], dtype=np.uint32)
    existing.data[1].spike_times = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    project.add_experiment(existing)

    incoming = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(incoming.data[0], TraceData)
    assert isinstance(incoming.data[1], SpikeData)
    incoming.data[0].indices = partially_overlapping_ids
    incoming.data[0].traces = np.array([[9.0] * 10, [8.0] * 10], dtype=np.float32)
    incoming.data[1].spike_indices = partially_overlapping_ids
    incoming.data[1].spike_times = np.array([0.2, 0.3], dtype=np.float32)

    project.append_experimental_data(
        "test_experiment", incoming.duration, incoming.data, 0.0
    )

    trace = project.experiments["test_experiment"].data[0]
    assert isinstance(trace, TraceData)
    row_of = {int(neuron): i for i, neuron in enumerate(trace.indices)}
    assert set(map(int, trace.indices)) == {0, 1, 10}
    assert trace.traces.shape[0] == 3
    assert np.all(trace.traces[row_of[0], :] == 9.0)
    assert np.all(trace.traces[row_of[1], :] == 2.0)
    assert np.all(trace.traces[row_of[10], :] == 8.0)


# split trace and spike assertions for partial overlap behavior.
def test_append_experimental_data_partial_overlap_spikes_replaces_and_adds() -> None:
    project = Project()
    existing_ids = np.array([0, 1], dtype=np.uint32)
    partially_overlapping_ids = np.array([0, 10], dtype=np.uint32)

    existing = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(existing.data[0], TraceData)
    assert isinstance(existing.data[1], SpikeData)
    existing.data[0].indices = existing_ids
    existing.data[1].spike_indices = existing_ids
    existing.data[1].spike_times = np.array([0.1, 0.8], dtype=np.float32)
    project.add_experiment(existing)

    incoming = _make_experiment(n_neurons=2, n_timestamps=10, dt=0.1)
    assert isinstance(incoming.data[0], TraceData)
    assert isinstance(incoming.data[1], SpikeData)
    incoming.data[0].indices = partially_overlapping_ids
    incoming.data[1].spike_indices = partially_overlapping_ids
    incoming.data[1].spike_times = np.array([0.2, 0.3], dtype=np.float32)

    project.append_experimental_data(
        "test_experiment", incoming.duration, incoming.data, 0.0
    )

    spikes = project.experiments["test_experiment"].data[1]
    assert isinstance(spikes, SpikeData)
    assert set(map(int, spikes.spike_indices)) == {0, 1, 10}
    assert len(spikes.spike_indices) == 3
    assert np.sum(spikes.spike_indices == 1) == 1


def test_append_experimental_data_disjoint_neurons_with_offset_raises() -> None:
    project = Project()
    project.add_experiment(_make_experiment(n_neurons=6, dt=0.2))
    disjoint_data = _make_experiment(n_neurons=6, dt=0.2).data
    assert isinstance(disjoint_data[0], TraceData)
    assert isinstance(disjoint_data[1], SpikeData)
    disjoint_data[0].indices = disjoint_data[0].indices + 100
    disjoint_data[1].spike_indices = disjoint_data[1].spike_indices + 100
    with pytest.raises(ValueError) as excinfo:
        project.append_experimental_data("test_experiment", 7.4, disjoint_data, 0.2)
    assert "non-overlapping neuron IDs" in str(excinfo.value)
    trace_data = project.experiments["test_experiment"].data[0]
    assert isinstance(
        trace_data, TraceData
    )  # sanity check, and before we have pre-commit typing errors.
    assert trace_data.traces.shape == (6, 37)
    assert np.array_equal(
        trace_data.indices,
        np.arange(6, dtype=np.uint32),
    )


def test_save_load(
    project_nmda: Project, tmp_path: Path, test_experiment: Experiment
) -> None:
    tmp_file = tmp_path / "project.nseq"
    assert project_nmda.filepath is None

    # simulate for 1 second, save checkpoint, then simulate for another second
    project_nmda.run_simulation(0.1)
    project_nmda.b2_model.store("after_0.1s")
    project_nmda.run_simulation(0.1)

    # add some experimental data
    project_nmda.add_experiment(test_experiment)

    # save project including everything
    save_options = SaveLoadOptions(True, True, True, True)
    project_nmda.save(tmp_file, save_options)
    assert project_nmda.filepath == tmp_file

    reconstructed_project = Project(tmp_file)
    assert reconstructed_project.filepath == tmp_file

    assert reconstructed_project.b2_model.seed == project_nmda.b2_model.seed
    assert (
        reconstructed_project.b2_model.network["neurons"].dt
        == project_nmda.b2_model.network["neurons"].dt
    )

    # synapses
    for neurongroupname in ["neuron_input", "neurons"]:
        for var in project_nmda.b2_model.network.get_states()[neurongroupname]:
            assert_allclose(
                getattr(reconstructed_project.b2_model.network[neurongroupname], var),
                getattr(project_nmda.b2_model.network[neurongroupname], var),
                verbose=True,
            )

    assert_allclose(
        reconstructed_project.b2_model.network["neurons"].g,
        project_nmda.b2_model.network["neurons"].g,
        verbose=True,
    )
    assert_allclose(
        project_nmda.b2_model.network["S"].g,
        reconstructed_project.b2_model.network["S"].g,
    )

    # test synapses
    for var in project_nmda.b2_model.network.get_states()["S"]:
        assert_allclose(
            getattr(reconstructed_project.b2_model.network["S"], var),
            getattr(project_nmda.b2_model.network["S"], var),
            verbose=True,
        )

    for name in ["M", "Mn"]:
        for var in project_nmda.b2_model.network.get_states()[name]:
            assert_allclose(
                getattr(reconstructed_project.b2_model.network[name], var),
                getattr(project_nmda.b2_model.network[name], var),
                verbose=True,
            )

    assert reconstructed_project.b2_model.parameters == project_nmda.b2_model.parameters

    # simulation data at save time is restored
    assert len(reconstructed_project.b2_model.network["M"].t) == 2000
    assert reconstructed_project.b2_model.network["M"].t[-1] == pytest.approx(
        0.2, abs=0.001
    )
    assert_allclose(
        project_nmda.b2_model.network["M"].g,
        reconstructed_project.b2_model.network["M"].g,
    )
    assert_allclose(
        project_nmda.b2_model.network["Mn"].g,
        reconstructed_project.b2_model.network["Mn"].g,
    )

    # as are checkpoints
    assert len(reconstructed_project.b2_model.checkpoints) == 2
    assert "after_0.1s" in reconstructed_project.b2_model.checkpoints
    reconstructed_project.b2_model.restore("after_0.1s")
    assert len(reconstructed_project.b2_model.network["M"].t) == 1000
    assert reconstructed_project.b2_model.network["M"].t[-1] == pytest.approx(
        0.1, abs=0.001
    )

    # as well as experimental data
    assert len(reconstructed_project.experiments) == 1
    assert "test_exp" in reconstructed_project.experiments
    exp = reconstructed_project.experiments["test_exp"]
    assert exp.duration == pytest.approx(test_experiment.duration)
    assert len(exp.timepoints) == 3
    assert len(exp.data) == 2
    loaded_trace_data = exp.data[0]
    test_trace_data = test_experiment.data[0]
    assert isinstance(loaded_trace_data, TraceData)
    assert isinstance(test_trace_data, TraceData)
    assert loaded_trace_data.name == test_trace_data.name
    assert loaded_trace_data.units == test_trace_data.units
    assert loaded_trace_data.dt == pytest.approx(test_trace_data.dt)
    assert np.allclose(loaded_trace_data.traces, test_trace_data.traces)
    loaded_spike_data = exp.data[1]
    test_spike_data = test_experiment.data[1]
    assert isinstance(loaded_spike_data, SpikeData)
    assert isinstance(test_spike_data, SpikeData)
    assert loaded_spike_data.name == test_spike_data.name
    assert np.allclose(loaded_spike_data.spike_indices, test_spike_data.spike_indices)
    assert np.allclose(loaded_spike_data.spike_times, test_spike_data.spike_times)

    # and simulations can be continued: run for 1s so that both models have run for 0.2s from the same initial conditions
    reconstructed_project.run_simulation(0.1)
    assert_allclose(
        project_nmda.b2_model.network["M"].g,
        reconstructed_project.b2_model.network["M"].g,
    )
    assert_allclose(
        project_nmda.b2_model.network["Mn"].g,
        reconstructed_project.b2_model.network["Mn"].g,
    )


def test_save_load_without_simulation(tmp_path: Path) -> None:
    project = default_project()
    assert project.b2_model.must_be_rebuilt is True
    assert [obj.name for obj in project.b2_model.network.objects] == []

    tmp_file = tmp_path / "nosim.nseq"
    project.save(tmp_file, SaveLoadOptions(True, True, True, True))

    with zipfile.ZipFile(tmp_file) as zf:
        assert "b2_stored_state.pkl" not in zf.namelist()

    reconstructed_project = Project(tmp_file)
    assert reconstructed_project.b2_model.checkpoints == []
    assert reconstructed_project.filepath == tmp_file
    reconstructed_project.run_simulation(0.01)


def test_stim_array_simulation_padding() -> None:
    sim_time = 0.010
    stim_padding_seconds = 3
    arr = np.random.rand(100, 30)
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr.copy(), "mS", dt=0.001
    )
    assert project.stim_arrays["stim"].values.shape == (100, 30)
    project.run_simulation(sim_time, stim_array_padding_seconds=stim_padding_seconds)
    new_stim_array = project.stim_arrays["stim"]
    # timed array length is simulation length + padding
    assert new_stim_array.values.shape[0] == pytest.approx(
        sim_time / 0.001 + stim_padding_seconds / 0.001
    )
    # timed array is padded with zeros - original values are preserved
    assert np.allclose(new_stim_array.values[0:100, :], arr)
    assert np.max(new_stim_array.values[100:, :]) == pytest.approx(0)


def test_pad_stim_arrays_to_neuron_count() -> None:
    arr = np.random.rand(100, 30)
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr.copy(), "mS", dt=0.001
    )
    # widen the neuron axis: new columns are zero, existing data preserved
    project.pad_stim_arrays_to_neuron_count(50)
    widened = project.stim_arrays["stim"]
    assert widened.values.shape == (100, 50)
    assert np.allclose(widened.values[:, :30], arr)
    assert np.max(widened.values[:, 30:]) == pytest.approx(0)
    # never shrinks: a smaller target is a no-op
    project.pad_stim_arrays_to_neuron_count(10)
    assert project.stim_arrays["stim"].values.shape == (100, 50)


def test_run_simulation_widens_stim_arrays_to_neuron_count() -> None:
    # Reproduces the "neuron count locked" bug: editing the neuron count up
    # used to crash the run with an out-of-bounds stim index. The run must now
    # auto-widen the stim arrays to the model's neuron count.
    project = default_project()
    model = project.b2_model
    stim_names = list(project.stim_arrays.keys())
    assert stim_names, "default project is expected to define stim arrays"
    original_width = project.stim_arrays[stim_names[0]].values.shape[1]

    new_count = original_width + 17
    model.functions["neurongroups"]["P"]["neuronnumber"]["modify"](new_count)
    assert model.get_defined_neuron_count() == new_count

    project.run_simulation(0.01)

    for name in stim_names:
        assert project.stim_arrays[name].values.shape[1] >= new_count
    assert new_count == model.network["P"].N


def test_stim_array_padding() -> None:
    arr = np.random.rand(1000, 30)
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert project.stim_arrays["stim"].values.shape == (1000, 30)
    # pad to 1.2s
    project.pad_stim_arrays(1.2)
    new_stim_array = project.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(1200, abs=1)
    assert np.allclose(new_stim_array.values[0 : arr.shape[0], :], arr)
    assert np.max(new_stim_array.values[1000:, :]) == pytest.approx(0)
    # pad to 2s
    project.pad_stim_arrays(2)
    new_stim_array = project.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(2000, abs=1)
    assert np.allclose(new_stim_array.values[0 : arr.shape[0], :], arr)
    assert np.max(new_stim_array.values[1000:, :]) == pytest.approx(0)


def test_stim_array_trimming(project_ornstein: Project) -> None:
    arr = np.random.rand(1000, 30)
    # make that last 100 values (0.1s) zero
    arr[900:, :] = 0
    project_ornstein.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert project_ornstein.stim_arrays["stim"].values.shape == (1000, 30)
    assert np.sum(project_ornstein.stim_arrays["stim"].values[900:, :]) == 0
    # last 0.1 seconds have no non-zero stimuli
    assert project_ornstein.trim_stim_array_would_delete_data(1.0) is False
    assert project_ornstein.trim_stim_array_would_delete_data(0.901) is False
    assert project_ornstein.trim_stim_array_would_delete_data(0.899) is True
    # trim to 0.95s
    assert project_ornstein.trim_stim_array_would_delete_data(0.95) is False
    project_ornstein.trim_stim_arrays(0.95)
    new_stim_array = project_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(950, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )
    # trim to 0.5s
    assert project_ornstein.trim_stim_array_would_delete_data(0.5) is True
    project_ornstein.trim_stim_arrays(0.5)
    new_stim_array = project_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(500, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )
    # trim to 0s
    assert project_ornstein.trim_stim_array_would_delete_data(0) is True
    project_ornstein.trim_stim_arrays(0)
    new_stim_array = project_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(0, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )


def test_stim_array_properties() -> None:
    arr = np.random.rand(1000, 30)
    project = Project()
    project.add_stim_array(StimArray("stim", StimType.Excitation, arr, "mS", dt=0.001))
    assert project.stim_arrays["stim"].values.shape == (1000, 30)
    project.update_stim_array_properties(
        "stim", units="nS", dt=0.001, n_neurons=25, stim_type=StimType.Inhibition
    )
    updated_stim_array = project.stim_arrays["stim"]
    assert updated_stim_array.units == "nS"
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (1000, 25)
    assert updated_stim_array.stim_type == StimType.Inhibition
    project.update_stim_array_properties(
        "stim", units="nS", dt=0.001, n_neurons=44, stim_type=StimType.External
    )
    updated_stim_array = project.stim_arrays["stim"]
    assert updated_stim_array.units == "nS"
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (1000, 44)
    assert updated_stim_array.stim_type == StimType.External


def test_update_stim_array_properties_remaps_existing_stim_array() -> None:
    project = Project()
    values = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, values, "nS", dt=0.001)
    )

    project.update_stim_array_properties(
        "stim", n_neurons=1, stim_type=StimType.Excitation, units="nS", dt=0.002
    )

    updated_stim_array = project.stim_arrays["stim"]
    assert updated_stim_array.dt == pytest.approx(0.002)
    assert updated_stim_array.values.shape == (3, 1)
    assert np.array_equal(updated_stim_array.values[:, 0], np.array([1.0, 0.0, 0.0]))
    assert project.max_time_secs == pytest.approx(0.006)


def test_add_stim_array_coerces_1d_values_to_2d() -> None:
    project = Project()
    values = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    project.add_stim_array(
        StimArray("stim", StimType.Inhibition, values, "nS", dt=0.001)
    )

    updated_stim_array = project.stim_arrays["stim"]
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (5, 1)
    assert np.array_equal(updated_stim_array.values[:, 0], values)
    assert project.max_time_secs == pytest.approx(0.005)
    assert project.max_neurons == 1


def test_save_load_project_preserves_mixed_stim_dt(tmp_path: Path) -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim_a",
            StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "nS",
            0.001,
        )
    )
    project.add_stim_array(
        StimArray(
            "stim_b",
            StimType.Inhibition,
            np.zeros((12, 3), dtype=np.float32),
            "nS",
            0.0005208125,
        )
    )
    tmp_file = tmp_path / "project_with_mixed_stim_dt.nseq"

    project.save(tmp_file, SaveLoadOptions(True, True, False, False))

    reconstructed_project = Project(tmp_file)
    assert reconstructed_project.stim_arrays["stim_a"].dt == pytest.approx(0.001)
    assert reconstructed_project.stim_arrays["stim_b"].dt == pytest.approx(0.0005208125)
    assert reconstructed_project.max_time_secs == pytest.approx(0.01)


def test_update_limits_with_mixed_stim_dt() -> None:
    project = Project()
    project.stim_arrays = {
        "stim_a": StimArray(
            "stim_a",
            StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "nS",
            0.001,
        ),
        "stim_b": StimArray(
            "stim_b",
            StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "nS",
            0.002,
        ),
    }

    project.update_limits()

    assert project.max_time_secs == pytest.approx(0.02)
    assert project.max_neurons == 2


def test_update_limits_include_cellpose_neuron_count() -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((10, 40), dtype=np.float32),
            "nS",
            0.001,
        )
    )
    project.cellpose_data = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.arange(100, dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
    )

    project.update_limits()

    assert project.max_neurons == 100


def test_stim_array_rename() -> None:
    arr = np.random.rand(1000, 30)
    project = Project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    project.rename_stim_array("stim", "stim2")
    assert len(project.stim_arrays) == 1
    assert "stim" not in project.stim_arrays
    updated_stim_array = project.stim_arrays["stim2"]
    assert updated_stim_array.units == "mS"
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (1000, 30)
    assert updated_stim_array.stim_type == StimType.Excitation
    project.delete_stim_array("stim2")
    assert "stim2" not in project.stim_arrays
    assert len(project.stim_arrays) == 0


def test_issues_with_appending_experimental_data_valid(
    project_nmda: Project, make_trace_data, make_spike_data
) -> None:
    experiment = Experiment(
        "exp",
        1.0,
        [0.0, 0.5, 1.0],
        [
            make_trace_data(n_neurons=5, n_timestamps=1000, dt=0.001),
            make_spike_data(n_neurons=5, max_time_secs=1.0),
        ],
    )
    project_nmda.add_experiment(experiment)
    # number of neurons and dt match, so no issues appending this data
    assert (
        project_nmda.issues_with_appending_experimental_data(
            "exp",
            [
                make_trace_data(n_neurons=5, n_timestamps=100, dt=0.001),
                make_spike_data(n_neurons=5, max_time_secs=0.10),
            ],
        )
        == ""
    )


def test_issues_with_appending_experimental_data_invalid(
    project_nmda: Project, make_trace_data, make_spike_data
) -> None:
    experiment = Experiment(
        "exp",
        1.0,
        [0.0, 0.5, 1.0],
        [
            make_trace_data(n_neurons=5, n_timestamps=1000, dt=0.001),
            make_spike_data(n_neurons=5, max_time_secs=1.0),
        ],
    )
    project_nmda.add_experiment(experiment)

    # experiment doesn't exist
    assert "exist" in project_nmda.issues_with_appending_experimental_data(
        "nonexistent_exp",
        [],
    )

    # data names match but types do not
    assert "data" in project_nmda.issues_with_appending_experimental_data(
        "exp",
        [
            make_spike_data(name="test_traces", n_neurons=5, max_time_secs=0.20),
            make_spike_data(n_neurons=5, max_time_secs=0.20),
        ],
    )

    # data names don't match
    assert "name" in project_nmda.issues_with_appending_experimental_data(
        "exp",
        [
            make_trace_data(n_neurons=5, n_timestamps=100, dt=0.001, name="wrong_name"),
            make_spike_data(n_neurons=5, max_time_secs=0.10),
        ],
    )

    # number of neurons is now allowed to mismatch with current data with replacing

    # dt doesn't match
    assert "dt" in project_nmda.issues_with_appending_experimental_data(
        "exp",
        [
            make_trace_data(n_neurons=5, n_timestamps=100, dt=0.002),
            make_spike_data(n_neurons=5, max_time_secs=0.10),
        ],
    )


def test_project_dt_defaults_to_1ms() -> None:
    project = Project()
    assert project.dt == pytest.approx(0.001)


def test_remap_all_data_to_dt_remaps_stims_and_traces() -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim1", StimType.External, np.ones((10, 2), dtype=np.float32), "pA", 0.001
        )
    )
    trace = TraceData(
        "trace1",
        np.ones((2, 10), dtype=np.float32),
        np.array([0, 1], dtype=np.uint32),
        "mV",
        0.001,
    )
    project.add_experiment(Experiment("exp1", 0.01, [], [trace]))

    project.remap_all_data_to_dt(0.0005)

    assert project.dt == pytest.approx(0.0005)
    assert project.stim_arrays["stim1"].dt == pytest.approx(0.0005)
    # 10 samples at 1 ms = 0.01 s → at 0.5 ms = 20 samples
    assert project.stim_arrays["stim1"].values.shape == (20, 2)
    td = project.experiments["exp1"].data[0]
    assert isinstance(td, TraceData)
    assert td.dt == pytest.approx(0.0005)
    assert td.traces.shape == (2, 20)


def test_remap_all_data_to_dt_noop_when_dt_already_matches() -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim1", StimType.External, np.ones((10, 1), dtype=np.float32), "pA", 0.001
        )
    )
    original = project.stim_arrays["stim1"].values.copy()

    project.remap_all_data_to_dt(0.001)

    assert project.dt == pytest.approx(0.001)
    np.testing.assert_array_equal(project.stim_arrays["stim1"].values, original)


def test_project_dt_persists_through_save_load(tmp_path: Path) -> None:
    project = Project()
    project.dt = 0.0005
    filepath = tmp_path / "test.nseq"
    project.save(filepath, SaveLoadOptions(True, True, True, True))

    loaded = Project(filepath)
    assert loaded.dt == pytest.approx(0.0005)


def test_project_power_luts_persist_through_save_load(tmp_path: Path) -> None:
    project = Project()
    project.power_luts = {
        1100: PowerLut(
            description="synthetic",
            tuples=((0.0, 0.0), (100.0, 50.0)),
        )
    }
    filepath = tmp_path / "power_lut_project.nseq"

    project.save(filepath, SaveLoadOptions(True, True, True, True))

    loaded = Project(filepath)
    assert 1100 in loaded.power_luts
    assert loaded.power_luts[1100].description == "synthetic"
    assert loaded.power_luts[1100].tuples == ((0.0, 0.0), (100.0, 50.0))


def test_project_dt_falls_back_to_stim_dt_for_old_files(tmp_path: Path) -> None:
    """Loading an .nseq without project_metadata.json derives dt from stim arrays."""
    import zipfile as _zf

    project = Project()
    project.dt = 0.0005
    project.add_stim_array(
        StimArray(
            "s", StimType.External, np.zeros((4, 1), dtype=np.float32), "pA", 0.0005
        )
    )
    filepath = tmp_path / "old.nseq"
    project.save(filepath, SaveLoadOptions(True, True, True, True))

    # strip the project_metadata.json from the archive to simulate an old file

    stripped = tmp_path / "old_stripped.nseq"
    with _zf.ZipFile(filepath, "r") as src, _zf.ZipFile(stripped, "w") as dst:
        for item in src.infolist():
            if item.filename != "project_metadata.json":
                dst.writestr(item, src.read(item.filename))

    loaded = Project(stripped)
    assert loaded.dt == pytest.approx(0.0005)


def test_load_project_without_b2_model_json(tmp_path: Path) -> None:
    """A stim-only .nseq (no serialized b2_model) must load, not KeyError 'seed'."""
    import zipfile as _zf

    project = Project()
    project.add_stim_array(
        StimArray(
            "s", StimType.External, np.ones((8, 3), dtype=np.float32), "pA", 0.001
        )
    )
    filepath = tmp_path / "stim_only.nseq"
    project.save(filepath, SaveLoadOptions(True, True, True, True))

    # Strip the serialized model to simulate a stim-array-only archive.
    stripped = tmp_path / "stim_only_stripped.nseq"
    with _zf.ZipFile(filepath, "r") as src, _zf.ZipFile(stripped, "w") as dst:
        for item in src.infolist():
            if item.filename not in {"b2_model.json", "b2_stored_state.pkl"}:
                dst.writestr(item, src.read(item.filename))

    loaded = Project(stripped)
    assert "s" in loaded.stim_arrays
    assert loaded.stim_arrays["s"].values.shape == (8, 3)
    # A default model is created rather than crashing.
    assert loaded.b2_model.seed == 123


def test_remappable_data_description_counts_correctly() -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "s1", StimType.External, np.zeros((4, 1), dtype=np.float32), "pA", 0.001
        )
    )
    project.add_stim_array(
        StimArray(
            "s2", StimType.External, np.zeros((4, 1), dtype=np.float32), "pA", 0.001
        )
    )
    trace = TraceData(
        "t1",
        np.zeros((1, 4), dtype=np.float32),
        np.array([0], dtype=np.uint32),
        "mV",
        0.001,
    )
    project.add_experiment(Experiment("exp1", 0.004, [], [trace]))

    desc = project.remappable_data_description(0.0005)
    assert "2 stim array(s)" in desc
    assert "1 trace dataset(s)" in desc

    assert project.remappable_data_description(0.001) == ""
