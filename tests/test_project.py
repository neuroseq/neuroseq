from __future__ import annotations

import json
import zipfile
from pathlib import Path

import brian2 as b2
import brian2.numpy_ as np
import pytest
from brian2.tests.utils import assert_allclose

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.example_projects import default_project
from neuroseq.export.power_lut import PowerLut
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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


def test_trim_windows_simulated_trace_and_restores_on_regrow() -> None:
    """Trimming a stim windows its simulated trace; regrowing restores it.

    The stim array is the authoritative timeline, so ``get_data`` clips
    simulated traces to the longest stim. Because this is non-destructive, the
    full trace comes back when the stim length is restored (as undo does).
    """
    project = default_project()
    project.run_simulation(0.05)
    trace_name = project.get_data_names(TraceData)[0]

    full = project.get_data(trace_name, TraceData)
    assert full is not None
    full_samples = full.traces.shape[1]

    # Trim to 0.02 s: the trace is windowed to that many of its own samples.
    project.trim_stim_arrays(0.02)
    clipped = project.get_data(trace_name, TraceData)
    assert clipped is not None
    expected_samples = round(0.02 / full.dt)
    assert clipped.traces.shape[1] == expected_samples
    assert clipped.traces.shape[1] < full_samples

    # Regrowing the stim (what undo does) restores the full recording.
    project.pad_stim_arrays(0.05)
    restored = project.get_data(trace_name, TraceData)
    assert restored is not None
    assert restored.traces.shape[1] == full_samples


def test_trace_read_budget_downsamples_time_to_fit_ram() -> None:
    """Simulated traces are decimated in time to fit max_ram_trace_plot_matrix.

    Mirrors the synapse-plot RAM budget: the neuron axis is preserved and only
    the time axis is strided, with dt scaled up to match. Non-destructive --
    a larger budget yields the full-resolution trace again.
    """
    project = default_project()
    # 100 neurons for 0.3 s at 0.1 ms recording => ~1.1 MB trace, enough to
    # exceed a 1 MB budget and trigger downsampling (the simulation cost is per
    # time step, so many neurons for a short time keeps it quick).
    project.b2_model.define_neuronnumber("P", 100)
    project.run_simulation(0.3)
    trace_name = project.get_data_names(TraceData)[0]

    project.max_ram_trace_plot_matrix = 0  # no limit
    full = project.get_data(trace_name, TraceData)
    assert full is not None
    n_neurons, n_time = full.traces.shape
    assert full.traces.nbytes > 1024 * 1024  # exceeds the 1 MB budget below

    project.max_ram_trace_plot_matrix = 1
    budgeted = project.get_data(trace_name, TraceData)
    assert budgeted is not None
    # Neuron axis untouched; time strided down; result fits the budget.
    assert budgeted.traces.shape[0] == n_neurons
    assert budgeted.traces.shape[1] < n_time
    assert budgeted.traces.nbytes <= 1024 * 1024
    # dt scales with the stride, and the array is a contiguous copy (so the
    # full-resolution monitor buffer is not pinned by a strided view).
    assert budgeted.dt > full.dt
    assert budgeted.traces.flags["C_CONTIGUOUS"]

    # Raising the budget again returns the full-resolution trace.
    project.max_ram_trace_plot_matrix = 0
    restored = project.get_data(trace_name, TraceData)
    assert restored is not None
    assert restored.traces.shape[1] == n_time


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


def test_resize_stim_arrays_to_neuron_count_grows_and_trims() -> None:
    arr = np.random.rand(100, 30)
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr.copy(), "mS", dt=0.001
    )
    # grow to 50 (trailing zeros, existing data preserved)
    project.resize_stim_arrays_to_neuron_count(50)
    assert project.stim_arrays["stim"].values.shape == (100, 50)
    assert np.allclose(project.stim_arrays["stim"].values[:, :30], arr)
    # trim back to 10 (the reported case: neuron count reduced -> array cut)
    project.resize_stim_arrays_to_neuron_count(10)
    assert project.stim_arrays["stim"].values.shape == (100, 10)
    assert np.allclose(project.stim_arrays["stim"].values, arr[:, :10])


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


@pytest.mark.parametrize(
    "padded_rows,trim_seconds",
    [
        (4151, 0.5),
        (4151, 1.0),
        (4425, 2.5),
        (5247, 5.0),
        (8398, 5.0),
    ],
)
def test_trim_stim_arrays_exact_after_padding(
    padded_rows: int, trim_seconds: float
) -> None:
    """Trimming removes exactly the requested duration.

    Running a simulation pads stim arrays to non-round lengths (via ceil).
    Trimming those lengths used ``int(seconds / dt)``, which floor-truncated
    float-representation error and removed one sample too many. The
    conversion now rounds to the nearest sample so the trim is exact.
    """
    dt = 0.001
    values = np.zeros((padded_rows, 3), dtype=np.float32)
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, values, "mS", dt=dt
    )
    current_length_secs = dt * padded_rows
    new_length_secs = current_length_secs - trim_seconds
    project.trim_stim_arrays(new_length_secs)
    expected_rows = padded_rows - round(trim_seconds / dt)
    assert project.stim_arrays["stim"].values.shape[0] == expected_rows


def test_simulation_timeline_seconds_uses_longest_stim() -> None:
    project = Project()
    assert project.simulation_timeline_seconds() == pytest.approx(0.0)
    project.add_stim_array(
        StimArray("a", StimType.Excitation, np.zeros((1000, 2)), "mV", dt=0.001)
    )
    project.add_stim_array(
        StimArray("b", StimType.Excitation, np.zeros((3000, 2)), "mV", dt=0.001)
    )
    assert project.simulation_timeline_seconds() == pytest.approx(3.0)


def test_clip_simulated_data_windows_to_stim_timeline() -> None:
    project = Project()
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, np.zeros((4000, 2)), "mV", dt=0.001)
    )
    # 10 s of simulated data at 1 ms; the 4 s stim timeline windows it to 4000.
    trace = TraceData(
        "Simulation :: G / v",
        np.arange(20000).reshape(2, 10000).astype(np.float32),
        np.array([0, 1]),
        "mV",
        0.001,
    )
    clipped = project._clip_trace_to_timeline(trace)
    assert clipped is not None
    assert clipped.traces.shape == (2, 4000)

    spikes = SpikeData(
        "Simulation :: G",
        np.array([0, 1, 0, 1]),
        np.array([1.0, 3.0, 5.0, 7.0]),
    )
    clipped_spikes = project._clip_spike_to_timeline(spikes)
    assert clipped_spikes is not None
    assert clipped_spikes.spike_times.tolist() == [1.0, 3.0]
    assert clipped_spikes.spike_indices.tolist() == [0, 1]

    synapse = SynapseData(
        "Simulation :: S / w",
        np.zeros((2, 2, 5), dtype=np.float32),
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
    )
    clipped_synapse = project._clip_synapse_to_timeline(synapse)
    assert clipped_synapse is not None
    assert clipped_synapse.timestamps.tolist() == [0.0, 2.0, 4.0]
    assert clipped_synapse.values.shape == (2, 2, 3)


def test_clip_simulated_data_no_op_without_stim_arrays() -> None:
    project = Project()
    trace = TraceData(
        "Simulation :: G / v",
        np.zeros((2, 100), dtype=np.float32),
        np.array([0, 1]),
        "mV",
        0.001,
    )
    # No stim array means no authoritative timeline; data is returned unclipped.
    clipped = project._clip_trace_to_timeline(trace)
    assert clipped is not None
    assert clipped.traces.shape == (2, 100)


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


def test_update_stim_array_properties_decrease_trims_whole_project_consistently() -> (
    None
):
    """Regression: decreasing a stim array's neuron count via "Edit stim
    array" used to only truncate that one array's own columns, leaving
    every other stim array and experiment's trace/spike data referencing
    neurons that no longer existed anywhere else. It must trim the whole
    project consistently - the same behavior as Trim to Mapped Neurons."""
    project = Project()
    project.cellpose_data = _cellpose_result_with_mapping({0: 10, 1: 11, 2: 12, 3: 13})
    project.add_stim_array(
        StimArray(
            "edited",
            StimType.Excitation,
            np.array([[0, 1, 2, 3]], dtype=np.float32),
            "mV",
            dt=0.001,
        )
    )
    project.add_stim_array(
        StimArray(
            "other",
            StimType.Excitation,
            np.array([[10, 11, 12, 13]], dtype=np.float32),
            "mV",
            dt=0.001,
        )
    )
    trace = TraceData(
        name="trace",
        traces=np.array([[0], [1], [2], [3]], dtype=np.float32),
        indices=np.array([0, 1, 2, 3], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    spikes = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1, 2, 3], dtype=np.uint32),
        spike_times=np.array([0, 1, 2, 3], dtype=np.float32) * 0.001,
    )
    project.add_experiment(
        Experiment(name="exp", duration=0.004, timepoints=[], data=[trace, spikes])
    )

    project.update_stim_array_properties(
        "edited", n_neurons=2, stim_type=StimType.Excitation, units="mV", dt=0.001
    )

    # the edited array itself is trimmed (as before)
    assert project.stim_arrays["edited"].values.shape == (1, 2)
    assert np.allclose(project.stim_arrays["edited"].values, [[0, 1]])
    # the OTHER stim array is now also trimmed to match, not left stale
    assert project.stim_arrays["other"].values.shape == (1, 2)
    assert np.allclose(project.stim_arrays["other"].values, [[10, 11]])
    # experiment data is trimmed too
    reloaded_trace = project.experiments["exp"].data[0]
    assert isinstance(reloaded_trace, TraceData)
    assert np.array_equal(reloaded_trace.indices, [0, 1])
    assert np.allclose(reloaded_trace.traces, [[0], [1]])
    reloaded_spikes = project.experiments["exp"].data[1]
    assert isinstance(reloaded_spikes, SpikeData)
    assert np.array_equal(reloaded_spikes.spike_indices, [0, 1])
    # the cellpose mapping drops entries for the now-removed neurons (2, 3)
    assert project.cellpose_data is not None
    assert project.cellpose_data.stimulus_to_cellpose_mapping == {0: 10, 1: 11}


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


def test_update_limits_uses_mapping_size_once_cellpose_ids_are_mapped() -> None:
    # A mapping picks a subset of segmented ROIs to actually be neurons - the
    # rest of the segmentation is only ever addressed by Cellpose id, never
    # by neuron index, so it must not inflate max_neurons (issue: trace/stim
    # plots showed a big empty region above the real neurons, hover-labeled
    # as the last real neuron, because the camera was sized to the full
    # segmentation instead of the mapping).
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((10, 3), dtype=np.float32),
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
        stimulus_to_cellpose_mapping={0: 5, 1: 7, 2: 9},
    )

    project.update_limits()

    assert project.max_neurons == 3


def _cellpose_data_with_mapping(
    mapping: dict[int, int] | None,
) -> CellposeImportResult:
    return CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        stimulus_to_cellpose_mapping=mapping,
    )


def test_remap_cellpose_indices_updates_trace_and_spike_data() -> None:
    project = Project()
    project.cellpose_data = _cellpose_data_with_mapping({4: 0, 7: 1})
    experiment = Experiment(
        "exp",
        1.0,
        [0.0, 1.0],
        [
            TraceData(
                "traces",
                np.zeros((2, 10), dtype=np.float32),
                np.array([4, 7], dtype=np.uint32),
                "mV",
                0.1,
                np.array([0, 1], dtype=np.uint32),
            ),
            SpikeData(
                "spikes",
                np.array([4, 7, 4], dtype=np.uint32),
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0, 1, 0], dtype=np.uint32),
            ),
        ],
    )
    project.add_experiment(experiment)

    project.cellpose_data = _cellpose_data_with_mapping({9: 0, 2: 1})
    failed = project.remap_cellpose_indices()

    assert failed == []
    remapped_trace, remapped_spike = project.experiments["exp"].data
    assert isinstance(remapped_trace, TraceData)
    assert isinstance(remapped_spike, SpikeData)
    np.testing.assert_array_equal(
        remapped_trace.indices, np.array([9, 2], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        remapped_spike.spike_indices, np.array([9, 2, 9], dtype=np.uint32)
    )


def test_remap_cellpose_indices_leaves_data_without_raw_ids_untouched() -> None:
    project = Project()
    project.cellpose_data = _cellpose_data_with_mapping({4: 0})
    patch_clamp_trace = TraceData(
        "patch_clamp",
        np.zeros((1, 10), dtype=np.float32),
        np.array([12], dtype=np.uint32),
        "mV",
        0.1,
    )
    project.add_experiment(Experiment("exp", 1.0, [0.0, 1.0], [patch_clamp_trace]))

    failed = project.remap_cellpose_indices()

    assert failed == []
    assert patch_clamp_trace.indices.tolist() == [12]


def test_remap_cellpose_indices_reports_experiments_with_invalid_mapping() -> None:
    project = Project()
    project.cellpose_data = _cellpose_data_with_mapping({4: 0, 7: 1})
    trace_data = TraceData(
        "traces",
        np.zeros((2, 10), dtype=np.float32),
        np.array([4, 7], dtype=np.uint32),
        "mV",
        0.1,
        np.array([0, 1], dtype=np.uint32),
    )
    project.add_experiment(Experiment("exp", 1.0, [0.0, 1.0], [trace_data]))

    # Two stimulus IDs target Cellpose id 0 - a duplicate that
    # _resolve_suite2p_indices rejects.
    project.cellpose_data = _cellpose_data_with_mapping({0: 0, 1: 0})
    failed = project.remap_cellpose_indices()

    assert failed == ["exp"]
    # left unchanged rather than corrupted
    np.testing.assert_array_equal(trace_data.indices, np.array([4, 7], dtype=np.uint32))


def test_remap_cellpose_indices_applies_linear_offset() -> None:
    """Regression test for #554's Linear-strategy blind spot: the "Linear"
    mapping strategy stores only linear_offset, not a materialized
    stimulus_to_cellpose_mapping, so remap_cellpose_indices() used to read a
    bare None there and silently treat it as no mapping (pure identity),
    dropping the offset entirely."""
    project = Project()
    project.cellpose_data = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1, 2, 3], dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
        mapping_strategy="Linear",
        linear_offset=2,
    )
    trace_data = TraceData(
        "traces",
        np.zeros((4, 10), dtype=np.float32),
        np.array([0, 1, 2, 3], dtype=np.uint32),
        "mV",
        0.1,
        np.array([0, 1, 2, 3], dtype=np.uint32),
    )
    project.add_experiment(Experiment("exp", 1.0, [0.0, 1.0], [trace_data]))

    failed = project.remap_cellpose_indices()

    assert failed == []
    # Linear offset 2 over ids [0,1,2,3] -> [2,3,0,1] (see
    # generate_linear_stimulus_mapping), applied to raw ids [0,1,2,3].
    np.testing.assert_array_equal(
        trace_data.indices, np.array([2, 3, 0, 1], dtype=np.uint32)
    )


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


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_save_load_preserves_independent_dataset_timing(
    tmp_path: Path, legacy_metadata: bool
) -> None:
    project = Project()
    for name, dt in (("fast", 0.0005), ("slow", 0.002)):
        project.add_stim_array(
            StimArray(name, StimType.External, np.arange(8).reshape(8, 1), "pA", dt)
        )
    trace = TraceData("calcium", np.array([[1, 2, 3]]), np.array([0]), "units", 1 / 30)
    project.add_experiment(Experiment("exp", 0.1, [], [trace]))
    filepath = tmp_path / "mixed.nseq"
    project.save(filepath, SaveLoadOptions(True, True, True, True))
    with zipfile.ZipFile(filepath) as archive:
        metadata = json.loads(archive.read("project_metadata.json"))
        assert "dt" not in metadata
        if legacy_metadata:
            metadata["dt"] = 0.01
            legacy_path = tmp_path / "legacy.nseq"
            with zipfile.ZipFile(legacy_path, "w") as legacy:
                for item in archive.infolist():
                    payload = (
                        json.dumps(metadata).encode()
                        if item.filename == "project_metadata.json"
                        else archive.read(item.filename)
                    )
                    legacy.writestr(item, payload)
    loaded = Project(legacy_path if legacy_metadata else filepath)
    assert not hasattr(loaded, "dt")
    for name, original in project.stim_arrays.items():
        assert loaded.stim_arrays[name].dt == original.dt
        np.testing.assert_array_equal(loaded.stim_arrays[name].values, original.values)
    loaded_trace = loaded.experiments["exp"].data[0]
    assert isinstance(loaded_trace, TraceData)
    assert loaded_trace.dt == trace.dt
    np.testing.assert_array_equal(loaded_trace.traces, trace.traces)


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


def test_load_without_metadata_preserves_stim_dt(tmp_path: Path) -> None:
    """Loading an archive without project metadata preserves per-array timing."""
    import zipfile as _zf

    project = Project()
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
    assert loaded.stim_arrays["s"].dt == pytest.approx(0.0005)
    assert not hasattr(loaded, "dt")


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


def _cellpose_result_with_mapping(
    mapping: dict[int, int] | None,
) -> CellposeImportResult:
    return CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=np.zeros((4, 4), dtype=np.int32),
        overlay_image=np.zeros((4, 4, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        stimulus_to_cellpose_mapping=mapping,
    )


def test_trim_to_mapped_neurons_raises_without_a_mapping() -> None:
    project = Project()
    with pytest.raises(ValueError):
        project.trim_to_mapped_neurons()

    project.cellpose_data = _cellpose_result_with_mapping(None)
    with pytest.raises(ValueError):
        project.trim_to_mapped_neurons()

    project.cellpose_data = _cellpose_result_with_mapping({})
    with pytest.raises(ValueError):
        project.trim_to_mapped_neurons()


def test_trim_to_mapped_neurons_trims_everything_consistently() -> None:
    # stim ids 0, 1, 3 are mapped (-> cellpose ids 10, 11, 13); 2, 4, 5 are not
    project = Project()
    project.cellpose_data = _cellpose_result_with_mapping({0: 10, 1: 11, 3: 13})
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], dtype=np.float32),
            "mV",
            dt=0.001,
        )
    )
    trace = TraceData(
        name="trace",
        traces=np.array([[0], [1], [2], [3]], dtype=np.float32),
        indices=np.array([0, 2, 3, 5], dtype=np.uint32),
        units="mV",
        dt=0.001,
        raw_cellpose_ids=np.array([100, 102, 103, 105], dtype=np.uint32),
    )
    spikes = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 2, 3, 4, 1, 5], dtype=np.uint32),
        spike_times=np.array([0, 1, 2, 3, 4, 5], dtype=np.float32) * 0.001,
    )
    original_synapse_values = np.arange(6 * 6 * 2, dtype=np.float32).reshape(6, 6, 2)
    synapses = SynapseData(
        name="synapses",
        values=original_synapse_values.copy(),
        timestamps=np.array([0.0, 0.001], dtype=np.float32),
    )
    non_square_synapses = SynapseData(
        name="non_square_synapses",
        values=np.zeros((6, 4, 2), dtype=np.float32),
        timestamps=np.array([0.0, 0.001], dtype=np.float32),
    )
    project.add_experiment(
        Experiment(
            name="exp",
            duration=0.006,
            timepoints=[],
            data=[trace, spikes, synapses, non_square_synapses],
        )
    )

    project.trim_to_mapped_neurons()

    # stim array: only columns 0, 1, 3 survive
    trimmed_stim = project.stim_arrays["stim"]
    assert trimmed_stim.values.shape == (2, 3)
    assert np.allclose(trimmed_stim.values, [[0, 1, 3], [10, 11, 13]])

    # trace: rows for indices 0 and 3 survive (renumbered 0 -> 0, 3 -> 2)
    reloaded_trace = project.experiments["exp"].data[0]
    assert isinstance(reloaded_trace, TraceData)
    assert np.allclose(reloaded_trace.traces, [[0], [2]])
    assert np.array_equal(reloaded_trace.indices, [0, 2])
    assert np.array_equal(reloaded_trace.raw_cellpose_ids, [100, 103])

    # spikes: events for indices 0, 3, 1 survive (renumbered 0 -> 0, 3 -> 2, 1 -> 1)
    reloaded_spikes = project.experiments["exp"].data[1]
    assert isinstance(reloaded_spikes, SpikeData)
    assert np.array_equal(reloaded_spikes.spike_indices, [0, 2, 1])
    assert np.allclose(reloaded_spikes.spike_times, np.array([0, 2, 4]) * 0.001)

    # synapses: both axes trimmed to [0, 1, 3], reordered/renumbered to match
    reloaded_synapses = project.experiments["exp"].data[2]
    assert isinstance(reloaded_synapses, SynapseData)
    assert reloaded_synapses.values.shape == (3, 3, 2)
    expected = original_synapse_values[np.ix_([0, 1, 3], [0, 1, 3], [0, 1])]
    assert np.allclose(reloaded_synapses.values, expected)

    # non-square synapse data is left untouched (skipped, not crashed)
    reloaded_non_square = project.experiments["exp"].data[3]
    assert isinstance(reloaded_non_square, SynapseData)
    assert reloaded_non_square.values.shape == (6, 4, 2)

    # the cellpose mapping itself is renumbered to the new contiguous ids
    assert project.cellpose_data is not None
    assert project.cellpose_data.stimulus_to_cellpose_mapping == {
        0: 10,
        1: 11,
        2: 13,
    }


def test_trim_to_mapped_neurons_is_a_noop_when_already_fully_mapped() -> None:
    project = Project()
    project.cellpose_data = _cellpose_result_with_mapping({0: 10, 1: 11})
    original_values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, original_values.copy(), "mV", dt=0.001)
    )

    project.trim_to_mapped_neurons()

    assert np.allclose(project.stim_arrays["stim"].values, original_values)


def test_fresh_and_loaded_projects_start_clean() -> None:
    """No unsaved-changes prompt should fire for a project nobody has

    touched yet (issue #535)."""
    assert Project().dirty is False
    assert default_project().dirty is False


def test_mark_dirty_and_save_roundtrip(tmp_path: Path) -> None:
    project = default_project()
    assert project.dirty is False

    project.mark_dirty()
    assert project.dirty is True

    project.save(tmp_path / "project.nseq", SaveLoadOptions(True, True, True, True))
    assert project.dirty is False

    reloaded = Project(tmp_path / "project.nseq")
    assert reloaded.dirty is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.add_stim_array(
            StimArray("s", StimType.Excitation, np.zeros((10, 2)), "mV", dt=0.001)
        ),
        lambda p: p.pad_stim_arrays(1.0),
        lambda p: p.pad_stim_arrays_to_neuron_count(5),
        lambda p: p.resize_stim_arrays_to_neuron_count(5),
        lambda p: p.trim_stim_arrays(0.0),
        lambda p: p.loop_stim_arrays(2),
        lambda p: p.tile_stim_arrays_to_length(0.5),
        lambda p: p.add_experiment(
            Experiment(
                "e",
                1.0,
                [],
                [TraceData("e_trace", np.zeros((1, 10)), np.array([0]), "mV", 0.1)],
            )
        ),
        lambda p: p.run_simulation(0.01),
    ],
)
def test_mutating_methods_mark_the_project_dirty(mutate) -> None:
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, np.zeros((10, 2)), "mV", dt=0.001
    )
    assert project.dirty is False

    mutate(project)

    assert project.dirty is True


def test_import_project_marks_dirty(tmp_path: Path) -> None:
    saved = default_project()
    saved.add_stim_array(
        StimArray("s", StimType.Excitation, np.zeros((10, 2)), "mV", dt=0.001)
    )
    tmp_file = tmp_path / "source.nseq"
    saved.save(tmp_file, SaveLoadOptions(True, True, True, True))

    project = default_project()
    assert project.dirty is False
    project.import_project(tmp_file, SaveLoadOptions(True, True, True, True))
    assert project.dirty is True


def test_device_settings_are_saved_with_the_project(tmp_path):
    project = Project()
    project.syscon_export_settings = SysconExportSettings(
        config_text="cfg", target_power_mw=23.0, laser_wavelength_nm=1100
    )
    project.device_folders = {"scanimage_recording": "D:/data", "syscon_export": "E:/"}
    path = tmp_path / "devices.nseq"
    project.save(path, SaveLoadOptions(True, True, False, False))
    loaded = Project(path)
    assert loaded.syscon_export_settings == project.syscon_export_settings
    assert loaded.device_folders == project.device_folders


def test_device_settings_setters_mark_the_project_dirty_only_on_change():
    project = Project()
    project.dirty = False
    settings = SysconExportSettings(config_text="cfg")
    project.set_syscon_export_settings(settings)
    project.set_device_folder("syscon_export", "E:/exports")
    assert project.dirty
    assert project.syscon_export_settings is settings
    assert project.device_folders == {"syscon_export": "E:/exports"}
    project.dirty = False
    project.set_syscon_export_settings(SysconExportSettings(config_text="cfg"))
    project.set_device_folder("syscon_export", "E:/exports")
    assert not project.dirty


def test_block_data_of_a_run_is_collected_in_one_group():
    """Each block of a routine run lands in one group at its own time."""
    project = Project()
    for offset, spike_time in ((0.0, 0.05), (0.3, 0.1)):
        spikes = SpikeData("chip spikes", np.array([1]), np.array([spike_time]))
        trace = TraceData(
            "v", np.full((1, 10), offset + 1.0), np.array([1]), "a.u.", 0.01
        )
        project.place_block_data("Run", [spikes, trace], offset, 0.2)
    experiment = project.experiments["Run"]
    assert experiment.duration == pytest.approx(0.5)
    spikes = next(d for d in experiment.data if isinstance(d, SpikeData))
    np.testing.assert_allclose(spikes.spike_times, [0.05, 0.4])
    trace = next(d for d in experiment.data if isinstance(d, TraceData))
    # 10 samples, a gap held at the last value, then the second block at 0.3 s.
    assert trace.traces.shape == (1, 40)
    np.testing.assert_allclose(trace.traces[0, 10:30], 1.0)
    np.testing.assert_allclose(trace.traces[0, 30:], 1.3)
    project.history.undo()
    spikes = next(
        d for d in project.experiments["Run"].data if isinstance(d, SpikeData)
    )
    np.testing.assert_allclose(spikes.spike_times, [0.05])
    project.history.undo()
    assert "Run" not in project.experiments


def test_block_trace_is_resampled_to_the_groups_interval():
    project = Project()
    first = TraceData("v", np.zeros((1, 10)), np.array([0]), "a.u.", 0.01)
    project.place_block_data("Run", [first], 0.0, 0.1)
    second = TraceData(
        "v", np.linspace(0, 1, 21)[None, :], np.array([0]), "a.u.", 0.005
    )
    project.place_block_data("Run", [second], 0.1, 0.1)
    trace = project.experiments["Run"].data[0]
    assert trace.dt == 0.01
    np.testing.assert_allclose(trace.traces[0, 10:], np.linspace(0, 1, 11), atol=1e-6)
    other = TraceData("v", np.zeros((1, 5)), np.array([3]), "a.u.", 0.01)
    with pytest.raises(ValueError, match="other neurons"):
        project.place_block_data("Run", [other], 0.2, 0.05)


def test_placed_spikes_keep_their_cellpose_ids():
    project = Project()
    project.place_block_data(
        "Run",
        [SpikeData("s", np.array([0, 1]), np.array([0.3, 0.1]), np.array([5, 6]))],
        0.0,
        1.0,
    )
    project.place_block_data(
        "Run",
        [SpikeData("s", np.array([2]), np.array([0.05]), np.array([7]))],
        0.1,
        1.0,
    )
    spikes = project.experiments["Run"].data[0]
    np.testing.assert_allclose(spikes.spike_times, [0.1, 0.15, 0.3])
    np.testing.assert_array_equal(spikes.spike_indices, [1, 2, 0])
    np.testing.assert_array_equal(spikes.raw_cellpose_ids, [6, 7, 5])
    project.place_block_data(
        "Run", [SpikeData("s", np.array([3]), np.array([0.0]))], 2.0, 1.0
    )
    assert project.experiments["Run"].data[0].raw_cellpose_ids is None


def test_placed_block_can_replace_spikes_in_its_span():
    """A recording's tail running into the next block is replaced by that block."""
    project = Project()
    project.place_block_data(
        "Run",
        [SpikeData("s", np.array([0, 1]), np.array([0.2, 1.5]), np.array([5, 6]))],
        0.0,
        2.0,
    )
    project.place_block_data(
        "Run",
        [SpikeData("s", np.array([2]), np.array([0.1]), np.array([7]))],
        1.0,
        2.0,
        replace_spikes=True,
    )
    spikes = project.experiments["Run"].data[0]
    np.testing.assert_allclose(spikes.spike_times, [0.2, 1.1])
    np.testing.assert_array_equal(spikes.spike_indices, [0, 2])
    np.testing.assert_array_equal(spikes.raw_cellpose_ids, [5, 7])


def test_empty_block_trace_is_skipped():
    project = Project()
    empty = TraceData("v", np.zeros((1, 0)), np.array([0]), "a.u.", 0.01)
    spikes = SpikeData("s", np.array([0]), np.array([0.1]))
    project.place_block_data("Run", [spikes, empty], 0.0, 0.2)
    assert [type(d) for d in project.experiments["Run"].data] == [SpikeData]


def test_exported_simulated_traces_ignore_the_plotting_ram_budget() -> None:
    project = default_project()
    project.b2_model.define_neuronnumber("P", 100)
    project.run_simulation(0.3)
    trace_name = project.get_data_names(TraceData)[0]
    project.max_ram_trace_plot_matrix = 1  # plots are downsampled
    budgeted = project.get_data(trace_name, TraceData)
    full = project.get_full_resolution_trace(trace_name)
    assert budgeted is not None and full is not None
    assert full.traces.shape[1] > budgeted.traces.shape[1]
    assert project.data_duration_seconds(trace_name) == pytest.approx(0.3)
    assert project.data_duration_seconds("Missing :: x") == 0.0


def test_relocate_calcium_video_finds_the_other_files_by_name(tmp_path):
    project = Project()
    old = [Path("/old/part_0.tif"), Path("/old/part_1.tif")]
    video = project.add_calcium_video(old, 30.0)
    video["index"] = {"paths": [str(p) for p in old], "frames": {}}
    new_folder = tmp_path / "new"
    new_folder.mkdir()
    (new_folder / "part_1.tif").write_bytes(b"")
    project.dirty = False
    still_missing = project.relocate_calcium_video(None, new_folder / "part_1.tif")
    assert project.calcium_video_paths == [
        new_folder / "part_0.tif",
        new_folder / "part_1.tif",
    ]
    assert still_missing == [new_folder / "part_0.tif"]
    assert video["index"]["paths"] == [str(p) for p in project.calcium_video_paths]
    assert project.dirty


def test_relocate_calcium_video_accepts_a_renamed_single_file(tmp_path):
    project = Project()
    project.add_calcium_video([Path("/old/movie.tif")], 30.0)
    renamed = tmp_path / "renamed.tif"
    renamed.write_bytes(b"")
    assert project.relocate_calcium_video(None, renamed) == []
    assert project.calcium_video_paths == [renamed]

    project.add_calcium_video([Path("/old/a.tif"), Path("/old/b.tif")], 30.0)
    with pytest.raises(ValueError, match="not one of"):
        project.relocate_calcium_video(None, renamed)
