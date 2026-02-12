from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from brian2.tests.utils import assert_allclose

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_projects import default_project
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


def test_add_experimental_data_append_invalid() -> None:
    project = Project()
    project.add_experiment(_make_experiment(n_neurons=6, dt=0.2))
    with pytest.raises(ValueError) as excinfo:
        project.append_experimental_data(
            "test_experiment", 3.7, _make_experiment(n_neurons=7, dt=0.2).data, 3.7
        )
    assert "neurons" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        project.append_experimental_data(
            "test_experiment", 3.7, _make_experiment(n_neurons=6, dt=0.15).data, 3.7
        )
    assert "dt" in str(excinfo.value)


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
    project = default_project()
    project.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert project.stim_arrays["stim"].values.shape == (1000, 30)
    project.update_stim_array_properties(
        "stim", units="nS", dt=0.01, n_neurons=25, stim_type=StimType.Inhibition
    )
    updated_stim_array = project.stim_arrays["stim"]
    assert updated_stim_array.units == "nS"
    assert updated_stim_array.dt == pytest.approx(0.01)
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

    # number of neurons don't match
    assert "neurons" in project_nmda.issues_with_appending_experimental_data(
        "exp",
        [
            make_trace_data(n_neurons=6, n_timestamps=100, dt=0.001),
            make_spike_data(n_neurons=6, max_time_secs=0.10),
        ],
    )

    # dt doesn't match
    assert "dt" in project_nmda.issues_with_appending_experimental_data(
        "exp",
        [
            make_trace_data(n_neurons=5, n_timestamps=100, dt=0.002),
            make_spike_data(n_neurons=5, max_time_secs=0.10),
        ],
    )
