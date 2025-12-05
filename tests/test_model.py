from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from brian2.tests.utils import assert_allclose

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_models import default_model
from neuroseq.model import Model
from neuroseq.save_options import SaveOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def _make_experimental_data(n_neurons=12, dt=0.1) -> list[TraceData | SpikeData]:
    trace_data = TraceData(
        name="test_trace",
        traces=np.random.rand(n_neurons, 37),
        indices=np.arange(0, n_neurons, dtype=np.uint32),
        units="mV",
        dt=dt,
    )
    spike_data = SpikeData(
        name="test_spike",
        spike_indices=np.array([1, 4, 5]),
        spike_times=np.array([0.1, 0.6, 0.99]),
    )
    return [trace_data, spike_data]


def test_add_experimental_data_overwrite() -> None:
    model = Model()
    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=False
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 37)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (3,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (3,)
    assert model.max_time_secs == pytest.approx(3.7, abs=0.01)

    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=False
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 37)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (3,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (3,)
    assert model.max_time_secs == pytest.approx(3.7, abs=0.01)


def test_add_experimental_data_append() -> None:
    model = Model()
    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=False
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 37)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (3,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (3,)
    assert model.max_time_secs == pytest.approx(3.7, abs=0.01)

    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=True
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 74)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (6,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (6,)
    assert model.max_time_secs == pytest.approx(7.4, abs=0.01)

    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=True
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 111)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (9,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (9,)
    assert model.max_time_secs == pytest.approx(11.1, abs=0.01)

    model.add_experimental_data(
        "test_experiment", _make_experimental_data(), append=False
    )

    assert len(model.experiments) == 1
    assert "test_experiment" in model.experiments
    assert isinstance(model.experiments["test_experiment"][0], TraceData)
    assert model.experiments["test_experiment"][0].traces.shape == (12, 37)
    assert isinstance(model.experiments["test_experiment"][1], SpikeData)
    assert model.experiments["test_experiment"][1].spike_indices.shape == (3,)
    assert model.experiments["test_experiment"][1].spike_times.shape == (3,)
    assert model.max_time_secs == pytest.approx(3.7, abs=0.01)


def test_add_experimental_data_append_invalid() -> None:
    model = Model()
    model.add_experimental_data(
        "test_experiment", _make_experimental_data(n_neurons=6, dt=0.2), append=False
    )
    with pytest.raises(ValueError) as excinfo:
        model.add_experimental_data(
            "test_experiment", _make_experimental_data(n_neurons=7, dt=0.2), append=True
        )
    assert "neurons" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        model.add_experimental_data(
            "test_experiment",
            _make_experimental_data(n_neurons=6, dt=0.15),
            append=True,
        )
    assert "dt" in str(excinfo.value)


def test_save_load(model_nmda: Model, tmp_path: Path) -> None:
    tmp_file = tmp_path / "model.nseq"

    # simulate for 1 second, save checkpoint, then simulate for another second
    model_nmda.run_simulation(0.1)
    model_nmda.b2_model.store("after_0.1s")
    model_nmda.run_simulation(0.1)

    # add some experimental data
    trace_data = TraceData(
        name="test_trace",
        traces=np.random.rand(12, 3),
        indices=np.arange(0, 12, dtype=np.uint32),
        units="mV",
        dt=0.01,
    )
    spike_data = SpikeData(
        name="test_spike",
        spike_indices=np.array([1, 4, 5]),
        spike_times=np.array([0.1, 0.6, 0.99]),
    )
    experimental_data: list[TraceData | SpikeData] = [trace_data, spike_data]
    model_nmda.add_experimental_data("test_experiment", experimental_data)

    # save model including everything
    save_options = SaveOptions(True, True, True, True)
    model_nmda.save(tmp_file, save_options)

    reconstructed_model = Model(tmp_file)

    assert reconstructed_model.b2_model.seed == model_nmda.b2_model.seed
    assert (
        reconstructed_model.b2_model.network["neurons"].dt
        == model_nmda.b2_model.network["neurons"].dt
    )

    # synapses
    for neurongroupname in ["neuron_input", "neurons"]:
        for var in model_nmda.b2_model.network.get_states()[neurongroupname]:
            assert_allclose(
                getattr(reconstructed_model.b2_model.network[neurongroupname], var),
                getattr(model_nmda.b2_model.network[neurongroupname], var),
                verbose=True,
            )

    assert_allclose(
        reconstructed_model.b2_model.network["neurons"].g,
        model_nmda.b2_model.network["neurons"].g,
        verbose=True,
    )
    assert_allclose(
        model_nmda.b2_model.network["S"].g, reconstructed_model.b2_model.network["S"].g
    )

    # test synapses
    for var in model_nmda.b2_model.network.get_states()["S"]:
        assert_allclose(
            getattr(reconstructed_model.b2_model.network["S"], var),
            getattr(model_nmda.b2_model.network["S"], var),
            verbose=True,
        )

    for name in ["M", "Mn"]:
        for var in model_nmda.b2_model.network.get_states()[name]:
            assert_allclose(
                getattr(reconstructed_model.b2_model.network[name], var),
                getattr(model_nmda.b2_model.network[name], var),
                verbose=True,
            )

    assert reconstructed_model.b2_model.parameters == model_nmda.b2_model.parameters

    # simulation data at save time is restored
    assert len(reconstructed_model.b2_model.network["M"].t) == 2000
    assert reconstructed_model.b2_model.network["M"].t[-1] == pytest.approx(
        0.2, abs=0.001
    )
    assert_allclose(
        model_nmda.b2_model.network["M"].g, reconstructed_model.b2_model.network["M"].g
    )
    assert_allclose(
        model_nmda.b2_model.network["Mn"].g,
        reconstructed_model.b2_model.network["Mn"].g,
    )

    # as are checkpoints
    assert len(reconstructed_model.b2_model.checkpoints) == 2
    assert "after_0.1s" in reconstructed_model.b2_model.checkpoints
    reconstructed_model.b2_model.restore("after_0.1s")
    assert len(reconstructed_model.b2_model.network["M"].t) == 1000
    assert reconstructed_model.b2_model.network["M"].t[-1] == pytest.approx(
        0.1, abs=0.001
    )

    # as well as experimental data
    assert len(reconstructed_model.experiments) == 1
    assert "test_experiment" in reconstructed_model.experiments
    loaded_experimental_data = reconstructed_model.experiments["test_experiment"]
    assert len(loaded_experimental_data) == 2
    loaded_trace_data = loaded_experimental_data[0]
    assert isinstance(loaded_trace_data, TraceData)
    assert loaded_trace_data.name == trace_data.name
    assert loaded_trace_data.units == trace_data.units
    assert loaded_trace_data.dt == pytest.approx(trace_data.dt)
    assert np.allclose(loaded_trace_data.traces, trace_data.traces)
    loaded_spike_data = loaded_experimental_data[1]
    assert isinstance(loaded_spike_data, SpikeData)
    assert loaded_spike_data.name == spike_data.name
    assert np.allclose(loaded_spike_data.spike_indices, spike_data.spike_indices)
    assert np.allclose(loaded_spike_data.spike_times, spike_data.spike_times)

    # and simulations can be continued: run for 1s so that both models have run for 0.2s from the same initial conditions
    reconstructed_model.run_simulation(0.1)
    assert_allclose(
        model_nmda.b2_model.network["M"].g, reconstructed_model.b2_model.network["M"].g
    )
    assert_allclose(
        model_nmda.b2_model.network["Mn"].g,
        reconstructed_model.b2_model.network["Mn"].g,
    )


def test_stim_array_simulation_padding() -> None:
    sim_time = 0.010
    stim_padding_seconds = 3
    arr = np.random.rand(100, 30)
    model = default_model()
    model.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr.copy(), "mS", dt=0.001
    )
    assert model.stim_arrays["stim"].values.shape == (100, 30)
    model.run_simulation(sim_time, stim_array_padding_seconds=stim_padding_seconds)
    new_stim_array = model.stim_arrays["stim"]
    # timed array length is simulation length + padding
    assert new_stim_array.values.shape[0] == pytest.approx(
        sim_time / 0.001 + stim_padding_seconds / 0.001
    )
    # timed array is padded with zeros - original values are preserved
    assert np.allclose(new_stim_array.values[0:100, :], arr)
    assert np.max(new_stim_array.values[100:, :]) == pytest.approx(0)


def test_stim_array_padding() -> None:
    arr = np.random.rand(1000, 30)
    model = default_model()
    model.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert model.stim_arrays["stim"].values.shape == (1000, 30)
    # pad to 1.2s
    model.pad_stim_arrays(1.2)
    new_stim_array = model.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(1200, abs=1)
    assert np.allclose(new_stim_array.values[0 : arr.shape[0], :], arr)
    assert np.max(new_stim_array.values[1000:, :]) == pytest.approx(0)
    # pad to 2s
    model.pad_stim_arrays(2)
    new_stim_array = model.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(2000, abs=1)
    assert np.allclose(new_stim_array.values[0 : arr.shape[0], :], arr)
    assert np.max(new_stim_array.values[1000:, :]) == pytest.approx(0)


def test_stim_array_trimming(mfx_ornstein: Model) -> None:
    arr = np.random.rand(1000, 30)
    # make that last 100 values (0.1s) zero
    arr[900:, :] = 0
    mfx_ornstein.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert mfx_ornstein.stim_arrays["stim"].values.shape == (1000, 30)
    assert np.sum(mfx_ornstein.stim_arrays["stim"].values[900:, :]) == 0
    # last 0.1 seconds have no non-zero stimuli
    assert mfx_ornstein.trim_stim_array_would_delete_data(1.0) is False
    assert mfx_ornstein.trim_stim_array_would_delete_data(0.901) is False
    assert mfx_ornstein.trim_stim_array_would_delete_data(0.899) is True
    # trim to 0.95s
    assert mfx_ornstein.trim_stim_array_would_delete_data(0.95) is False
    mfx_ornstein.trim_stim_arrays(0.95)
    new_stim_array = mfx_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(950, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )
    # trim to 0.5s
    assert mfx_ornstein.trim_stim_array_would_delete_data(0.5) is True
    mfx_ornstein.trim_stim_arrays(0.5)
    new_stim_array = mfx_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(500, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )
    # trim to 0s
    assert mfx_ornstein.trim_stim_array_would_delete_data(0) is True
    mfx_ornstein.trim_stim_arrays(0)
    new_stim_array = mfx_ornstein.stim_arrays["stim"]
    assert new_stim_array.values.shape[0] == pytest.approx(0, abs=1)
    assert np.allclose(
        new_stim_array.values,
        arr[0 : new_stim_array.values.shape[0], :],
    )


def test_stim_array_properties() -> None:
    arr = np.random.rand(1000, 30)
    model = default_model()
    model.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    assert model.stim_arrays["stim"].values.shape == (1000, 30)
    model.update_stim_array_properties(
        "stim", units="nS", dt=0.01, n_neurons=25, stim_type=StimType.Inhibition
    )
    updated_stim_array = model.stim_arrays["stim"]
    assert updated_stim_array.units == "nS"
    assert updated_stim_array.dt == pytest.approx(0.01)
    assert updated_stim_array.values.shape == (1000, 25)
    assert updated_stim_array.stim_type == StimType.Inhibition
    model.update_stim_array_properties(
        "stim", units="nS", dt=0.001, n_neurons=44, stim_type=StimType.External
    )
    updated_stim_array = model.stim_arrays["stim"]
    assert updated_stim_array.units == "nS"
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (1000, 44)
    assert updated_stim_array.stim_type == StimType.External


def test_stim_array_rename() -> None:
    arr = np.random.rand(1000, 30)
    model = Model()
    model.stim_arrays["stim"] = StimArray(
        "stim", StimType.Excitation, arr, "mS", dt=0.001
    )
    model.rename_stim_array("stim", "stim2")
    assert len(model.stim_arrays) == 1
    assert "stim" not in model.stim_arrays
    updated_stim_array = model.stim_arrays["stim2"]
    assert updated_stim_array.units == "mS"
    assert updated_stim_array.dt == pytest.approx(0.001)
    assert updated_stim_array.values.shape == (1000, 30)
    assert updated_stim_array.stim_type == StimType.Excitation
    model.delete_stim_array("stim2")
    assert "stim2" not in model.stim_arrays
    assert len(model.stim_arrays) == 0
