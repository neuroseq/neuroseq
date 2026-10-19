import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.project import Project
from neuroseq.synthetic_data_dialog import SyntheticDataDialog


@pytest.fixture
def dialog(qtbot):
    widget = SyntheticDataDialog(Project())
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("rate", [1000, 3000, 10000])
def test_patch_sampling_rate(dialog, rate):
    t = np.arange(10000) / 10000
    source = TraceData(
        "source",
        np.array([np.sin(2 * np.pi * 50 * t), np.ones(t.size)]),
        np.array([0, 1]),
        "mV",
        0.0001,
    )
    original = source.traces.copy()
    dialog.patch_sampling_rate_spinbox.setValue(rate)
    result = dialog._generate_synthetic_patch_clamp_traces(source, np.array([1, 0]), 0)
    assert result is not None
    traces, dt = result
    assert dt == pytest.approx(1 / rate)
    assert traces.shape == (2, rate)
    np.testing.assert_allclose(traces[0], 1, atol=0.002)
    output_t = np.arange(rate) * dt
    np.testing.assert_allclose(
        traces[1, 20:-20], np.sin(2 * np.pi * 50 * output_t[20:-20]), atol=0.003
    )
    np.testing.assert_array_equal(source.traces, original)


def test_patch_downsampling_suppresses_aliasing(dialog):
    t = np.arange(10000) / 10000
    source = TraceData(
        "source", np.sin(2 * np.pi * 1800 * t)[None, :], np.array([0]), "mV", 0.0001
    )
    dialog.patch_sampling_rate_spinbox.setValue(1000)
    result = dialog._generate_synthetic_patch_clamp_traces(source, None, 0)
    assert result is not None
    assert np.max(np.abs(result[0][0, 20:-20])) < 0.01


def test_patch_sampling_rejects_upsampling(dialog, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "neuroseq.synthetic_data_dialog.QtWidgets.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )
    source = TraceData("source", np.ones((1, 10)), np.array([0]), "mV", 0.01)
    dialog.patch_sampling_rate_spinbox.setValue(1000)
    assert dialog._generate_synthetic_patch_clamp_traces(source, None, 0) is None
    assert "100 Hz" in warnings[0][2]


def test_patch_add_and_export_use_output_rate(dialog, monkeypatch, tmp_path):
    source = TraceData("source", np.ones((1, 10000)), np.array([0]), "mV", 0.0001)
    monkeypatch.setattr(dialog.model, "get_data", lambda *args: source)
    dialog.data_patch_clamp_combo.addItem("source")
    dialog.data_patch_clamp_combo.setCurrentText("source")
    dialog.patch_sampling_rate_spinbox.setValue(3000)
    dialog.noise_spinbox.setValue(0)
    added = []
    monkeypatch.setattr(
        dialog.model, "add_experimental_data", lambda name, data: added.extend(data)
    )
    monkeypatch.setattr(
        "neuroseq.synthetic_data_dialog.QtWidgets.QMessageBox.information",
        lambda *args: None,
    )
    dialog._add_patch_clamp_to_project()
    assert added[0].dt == pytest.approx(1 / 3000)
    assert added[0].traces.shape == (1, 3000)
    for extension in ("npy", "csv"):
        path = tmp_path / f"traces.{extension}"
        monkeypatch.setattr(
            "neuroseq.synthetic_data_dialog.QtWidgets.QFileDialog.getSaveFileName",
            lambda *args, path=path: (str(path), ""),
        )
        dialog._export_patch_clamp_data()
        values = (
            np.load(path) if extension == "npy" else np.loadtxt(path, delimiter=",")
        )
        np.testing.assert_allclose(values.reshape(1, -1), added[0].traces)


def test_indicator_controls(dialog):
    assert dialog.indicator_response_combo.currentData() == "decay"
    assert not dialog.rise_time_spinbox.isEnabled()
    dialog.indicator_response_combo.setCurrentIndex(1)
    assert dialog.rise_time_spinbox.isEnabled()
    assert dialog.rise_time_spinbox.value() == 0.01
    assert dialog.btn_add_imaging.isEnabled()
    for invalid_rise in (0.05, 0.06):
        dialog.rise_time_spinbox.setValue(invalid_rise)
        assert not dialog.btn_add_imaging.isEnabled()
        assert not dialog.btn_export_imaging.isEnabled()
        assert dialog.indicator_validation_label.text()
    dialog.decay_time_spinbox.setValue(0.1)
    assert dialog.btn_add_imaging.isEnabled()
    dialog.decay_time_spinbox.setValue(0.05)
    dialog.indicator_response_combo.setCurrentIndex(0)
    assert dialog.btn_add_imaging.isEnabled()
    assert dialog.btn_export_imaging.isEnabled()
    assert not dialog.indicator_validation_label.text()


def test_rise_decay_impulse_and_superposition(dialog):
    dt = 0.0001
    spikes = np.zeros((3, 5000))
    spikes[0, 0] = 0.4
    spikes[1, 100] = 0.4
    spikes[2] = spikes[0] + spikes[1]
    result = dialog._apply_rise_and_decay(spikes, dt, 0.01, 0.05)
    t = np.arange(5000) * dt
    peak_time = np.log(5) / 80
    peak = np.exp(-peak_time / 0.05) - np.exp(-peak_time / 0.01)
    expected = 0.4 * (np.exp(-t / 0.05) - np.exp(-t / 0.01)) / peak
    np.testing.assert_allclose(result[0], expected, atol=1e-12)
    assert result[0, 0] == 0
    assert np.argmax(result[0]) * dt == pytest.approx(peak_time, abs=dt)
    assert result[0].max() == pytest.approx(0.4, rel=1e-5)
    assert result[0, -1] < 0.0001
    np.testing.assert_allclose(result[1, :100], 0)
    np.testing.assert_allclose(result[1, 100:], result[0, :-100])
    np.testing.assert_allclose(result[2], result[0] + result[1], atol=1e-12)


@pytest.mark.parametrize("rise", [0, -0.01, 0.05, 0.1])
def test_invalid_rise_time(dialog, rise):
    with pytest.raises(ValueError, match="Rise time"):
        dialog._apply_rise_and_decay(np.zeros((1, 10)), 0.0001, rise, 0.05)


@pytest.mark.parametrize("mode", [0, 1])
def test_generate_imaging_response(dialog, monkeypatch, mode):
    dialog.indicator_response_combo.setCurrentIndex(mode)
    dialog.recording_duration_spinbox.setValue(0.2)
    dialog.num_neurons_spinbox.setValue(2)
    dialog.sampling_rate_spinbox.setValue(1000)
    dialog.df_f_var_spinbox.setValue(0)
    dialog.lognormal_sigma_spinbox.setValue(0)
    # Inspect expected fluorescence before random photoelectron sampling.
    monkeypatch.setattr(np.random, "poisson", lambda values: values)
    spikes = SpikeData("spikes", np.array([1]), np.array([0.0]))
    traces, indices, dt = dialog._generate_synthetic_imaging_traces(
        spikes, np.array([1, 0])
    )
    t = np.arange(traces.shape[1]) * dt
    if mode == 0:
        response = np.exp(-t / 0.05)
    else:
        peak_time = np.log(5) / 80
        peak = np.exp(-peak_time / 0.05) - np.exp(-peak_time / 0.01)
        response = (np.exp(-t / 0.05) - np.exp(-t / 0.01)) / peak
    np.testing.assert_allclose(traces[0], 1000 * (1 + 0.4 * response))
    np.testing.assert_allclose(traces[1], 1000)
    np.testing.assert_array_equal(indices, [1, 0])
    assert dt == 0.001


def test_add_experimental_data_duration_uses_sample_count() -> None:
    project = Project()
    traces = TraceData("t", np.zeros((3, 500)), np.arange(3), "mV", 0.002)
    project.add_experimental_data("exp", [traces])
    # 500 samples x 2 ms, not 3 neurons x 2 ms.
    assert project.experiments["exp"].duration == pytest.approx(1.0)
    assert project.max_time_secs == pytest.approx(1.0)


def test_synthetic_patch_clamp_experiment_duration(dialog, monkeypatch):
    source = TraceData("source", np.ones((2, 10000)), np.array([0, 1]), "mV", 0.0001)
    monkeypatch.setattr(dialog.model, "get_data", lambda *args: source)
    dialog.data_patch_clamp_combo.addItem("source")
    dialog.data_patch_clamp_combo.setCurrentText("source")
    dialog.patch_sampling_rate_spinbox.setValue(1000)
    monkeypatch.setattr(
        "neuroseq.synthetic_data_dialog.QtWidgets.QMessageBox.information",
        lambda *args: None,
    )
    dialog._add_patch_clamp_to_project()
    (experiment,) = dialog.model.experiments.values()
    assert experiment.duration == pytest.approx(1.0)
