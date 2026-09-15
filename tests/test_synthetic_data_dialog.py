import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.project import Project
from neuroseq.synthetic_data_dialog import SyntheticDataDialog


@pytest.fixture
def dialog(qtbot):
    widget = SyntheticDataDialog(Project())
    qtbot.addWidget(widget)
    return widget


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
