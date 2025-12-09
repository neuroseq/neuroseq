from __future__ import annotations

import numpy as np

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog


def _generate_patchclamp_trace(
    duration_sec: float = 30.0,
    sampling_rate: float = 10000.0,
    baseline: float = -60.0,
    spike_amplitude: float = 100.0,
    num_spikes: int = 5,
    noise_std: float = 2.0,
) -> np.ndarray:
    """Generate synthetic patch clamp recording/noise with spikes, AI generated."""
    n_samples = int(duration_sec * sampling_rate)

    trace = np.random.normal(baseline, noise_std, n_samples)

    # Random interval spikes
    if num_spikes > 0:
        min_interval_samples = int(0.1 * sampling_rate)
        spike_positions = []

        for _ in range(num_spikes):
            # Find valid position
            attempts = 0
            while attempts < 100:
                pos = np.random.randint(
                    min_interval_samples, n_samples - min_interval_samples
                )
                if (
                    not spike_positions
                    or min(abs(pos - sp) for sp in spike_positions)
                    >= min_interval_samples
                ):
                    spike_positions.append(pos)
                    break
                attempts += 1

        # Add spike waveforms
        for pos in spike_positions:
            spike_width = int(0.002 * sampling_rate)  # 2ms spike
            start_idx = max(0, pos - spike_width // 4)
            end_idx = min(n_samples, pos + spike_width)

            for i in range(start_idx, end_idx):
                if i < pos:
                    # Rising phase
                    progress = (i - start_idx) / (pos - start_idx)
                    amplitude = spike_amplitude * progress
                else:
                    # Falling phase
                    progress = (i - pos) / (end_idx - pos)
                    amplitude = spike_amplitude * np.exp(
                        -progress * 5
                    )  # exponential decay

                trace[i] += amplitude

    return trace.astype(np.float64)


def _trace_data_exists_for_index(
    trace_data_list: list[TraceData], neuron_index: int
) -> bool:
    """Check if trace data exists for a given neuron index."""
    for trace_data in trace_data_list:
        if neuron_index in trace_data.indices:
            return True
    return False


def _spike_data_exists_for_index(
    spike_data_list: list[SpikeData], neuron_index: int
) -> bool:
    """Check if spike data exists for a given neuron index."""
    for spike_data in spike_data_list:
        if neuron_index in spike_data.spike_indices:
            return True
    return False


def test_load_single_npy_file(tmp_path, qtbot):
    """Test loading a single neuron patch clamp .npy file."""
    # Generate synthetic patch clamp data
    trace_data = _generate_patchclamp_trace(
        duration_sec=5.0, sampling_rate=10000.0, num_spikes=3, noise_std=1.5
    )

    # Save to temporary .npy file with neuron ID in filename
    npy_path = tmp_path / "patch_clamp_recording_neuronID_42.npy"
    np.save(npy_path, trace_data)

    # Create dialog and load file
    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    # Verify data loaded correctly
    assert dialog.npy_data is not None
    assert dialog.npy_data.shape == trace_data.shape
    assert dialog.npy_data.dtype == trace_data.dtype
    np.testing.assert_array_equal(dialog.npy_data, trace_data)

    # Verify neuron ID extracted from filename
    assert dialog.extracted_neuron_id == 42
    assert dialog.custom_neuron_id_spin.value() == 42

    # Verify UI state
    assert dialog.import_button.isEnabled()
    assert "patch_clamp_recording_neuronID_42.npy" in dialog.file_path_label.text()


def test_load_npy_without_neuron_id(tmp_path, qtbot):
    """Test loading patch clamp file without neuron ID in filename."""
    trace_data = _generate_patchclamp_trace(duration_sec=2.0, num_spikes=1)

    npy_path = tmp_path / "recording_data.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    assert dialog.npy_data is not None
    assert dialog.extracted_neuron_id is None
    # Default value should remain - visually htis sometimes diappears
    assert dialog.custom_neuron_id_spin.value() == 10


def test_load_invalid_npy_file(tmp_path, qtbot):
    """Test loading invalid .npy file (wrong dimensions)."""
    # Create 2D array (invalid for patch clamp)
    invalid_data = np.random.rand(100, 50)
    npy_path = tmp_path / "invalid_data.npy"
    np.save(npy_path, invalid_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    # Should raise error and not enable import
    dialog.load_file(str(npy_path))

    assert not dialog.import_button.isEnabled()


def test_import_single_npy_to_correct_index(tmp_path, qtbot):
    """Test importing a single .npy file and verifying it exists at the correct index."""
    trace_data = _generate_patchclamp_trace(
        duration_sec=1.0,
        sampling_rate=1000.0,
        baseline=-70.0,
        spike_amplitude=120.0,
        num_spikes=3,
        noise_std=0.5,
    )

    npy_path = tmp_path / "patch_clamp_recording_neuronID_15.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    # Set Custom mapping to neuron ID 15
    for button in dialog.mapping_buttons.buttons():
        if button.text() == "Custom":
            button.setChecked(True)
            break
    dialog.custom_neuron_id_spin.setValue(15)

    # Set reasonable detection parameters
    dialog.sampling_rate_spin.setValue(1000.0)
    dialog.peak_height_spin.setValue(2.0)
    dialog.peak_prominence_spin.setValue(1.0)

    # Mock the dialog accept to avoid actual GUI interaction
    dialog.accept = lambda: None

    # Import the data
    dialog.import_data()

    # Verify imported data exists at index 15
    assert _trace_data_exists_for_index(dialog.imported_trace_data, 15)
    assert _spike_data_exists_for_index(dialog.imported_spike_data, 15)

    # Verify data does not exist at index 10
    assert not _trace_data_exists_for_index(dialog.imported_trace_data, 10)
    assert not _spike_data_exists_for_index(dialog.imported_spike_data, 10)

    # Verify correct neuron ID in imported data
    assert dialog.imported_trace_data[0].indices[0] == 15
    assert 15 in dialog.imported_spike_data[0].spike_indices


def test_import_data_functionality(tmp_path, qtbot):
    """Test the actual import functionality with detailed verification."""
    trace_data = _generate_patchclamp_trace(
        duration_sec=1.0,
        sampling_rate=1000.0,
        baseline=-70.0,
        spike_amplitude=120.0,
        num_spikes=3,
        noise_std=0.5,
    )

    npy_path = tmp_path / "patch_clamp_recording_neuronID_25.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))
    dialog.sampling_rate_spin.setValue(1000.0)
    dialog.peak_height_spin.setValue(2.0)
    dialog.peak_prominence_spin.setValue(1.0)

    dialog.accept = lambda: None
    dialog.import_data()

    # Verify imported data
    assert len(dialog.imported_spike_data) == 1
    assert len(dialog.imported_trace_data) == 1

    spike_data = dialog.imported_spike_data[0]
    trace_data_imported = dialog.imported_trace_data[0]

    # Check spike data properties
    assert spike_data.name == "Patch Clamp Spikes"
    assert 25 in spike_data.spike_indices

    # Check trace data properties
    assert trace_data_imported.name == "Patch Clamp Traces"
    assert trace_data_imported.traces.shape[0] == 1  # Single neuron
    assert trace_data_imported.traces.shape[1] == len(trace_data)  # All timepoints
    assert trace_data_imported.indices[0] == 25
    assert trace_data_imported.units == "mV"
    assert abs(trace_data_imported.dt - 0.001) < 1e-6  # 1/1000


def test_different_sampling_rates(tmp_path, qtbot):
    """Test with different sampling rates."""
    for sampling_rate in [1000.0, 5000.0, 10000.0, 20000.0]:
        trace_data = _generate_patchclamp_trace(
            duration_sec=0.5, sampling_rate=sampling_rate, num_spikes=1
        )

        npy_path = tmp_path / f"test_{int(sampling_rate)}hz.npy"
        np.save(npy_path, trace_data)

        dialog = PatchClampImportDialog()
        qtbot.addWidget(dialog)

        dialog.load_file(str(npy_path))
        dialog.sampling_rate_spin.setValue(sampling_rate)

        expected_dt = 1.0 / sampling_rate

        dialog.accept = lambda: None
        dialog.import_data()

        if len(dialog.imported_trace_data) > 0:
            actual_dt = dialog.imported_trace_data[0].dt
            assert abs(actual_dt - expected_dt) < 1e-10


def test_no_spikes_detected(tmp_path, qtbot):
    """Test behavior when no spikes are detected."""
    # Generate flat trace with no spikes
    trace_data = np.full(10000, -70.0) + np.random.normal(0, 0.1, 10000)

    npy_path = tmp_path / "flat_trace.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))
    dialog.accept = lambda: None

    # Import should still work, just with no spikes
    dialog.import_data()

    assert len(dialog.imported_spike_data) == 1
    assert len(dialog.imported_trace_data) == 1

    spike_data = dialog.imported_spike_data[0]
    assert len(spike_data.spike_times) == 0  # No spikes detected


def test_preview_text_content(tmp_path, qtbot):
    """Test that preview shows correct information about loaded data."""
    trace_data = _generate_patchclamp_trace(
        duration_sec=3.0, sampling_rate=5000.0, baseline=-65.0, num_spikes=2
    )

    npy_path = tmp_path / "patch_clamp_recording_neuronID_123.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    preview_text = dialog.preview_text.toPlainText()

    # Check that preview contains expected information
    assert f"{trace_data.shape[0]} timepoints" in preview_text
    assert "extracted neuron ID: 123" in preview_text
    assert "duration:" in preview_text
    assert str(np.mean(trace_data))[:5] in preview_text
