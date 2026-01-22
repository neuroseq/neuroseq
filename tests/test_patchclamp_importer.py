from __future__ import annotations

import numpy as np
import pytest

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
        spike_positions: list[int] = []
        # Clamp interval so randint has a valid range even for short traces
        max_valid_start = n_samples - min_interval_samples
        if max_valid_start <= min_interval_samples:
            min_interval_samples = max(1, n_samples // 4)
            max_valid_start = max(min_interval_samples + 1, n_samples)

        for _ in range(num_spikes):
            # Find valid position
            attempts = 0
            while attempts < 100:
                pos = np.random.randint(min_interval_samples, max_valid_start)
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
    return any(neuron_index in trace_data.indices for trace_data in trace_data_list)


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

    # Filename neuron ID should be used as the default target
    assert dialog.extracted_neuron_id == 42
    assert dialog.custom_mapping_line.text() == "42"
    assert dialog.custom_neuron_id_line.text() == "42"

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
    assert dialog.custom_mapping_line.text() == "1"
    assert dialog.custom_neuron_id_line.text() == "1"


def test_single_row_uses_filename_for_stimulus_id(tmp_path, qtbot):
    """Single-row files should default to the filename-derived stimulus ID when available."""
    trace_data = _generate_patchclamp_trace(duration_sec=1.0, sampling_rate=1000.0)

    npy_path = tmp_path / "single_trace_neuronID_77.npy"
    np.save(npy_path, trace_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    # Pre-fill with an invalid multi-ID mapping to ensure it gets reduced
    dialog.custom_mapping_line.setText("1,2,3")

    dialog.load_file(str(npy_path))

    assert dialog.custom_mapping_line.text() == "77"
    assert dialog.custom_neuron_id_line.text() == "77"


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

    # Set target stimulus ID to 15
    dialog._set_mapping_strategy("Custom")
    dialog.custom_mapping_line.setText("15")

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
    dialog._set_mapping_strategy("Custom")
    dialog.custom_mapping_line.setText("25")
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


@pytest.mark.parametrize("sampling_rate", [1000.0, 5000.0, 10000.0])
def test_different_sampling_rates(tmp_path, qtbot, sampling_rate: float):
    """Test with different sampling rates."""
    trace_data = _generate_patchclamp_trace(
        duration_sec=2.195, sampling_rate=sampling_rate, num_spikes=1
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

    actual_dt = dialog.imported_trace_data[0].dt
    assert actual_dt == pytest.approx(expected_dt)
    assert dialog.experiment_duration == pytest.approx(2.195)


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


def test_import_single_then_extend_timeline(tmp_path, qtbot):
    """Test importing single neuron data, then extending it with more data."""
    from unittest.mock import MagicMock

    # Create first recording (1 second at 1000Hz = 1000 samples)
    first_trace = _generate_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0, baseline=-70.0, num_spikes=2
    )
    first_npy = tmp_path / "patch_neuronID_15_part1.npy"
    np.save(first_npy, first_trace)

    # Create second recording (0.5 seconds at same rate = 500 samples)
    second_trace = _generate_patchclamp_trace(
        duration_sec=0.5, sampling_rate=1000.0, baseline=-70.0, num_spikes=1
    )
    second_npy = tmp_path / "patch_neuronID_15_part2.npy"
    np.save(second_npy, second_trace)

    # Mock model to simulate existing data
    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import (initial data)
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(first_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.custom_mapping_line.setText("15")
    dialog1.replace_checkbox.setChecked(True)  # Replace mode for first import

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Verify first import
    assert len(dialog1.imported_trace_data) == 1
    first_imported = dialog1.imported_trace_data[0]
    assert first_imported.indices[0] == 15
    assert first_imported.traces.shape[1] == 1000  # 1000 samples
    assert abs(first_imported.dt - 0.001) < 1e-6  # 1/1000 Hz

    # Simulate data being added to model (normally done by GUI)
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import in EXTEND mode
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(second_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.custom_mapping_line.setText("15")
    dialog2.replace_checkbox.setChecked(False)  # EXTEND mode

    dialog2.accept = lambda: None
    dialog2.import_data()

    # Verify second import extends the data
    assert len(dialog2.imported_trace_data) == 1
    second_imported = dialog2.imported_trace_data[0]
    assert second_imported.indices[0] == 15
    assert second_imported.traces.shape[1] == 500  # New data is 500 samples
    assert abs(second_imported.dt - 0.001) < 1e-6

    dialog2.close()


def test_import_single_then_replace_timeline(tmp_path, qtbot):
    """Test importing single neuron data, then replacing it with different data."""
    from unittest.mock import MagicMock

    # Create first recording (2 seconds at 1000Hz = 2000 samples)
    first_trace = _generate_patchclamp_trace(
        duration_sec=2.0, sampling_rate=1000.0, baseline=-70.0, num_spikes=4
    )
    first_npy = tmp_path / "patch_neuronID_15_original.npy"
    np.save(first_npy, first_trace)

    # Create replacement recording (1 second at same rate = 1000 samples)
    replacement_trace = _generate_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0, baseline=-70.0, num_spikes=2
    )
    replacement_npy = tmp_path / "patch_neuronID_15_replacement.npy"
    np.save(replacement_npy, replacement_trace)

    # Mock model
    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(first_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.custom_mapping_line.setText("15")
    dialog1.replace_checkbox.setChecked(True)  # Replace mode

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Verify first import (2 seconds)
    assert len(dialog1.imported_trace_data) == 1
    first_imported = dialog1.imported_trace_data[0]
    assert first_imported.traces.shape[1] == 2000  # 2000 samples

    # Simulate data being added to model
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import in REPLACE mode
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(replacement_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.custom_mapping_line.setText("15")
    dialog2.replace_checkbox.setChecked(True)  # REPLACE mode

    dialog2.accept = lambda: None
    dialog2.import_data()

    # Verify replacement data (should be 1 second, not 2)
    assert len(dialog2.imported_trace_data) == 1
    second_imported = dialog2.imported_trace_data[0]
    assert second_imported.traces.shape[1] == 1000  # Only 1000 samples (replacement)
    assert 15 in dialog2.neurons_to_replace  # Neuron marked for replacement

    dialog2.close()


def _generate_two_neuron_patchclamp_trace(
    duration_sec: float = 30.0,
    sampling_rate: float = 10000.0,
    baseline: float = -60.0,
    spike_amplitude: float = 100.0,
    num_spikes_per_neuron: int = 5,
    noise_std: float = 2.0,
) -> np.ndarray:
    """Generate synthetic patch clamp recording for two neurons (2D array)."""
    # Create two neuron traces
    trace_1 = _generate_patchclamp_trace(
        duration_sec,
        sampling_rate,
        baseline,
        spike_amplitude,
        num_spikes_per_neuron,
        noise_std,
    )
    trace_2 = _generate_patchclamp_trace(
        duration_sec,
        sampling_rate,
        baseline - 5,
        spike_amplitude * 0.8,
        num_spikes_per_neuron,
        noise_std,
    )

    # Stack into 2D array (2 neurons x n_samples)
    return np.vstack([trace_1, trace_2])


def test_multi_row_mapping_trims_to_row_count(tmp_path, qtbot):
    """Target stimulus IDs should not exceed the number of rows in the file."""
    two_neuron_data = _generate_two_neuron_patchclamp_trace(
        duration_sec=0.2, sampling_rate=1000.0, num_spikes_per_neuron=1
    )

    npy_path = tmp_path / "two_neuron_patch_recording.npy"
    np.save(npy_path, two_neuron_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    # Pretend the UI had more IDs than rows before loading
    dialog.custom_mapping_line.setText("5,6,7,8")

    dialog.load_file(str(npy_path))

    assert dialog.custom_mapping_line.text() == "5,6"
    assert dialog.custom_neuron_id_line.text() == "5,6"


def test_import_two_neuron_npy_file(tmp_path, qtbot):
    """Test importing .npy file with two neuron traces (2D array)."""
    # Generate two neuron traces
    two_neuron_data = _generate_two_neuron_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0, num_spikes_per_neuron=2
    )

    npy_path = tmp_path / "two_neuron_patch_recording.npy"
    np.save(npy_path, two_neuron_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    # Set up for two neurons
    dialog.custom_neuron_id_line.setText("15,17")
    dialog.sampling_rate_spin.setValue(1000.0)
    dialog.peak_height_spin.setValue(2.0)

    dialog.accept = lambda: None
    dialog.import_data()

    # Verify both neurons imported
    assert len(dialog.imported_trace_data) == 1  # One TraceData object
    trace_data = dialog.imported_trace_data[0]
    assert trace_data.traces.shape[0] == 2  # Two neurons
    assert trace_data.traces.shape[1] == 1000  # 1000 samples each
    assert 15 in trace_data.indices and 17 in trace_data.indices

    # Should have spike data for both neurons
    assert len(dialog.imported_spike_data) == 2  # Two SpikeData objects
    spike_neuron_ids = [
        15 if 15 in sd.spike_indices else 17 for sd in dialog.imported_spike_data
    ]
    assert 15 in spike_neuron_ids and 17 in spike_neuron_ids

    assert dialog.experiment_duration == pytest.approx(1.0)

    dialog.close()


def test_import_two_neuron_wrong_id_count(tmp_path, qtbot):
    """Test error when importing 2D npy with wrong number of neuron IDs."""
    two_neuron_data = _generate_two_neuron_patchclamp_trace(duration_sec=0.5)

    npy_path = tmp_path / "two_neuron_data.npy"
    np.save(npy_path, two_neuron_data)

    dialog = PatchClampImportDialog()
    qtbot.addWidget(dialog)

    dialog.load_file(str(npy_path))

    # Try to import with wrong number of IDs (should be 2, not 1)
    dialog.custom_neuron_id_line.setText("15")  # Only one ID for two traces
    dialog.sampling_rate_spin.setValue(1000.0)

    dialog.accept = lambda: None
    try:
        dialog.import_data()
        assert (
            False
            == "Should raise error for dimension mismatch (insufficent number of neuron IDs in mapping)"
        )
    except Exception:
        pass  # Expected error

    dialog.close()


def test_import_single_then_double_replace_mode(tmp_path, qtbot):
    """Test importing single neuron, then double (including same neuron) in REPLACE mode - Should always be fine"""
    from unittest.mock import MagicMock

    # Single neuron trace (neuron 15)
    single_trace = _generate_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0, num_spikes=2
    )
    single_npy = tmp_path / "single_neuron_15.npy"
    np.save(single_npy, single_trace)

    # Double neuron trace (neurons 15 and 17)
    double_trace = _generate_two_neuron_patchclamp_trace(
        duration_sec=1.5, sampling_rate=1000.0, num_spikes_per_neuron=3
    )
    double_npy = tmp_path / "double_neuron_15_17.npy"
    np.save(double_npy, double_trace)

    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import: single neuron 15
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(single_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.custom_mapping_line.setText("15")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.replace_checkbox.setChecked(True)  # Replace mode

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Verify first import
    assert len(dialog1.imported_trace_data) == 1
    first_trace = dialog1.imported_trace_data[0]
    assert 15 in first_trace.indices
    assert first_trace.traces.shape[1] == 1000  # 1 second

    # Simulate data added to model
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import: double neurons 15,17 in REPLACE mode
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(double_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.custom_mapping_line.setText("15,17")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.replace_checkbox.setChecked(True)  # REPLACE mode

    dialog2.accept = lambda: None
    dialog2.import_data()

    # Verify replace mode worked
    assert len(dialog2.imported_trace_data) == 1
    double_trace_imported = dialog2.imported_trace_data[0]
    assert double_trace_imported.traces.shape[0] == 2  # Two neurons
    assert double_trace_imported.traces.shape[1] == 1500  # 1.5 seconds
    assert 15 in double_trace_imported.indices and 17 in double_trace_imported.indices
    assert 15 in dialog2.neurons_to_replace  # Neuron 15 marked for replacement

    dialog2.close()


def test_import_single_then_double_extend_mode_error(tmp_path, qtbot):
    """Test importing single neuron, then double (mixed existing/new) in EXTEND mode should fail."""
    from unittest.mock import MagicMock

    # Single neuron trace (neuron 15)
    single_trace = _generate_patchclamp_trace(duration_sec=1.0, sampling_rate=1000.0)
    single_npy = tmp_path / "single_neuron_15.npy"
    np.save(single_npy, single_trace)

    # Double neuron trace (neurons 15 and 17)
    double_trace = _generate_two_neuron_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0
    )
    double_npy = tmp_path / "double_neuron_15_17.npy"
    np.save(double_npy, double_trace)

    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import: single neuron 15
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(single_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.custom_mapping_line.setText("15")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.replace_checkbox.setChecked(True)  # Replace mode

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Simulate data added to model
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import: try double neurons 15,17 in EXTEND mode (should fail)
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(double_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.custom_mapping_line.setText("15,17")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.replace_checkbox.setChecked(False)  # EXTEND mode

    dialog2.accept = lambda: None

    # Should raise validation error due to mixed existing (15) and new (17) neurons
    with pytest.raises(ValueError):
        dialog2.import_data()
    dialog2.close()


def test_import_new_neurons_extend_mode_with_existing_data_error(
    tmp_path, qtbot, caplog
):
    """Test importing new neurons in EXTEND mode when existing data exists should fail."""
    from unittest.mock import MagicMock

    # Single neuron trace (neuron 15)
    single_trace = _generate_patchclamp_trace(duration_sec=1.0, sampling_rate=1000.0)
    single_npy = tmp_path / "single_neuron_15.npy"
    np.save(single_npy, single_trace)

    # Different neurons trace (neurons 20,25)
    different_trace = _generate_two_neuron_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0
    )
    different_npy = tmp_path / "different_neurons_20_25.npy"
    np.save(different_npy, different_trace)

    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import: neuron 15
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(single_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.custom_mapping_line.setText("15")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.replace_checkbox.setChecked(True)

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Simulate data added to model
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import: try new neurons 20,25 in EXTEND mode (should fail because 15 exists)
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(different_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.custom_mapping_line.setText("20,25")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.replace_checkbox.setChecked(False)  # EXTEND mode

    dialog2.accept = lambda: None
    with pytest.raises(ValueError):
        dialog2.import_data()
    assert len(dialog2.imported_trace_data) == 0

    dialog2.close()


def test_import_existing_neurons_only_extend_mode_success(tmp_path, qtbot):
    """Test importing only existing neurons in EXTEND mode should succeed."""
    from unittest.mock import MagicMock

    # Two neuron trace (neurons 15,17)
    original_trace = _generate_two_neuron_patchclamp_trace(
        duration_sec=1.0, sampling_rate=1000.0
    )
    original_npy = tmp_path / "original_neurons_15_17.npy"
    np.save(original_npy, original_trace)

    # Extension for same neurons
    extension_trace = _generate_two_neuron_patchclamp_trace(
        duration_sec=0.5, sampling_rate=1000.0
    )
    extension_npy = tmp_path / "extension_neurons_15_17.npy"
    np.save(extension_npy, extension_trace)

    mock_model = MagicMock()
    mock_model.experiments = {}

    # First import: neurons 15,17
    dialog1 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog1)

    dialog1.load_file(str(original_npy))
    dialog1._set_mapping_strategy("Custom")
    dialog1.custom_mapping_line.setText("15,17")
    dialog1.sampling_rate_spin.setValue(1000.0)
    dialog1.replace_checkbox.setChecked(True)

    dialog1.accept = lambda: None
    dialog1.import_data()

    # Simulate data added to model
    mock_model.experiments["PatchClamp Import"] = dialog1.imported_data

    dialog1.close()

    # Second import: same neurons 15,17 in EXTEND mode (should work)
    dialog2 = PatchClampImportDialog(project=mock_model)
    qtbot.addWidget(dialog2)

    dialog2.load_file(str(extension_npy))
    dialog2._set_mapping_strategy("Custom")
    dialog2.custom_mapping_line.setText("15,17")
    dialog2.sampling_rate_spin.setValue(1000.0)
    dialog2.replace_checkbox.setChecked(False)  # EXTEND mode

    dialog2.accept = lambda: None
    dialog2.import_data()  # Should not raise error

    # Verify extension worked
    assert len(dialog2.imported_trace_data) == 1
    extended_trace = dialog2.imported_trace_data[0]
    assert extended_trace.traces.shape[0] == 2  # Two neurons
    assert extended_trace.traces.shape[1] == 500  # 0.5 seconds extension
    assert 15 in extended_trace.indices and 17 in extended_trace.indices

    dialog2.close()
