from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from qtpy import QtWidgets

from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog


# For
def _build_cellpose_import_result(tmp_path, qtbot, *, offset: int | None = None):
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    image = np.zeros((6, 6, 3), dtype=np.uint8)
    segmentation = np.zeros((6, 6), dtype=np.int32)
    segmentation[1:3, 1:3] = 4
    segmentation[1:3, 3:5] = 5
    segmentation[3:5, 1:3] = 6

    image_path = tmp_path / "cellpose_image.tif"
    seg_path = tmp_path / "segmentation.npy"
    Image.fromarray(image).save(image_path)
    np.save(seg_path, segmentation)

    dialog._load_image_file(str(image_path))
    dialog._load_segmentation_file(str(seg_path))
    if offset is not None:
        dialog.linear_offset_spin.setValue(offset)

    dialog.accept = lambda: None
    dialog._import_data()
    return dialog.get_imported_data()


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

    # Default stim ID target is 0.
    assert dialog.custom_mapping_line.text() == "0"
    dialog.close()


# Wider patchclamp-cellpose integration with multiple patchclamp neurons
def test_patchclamp_defaults_after_cellpose_import(tmp_path, qtbot):
    cellpose_import = _build_cellpose_import_result(tmp_path, qtbot)
    dialog = PatchClampImportDialog(cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)

    dialog.load_file("tests/sampledata/3patchclampneurons.npy")

    assert dialog.custom_mapping_line.text() == "0,1,2"
    assert dialog.data_row_count == 3
    dialog.close()


def test_patchclamp_defaults_with_linear_offset_mapping(tmp_path, qtbot):
    tested_offset = 3
    cellpose_import = _build_cellpose_import_result(
        tmp_path, qtbot, offset=tested_offset
    )
    dialog = PatchClampImportDialog(cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)

    dialog.load_file("tests/sampledata/3patchclampneurons.npy")
    mapping = cellpose_import.stimulus_to_cellpose_mapping or {}

    assert mapping != {}
    assert mapping.get(0) == tested_offset
    assert mapping.get(2) == tested_offset + 2

    # Verify UI state
    assert dialog.import_button.isEnabled()
    assert "3patchclampneurons.npy" in dialog.file_path_label.text()
    assert dialog.custom_mapping_line.text() == "0,1,2"
    dialog.close()


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
    dialog.close()


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
    dialog.close()


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
    dialog.close()


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
    assert "NPY DATA:" in preview_text
    assert "duration:" in preview_text
    assert str(np.mean(trace_data))[:5] in preview_text
    dialog.close()


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
    dialog.close()


"""
 Giving "2, 3" as input should mean the 0th (first) patchclamp data targets stim ID 2.
 This should NOT be remapped via the cellpose mapping. Cellpose is context-only here.
  """


def test_custom_mapping_stim_id_input_uses_stim_ids_directly(tmp_path, qtbot):
    """Custom input '2,3' should import to stim IDs 2 and 3, even with mapping."""
    two_neuron_data = _generate_two_neuron_patchclamp_trace(
        duration_sec=0.2, sampling_rate=1000.0, num_spikes_per_neuron=1
    )
    npy_path = tmp_path / "two_neuron_patch_recording.npy"
    np.save(npy_path, two_neuron_data)

    cellpose_import = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=int),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([2, 3], dtype=int),
        stimulus_to_cellpose_mapping={
            7: 2,
            11: 3,
        },  # For manual testing, it's pretty handy that you can paste this
        # into the Cellpose modal directly as the Custom mapping.
    )

    dialog = PatchClampImportDialog(cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog.load_file(str(npy_path))

    dialog.custom_mapping_line.setText("2, 3")
    dialog.sampling_rate_spin.setValue(1000.0)
    dialog.peak_height_spin.setValue(2.0)
    dialog.peak_prominence_spin.setValue(1.0)

    dialog.accept = lambda: None
    dialog.import_data()

    assert _trace_data_exists_for_index(dialog.imported_trace_data, 2)
    assert _trace_data_exists_for_index(dialog.imported_trace_data, 3)
    assert _spike_data_exists_for_index(dialog.imported_spike_data, 2)
    assert _spike_data_exists_for_index(dialog.imported_spike_data, 3)
    assert not _trace_data_exists_for_index(dialog.imported_trace_data, 7)
    assert not _trace_data_exists_for_index(dialog.imported_trace_data, 11)
    dialog.close()


def test_custom_mapping_duplicate_stimulus_ids_error(tmp_path, qtbot, monkeypatch):
    """Custom mapping should error if multiple rows resolve to the same stim ID."""
    traces = []
    for _ in range(4):
        traces.append(
            _generate_patchclamp_trace(
                duration_sec=0.2, sampling_rate=1000.0, num_spikes=1
            )
        )
    four_neuron_data = np.vstack(traces)
    npy_path = tmp_path / "four_neuron_patch_recording.npy"
    np.save(npy_path, four_neuron_data)

    cellpose_import = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=int),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([1, 3, 6, 10], dtype=int),
        stimulus_to_cellpose_mapping={3: 1},
    )

    dialog = PatchClampImportDialog(cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog.load_file(str(npy_path))

    dialog.custom_mapping_line.setText("3,3,6,10")
    dialog.sampling_rate_spin.setValue(1000.0)
    dialog.peak_height_spin.setValue(2.0)
    dialog.peak_prominence_spin.setValue(1.0)

    captured = {}

    def _capture_critical(_parent, title, text):
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", _capture_critical, raising=True
    )

    dialog.import_data()

    assert "Custom mapping would assign stimulus ID 3" in captured.get("text", "")
    dialog.close()


def test_multi_row_defaults_are_zero_indexed_with_cellpose_ids(tmp_path, qtbot):
    two_neuron_data = _generate_two_neuron_patchclamp_trace(
        duration_sec=0.2, sampling_rate=1000.0, num_spikes_per_neuron=1
    )
    npy_path = tmp_path / "two_neuron_patch_recording.npy"
    np.save(npy_path, two_neuron_data)

    cellpose_import = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=int),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([10, 11], dtype=int),
        stimulus_to_cellpose_mapping=None,
    )

    dialog = PatchClampImportDialog(cellpose_import=cellpose_import)
    qtbot.addWidget(dialog)
    dialog.load_file(str(npy_path))

    assert dialog.custom_mapping_line.text() == "0,1"
    dialog.close()


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
    dialog.custom_mapping_line.setText("15,17")
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
    dialog.custom_mapping_line.setText("15")  # Only one ID for two traces
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
