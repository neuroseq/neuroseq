from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.suite2p_import_dialog import Suite2pImportDialog


class TestSuite2pImportDialog:
    """Test cases for the combined Suite2p import dialog"""

    def test_dialog_initialization(self, qtbot: QtBot):
        """Test that the dialog initializes correctly with all UI elements"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        assert dialog.f_file_path_label is not None
        assert dialog.import_button is not None

        assert not dialog.import_button.isEnabled()
        assert dialog.max_neurons_spin.value() == 2
        assert dialog.spike_threshold_spin.value() == 100.0
        assert dialog.sampling_rate_spin.value() == 30.0

    def test_file_loading_and_preview(self, qtbot: QtBot):
        """Test loading F.npy file and showing preview"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        test_f_path = test_data_dir / "F.npy"
        dialog.load_f_file(str(test_f_path))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        assert dialog.import_button.isEnabled()
        preview = dialog.preview_text.toPlainText()
        assert "Contrasting traces: 5 neurons × 900 timepoints" in preview

    def test_data_import_with_valid_files(self, qtbot: QtBot):
        """Test importing data from F.npy and Fneu.npy files"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))

        dialog.import_data()

        assert "suite2p_contrasting_traces" in imported_data

        trace_data = imported_data["suite2p_contrasting_traces"]
        assert "traces" in trace_data
        assert "timestamps" in trace_data

        traces = trace_data["traces"]
        assert traces.shape == (5, 900)  # 5 neurons, 900 time steps
        assert np.count_nonzero(traces) > 0  # non-zero values present
        assert np.all(np.isfinite(traces))  # no NaN or inf values
        assert trace_data["sampling_rate"] == 30.0
        assert trace_data["dt"] == pytest.approx(1.0 / 30.0)
        assert "suite2p_spikes" not in imported_data

    def test_contrasting_calculation(self, qtbot: QtBot):
        """Test that F - 0.7 * Fneu calculation is performed correctly"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_contrasted = f_data - 0.7 * fneu_data

        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))

        dialog.import_data()

        traces = imported_data["suite2p_contrasting_traces"]["traces"]
        np.testing.assert_array_almost_equal(traces, expected_contrasted, decimal=5)

    def test_optional_spike_import_with_threshold(self, qtbot: QtBot):
        """Test importing optional spks file with adjustable threshold"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        dialog.load_spks_file(str(test_data_dir / "spks.npy"))

        dialog.spike_threshold_spin.setValue(0.01)

        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))

        dialog.import_data()

        spike_data = imported_data["suite2p_spikes"]
        assert spike_data["threshold"] == dialog.spike_threshold_spin.value()
        assert len(spike_data["spikes"]) > 0
        assert len(spike_data["timestamps"]) == len(spike_data["spikes"])
        assert spike_data["dt"] == pytest.approx(
            1.0 / dialog.sampling_rate_spin.value()
        )

    def test_sampling_rate_adjustment_affects_timeline(self, qtbot: QtBot):
        """Changing the sampling rate should scale timestamps (base unit ms)."""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        dialog.sampling_rate_spin.setValue(10.0)  # Hz

        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))
        dialog.import_data()

        trace_data = imported_data["suite2p_contrasting_traces"]
        timestamps = trace_data["timestamps"]
        dt = trace_data["dt"]
        assert dt == pytest.approx(0.1)
        assert trace_data["sampling_rate"] == pytest.approx(10.0)

        assert timestamps[1] - timestamps[0] == pytest.approx(dt)
        total_duration = timestamps[-1] - timestamps[0]
        expected_duration = (trace_data["traces"].shape[1] - 1) * dt
        assert total_duration == pytest.approx(expected_duration)

    def test_replace_checkbox_sets_flag(self, qtbot: QtBot):
        """Checking the replace option propagates to emitted payloads."""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = Path(__file__).parent / "imported_data"
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        dialog.replace_checkbox.setChecked(True)

        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))
        dialog.import_data()

        assert imported_data.get("_replace_existing") is True


class TestSuite2pDataValidation:
    """Test data validation for Suite2p imports"""

    def test_data_dimensions_validation(self):
        """Test that imported data has correct dimensions"""
        test_data_dir = Path(__file__).parent / "imported_data"

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape == (5, 900)
        assert fneu_data.shape == (5, 900)
        assert f_data.shape == fneu_data.shape

    def test_data_content_validation(self):
        """Test that data contains expected non-zero values"""
        test_data_dir = Path(__file__).parent / "imported_data"

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert np.count_nonzero(f_data) > 0
        assert np.count_nonzero(fneu_data) > 0

        assert np.all(np.isfinite(f_data))
        assert np.all(np.isfinite(fneu_data))

        contrasted_data = f_data - 0.7 * fneu_data
        assert np.count_nonzero(contrasted_data) > 0

    def test_timesteps_length(self):
        """Test that data has exactly 900 timesteps"""
        test_data_dir = Path(__file__).parent / "imported_data"

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape[1] == 900
        assert fneu_data.shape[1] == 900
