from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.experiment import SpikeData
from neuroseq.experiment import TraceData
from neuroseq.import_dialog import Suite2pImportDialog
from neuroseq.importer import compute_normalized_suite2p_traces


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
        assert dialog.spike_threshold_spin.value() == 0.02
        assert dialog.sampling_rate_spin.value() == 30.0
        assert dialog.replace_checkbox.isChecked() is True

    def test_file_loading_and_preview(self, qtbot: QtBot, suite2p_import_data_dir):
        """Test loading F.npy file and showing preview"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        test_f_path = test_data_dir / "F.npy"
        dialog.load_f_file(str(test_f_path))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        assert dialog.import_button.isEnabled()
        preview = dialog.preview_text.toPlainText()
        assert "Contrasting traces: 5 neurons × 900 timepoints" in preview

    def test_data_import_with_valid_files(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test importing data from F.npy and Fneu.npy files"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        # only get traces:
        assert len(dialog.imported_data) == 1
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        assert traces.shape == (5, 900)  # 5 neurons, 900 time steps
        assert np.all((traces >= 0.0) & (traces <= 1.0))
        assert np.all(np.isfinite(traces))  # no NaN or inf values
        assert dialog.f_data is not None
        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
        np.testing.assert_allclose(traces, expected_traces[: traces.shape[0], :])
        assert trace_data.dt == pytest.approx(1.0 / 30.0)

    def test_contrasting_calculation(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test that F - 0.7 * Fneu calculation is performed correctly"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        assert dialog.f_data is not None
        assert dialog.fneu_data is not None
        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_contrasted, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        assert len(dialog.imported_data) == 1
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        np.testing.assert_array_almost_equal(traces, expected_contrasted, decimal=5)

    def test_import_without_fneu_file(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """F.npy-only imports should normalize fluorescence without neuropil subtraction."""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))

        assert dialog.import_button.isEnabled()

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        assert len(dialog.imported_data) == 1
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        assert dialog.f_data is not None
        f_data = dialog.f_data
        expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, None)
        np.testing.assert_allclose(traces, expected_traces[: traces.shape[0], :])

    def test_optional_spike_import_with_threshold(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test importing optional spks file with adjustable threshold"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        dialog.load_spks_file(str(test_data_dir / "spks.npy"))

        dialog.spike_threshold_spin.setValue(
            0.02
        )  # realistic default, this will leave 36 spikes.

        dialog.import_data()
        assert len(dialog.imported_data) == 2
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        spike_data = dialog.imported_data[1]
        assert isinstance(spike_data, SpikeData)
        assert len(spike_data.spike_indices) > 0
        assert len(spike_data.spike_times) == len(spike_data.spike_indices)

    def test_replace_checkbox_sets_flag(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Checking the replace option propagates to emitted payloads."""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        dialog.replace_checkbox.setChecked(True)

        dialog.import_data()
        assert len(dialog.imported_data) == 1
        assert dialog.append_data is False


class TestSuite2pDataValidation:
    """Test data validation for Suite2p imports"""

    def test_data_dimensions_validation(self, suite2p_import_data_dir):
        """Test that imported data has correct dimensions"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape == (5, 900)
        assert fneu_data.shape == (5, 900)
        assert f_data.shape == fneu_data.shape

    def test_data_content_validation(self, suite2p_import_data_dir):
        """Test that data contains expected non-zero values"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert np.count_nonzero(f_data) > 0
        assert np.count_nonzero(fneu_data) > 0

        assert np.all(np.isfinite(f_data))
        assert np.all(np.isfinite(fneu_data))

        contrasted_data = f_data - 0.7 * fneu_data
        assert np.count_nonzero(contrasted_data) > 0

    def test_timesteps_length(self, suite2p_import_data_dir):
        """Test that data has exactly 900 timesteps"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape[1] == 900
        assert fneu_data.shape[1] == 900
