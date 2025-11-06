from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from pytestqt.qtbot import QtBot
from qtpy import QtCore, QtWidgets
from unittest.mock import patch, MagicMock

from neuroseq.fluorescent_suite2p_contrasting_traces_import_dialog import FluorescentSuite2pContrastingTracesImportDialog
from neuroseq.gui import Gui


class TestFluorescentSuite2pContrastingTracesImportDialog:
    """Test cases for the Suite2p contrasting traces import dialog"""
    
    def test_dialog_initialization(self, qtbot: QtBot):
        """Test that the dialog initializes correctly with all UI elements"""
        dialog = FluorescentSuite2pContrastingTracesImportDialog()
        qtbot.addWidget(dialog)
        
        assert dialog.f_file_path_label is not None
        assert dialog.import_button is not None
        
        assert not dialog.import_button.isEnabled()
        assert dialog.max_neurons_spin.value() == 50

    def test_file_loading_and_preview(self, qtbot: QtBot):
        """Test loading F.npy file and showing preview"""
        dialog = FluorescentSuite2pContrastingTracesImportDialog()
        qtbot.addWidget(dialog)
        
        test_f_path = Path("/home/james/Work/neuroseq/src/imported_data/F.npy")
        dialog.load_f_file(str(test_f_path))
        dialog.load_fneu_file(str(test_f_path.parent / "Fneu.npy"))
        
        assert dialog.import_button.isEnabled()
        assert "Shape: (5, 900)" in dialog.preview_text.toPlainText() # handy to know hwo many neurons x time steps are in the import

    def test_data_import_with_valid_files(self, qtbot: QtBot):
        """Test importing data from F.npy and Fneu.npy files"""
        dialog = FluorescentSuite2pContrastingTracesImportDialog()
        qtbot.addWidget(dialog)
        
        test_data_dir = Path("/home/james/Work/neuroseq/src/imported_data")
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        
        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))
        
        dialog.import_data()
        
        assert "suite2p_contrasting_traces" in imported_data
        assert "_metadata" in imported_data
        
        trace_data = imported_data["suite2p_contrasting_traces"]
        assert "traces" in trace_data
        assert "timestamps" in trace_data
        
        traces = trace_data["traces"]
        assert traces.shape == (5, 900)  # 5 neurons, 900 time steps
        assert np.count_nonzero(traces) > 0  # non-zero values present
        assert np.all(np.isfinite(traces))  # no NaN or inf values

    def test_contrasting_calculation(self, qtbot: QtBot):
        """Test that F - 0.7 * Fneu calculation is performed correctly"""
        dialog = FluorescentSuite2pContrastingTracesImportDialog()
        qtbot.addWidget(dialog)
        
        test_data_dir = Path("/home/james/Work/neuroseq/src/imported_data")
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        
        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_contrasted = f_data - 0.7 * fneu_data
        
        imported_data = {}
        dialog.data_imported.connect(lambda data: imported_data.update(data))
        
        dialog.import_data()
        
        traces = imported_data["suite2p_contrasting_traces"]["traces"]
        np.testing.assert_array_almost_equal(
            traces, 
            expected_contrasted, 
            decimal=5
        )

    # todo when implemented: sample rate test



class TestSuite2pDataValidation:
    """Test data validation for Suite2p imports"""
    
    def test_data_dimensions_validation(self):
        """Test that imported data has correct dimensions"""
        test_data_dir = Path("/home/james/Work/neuroseq/src/imported_data")
        
        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")
        
        assert f_data.shape == (5, 900)
        assert fneu_data.shape == (5, 900)
        assert f_data.shape == fneu_data.shape

    def test_data_content_validation(self):
        """Test that data contains expected non-zero values"""
        test_data_dir = Path("/home/james/Work/neuroseq/src/imported_data")
        
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
        test_data_dir = Path("/home/james/Work/neuroseq/src/imported_data")
        
        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")
        
        assert f_data.shape[1] == 900
        assert fneu_data.shape[1] == 900