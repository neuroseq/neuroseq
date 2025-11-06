from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename


class FluorescentSuite2pContrastingTracesImportDialog(QtWidgets.QDialog):
    """Dialog for importing Suite2p fluorescence trace data from F.npy and Fneu.npy files"""
    
    # Signal emitted when data is successfully imported
    data_imported = QtCore.Signal(dict)
    
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Import Suite2p Fluorescence Traces")
        self.setModal(True)
        self.resize(400, 250)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Select both F.npy and Fneu.npy files from Suite2p output.\n"
            "Traces will be calculated as F - 0.7 * Fneu for each neuron/timestep."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # File selection
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        
        # F.npy file selection
        f_file_layout = QtWidgets.QHBoxLayout()
        f_file_layout.addWidget(QtWidgets.QLabel("F.npy:"))
        self.f_file_path_label = QtWidgets.QLabel("No file selected")
        self.f_file_path_label.setStyleSheet("QLabel { color: gray; }")
        f_file_layout.addWidget(self.f_file_path_label)
        
        browse_f_button = QtWidgets.QPushButton("Browse...")
        browse_f_button.clicked.connect(self.browse_f_file)
        f_file_layout.addWidget(browse_f_button)
        file_layout.addLayout(f_file_layout)
        
        # Fneu.npy file selection
        fneu_file_layout = QtWidgets.QHBoxLayout()
        fneu_file_layout.addWidget(QtWidgets.QLabel("Fneu.npy:"))
        self.fneu_file_path_label = QtWidgets.QLabel("No file selected")
        self.fneu_file_path_label.setStyleSheet("QLabel { color: gray; }")
        fneu_file_layout.addWidget(self.fneu_file_path_label)
        
        browse_fneu_button = QtWidgets.QPushButton("Browse...")
        browse_fneu_button.clicked.connect(self.browse_fneu_file)
        fneu_file_layout.addWidget(browse_fneu_button)
        file_layout.addLayout(fneu_file_layout)
        
        layout.addWidget(file_group)
        
        # Data preview
        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # Import options
        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        
        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setMinimum(0.001)
        self.sampling_rate_spin.setMaximum(100000.0)
        self.sampling_rate_spin.setValue(0.001)  # Typical for calcium imaging
        self.sampling_rate_spin.setSuffix(" Hz")
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)
        
        self.max_neurons_spin = QtWidgets.QSpinBox()
        self.max_neurons_spin.setMinimum(1)
        self.max_neurons_spin.setMaximum(10000)
        self.max_neurons_spin.setValue(50)
        options_layout.addRow("Max neurons to import:", self.max_neurons_spin)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)
        
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def browse_f_file(self):
        """Open file browser to select F.npy file"""
        filename, _ = getopenfilename(
            self,
            "Select Suite2p F.npy Data File",
            "",
            "NumPy files (*.npy);;All files (*)"
        )
        
        if filename:
            self.load_f_file(filename)
            
    def browse_fneu_file(self):
        """Open file browser to select Fneu.npy file"""
        filename, _ = getopenfilename(
            self,
            "Select Suite2p Fneu.npy Data File",
            "",
            "NumPy files (*.npy);;All files (*)"
        )
        
        if filename:
            self.load_fneu_file(filename)
            
    def load_f_file(self, filename: str):
        """Load and validate F.npy file"""
        try:
            self.f_file_path = Path(filename)
            self.f_file_path_label.setText(str(self.f_file_path.name))
            self.f_file_path_label.setStyleSheet("QLabel { color: black; }")
            
            # Load the F.npy file
            self.f_data = np.load(filename)
            
            # Validate data format
            if self.f_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.f_data.ndim}D array")
            
            self.logger.info(f"Loaded F.npy: shape={self.f_data.shape}, dtype={self.f_data.dtype}")
            self._check_files_loaded()
            
        except Exception as e:
            self.logger.exception(f"Error loading F.npy file {filename}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading F.npy File",
                f"Could not load F.npy file {filename}:\n{str(e)}"
            )
            self.import_button.setEnabled(False)
            
    def load_fneu_file(self, filename: str):
        """Load and validate Fneu.npy file"""
        try:
            self.fneu_file_path = Path(filename)
            self.fneu_file_path_label.setText(str(self.fneu_file_path.name))
            self.fneu_file_path_label.setStyleSheet("QLabel { color: black; }")
            
            # Load the Fneu.npy file
            self.fneu_data = np.load(filename)
            
            # Validate data format
            if self.fneu_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.fneu_data.ndim}D array")
            
            self.logger.info(f"Loaded Fneu.npy: shape={self.fneu_data.shape}, dtype={self.fneu_data.dtype}")
            self._check_files_loaded()
            
        except Exception as e:
            self.logger.exception(f"Error loading Fneu.npy file {filename}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Fneu.npy File",
                f"Could not load Fneu.npy file {filename}:\n{str(e)}"
            )
            self.import_button.setEnabled(False)
            
    def _check_files_loaded(self):
        """Check if both files are loaded and validate compatibility"""
        if hasattr(self, 'f_data') and hasattr(self, 'fneu_data'):
            # Validate shapes match
            if self.f_data.shape != self.fneu_data.shape:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Shape Mismatch",
                    f"F.npy shape {self.f_data.shape} does not match Fneu.npy shape {self.fneu_data.shape}"
                )
                self.import_button.setEnabled(False)
                return
            
            self.logger.info(f"Both files loaded successfully with matching shapes: {self.f_data.shape}")
            self.update_preview()
            self.import_button.setEnabled(True)
            
    def update_preview(self):
        """Update the data preview"""
        if not (hasattr(self, 'f_data') and hasattr(self, 'fneu_data')):
            return
        
        # Calculate contrasting traces: F - 0.7 * Fneu
        contrasting_traces = self.f_data - 0.7 * self.fneu_data
        
        n_neurons, n_timepoints = self.f_data.shape
        
        preview_text = [
            f"F.npy shape: {self.f_data.shape}",
            f"F.npy range: {np.min(self.f_data):.3f} to {np.max(self.f_data):.3f}",
            f"Fneu.npy shape: {self.fneu_data.shape}",
            f"Fneu.npy range: {np.min(self.fneu_data):.3f} to {np.max(self.fneu_data):.3f}",
            f"Contrasting traces (F - 0.7*Fneu):",
            f"  Shape: {contrasting_traces.shape} ({n_neurons} neurons × {n_timepoints} timepoints)",
            f"  Range: {np.min(contrasting_traces):.3f} to {np.max(contrasting_traces):.3f}",
            f"  Mean: {np.mean(contrasting_traces):.3f} ± {np.std(contrasting_traces):.3f}"
        ]
        
        # Debug logging
        self.logger.info(f"F data stats: min={np.min(self.f_data):.3f}, max={np.max(self.f_data):.3f}, mean={np.mean(self.f_data):.3f}")
        self.logger.info(f"Fneu data stats: min={np.min(self.fneu_data):.3f}, max={np.max(self.fneu_data):.3f}, mean={np.mean(self.fneu_data):.3f}")
        self.logger.info(f"Contrasting traces stats: min={np.min(contrasting_traces):.3f}, max={np.max(contrasting_traces):.3f}, mean={np.mean(contrasting_traces):.3f}")
        
        self.preview_text.setPlainText("\n".join(preview_text))
        
    def import_data(self):
        """Import the contrasting fluorescence data and emit signal"""
        try:
            if not (hasattr(self, 'f_data') and hasattr(self, 'fneu_data')):
                return
                
            # Get import settings
            sampling_rate = self.sampling_rate_spin.value()
            max_neurons = self.max_neurons_spin.value()
            dt = 1.0 / sampling_rate
            
            n_neurons, n_timepoints = self.f_data.shape
            
            # Calculate contrasting traces: F - 0.7 * Fneu
            contrasting_traces = self.f_data - 0.7 * self.fneu_data
            
            # Debug logging with detailed trace calculations
            self.logger.info(f"Calculating contrasting traces: F - 0.7 * Fneu")
            self.logger.info(f"Original F shape: {self.f_data.shape}, range: {np.min(self.f_data):.6f} to {np.max(self.f_data):.6f}")
            self.logger.info(f"Original Fneu shape: {self.fneu_data.shape}, range: {np.min(self.fneu_data):.6f} to {np.max(self.fneu_data):.6f}")
            self.logger.info(f"Calculated contrasting traces range: {np.min(contrasting_traces):.6f} to {np.max(contrasting_traces):.6f}")
            
            # Sample some values for detailed logging
            for neuron_idx in range(min(3, n_neurons)):
                for time_idx in range(min(3, n_timepoints)):
                    f_val = self.f_data[neuron_idx, time_idx]
                    fneu_val = self.fneu_data[neuron_idx, time_idx]
                    contrast_val = contrasting_traces[neuron_idx, time_idx]
                    self.logger.debug(f"Neuron {neuron_idx}, time {time_idx}: F={f_val:.6f}, Fneu={fneu_val:.6f}, F-0.7*Fneu={contrast_val:.6f}")
            
            # Limit number of neurons
            n_neurons_to_import = min(max_neurons, n_neurons)
            final_traces = contrasting_traces[:n_neurons_to_import, :]
            
            # Create timestamps
            timestamps = np.arange(n_timepoints) * dt
            
            # Prepare data for Neuroseq
            timestamp = int(time.time())
            imported_data = {
                f"suite2p_contrasting_traces_{timestamp}": {
                    "widget": "trace",
                    "timestamps": timestamps,
                    "traces": final_traces
                },
                "_metadata": {
                    "n_timepoints": n_timepoints,
                    "dt": dt,
                    "sampling_rate": sampling_rate,
                    "n_neurons_imported": n_neurons_to_import,
                    "n_neurons_total": n_neurons,
                    "data_type": "suite2p_contrasting_traces",
                    "calculation": "F - 0.7 * Fneu"
                }
            }
            
            self.logger.info(f"Imported Suite2p contrasting traces: {n_neurons_to_import} neurons, {n_timepoints} timepoints")
            self.logger.info(f"Final imported traces range: {np.min(final_traces):.6f} to {np.max(final_traces):.6f}")
            
            # Emit signal with the imported data
            self.data_imported.emit(imported_data)
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Successfully imported Suite2p contrasting fluorescence traces (F - 0.7 * Fneu).\n"
                f"{n_neurons_to_import} neurons with {n_timepoints} timepoints at {sampling_rate} Hz."
            )
            
            self.accept()
            
        except Exception as e:
            self.logger.exception("Error importing contrasting fluorescence data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing contrasting fluorescence data:\n{str(e)}"
            )