from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename


class FluorescentSuite2pImportDialog(QtWidgets.QDialog):
    """Dialog for importing Suite2p fluorescence spike data from .npy files"""
    
    # Signal emitted when data is successfully imported
    data_imported = QtCore.Signal(dict)
    
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Import Suite2p Spike Data")
        self.setModal(True)
        self.resize(400, 200)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Select the spks.npy file from Suite2p output.\n"
            "This contains inferred spike data from calcium imaging."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # File selection
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        
        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        file_layout.addWidget(self.file_path_label)
        
        browse_button = QtWidgets.QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)
        
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
        self.sampling_rate_spin.setMinimum(0.1)
        self.sampling_rate_spin.setMaximum(100000.0)
        self.sampling_rate_spin.setValue(30000.0)  # This aligns with the 900 data points exactly on a time scale.
        # but I think that's really because we "accidentally" multiply patchclamp data by 30hz when it's already 1:1 hz
        # Ask Andrey: so in other words this should sample at 1000, patchclamp should sample at 1 hz (?)
        self.sampling_rate_spin.setSuffix(" Hz")
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)
        
        self.spike_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.spike_threshold_spin.setMinimum(0.0)
        self.spike_threshold_spin.setMaximum(100.0)
        self.spike_threshold_spin.setValue(0.1)
        self.spike_threshold_spin.setDecimals(3)
        options_layout.addRow("Spike threshold:", self.spike_threshold_spin)
        
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
        
    def browse_file(self):
        """Open file browser to select .npy file"""
        filename, _ = getopenfilename(
            self,
            "Select Suite2p Spike Data File",
            "",
            "NumPy files (*.npy);;All files (*)"
        )
        
        if filename:
            self.load_file(filename)
            
    def load_file(self, filename: str):
        """Load and preview the .npy file"""
        try:
            self.file_path = Path(filename)
            self.file_path_label.setText(str(self.file_path.name))
            self.file_path_label.setStyleSheet("QLabel { color: black; }")
            
            # Load the .npy file
            self.spike_data = np.load(filename)
            
            # Validate data format
            if self.spike_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.spike_data.ndim}D array")
            
            # Update preview
            self.update_preview()
            self.import_button.setEnabled(True)
            
        except Exception as e:
            self.logger.exception(f"Error loading file {filename}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{str(e)}"
            )
            self.import_button.setEnabled(False)
            
    def update_preview(self):
        """Update the data preview"""
        if not hasattr(self, 'spike_data'):
            return
            
        data = self.spike_data
        n_neurons, n_timepoints = data.shape
        non_zero_count = np.sum(data > 0)
        
        preview_text = [
            f"Shape: {data.shape} ({n_neurons} neurons × {n_timepoints} timepoints)",
            f"Type: {data.dtype}",
            f"Range: {np.min(data):.3f} to {np.max(data):.3f}",
            f"Non-zero values: {non_zero_count} / {data.size} ({non_zero_count/data.size*100:.1f}%)",
            f"Mean spike rate: {np.mean(data):.3f}"
        ]
        
        self.preview_text.setPlainText("\n".join(preview_text))
        
    def convert_spks_to_spike_events(self, spks_array: np.ndarray, dt: float, threshold: float) -> dict:
        """
        Convert Suite2p spike probability array to spike events.
        
        Args:
            spks_array: 2D array (neurons × timepoints) with spike probabilities
            dt: time step in seconds
            threshold: minimum value to consider as spike
            
        Returns:
            Dict compatible with spike widget format
        """
        spike_times = []
        spike_indices = []
        
        n_neurons, n_timesteps = spks_array.shape
        
        # Find all timepoints where spike probability exceeds threshold
        for neuron_idx in range(n_neurons):
            for time_idx in range(n_timesteps):
                spike_prob = spks_array[neuron_idx, time_idx]
                if spike_prob > threshold:
                    spike_times.append(time_idx * dt)
                    spike_indices.append(neuron_idx)
        
        return {
            "timestamps": np.array(spike_times),
            "spikes": np.array(spike_indices),
            "dt": dt,
            "n_neurons": n_neurons,
            "n_timesteps": n_timesteps
        }
        
    def import_data(self):
        """Import the spike data and emit signal"""
        try:
            if not hasattr(self, 'spike_data'):
                return
                
            # Get import settings
            sampling_rate = self.sampling_rate_spin.value()
            spike_threshold = self.spike_threshold_spin.value()
            dt = 1.0 / sampling_rate
            
            n_neurons, n_timepoints = self.spike_data.shape
            
            # Convert spks array to spike events
            spike_events = self.convert_spks_to_spike_events(self.spike_data, dt, spike_threshold)
            
            # Prepare data for Neuroseq
            imported_data = {
                "suite2p_spikes": {
                    "widget": "spike",
                    **spike_events
                },
                "_metadata": {
                    "n_timepoints": n_timepoints,
                    "dt": dt,
                    "sampling_rate": sampling_rate,
                    "spike_threshold": spike_threshold,
                    "n_neurons": n_neurons,
                    "data_type": "suite2p_spikes"
                }
            }
            
            spike_count = len(spike_events["spikes"])
            self.logger.info(f"Imported Suite2p spike data: {n_neurons} neurons, {spike_count} spike events")
            
            # Emit signal with the imported data
            self.data_imported.emit(imported_data)
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Successfully imported Suite2p spike data.\n"
                f"{n_neurons} neurons, {spike_count} spike events detected\n"
                f"from {n_timepoints} timepoints at {sampling_rate} Hz."
            )
            
            self.accept()
            
        except Exception as e:
            self.logger.exception("Error importing spike data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing spike data:\n{str(e)}"
            )