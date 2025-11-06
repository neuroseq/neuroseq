from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
import scipy.io
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp data from .mat files"""
    
    # Signal emitted when data is successfully imported
    # Emits dict with traces data compatible with Neuroseq format
    data_imported = QtCore.Signal(dict)
    
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.imported_data: dict[str, Any] = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)
        self.resize(500, 300)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        instructions = QtWidgets.QLabel(
            "Select a .mat file containing patch clamp data.\n"
            "Expected format: MATLAB file with 'neur_act' structure containing 'soma', 'dend', 'bg' fields."
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
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # Import options
        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        
        self.import_type_combo = QtWidgets.QComboBox()
        self.import_type_combo.addItems(["traces", "spikes"])
        self.import_type_combo.setCurrentText("traces")
        options_layout.addRow("Import as:", self.import_type_combo)
        
        self.data_source_combo = QtWidgets.QComboBox()
        self.data_source_combo.addItems(["soma", "dend", "bg"])
        self.data_source_combo.setCurrentText("soma") # this lets user choose what to select when importing patch clamp for traces.
        options_layout.addRow("Data source:", self.data_source_combo)
        
        self.max_neurons_spin = QtWidgets.QSpinBox()
        self.max_neurons_spin.setMinimum(1)
        self.max_neurons_spin.setMaximum(1000)
        self.max_neurons_spin.setValue(20)
        options_layout.addRow("Max neurons to import:", self.max_neurons_spin)
        
        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setMinimum(0.1)
        self.sampling_rate_spin.setMaximum(10000.0)
        self.sampling_rate_spin.setValue(30.0)
        self.sampling_rate_spin.setSuffix(" Hz")
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)
        
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
        """Open file browser to select .mat file"""
        filename, _ = getopenfilename(
            self,
            "Select Patch Clamp Data File",
            "",
            "MATLAB files (*.mat);;All files (*)"
        )
        
        if filename:
            self.load_file(filename)
            
    def load_file(self, filename: str):
        """Load and preview the .mat file"""
        try:
            self.file_path = Path(filename)
            self.file_path_label.setText(str(self.file_path.name))
            self.file_path_label.setStyleSheet("QLabel { color: black; }")
            
            # Load the .mat file
            mat_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
            
            if 'neur_act' not in mat_data:
                raise ValueError("File does not contain 'neur_act' variable")
                
            neur_act = mat_data['neur_act']
            
            # Check for required fields
            required_fields = ['soma', 'dend', 'bg']
            missing_fields = []
            for field in required_fields:
                if not hasattr(neur_act, field):
                    missing_fields.append(field)
                    
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Store the data
            self.mat_data = {
                'soma': neur_act.soma,
                'dend': neur_act.dend, 
                'bg': neur_act.bg
            }
            
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
        if not hasattr(self, 'mat_data'):
            return
            
        preview_text = []
        
        for field_name, data in self.mat_data.items():
            preview_text.append(f"{field_name.upper()}:")
            preview_text.append(f"  Shape: {data.shape}")
            preview_text.append(f"  Type: {data.dtype}")
            preview_text.append(f"  Range: {np.min(data):.3f} to {np.max(data):.3f}")
            preview_text.append(f"  Mean: {np.mean(data):.3f} ± {np.std(data):.3f}")
            preview_text.append("")
            
        self.preview_text.setPlainText("\n".join(preview_text))
        
    def extract_spikes_from_voltage(self, voltage_traces: np.ndarray, dt: float) -> dict:
        """
        Extract spike events from voltage traces for spike widget.
        
        Args:
            voltage_traces: 2D array (neurons × timepoints)
            dt: time step in seconds
            
        Returns:
            Dict compatible with spike widget format
        """
        spike_times = []
        spike_indices = []
        
        n_neurons, n_timesteps = voltage_traces.shape
        
        # Simple threshold-based spike detection
        for neuron_idx in range(n_neurons):
            trace = voltage_traces[neuron_idx, :]
            
            # Dynamic threshold: mean + 3*std
            threshold = np.mean(trace) + 2 * np.std(trace)
            
            # Find crossings
            above_threshold = trace > threshold
            spike_timepoints = np.where(above_threshold)[0]
            
            # Convert to times and add to lists
            for timepoint in spike_timepoints:
                spike_times.append(timepoint * dt)
                spike_indices.append(neuron_idx)
        
        return {
            "timestamps": np.array(spike_times),
            "spikes": np.array(spike_indices),
            "dt": dt,
            "n_neurons": n_neurons,
            "n_timesteps": n_timesteps
        }
        
    def import_data(self):
        """Import the selected data and emit signal"""
        try:
            if not hasattr(self, 'mat_data'):
                return
                
            # Get import settings
            import_type = self.import_type_combo.currentText()  # "traces" or "spikes"
            data_source = self.data_source_combo.currentText() # allows user to select soma/dend/bg
            max_neurons = self.max_neurons_spin.value()
            sampling_rate = self.sampling_rate_spin.value()
            dt = 1.0 / sampling_rate

            voltage_traces = self.mat_data[data_source]
            n_neurons, n_timepoints = voltage_traces.shape
            
            # Limit number of neurons for testing
            n_neurons_to_import = min(max_neurons, n_neurons)
            voltage_traces = voltage_traces[:n_neurons_to_import, :]
            
            # Import data based on user choice
            imported_data = {}
            
            if import_type == "traces":
                timestamps = np.arange(n_timepoints) * dt
                self.logger.info(f"Min voltage scaled: {voltage_traces - np.min(voltage_traces)}")
                self.logger.info(f"Max voltage range: {np.max(voltage_traces) - np.min(voltage_traces)}")
                # Shim: Scale voltage traces to mV approximate range (asking Andrey about this...)
                scaled_traces = -0.07 + (voltage_traces - np.min(voltage_traces)) * (0.03 - (-0.07)) / (np.max(voltage_traces) - np.min(voltage_traces))
                imported_data[f"patchclamp_{data_source}_traces"] = {
                    "widget": "trace",
                    "timestamps": timestamps,
                    "traces": scaled_traces
                }
                
            elif import_type == "spikes":
                # Extract and import as spikes
                spike_data = self.extract_spikes_from_voltage(voltage_traces, dt)
                imported_data[f"patchclamp_{data_source}_spikes"] = {
                    "widget": "spike",
                    **spike_data
                }

            
            # Store metadata for stim expansion
            imported_data["_metadata"] = {
                "n_timepoints": n_timepoints,
                "dt": dt,
                "sampling_rate": sampling_rate,
                "data_source": data_source,
                "import_type": import_type,
                "n_neurons_imported": n_neurons_to_import,
                "n_neurons_total": n_neurons
            }
            
            self.logger.info(f"Imported {n_neurons_to_import} neurons from {data_source} as {import_type} with {n_timepoints} timepoints")
            
            # Emit signal with the imported data
            self.data_imported.emit(imported_data)
            
            # Show success message
            if import_type == "spikes":
                spike_count = len(imported_data[f"patchclamp_{data_source}_spikes"]["spikes"])
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Successfully imported {n_neurons_to_import} neurons from {data_source} as spike data.\n"
                    f"Detected {spike_count} spike events from {n_timepoints} timepoints at {sampling_rate} Hz."
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Successfully imported {n_neurons_to_import} neurons from {data_source} as trace data.\n"
                    f"Data contains {n_timepoints} timepoints at {sampling_rate} Hz."
                )
            
            self.accept()
            
        except Exception as e:
            self.logger.exception("Error importing data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing data:\n{str(e)}"
            )