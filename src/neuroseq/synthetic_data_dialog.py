from __future__ import annotations

import logging

import brian2 as b2
import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.model import Model


class SyntheticDataDialog(QtWidgets.QDialog):
    """Dialog to generate synthetic trace and spike data and apply to widgets.

    The dialog can be constructed with references to an existing
    `TraceWidget` and `SpikeWidget`. Generated data will be applied to those
    widgets when the user clicks "Generate & Apply".
    """

    def __init__(
        self,
        model: Model,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.data_added = False  # Track if data was added to the project

        self.setWindowTitle("Generate Synthetic Data")
        self.setModal(True)
        self.resize(520, 360)

        layout = QtWidgets.QVBoxLayout(self)

        # Patch clamp

        data_patch_clamp_group = QtWidgets.QGroupBox("Patch clamp")
        data_patch_clamp_group_layout = QtWidgets.QVBoxLayout()

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select traces:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_patch_clamp_combo = QtWidgets.QComboBox()
        self.data_patch_clamp_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(TraceData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_patch_clamp_combo.addItem("")
        for name in data_names:
            self.data_patch_clamp_combo.addItem(name)

        if data_names:
            self.data_patch_clamp_combo.setCurrentIndex(1)

        self.data_patch_clamp_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_patch_clamp_combo)
        data_patch_clamp_group_layout.addLayout(data_inner_layout)

        # Noise value selector
        noise_layout = QtWidgets.QHBoxLayout()
        noise_label = QtWidgets.QLabel("Noise (mV):")
        noise_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.noise_spinbox = QtWidgets.QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 10.0)
        self.noise_spinbox.setValue(0.5)
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setDecimals(2)
        self.noise_spinbox.setToolTip(
            "Noise level in millivolts to add to the trace data"
        )
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_spinbox)
        data_patch_clamp_group_layout.addLayout(noise_layout)

        # Index selector
        index_layout = QtWidgets.QHBoxLayout()
        index_label = QtWidgets.QLabel("Indexes to import:")
        index_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.index_lineedit = QtWidgets.QLineEdit()
        self.index_lineedit.setPlaceholderText("e.g., 0,1,5-15,20")
        self.index_lineedit.setToolTip(
            "Comma-separated neuron indexes and/or ranges (e.g., 0,1,5-15,20). Leave empty for all."
        )
        index_layout.addWidget(index_label)
        index_layout.addWidget(self.index_lineedit)
        data_patch_clamp_group_layout.addLayout(index_layout)

        # Buttons for patch clamp
        data_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_export_trace = QtWidgets.QPushButton("Export as a file")
        self.btn_export_trace.clicked.connect(self._export_trace_data)
        self.btn_add_trace = QtWidgets.QPushButton("Add to the project")
        self.btn_add_trace.clicked.connect(self._add_trace_to_project)
        data_buttons_layout.addWidget(self.btn_export_trace)
        data_buttons_layout.addWidget(self.btn_add_trace)
        data_patch_clamp_group_layout.addLayout(data_buttons_layout)

        data_patch_clamp_group.setLayout(data_patch_clamp_group_layout)

        layout.addWidget(data_patch_clamp_group)

        ### Imaging

        data_imaging_group = QtWidgets.QGroupBox("Imaging")
        data_imaging_group_layout = QtWidgets.QVBoxLayout()

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select traces:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_imaging_combo = QtWidgets.QComboBox()
        self.data_imaging_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(TraceData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_imaging_combo.addItem("")
        for name in data_names:
            self.data_imaging_combo.addItem(name)

        if data_names:
            self.data_imaging_combo.setCurrentIndex(1)

        self.data_imaging_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_imaging_combo)
        data_imaging_group_layout.addLayout(data_inner_layout)

        # Noise value selector for imaging
        imaging_noise_layout = QtWidgets.QHBoxLayout()
        imaging_noise_label = QtWidgets.QLabel("Noise (mV):")
        imaging_noise_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.imaging_noise_spinbox = QtWidgets.QDoubleSpinBox()
        self.imaging_noise_spinbox.setRange(0.0, 10.0)
        self.imaging_noise_spinbox.setValue(0.5)
        self.imaging_noise_spinbox.setSingleStep(0.1)
        self.imaging_noise_spinbox.setDecimals(2)
        self.imaging_noise_spinbox.setToolTip(
            "Noise level in millivolts to add to the trace data"
        )
        imaging_noise_layout.addWidget(imaging_noise_label)
        imaging_noise_layout.addWidget(self.imaging_noise_spinbox)
        data_imaging_group_layout.addLayout(imaging_noise_layout)

        # Index selector for imaging
        imaging_index_layout = QtWidgets.QHBoxLayout()
        imaging_index_label = QtWidgets.QLabel("Indexes to import:")
        imaging_index_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.imaging_index_lineedit = QtWidgets.QLineEdit()
        self.imaging_index_lineedit.setPlaceholderText("e.g., 0,1,5-15,20")
        self.imaging_index_lineedit.setToolTip(
            "Comma-separated neuron indexes and/or ranges (e.g., 0,1,5-15,20). Leave empty for all."
        )
        imaging_index_layout.addWidget(imaging_index_label)
        imaging_index_layout.addWidget(self.imaging_index_lineedit)
        data_imaging_group_layout.addLayout(imaging_index_layout)

        # Buttons for imaging
        imaging_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_export_imaging = QtWidgets.QPushButton("Export as a file")
        self.btn_export_imaging.clicked.connect(self._export_imaging_data)
        self.btn_add_imaging = QtWidgets.QPushButton("Add to the project")
        self.btn_add_imaging.clicked.connect(self._add_imaging_to_project)
        imaging_buttons_layout.addWidget(self.btn_export_imaging)
        imaging_buttons_layout.addWidget(self.btn_add_imaging)
        data_imaging_group_layout.addLayout(imaging_buttons_layout)

        data_imaging_group.setLayout(data_imaging_group_layout)
        layout.addWidget(data_imaging_group)

        # Data Preview
        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        # Initial preview update
        self._update_preview()

    def _parse_indices(self, index_string: str) -> np.ndarray | None:
        """Parse comma-separated indices and ranges into a numpy array.
        Args:
            index_string: String containing indices and/or ranges (e.g., "0,1,5-15,20")
        Returns:
            Numpy array of parsed indices, or None if empty string
        """
        index_string = index_string.strip()
        if not index_string:
            return None

        indices: list[int] = []
        parts = index_string.split(",")

        for part in parts:
            part = part.strip()
            if "-" in part:
                # Handle range (e.g., "5-15")
                try:
                    start, end = part.split("-")
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                    if start_idx > end_idx:
                        raise ValueError(
                            f"Invalid range '{part}': start of the range must be <= than the end of the range"
                        )
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid range format: '{part}'") from None
                    raise
            else:
                # Handle single index
                try:
                    indices.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid index: '{part}'") from None

        # Remove duplicates and sort
        indices = sorted(set(indices))
        return np.array(indices, dtype=int)

    def _update_preview(self) -> None:
        """Update the data preview based on current selections."""
        preview_lines = []

        # Patch clamp preview
        trace_name = self.data_patch_clamp_combo.currentText()
        if trace_name:
            trace_data = self.model.get_data(trace_name, TraceData)
            if trace_data:
                shape = trace_data.traces.shape
                dt = trace_data.dt
                # units = trace_data.units
                indices = trace_data.indices
                preview_lines.append(f"Traces: {trace_name}")
                preview_lines.append(f"  Number of neurons: {shape[0]}")
                preview_lines.append(f"  Duration in seconds: {shape[1] * dt:.2f}")
                # preview_lines.append(f"  Units: {units}")
                preview_lines.append("  Units: mV")  # hardcoded
                preview_lines.append(
                    f"  Neuron indices (range): from {indices[0]} to {indices[-1]}"
                )
                preview_lines.append(f"  Neuron indices (full list): {indices}")

        # Imaging preview
        imaging_name = self.data_imaging_combo.currentText()
        if imaging_name:
            imaging_data = self.model.get_data(imaging_name, TraceData)
            if imaging_data:
                if preview_lines:
                    preview_lines.append("")

                shape = imaging_data.traces.shape
                dt = imaging_data.dt
                indices = imaging_data.indices
                preview_lines.append(f"Imaging Traces: {imaging_name}")
                preview_lines.append(f"  Number of neurons: {shape[0]}")
                preview_lines.append(f"  Duration in seconds: {shape[1] * dt:.2f}")
                preview_lines.append("  Units: mV")  # hardcoded
                preview_lines.append(
                    f"  Neuron indices (range): from {indices[0]} to {indices[-1]}"
                )
                preview_lines.append(f"  Neuron indices (full list): {indices}")

        self.preview_text.setPlainText(
            "\n".join(preview_lines) if preview_lines else "No data selected"
        )

    def _generate_synthetic_patch_clamp_traces(
        self,
        trace_data: TraceData,
        indices: np.ndarray | None,
        noise_amplitude: float,
    ) -> np.ndarray | None:
        """Filter trace data by indices and add Gaussian noise.

        Args:
            trace_data: The trace data to filter.
            indices: Array of indices to extract, or None to export all traces.
            noise_amplitude: Standard deviation of Gaussian noise to add (in mV).

        Returns:
            The filtered and noised traces, or None if validation fails.
        """
        # Check if trace data has non-consecutive indices
        if len(trace_data.indices) > 1:
            indices_diff = np.diff(trace_data.indices)
            if not np.all(indices_diff == 1):
                reply = QtWidgets.QMessageBox.warning(
                    self,
                    "Non-consecutive Indices",
                    f"The trace data has non-consecutive neuron indices: {trace_data.indices}\n\n"
                    "This may cause issues when interpreting the exported data as the current implementation expects consecutive indices. "
                    "Do you want to continue with the export?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.No:
                    return None

        # Filter trace data to only include selected indices
        if indices is not None:
            # Validate that all requested indices are within bounds
            max_idx = trace_data.traces.shape[0] - 1
            invalid_indices = indices[indices > max_idx]
            if len(invalid_indices) > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Indices",
                    f"Some indices are out of bounds. Maximum index is {max_idx}.\n"
                    f"Invalid indices: {invalid_indices}",
                )
                return None
            traces_to_export = trace_data.traces[indices]
        else:
            # Export all traces if no indices specified
            traces_to_export = trace_data.traces

        # Add gaussian noise with amplitude from the noise spinbox
        noise_amplitude_with_units = noise_amplitude * b2.mV
        traces_to_export = traces_to_export + np.random.normal(
            0, noise_amplitude_with_units, traces_to_export.shape
        )

        return traces_to_export

    def _export_trace_data(self) -> None:
        """Export selected trace data to a file."""
        trace_name = self.data_patch_clamp_combo.currentText()
        if not trace_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select trace data to export."
            )
            return

        trace_data = self.model.get_data(trace_name, TraceData)
        if not trace_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {trace_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        traces_to_export = self._generate_synthetic_patch_clamp_traces(
            trace_data, indices, self.noise_spinbox.value()
        )
        if traces_to_export is None:
            return

        # Open file dialog

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Trace Data",
            f"{trace_name}_patch_clamp.npy",
            "NumPy Array (*.npy);;CSV File (*.csv)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npy"):
                np.save(file_path, traces_to_export)
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, traces_to_export, delimiter=",")
            else:
                np.savetxt(file_path, traces_to_export)

            self.logger.info(f"Exported trace data to {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Trace data exported to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to export trace data: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export trace data:\n{str(e)}"
            )

    def _add_trace_to_project(self) -> None:
        """Add generated synthetic trace data to the project as experimental data."""
        trace_name = self.data_patch_clamp_combo.currentText()
        if not trace_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select trace data to generate."
            )
            return

        trace_data = self.model.get_data(trace_name, TraceData)
        if not trace_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {trace_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        traces_to_add = self._generate_synthetic_patch_clamp_traces(
            trace_data, indices, self.noise_spinbox.value()
        )
        if traces_to_add is None:
            return

        # Determine the indices for the new data
        new_indices = indices if indices is not None else trace_data.indices

        # Create a unique experiment name
        base_experiment_name = "Synthetic data"
        experiment_name = base_experiment_name
        counter = 1
        while experiment_name in self.model.experiments:
            experiment_name = f"{base_experiment_name} {counter}"
            counter += 1

        # Create new TraceData object with the synthetic data
        new_trace_data = TraceData(
            name="Patch clamp",
            traces=traces_to_add,
            indices=new_indices,
            dt=trace_data.dt,
            units=trace_data.units,
        )

        # Add to model
        try:
            self.model.add_experimental_data(experiment_name, [new_trace_data])
            self.data_added = True  # Mark that data was added
            self.logger.info(
                f"Added synthetic trace data to experiment '{experiment_name}'"
            )
            QtWidgets.QMessageBox.information(
                self,
                "Data Added",
                f"Synthetic trace data added to experiment:\n{experiment_name}",
            )
        except Exception as e:
            self.logger.error(f"Failed to add trace data to project: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Add Failed",
                f"Failed to add trace data to project:\n{str(e)}",
            )

    def _export_imaging_data(self) -> None:
        """Export selected imaging trace data to a file."""
        imaging_name = self.data_imaging_combo.currentText()
        if not imaging_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select imaging trace data to export."
            )
            return

        imaging_data = self.model.get_data(imaging_name, TraceData)
        if not imaging_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {imaging_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.imaging_index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        traces_to_export = self._generate_synthetic_patch_clamp_traces(
            imaging_data, indices, self.imaging_noise_spinbox.value()
        )
        if traces_to_export is None:
            return

        # Open file dialog
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Imaging Trace Data",
            f"{imaging_name}_imaging.npy",
            "NumPy Array (*.npy);;CSV File (*.csv)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npy"):
                np.save(file_path, traces_to_export)
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, traces_to_export, delimiter=",")
            else:
                np.savetxt(file_path, traces_to_export)

            self.logger.info(f"Exported imaging trace data to {file_path}")
            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Imaging trace data exported to:\n{file_path}",
            )
        except Exception as e:
            self.logger.error(f"Failed to export imaging trace data: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export imaging trace data:\n{str(e)}"
            )

    def _add_imaging_to_project(self) -> None:
        """Add generated synthetic imaging trace data to the project as experimental data."""
        imaging_name = self.data_imaging_combo.currentText()
        if not imaging_name:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Selected",
                "Please select imaging trace data to generate.",
            )
            return

        imaging_data = self.model.get_data(imaging_name, TraceData)
        if not imaging_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {imaging_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.imaging_index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        traces_to_add = self._generate_synthetic_patch_clamp_traces(
            imaging_data, indices, self.imaging_noise_spinbox.value()
        )
        if traces_to_add is None:
            return

        # Determine the indices for the new data
        new_indices = indices if indices is not None else imaging_data.indices

        # Create a unique experiment name
        base_experiment_name = "Synthetic imaging data"
        experiment_name = base_experiment_name
        counter = 1
        while experiment_name in self.model.experiments:
            experiment_name = f"{base_experiment_name} {counter}"
            counter += 1

        # Create new TraceData object with the synthetic data
        new_trace_data = TraceData(
            name="Imaging",
            traces=traces_to_add,
            indices=new_indices,
            dt=imaging_data.dt,
            units=imaging_data.units,
        )

        # Add to model
        try:
            self.model.add_experimental_data(experiment_name, [new_trace_data])
            self.data_added = True  # Mark that data was added
            self.logger.info(
                f"Added synthetic imaging trace data to experiment '{experiment_name}'"
            )
            QtWidgets.QMessageBox.information(
                self,
                "Data Added",
                f"Synthetic imaging trace data added to experiment:\n{experiment_name}",
            )
        except Exception as e:
            self.logger.error(f"Failed to add imaging trace data to project: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Add Failed",
                f"Failed to add imaging trace data to project:\n{str(e)}",
            )
