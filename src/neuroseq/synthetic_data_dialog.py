from __future__ import annotations

import logging

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SpikeData
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
        data_buttons_layout.addWidget(self.btn_export_trace)
        data_buttons_layout.addWidget(self.btn_add_trace)
        data_patch_clamp_group_layout.addLayout(data_buttons_layout)

        data_patch_clamp_group.setLayout(data_patch_clamp_group_layout)

        layout.addWidget(data_patch_clamp_group)

        ### Imaging

        data_imaging_group = QtWidgets.QGroupBox("Imaging")

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select spikes:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_imaging_combo = QtWidgets.QComboBox()
        self.data_imaging_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(SpikeData)
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
        data_imaging_group.setLayout(data_inner_layout)
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
        spike_name = self.data_imaging_combo.currentText()
        if spike_name:
            spike_data = self.model.get_data(spike_name, SpikeData)
            if spike_data:
                if preview_lines:
                    preview_lines.append("")

                shape = spike_data.spike_times.shape
                preview_lines.append(f"Spikes: {spike_name}")
                preview_lines.append(f"  Number of spikes: {shape[0]}")

        self.preview_text.setPlainText(
            "\n".join(preview_lines) if preview_lines else "No data selected"
        )

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
                np.save(file_path, trace_data.traces)
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, trace_data.traces, delimiter=",")
            else:
                np.savetxt(file_path, trace_data.traces)

            self.logger.info(f"Exported trace data to {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Trace data exported to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to export trace data: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export trace data:\n{str(e)}"
            )
