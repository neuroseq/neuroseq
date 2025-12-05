from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import brian2.numpy_ as np
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.experiment import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spike_detection import detect_spikes_voltage_trace


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp data from .npy files."""

    def __init__(
        self,
        cellpose_import: CellposeImportResult | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.cellpose_import = cellpose_import
        self.imported_spike_data: list = []
        self.imported_trace_data: list = []
        self.experiment_name = "PatchClamp Import"
        self.append_data = False
        self.file_path: Path | None = None
        self.npy_data: np.ndarray | None = None
        self.extracted_neuron_id: int | None = None
        self.overlay_widget = CellOverlayWidget("Cell overlay preview")
        self._setup_ui()
        self._load_overlay_from_cellpose()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)

        # Set width to 1.2k pixels or 90% of screen width, whichever is smaller
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width()
        target_width = min(1200, int(0.9 * screen_width))
        self.resize(target_width, 540)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Select a .npy file containing patch clamp voltage recording data."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addWidget(self.overlay_widget)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        layout.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)

        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setDecimals(2)
        self.sampling_rate_spin.setMinimum(0.1)
        self.sampling_rate_spin.setMaximum(10000.0)
        self.sampling_rate_spin.setSingleStep(1.0)
        self.sampling_rate_spin.setValue(30.0)
        self.sampling_rate_spin.setSuffix(" Hz")
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)

        mapping_layout = QtWidgets.QHBoxLayout()
        self.mapping_buttons = QtWidgets.QButtonGroup(self)
        for option in ("Linear", "Random", "Custom"):
            button = QtWidgets.QRadioButton(option)
            self.mapping_buttons.addButton(button)
            mapping_layout.addWidget(button)
        first_button = self.mapping_buttons.buttons()[0]
        first_button.setChecked(True)
        options_layout.addRow("Mapping strategy:", mapping_layout)

        self.custom_neuron_id_spin = QtWidgets.QSpinBox()
        self.custom_neuron_id_spin.setMinimum(0)
        self.custom_neuron_id_spin.setMaximum(999)
        self.custom_neuron_id_spin.setValue(10)

        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setPlaceholderText(
            "e.g., 1:12,2:3 (stimkey:cellposeId)"
        )
        self.custom_mapping_line.setVisible(False)

        custom_layout = QtWidgets.QVBoxLayout()
        custom_layout.addWidget(self.custom_neuron_id_spin)
        custom_layout.addWidget(self.custom_mapping_line)
        custom_widget = QtWidgets.QWidget()
        custom_widget.setLayout(custom_layout)

        options_layout.addRow("Custom mapping:", custom_widget)

        # Connect radio buttons to show/hide custom mapping input
        for button in self.mapping_buttons.buttons():
            button.toggled.connect(self._on_mapping_changed)

        spike_group = QtWidgets.QGroupBox("Spike Detection Parameters")
        spike_layout = QtWidgets.QFormLayout(spike_group)

        threshold_label = QtWidgets.QLabel("30.0 mV (fixed)")
        threshold_label.setToolTip(
            "Spikes detected above this absolute voltage threshold"
        )
        spike_layout.addRow("Voltage threshold:", threshold_label)

        self.min_interval_spin = QtWidgets.QDoubleSpinBox()
        self.min_interval_spin.setDecimals(3)
        self.min_interval_spin.setMinimum(0.0001)
        self.min_interval_spin.setMaximum(0.1)
        self.min_interval_spin.setSingleStep(0.001)
        self.min_interval_spin.setValue(0.001)
        self.min_interval_spin.setSuffix(" s")
        self.min_interval_spin.setToolTip("Minimum interval between detected spikes")
        spike_layout.addRow("Min spike interval:", self.min_interval_spin)

        layout.addWidget(options_group)
        layout.addWidget(spike_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def _on_mapping_changed(self) -> None:
        mapping_strategy = self._get_mapping_strategy()
        if mapping_strategy == "Custom" and self.cellpose_import is not None:
            self.custom_mapping_line.setVisible(True)
            self.custom_neuron_id_spin.setVisible(False)
        else:
            self.custom_mapping_line.setVisible(False)
            self.custom_neuron_id_spin.setVisible(True)

    def _load_overlay_from_cellpose(self) -> None:
        if self.cellpose_import is None:
            self.overlay_widget.set_empty_message(
                "Import Cellpose segmentation to see the overlay here."
            )
            return

        try:
            self.overlay_widget.set_overlay(
                self.cellpose_import.overlay_image,
                self.cellpose_import.segmentation_array,
                original_cellpose_ids=self.cellpose_import.original_cellpose_ids,
                stimulus_to_cellpose_mapping=self.cellpose_import.stimulus_to_cellpose_mapping,
            )
        except Exception as exc:
            self.logger.warning("Could not display Cellpose overlay: %s", exc)
            self.overlay_widget.set_empty_message(
                "Cellpose overlay could not be displayed."
            )

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Patch Clamp Data File",
            "",
            "NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, filename: str) -> None:
        try:
            self.file_path = Path(filename)
            self.file_path_label.setText(self.file_path.name)
            self.file_path_label.setStyleSheet("QLabel { color: black; }")

            # Clear previous data
            self.npy_data = None
            self.extracted_neuron_id = None

            if self.file_path.suffix.lower() == ".npy":
                self._load_npy_file(filename)
            else:
                raise ValueError("Unsupported file format. Expected .npy file.")

            self._update_preview()
            self.import_button.setEnabled(True)
        except Exception as exc:
            self.logger.exception("Error loading patch clamp file")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{exc}",
            )
            self.import_button.setEnabled(False)

    def _load_npy_file(self, filename: str) -> None:
        self.npy_data = np.load(filename)

        # Extract neuron ID from filename if possible
        neuron_id_match = re.search(r"neuronID_(\d+)", Path(filename).name)
        if neuron_id_match:
            self.extracted_neuron_id = int(neuron_id_match.group(1))
            # Auto-select Custom mapping and set the neuron ID
            for button in self.mapping_buttons.buttons():
                if button.text() == "Custom":
                    button.setChecked(True)
                    break
            self.custom_neuron_id_spin.setValue(self.extracted_neuron_id)

        if self.npy_data.ndim != 1:
            raise ValueError(
                "NumPy patchclamp data must be 1D (single neuron recording)"
            )

        # Set default sampling rate for patch clamp recordings (10kHz as mentioned)
        self.sampling_rate_spin.setValue(10000.0)

    def _update_preview(self) -> None:
        lines = []

        if self.npy_data is not None:
            lines.append(f"NPY DATA: {self.npy_data.shape[0]} timepoints")
            lines.append(
                f"  mean {np.mean(self.npy_data):.3f} ± {np.std(self.npy_data):.3f}"
            )
            lines.append(
                f"  range {np.min(self.npy_data):.3f} to {np.max(self.npy_data):.3f}"
            )
            if self.extracted_neuron_id is not None:
                lines.append(f"  extracted neuron ID: {self.extracted_neuron_id}")

            # Calculate duration at 10kHz
            duration_sec = self.npy_data.shape[0] / 10000.0
            lines.append(f"  duration: {duration_sec:.1f} seconds at 10kHz")
            lines.append("")
        else:
            self.preview_text.clear()
            return

        self.preview_text.setPlainText("\n".join(lines).strip())

    def _get_mapping_strategy(self) -> str:
        checked = self.mapping_buttons.checkedButton()
        return checked.text() if checked is not None else "Linear"

    def _parse_custom_mapping(self, mapping_str: str) -> dict[int, int]:
        """Parse custom mapping string like '1:12,2:3' into {1: 12, 2: 3}"""
        mapping = {}
        if not mapping_str.strip():
            return mapping

        try:
            pairs = mapping_str.split(",")
            for pair in pairs:
                pair = pair.strip()
                if ":" not in pair:
                    continue
                stim_key, cellpose_id = pair.split(":", 1)
                stim_key = int(stim_key.strip())
                cellpose_id = int(cellpose_id.strip())
                mapping[stim_key] = cellpose_id
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid custom mapping format: {exc}") from exc

        return mapping

    def _generate_stimulus_mapping(self, strategy: str) -> dict[int, int]:
        """Generate stimulus-to-cellpose mapping based on strategy"""
        if (
            self.cellpose_import is None
            or self.cellpose_import.original_cellpose_ids is None
        ):
            return {}

        original_ids = self.cellpose_import.original_cellpose_ids

        if strategy == "Linear":
            # Map stimulus IDs 1,2,3... to cellpose IDs in order
            return {
                i + 1: int(cellpose_id) for i, cellpose_id in enumerate(original_ids)
            }
        elif strategy == "Random":
            # Map stimulus IDs 1,2,3... to cellpose IDs randomly
            shuffled_ids = list(original_ids)
            random.shuffle(shuffled_ids)
            return {
                i + 1: int(cellpose_id) for i, cellpose_id in enumerate(shuffled_ids)
            }
        elif strategy == "Custom":
            # Parse custom mapping string
            return self._parse_custom_mapping(self.custom_mapping_line.text())
        else:
            return {}

    def import_data(self) -> None:
        if self.npy_data is None:
            return

        try:
            sampling_rate = self.sampling_rate_spin.value()
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be positive")
            dt = 1.0 / sampling_rate

            # Generate and store the stimulus-to-cellpose mapping
            mapping_strategy = self._get_mapping_strategy()
            stimulus_mapping = self._generate_stimulus_mapping(mapping_strategy)

            # Update cellpose import with the mapping
            if self.cellpose_import is not None:
                self.cellpose_import.stimulus_to_cellpose_mapping = stimulus_mapping

            self._import_npy_data(sampling_rate, dt)
            self.accept()
        except Exception as exc:
            self.logger.exception("Error importing patch clamp data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing data:\n{exc}",
            )

    # todo: needs to be able to accomodate multiple neurons
    def _import_npy_data(self, sampling_rate: float, dt: float) -> None:
        if self.npy_data is None:
            raise ValueError("No NPY data loaded")

        neuron_id = self.custom_neuron_id_spin.value()

        # Use absolute voltage threshold for spike detection
        spike_data = detect_spikes_voltage_trace(
            self.npy_data,
            neuron_id=neuron_id,
            dt=dt,
            detection_method="threshold",
            threshold_method="absolute",
            threshold_multiplier=30.0,  # 30mV absolute threshold
            min_spike_interval=self.min_interval_spin.value(),
        )

        # Create trace data (single neuron, so reshape to 2D)
        trace_array = self.npy_data.reshape(1, -1)
        trace_data = TraceData(
            name=f"Voltage trace (Neuron {neuron_id})",
            traces=trace_array,
            units="mV",
            dt=dt,
        )

        self.imported_spike_data.append(spike_data)
        self.imported_trace_data.append(trace_data)
        # todo: Liam/James to discuss/figure out saving the trace data in a sparse way (but still with gaps between, most likely by mapping e.g. 27:0, 5:1 would mean: show in stim arr pos 27 the value of traces data arr itself 0, and 5 value of traces data arr 1, and there would be in that case a 2x(timestamps) length array.

        if len(spike_data.spike_times) > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {len(spike_data.spike_times)} spikes and trace data from neuron {neuron_id}\n"
                f"Duration: {len(self.npy_data) / sampling_rate:.1f} seconds at {sampling_rate:.0f} Hz.",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "No Spikes Detected",
                f"No spikes were detected in the recording for neuron {neuron_id}.\n"
                f"You may need to adjust the detection parameters.",
            )
