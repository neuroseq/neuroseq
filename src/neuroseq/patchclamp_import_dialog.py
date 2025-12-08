from __future__ import annotations

import logging
import re
from pathlib import Path

import brian2.numpy_ as np
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spike_detection import detect_spikes_voltage_trace


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp data from .npy files."""

    def __init__(
        self,
        cellpose_import: CellposeImportResult | None = None,
        model=None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.cellpose_import = cellpose_import
        self.model = model
        self.imported_spike_data: list = []
        self.imported_trace_data: list = []
        self.experiment_name = "PatchClamp Import"
        self.append_data = False
        self.neurons_to_replace: set[int] = (
            set()
        )  # Track which neurons should be replaced
        self.file_path: Path | None = None
        self.npy_data: np.ndarray | None = None
        self.extracted_neuron_id: int | None = None
        self.overlay_widget: CellOverlayWidget | None = None
        self.preview_group: QtWidgets.QGroupBox | None = None
        self.options_group: QtWidgets.QGroupBox | None = None
        self.spike_group: QtWidgets.QGroupBox | None = None
        self._setup_ui()
        self._load_overlay_from_cellpose()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)

        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width()
        self.full_width = min(1200, int(0.9 * screen_width))
        self.compact_width = min(820, int(0.55 * screen_width))
        self.full_height = 540
        self.compact_height = 320

        initial_width = (
            self.full_width if self.cellpose_import is not None else self.compact_width
        )
        initial_height = (
            self.full_height
            if self.cellpose_import is not None
            else self.compact_height
        )
        self.resize(initial_width, initial_height)

        layout = QtWidgets.QVBoxLayout(self)

        if self.cellpose_import is not None:
            self.overlay_widget = CellOverlayWidget("Cell overlay preview")
            layout.addWidget(self.overlay_widget)

        file_group = QtWidgets.QGroupBox("Select .npy patchclamp file")
        file_layout = QtWidgets.QHBoxLayout(file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        layout.addWidget(file_group)

        self.preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(self.preview_group)

        self.options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(self.options_group)

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
        for option in ("Linear", "Custom"):
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
        self.custom_widget = QtWidgets.QWidget()
        self.custom_widget.setLayout(custom_layout)

        self.custom_mapping_label = QtWidgets.QLabel("Custom mapping:")
        self.custom_mapping_label.setVisible(False)
        self.custom_widget.setVisible(False)

        options_layout.addRow(self.custom_mapping_label, self.custom_widget)

        # Replace/extend data option
        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing data for this neuron ID instead of extending"
        )
        self.replace_checkbox.setChecked(not self.append_data)
        self.replace_checkbox.setToolTip(
            "When enabled, imported data will overwrite existing traces/spikes "
            "for the specified neuron ID. When disabled, data will be appended/extended."
        )
        self.replace_checkbox.toggled.connect(self._on_replace_toggled)
        options_layout.addRow(self.replace_checkbox)

        # Connect radio buttons to show/hide custom mapping input
        for button in self.mapping_buttons.buttons():
            button.toggled.connect(self._on_mapping_changed)

        self.spike_group = QtWidgets.QGroupBox("Peak Detection Parameters")
        spike_layout = QtWidgets.QFormLayout(self.spike_group)

        self.peak_height_spin = QtWidgets.QDoubleSpinBox()
        self.peak_height_spin.setDecimals(2)
        self.peak_height_spin.setMinimum(0.1)
        self.peak_height_spin.setMaximum(10.0)
        self.peak_height_spin.setSingleStep(0.1)
        self.peak_height_spin.setValue(3.0)
        self.peak_height_spin.setSuffix(" std")
        self.peak_height_spin.setToolTip(
            "Minimum height above baseline in standard deviations"
        )
        spike_layout.addRow("Peak height threshold:", self.peak_height_spin)

        self.peak_prominence_spin = QtWidgets.QDoubleSpinBox()
        self.peak_prominence_spin.setDecimals(2)
        self.peak_prominence_spin.setMinimum(0.1)
        self.peak_prominence_spin.setMaximum(5.0)
        self.peak_prominence_spin.setSingleStep(0.1)
        self.peak_prominence_spin.setValue(0.5)
        self.peak_prominence_spin.setSuffix(" std")
        self.peak_prominence_spin.setToolTip("Minimum prominence for peak detection")
        spike_layout.addRow("Peak prominence:", self.peak_prominence_spin)

        self.min_interval_spin = QtWidgets.QDoubleSpinBox()
        self.min_interval_spin.setDecimals(3)
        self.min_interval_spin.setMinimum(0.0001)
        self.min_interval_spin.setMaximum(0.1)
        self.min_interval_spin.setSingleStep(0.001)
        self.min_interval_spin.setValue(0.001)
        self.min_interval_spin.setSuffix(" s")
        self.min_interval_spin.setToolTip("Minimum interval between detected spikes")
        spike_layout.addRow("Min spike interval:", self.min_interval_spin)

        self.peak_distance_spin = QtWidgets.QSpinBox()
        self.peak_distance_spin.setMinimum(1)
        self.peak_distance_spin.setMaximum(1000)
        self.peak_distance_spin.setSingleStep(1)
        self.peak_distance_spin.setValue(10)
        self.peak_distance_spin.setSuffix(" samples")
        self.peak_distance_spin.setToolTip(
            "Minimum horizontal distance between peaks in samples"
        )
        spike_layout.addRow("Peak distance:", self.peak_distance_spin)

        self.peak_width_spin = QtWidgets.QSpinBox()
        self.peak_width_spin.setMinimum(1)
        self.peak_width_spin.setMaximum(100)
        self.peak_width_spin.setSingleStep(1)
        self.peak_width_spin.setValue(1)
        self.peak_width_spin.setSuffix(" samples")
        self.peak_width_spin.setToolTip("Minimum width of peaks in samples")
        spike_layout.addRow("Peak width:", self.peak_width_spin)

        layout.addWidget(self.options_group)
        layout.addWidget(self.spike_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self._set_file_sections_visible(False)

    def _on_mapping_changed(self) -> None:
        mapping_strategy = self._get_mapping_strategy()
        is_custom = mapping_strategy == "Custom"

        self.custom_mapping_label.setVisible(is_custom)
        self.custom_widget.setVisible(is_custom)

        if is_custom and self.cellpose_import is not None:
            self.custom_mapping_line.setVisible(True)
            self.custom_neuron_id_spin.setVisible(False)
        elif is_custom:
            self.custom_mapping_line.setVisible(False)
            self.custom_neuron_id_spin.setVisible(True)

    def _on_replace_toggled(self, checked: bool) -> None:
        self.append_data = not checked

    def _load_overlay_from_cellpose(self) -> None:
        if self.overlay_widget is None:
            return
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
            self._set_file_sections_visible(True)
        except Exception as exc:
            self.logger.exception("Error loading patch clamp file")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{exc}",
            )
            self.import_button.setEnabled(False)
            self._set_file_sections_visible(False)

    def _load_npy_file(self, filename: str) -> None:
        self.npy_data = np.load(filename)

        self.logger.debug(
            "Loaded patchclamp npy: path=%s shape=%s dtype=%s range=%.4f–%.4f",
            filename,
            getattr(self.npy_data, "shape", None),
            getattr(self.npy_data, "dtype", None),
            float(np.min(self.npy_data)),
            float(np.max(self.npy_data)),
        )

        # Extract neuron ID from filename if possible
        neuron_id_match = re.search(r"neuronID_(\d+)", Path(filename).name)
        if neuron_id_match:
            self.extracted_neuron_id = int(neuron_id_match.group(1))
            self.logger.debug(
                "Extracted neuron ID %d from filename %s",
                self.extracted_neuron_id,
                filename,
            )
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

    def _set_file_sections_visible(self, visible: bool) -> None:
        for group in (self.preview_group, self.options_group, self.spike_group):
            if group is not None:
                group.setVisible(visible)
        if visible:
            self.resize(
                max(self.width(), getattr(self, "full_width", self.width())),
                max(self.height(), getattr(self, "full_height", self.height())),
            )
        elif self.cellpose_import is None:
            self.resize(
                min(self.width(), getattr(self, "compact_width", self.width())),
                min(self.height(), getattr(self, "compact_height", self.height())),
            )

    def _parse_custom_mapping(self, mapping_str: str) -> dict[int, int]:
        """Parse custom mapping string like '1:12,2:3' into {1: 12, 2: 3}"""
        mapping: dict[int, int] = {}
        if not mapping_str.strip():
            return mapping

        try:
            pairs = mapping_str.split(",")
            for pair in pairs:
                pair = pair.strip()
                if ":" not in pair:
                    continue
                stim_key_str, cellpose_id_str = pair.split(":", 1)
                stim_key = int(stim_key_str.strip())
                cellpose_id = int(cellpose_id_str.strip())
                mapping[stim_key] = cellpose_id
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid custom mapping format: {exc}") from exc

        return mapping

    def _validate_append_import(self) -> None:
        """Validate that we can safely append data in extend mode."""
        if self.model is None:
            return

        # Get existing trace data for the same experiment name
        existing_experiments = getattr(self.model, "experiments", {})
        existing_data = existing_experiments.get(self.experiment_name, [])

        # Find existing trace data with our target name
        existing_trace_data = None
        for data in existing_data:
            if hasattr(data, "name") and data.name == "Patch Clamp Traces":
                from neuroseq.data import TraceData

                if isinstance(data, TraceData):
                    existing_trace_data = data
                    break

        if existing_trace_data is None:
            # No existing data, safe to proceed
            return

        # Check which neurons we're trying to import
        neuron_id = self.custom_neuron_id_spin.value()
        existing_neuron_indices = set(existing_trace_data.indices)

        if neuron_id in existing_neuron_indices:
            # This neuron already exists, which is fine for extension
            return

        # Check if there are multiple neurons being imported with mixed existing/new status
        # For now, we're importing one neuron at a time, but this logic prepares for multiple
        if len(existing_neuron_indices) > 0:
            # We have existing data for some neurons but not others
            # In extend mode, this could cause timestamp misalignment
            raise ValueError(
                f"Cannot extend data when importing neuron {neuron_id} - "
                f"existing neurons {sorted(existing_neuron_indices)} already have data. "
                f"This would cause timestamp misalignment. Use 'Replace' mode instead, "
                f"or import all neurons with the same timeline."
            )

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

            self.logger.debug(
                "Importing patchclamp data: sampling_rate=%.2fHz dt=%.6f mapping_strategy=%s mapping_keys=%s",
                sampling_rate,
                dt,
                mapping_strategy,
                sorted(stimulus_mapping.keys()),
            )

            # Update cellpose import with the mapping
            if self.cellpose_import is not None:
                self.cellpose_import.stimulus_to_cellpose_mapping = stimulus_mapping

            # Validate import if in append mode
            if self.append_data:
                self._validate_append_import()

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

        # Track neurons being imported for potential replacement
        if not self.append_data:
            self.neurons_to_replace.add(neuron_id)

        self.logger.debug(
            "Detecting spikes for patchclamp neuron %d: dt=%.6f height=%.2f prominence=%.2f min_interval=%.4f distance=%d width=%d",
            neuron_id,
            dt,
            self.peak_height_spin.value(),
            self.peak_prominence_spin.value(),
            self.min_interval_spin.value(),
            self.peak_distance_spin.value(),
            self.peak_width_spin.value(),
        )

        # Use peak detection for spike detection
        spike_data = detect_spikes_voltage_trace(
            self.npy_data,
            neuron_id=neuron_id,
            dt=dt,
            detection_method="peak",
            peak_height_threshold=self.peak_height_spin.value(),
            peak_prominence=self.peak_prominence_spin.value(),
            min_spike_interval=self.min_interval_spin.value(),
            peak_distance=self.peak_distance_spin.value(),
            peak_width=self.peak_width_spin.value(),
        )

        # Use consistent name for all patch clamp spike data to enable merging
        spike_data.name = "Patch Clamp Spikes"

        # Create trace data (single neuron, so reshape to 2D)
        trace_array = self.npy_data.reshape(1, -1)
        neuron_indices = np.array([neuron_id], dtype=np.uint32)
        trace_data = TraceData(
            name="Patch Clamp Traces",  # Use consistent name for all patch clamp data
            traces=trace_array,
            indices=neuron_indices,
            units="mV",
            dt=dt,
        )

        self.imported_spike_data.append(spike_data)
        self.imported_trace_data.append(trace_data)
        self.logger.debug(
            "Patchclamp import complete: neuron=%d spikes=%d trace_len=%d dt=%.6f",
            neuron_id,
            len(spike_data.spike_times),
            trace_array.shape[1],
            dt,
        )
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

    @property
    def imported_data(self) -> list:
        """Combined imported spike and trace data."""
        return self.imported_spike_data + self.imported_trace_data
