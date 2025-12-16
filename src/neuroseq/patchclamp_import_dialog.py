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
from neuroseq.importer_cellpose import clone_cellpose_result
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
        # Work on a copy so we don't mutate the caller's stored data
        self.cellpose_import = clone_cellpose_result(cellpose_import)
        self.model = model
        self.imported_spike_data: list = []
        self.imported_trace_data: list = []
        self.experiment_name = "PatchClamp Import"
        self.append_data = False
        self._auto_stimulus_ids: list[int] = []
        self._auto_from_cellpose = False
        self.neurons_to_replace: set[int] = (
            set()
        )  # Track which neurons should be replaced
        self.file_path: Path | None = None
        self.npy_data: np.ndarray | None = None
        self.extracted_neuron_id: int | None = None
        self.data_row_count: int = 0
        self.overlay_widget: CellOverlayWidget | None = None
        self.preview_group: QtWidgets.QGroupBox | None = None
        self.options_group: QtWidgets.QGroupBox | None = None
        self.spike_group: QtWidgets.QGroupBox | None = None
        self._setup_ui()
        self._apply_existing_stim_mapping()
        self._populate_cellpose_preview()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)

        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width()
        screen_height = screen.availableGeometry().height()
        self.dialog_width = min(1400, int(0.9 * screen_width))
        self.dialog_height = min(900, int(0.8 * screen_height))
        self.resize(self.dialog_width, self.dialog_height)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(12)

        max_half_width = int(screen_width * 0.5)

        self.overlay_widget = CellOverlayWidget("Cellpose Preview")
        self.overlay_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_widget.setMaximumWidth(max_half_width)
        layout.addWidget(self.overlay_widget, stretch=1)

        right_container = QtWidgets.QWidget()
        right_container.setMaximumWidth(max_half_width)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        file_group = QtWidgets.QGroupBox("Select .npy patchclamp file")
        file_layout = QtWidgets.QHBoxLayout(file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        right_layout.addWidget(file_group)

        self.preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        right_layout.addWidget(self.preview_group)

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

        self.custom_neuron_id_line = QtWidgets.QLineEdit()
        self.custom_neuron_id_line.setPlaceholderText("e.g., 10 or 12,15")
        self.custom_neuron_id_line.setText("10")
        self.custom_neuron_id_line.setToolTip(
            "Single neuron ID (e.g., 10) or comma-separated list (e.g., 12,15)"
        )

        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setPlaceholderText(
            "Comma-separated stimulus IDs (e.g., 12,14)"
        )
        self.custom_mapping_line.setVisible(False)

        custom_layout = QtWidgets.QVBoxLayout()
        custom_layout.addWidget(self.custom_neuron_id_line)
        custom_layout.addWidget(self.custom_mapping_line)
        self.custom_widget = QtWidgets.QWidget()
        self.custom_widget.setLayout(custom_layout)

        self.custom_mapping_label = QtWidgets.QLabel("Target stimulus IDs:")
        self.custom_mapping_label.setVisible(False)
        self.custom_widget.setVisible(False)

        options_layout.addRow(self.custom_mapping_label, self.custom_widget)

        # Replace/extend data option
        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing data for these neuron IDs instead of extending"
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

        right_layout.addWidget(self.options_group)
        right_layout.addWidget(self.spike_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        button_layout.addStretch()
        right_layout.addLayout(button_layout)
        layout.addWidget(right_container, stretch=1)
        self._set_file_sections_visible(False)

    def _populate_cellpose_preview(self) -> None:
        """Fill the left-hand preview with Cellpose data if available."""
        if self.overlay_widget is None:
            return

        if (
            self.cellpose_import is None
            or self.cellpose_import.overlay_image is None
            or self.cellpose_import.segmentation_array is None
        ):
            self.overlay_widget.set_empty_message(
                "Load Cellpose segmentation first to preview cells."
            )
            return

        self.overlay_widget.set_overlay(
            self.cellpose_import.overlay_image,
            None,
            self.cellpose_import.segmentation_array,
            self.cellpose_import.original_cellpose_ids,
            self.cellpose_import.stimulus_to_cellpose_mapping,
        )

    def _on_mapping_changed(self) -> None:
        mapping_strategy = self._get_mapping_strategy()
        is_custom = mapping_strategy == "Custom"

        self.custom_mapping_label.setVisible(is_custom)
        self.custom_widget.setVisible(is_custom)

        if is_custom:
            self.custom_mapping_line.setVisible(True)
            self.custom_neuron_id_line.setVisible(False)
        else:
            self.custom_mapping_line.setVisible(False)
            self.custom_neuron_id_line.setVisible(True)

    def _apply_existing_stim_mapping(self) -> None:
        """Prefill custom mapping when Cellpose data already has stimulus IDs."""
        if (
            self.cellpose_import is None
            or not self.cellpose_import.stimulus_to_cellpose_mapping
        ):
            return

        stim_ids = sorted(self.cellpose_import.stimulus_to_cellpose_mapping.keys())
        mapping_str = ",".join(str(stim) for stim in stim_ids)
        self.custom_mapping_line.setText(mapping_str)

        self._set_mapping_strategy("Custom")
        if len(stim_ids) > 1:
            self._auto_stimulus_ids = stim_ids
            self._auto_from_cellpose = True

    def _on_replace_toggled(self, checked: bool) -> None:
        self.append_data = not checked

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
            self.data_row_count = 0
            self._auto_stimulus_ids = []
            self._auto_from_cellpose = False

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
            self._set_mapping_strategy("Custom")
            self.custom_neuron_id_line.setText(str(self.extracted_neuron_id))
            if not self.custom_mapping_line.text().strip():
                # Default stim ID target to the detected neuron ID
                self.custom_mapping_line.setText(str(self.extracted_neuron_id))

        if self.npy_data.ndim not in (1, 2):
            raise ValueError(
                f"NumPy patchclamp data must be 1D (single neuron) or 2D (multiple neurons), got {self.npy_data.ndim}D"
            )

        self.data_row_count = (
            1 if self.npy_data.ndim == 1 else int(self.npy_data.shape[0])
        )
        self._apply_cellpose_mapping_defaults(self.data_row_count)
        self._update_target_stimulus_ids()

        # Pathclamp has a higher hz in general
        self.sampling_rate_spin.setValue(10000.0)

    def _update_preview(
        self,
    ) -> None:  # important to have so the user can figure out mistakes
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

            target_ids = []
            if self._get_mapping_strategy() == "Custom":
                target_ids = self._safe_parse_stimulus_ids()
            else:
                target_ids = self._safe_parse_neuron_ids()
            if len(target_ids) > 1:
                suffix = (
                    " (auto from Cellpose mapping)"
                    if self._auto_from_cellpose
                    and target_ids == self._auto_stimulus_ids
                    else ""
                )
                lines.append(
                    f"Target stimulus IDs: {', '.join(map(str, target_ids))}{suffix}"
                )
        else:
            self.preview_text.clear()
            return

        self.preview_text.setPlainText("\n".join(lines).strip())

    def _get_mapping_strategy(self) -> str:
        checked = self.mapping_buttons.checkedButton()
        return checked.text() if checked is not None else "Linear"

    def _target_neuron_ids(self) -> list[int]:
        """Get target neuron IDs based on current mapping strategy."""
        strategy = self._get_mapping_strategy()
        if strategy == "Custom":
            return self._parse_stim_neuron_ids(self.custom_mapping_line.text())
        return self._parse_neuron_ids(self.custom_neuron_id_line.text())

    def _apply_cellpose_mapping_defaults(self, row_count: int) -> None:
        """Use Cellpose stimulus IDs in ascending order to prefill multi-row targets."""
        stim_ids = self._get_cellpose_stim_ids()
        if len(stim_ids) < 2 or row_count < 2:
            return

        derived_ids = stim_ids[:row_count]
        self.custom_mapping_line.setText(",".join(str(stim) for stim in derived_ids))
        self.custom_neuron_id_line.setText(self.custom_mapping_line.text())
        self._set_mapping_strategy("Custom")
        self._auto_stimulus_ids = derived_ids
        self._auto_from_cellpose = True

    def _get_cellpose_stim_ids(self) -> list[int]:
        """Return sorted stimulus IDs from an existing Cellpose mapping."""
        if (
            self.cellpose_import is None
            or not self.cellpose_import.stimulus_to_cellpose_mapping
        ):
            return []
        return sorted(self.cellpose_import.stimulus_to_cellpose_mapping.keys())

    def _set_file_sections_visible(self, visible: bool) -> None:
        for group in (self.preview_group, self.options_group, self.spike_group):
            if group is not None:
                group.setVisible(visible)
        if visible:
            self.resize(
                max(self.width(), getattr(self, "dialog_width", self.width())),
                max(self.height(), getattr(self, "dialog_height", self.height())),
            )

    def _parse_stim_neuron_ids(self, mapping_str: str) -> list[int]:
        """Parse comma-separated stimulus IDs into a list of ints."""
        if not mapping_str.strip():
            raise ValueError("Stimulus IDs cannot be empty")

        try:
            return [
                int(part.strip()) for part in mapping_str.split(",") if part.strip()
            ]
        except ValueError as exc:
            raise ValueError(f"Invalid stimulus ID format: {exc}") from exc

    def _safe_parse_stimulus_ids(self) -> list[int]:
        """Parse stimulus IDs without raising, used for prefilling defaults."""
        text = self.custom_mapping_line.text()
        if not text.strip():
            return []
        try:
            return self._parse_stim_neuron_ids(text)
        except Exception:
            return []

    def _parse_neuron_ids(self, neuron_ids_str: str) -> list[int]:
        """Parse comma-separated neuron IDs string into list of integers."""
        if not neuron_ids_str.strip():
            raise ValueError("Neuron IDs cannot be empty")

        try:
            # Split by comma and strip whitespace
            parts = [part.strip() for part in neuron_ids_str.split(",") if part.strip()]
            if not parts:
                raise ValueError("No valid neuron IDs found")
            return [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid neuron ID format: {exc}") from exc

    def _safe_parse_neuron_ids(self) -> list[int]:
        """Parse neuron IDs without raising, used for prefilling defaults."""
        text = self.custom_neuron_id_line.text()
        if not text.strip():
            return []
        try:
            return self._parse_neuron_ids(text)
        except Exception:
            return []

    def _set_mapping_strategy(self, target: str) -> None:
        for button in self.mapping_buttons.buttons():
            if button.text() == target:
                button.setChecked(True)
                break

    def _update_target_stimulus_ids(self) -> None:
        """Limit target stimulus IDs to the number of traces in the loaded data."""
        if self.npy_data is None:
            return

        auto_from_cellpose = False
        row_count = 1 if self.npy_data.ndim == 1 else int(self.npy_data.shape[0])
        self.data_row_count = row_count

        if row_count == 1 and self.extracted_neuron_id is not None:
            target_ids = [self.extracted_neuron_id]
        elif row_count == 1:
            target_ids = self._safe_parse_stimulus_ids()
            if not target_ids:
                target_ids = self._safe_parse_neuron_ids()
            if not target_ids:
                target_ids = [1]
            else:
                target_ids = [target_ids[0]]
        else:
            target_ids = self._safe_parse_stimulus_ids()
            auto_from_cellpose = False
            if not target_ids:
                # Default to sorted Cellpose stim IDs when available
                cellpose_ids = self._get_cellpose_stim_ids()
                if cellpose_ids:
                    target_ids = cellpose_ids[:row_count]
                    auto_from_cellpose = True
            if target_ids:
                target_ids = target_ids[:row_count]
                # Pad with sequential IDs if we truncated a shorter list
                next_id = (target_ids[-1] + 1) if target_ids else 1
                while len(target_ids) < row_count:
                    target_ids.append(next_id)
                    next_id += 1
            else:
                target_ids = list(range(1, row_count + 1))

        mapping_str = ",".join(str(tid) for tid in target_ids)
        self.custom_mapping_line.setText(mapping_str)
        self.custom_neuron_id_line.setText(mapping_str)
        self._auto_from_cellpose = auto_from_cellpose and len(target_ids) > 1
        self._auto_stimulus_ids = target_ids if self._auto_from_cellpose else []

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

        # Deal with the multi-neuron extend/replace situation: Double check importing makes sense.
        # Check which neurons we're trying to import
        neuron_ids_to_import = self._target_neuron_ids()
        existing_neuron_indices = set(existing_trace_data.indices)

        # Check existing status for each neuron being imported
        importing_existing = [
            nid for nid in neuron_ids_to_import if nid in existing_neuron_indices
        ]
        importing_new = [
            nid for nid in neuron_ids_to_import if nid not in existing_neuron_indices
        ]

        # In extend mode, if we have both existing and new neurons in the same import,
        # this could cause timestamp misalignment
        if len(importing_existing) > 0 and len(importing_new) > 0:
            raise ValueError(
                f"Cannot extend data when importing both existing neurons {importing_existing} "
                f"and new neurons {importing_new} together. "
                f"This would cause timestamp misalignment. Use 'Replace' mode instead, "
                f"or import all neurons with the same timeline separately."
            )

        # If importing only new neurons but existing data exists for other neurons
        if (
            len(importing_new) == len(neuron_ids_to_import)
            and len(existing_neuron_indices) > 0
        ):
            raise ValueError(
                f"Cannot extend data when importing new neurons {importing_new} - "
                f"existing neurons {sorted(existing_neuron_indices)} already have data. "
                f"This would cause timestamp misalignment. Use 'Replace' mode instead, "
                f"or import all neurons with the same timeline."
            )

    def _generate_stimulus_mapping(self, strategy: str) -> dict[int, int]:
        """Return any existing Cellpose stimulus mapping (no manual editing here)."""
        if self.cellpose_import and self.cellpose_import.stimulus_to_cellpose_mapping:
            return dict(self.cellpose_import.stimulus_to_cellpose_mapping)
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

    def _import_npy_data(self, sampling_rate: float, dt: float) -> None:
        if self.npy_data is None:
            raise ValueError("No NPY data loaded")

        # Parse neuron IDs from the UI
        neuron_ids = self._target_neuron_ids()

        # Validate data dimensions match neuron count
        if self.npy_data.ndim == 1:
            # Single trace - can only import to one neuron
            if len(neuron_ids) > 1:
                raise ValueError(
                    f"Single trace data cannot be imported to multiple neurons {neuron_ids}"
                )
            trace_data_rows = [self.npy_data]
        elif self.npy_data.ndim == 2:
            # Multiple traces - check if we have enough rows
            if self.npy_data.shape[0] != len(neuron_ids):
                raise ValueError(
                    f"Data has {self.npy_data.shape[0]} rows but {len(neuron_ids)} neuron IDs specified"
                )
            trace_data_rows = [self.npy_data[i] for i in range(self.npy_data.shape[0])]
        else:
            raise ValueError(f"Unsupported data dimensions: {self.npy_data.shape}")

        # Track neurons being imported for potential replacement
        if not self.append_data:
            self.neurons_to_replace.update(neuron_ids)

        # Import data for each neuron
        all_traces = []
        all_neuron_indices = []

        for i, neuron_id in enumerate(neuron_ids):
            trace_data = trace_data_rows[i]

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
                trace_data,
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
            self.imported_spike_data.append(spike_data)

            # Collect trace data for combined TraceData object
            all_traces.append(trace_data)
            all_neuron_indices.append(neuron_id)

            self.logger.debug(
                "Patchclamp import complete for neuron=%d spikes=%d trace_len=%d dt=%.6f",
                neuron_id,
                len(spike_data.spike_times),
                len(trace_data),
                dt,
            )

        # Create combined trace data for all imported neurons
        trace_array = np.array(all_traces)
        neuron_indices_array = np.array(all_neuron_indices, dtype=np.uint32)
        trace_data = TraceData(
            name="Patch Clamp Traces",  # Use consistent name for all patch clamp data
            traces=trace_array,
            indices=neuron_indices_array,
            units="mV",
            dt=dt,
        )
        self.imported_trace_data.append(trace_data)
        # todo: Liam/James to discuss/figure out saving the trace data in a sparse way (but still with gaps between, most likely by mapping e.g. 27:0, 5:1 would mean: show in stim arr pos 27 the value of traces data arr itself 0, and 5 value of traces data arr 1, and there would be in that case a 2x(timestamps) length array.

        # Check if any neurons had spikes detected
        total_spikes = sum(
            len(sd.spike_times) for sd in self.imported_spike_data[-len(neuron_ids) :]
        )
        if total_spikes > 0:
            neuron_list = ", ".join(map(str, neuron_ids))
            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {total_spikes} spikes and trace data from neuron(s) {neuron_list}\n"
                f"Duration: {trace_array.shape[1] / sampling_rate:.1f} seconds at {sampling_rate:.0f} Hz.",
            )
        else:
            neuron_list = ", ".join(map(str, neuron_ids))
            QtWidgets.QMessageBox.warning(
                self,
                "No Spikes Detected",
                f"No spikes were detected in the recording for neuron(s) {neuron_list}.\n"
                f"You may need to adjust the detection parameters.",
            )

    @property
    def imported_data(self) -> list:
        """Combined imported spike and trace data."""
        return self.imported_spike_data + self.imported_trace_data
