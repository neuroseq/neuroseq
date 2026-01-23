from __future__ import annotations

import logging
import re
from pathlib import Path

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import clone_cellpose_result
from neuroseq.importer_patchclamp import PeakDetectionParams
from neuroseq.importer_patchclamp import generate_stimulus_mapping
from neuroseq.importer_patchclamp import import_patchclamp_data
from neuroseq.importer_patchclamp import validate_append_import


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp data from .npy files."""

    def __init__(
        self,
        cellpose_import: CellposeImportResult | None = None,
        project: object | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        # Work on a copy so we don't mutate the caller's stored data
        self.cellpose_import = clone_cellpose_result(cellpose_import)
        self.project = project
        self.imported_spike_data: list = []
        self.imported_trace_data: list = []
        self.experiment_name = "PatchClamp Import"
        self.experiment_duration: float = 0.0
        self._auto_stimulus_ids: list[int] = []
        self._auto_from_cellpose = False
        self.append_data = True
        self.neurons_to_replace: list[int] = []
        self.file_path: Path | None = None
        self.npy_data: np.ndarray | None = None
        self.extracted_neuron_id: int | None = None
        self.data_row_count: int = 0
        self.overlay_widget: CellOverlayWidget | None = None
        self.show_stimulus_checkbox: QtWidgets.QCheckBox | None = None
        self.preview_group: QtWidgets.QGroupBox | None = None
        self.options_group: QtWidgets.QGroupBox | None = None
        self.spike_group: QtWidgets.QGroupBox | None = None
        self.overlay_stack: QtWidgets.QStackedWidget | None = None
        self.overlay_placeholder: QtWidgets.QLabel | None = None
        self.file_group: QtWidgets.QGroupBox | None = None
        self.file_status_label: QtWidgets.QLabel | None = None
        self.button_row: QtWidgets.QWidget | None = None
        self._setup_ui()
        self._apply_existing_stim_mapping()
        self._populate_cellpose_preview()

    @staticmethod
    def _double_spinbox(
        *,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int,
        suffix: str = "",
        tooltip: str | None = None,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(f" {suffix}")
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    @staticmethod
    def _int_spinbox(
        *,
        minimum: int,
        maximum: int,
        step: int,
        value: int,
        suffix: str = "",
        tooltip: str | None = None,
    ) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(f" {suffix}")
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

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

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.overlay_stack = QtWidgets.QStackedWidget()
        self.overlay_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_stack.setMaximumWidth(max_half_width)
        self.overlay_widget = CellOverlayWidget("Cellpose Preview")
        self.overlay_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_widget.setMaximumWidth(max_half_width)
        self.overlay_placeholder = QtWidgets.QLabel("No cellpose data yet")
        self.overlay_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.overlay_placeholder.setStyleSheet("QLabel { color: gray; }")
        self.overlay_placeholder.setWordWrap(True)
        self.overlay_stack.addWidget(self.overlay_placeholder)
        self.overlay_stack.addWidget(self.overlay_widget)
        left_layout.addWidget(self.overlay_stack, stretch=1)
        left_layout.addStretch()

        self.file_group = QtWidgets.QGroupBox("Select .npy patchclamp file")
        file_layout = QtWidgets.QHBoxLayout(self.file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        left_layout.addWidget(self.file_group)
        layout.addWidget(left_container, stretch=1)

        right_container = QtWidgets.QWidget()
        right_container.setMaximumWidth(max_half_width)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        right_layout.addWidget(self.preview_group)

        self.options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(self.options_group)

        self.sampling_rate_spin = self._double_spinbox(
            minimum=0.1,
            maximum=10000.0,
            step=1.0,
            value=30.0,
            decimals=2,
            suffix="Hz",
        )
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
        self.custom_neuron_id_line.setPlaceholderText("e.g., 0 or 0,1")
        self.custom_neuron_id_line.setText("0")
        self.custom_neuron_id_line.setToolTip(
            "Single neuron ID (e.g., 10) or comma-separated list (e.g., 12,15)"
        )

        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setPlaceholderText(
            "Comma-separated cellpose IDs (e.g., 2,3 or 0:2,1:3)"
        )
        self.custom_mapping_line.textChanged.connect(
            lambda _text: self._update_custom_mapping_hint()
        )
        self.custom_mapping_line.setVisible(False)

        custom_layout = QtWidgets.QVBoxLayout()
        custom_layout.addWidget(self.custom_neuron_id_line)
        custom_layout.addWidget(self.custom_mapping_line)
        self.custom_widget = QtWidgets.QWidget()
        self.custom_widget.setLayout(custom_layout)

        self.custom_mapping_label = QtWidgets.QLabel("Target cellpose IDs:")
        self.custom_mapping_label.setVisible(False)
        self.custom_widget.setVisible(False)
        self.custom_mapping_hint = QtWidgets.QLabel("")
        self.custom_mapping_hint.setStyleSheet("QLabel { color: gray; }")
        self.custom_mapping_hint.setVisible(False)

        options_layout.addRow(self.custom_mapping_label, self.custom_widget)
        options_layout.addRow("", self.custom_mapping_hint)

        self.show_stimulus_checkbox = QtWidgets.QCheckBox(
            "Show Stimulus IDs in preview"
        )
        self.show_stimulus_checkbox.setChecked(True)
        self.show_stimulus_checkbox.toggled.connect(self._on_show_stimulus_toggled)
        options_layout.addRow(self.show_stimulus_checkbox)

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

        self.peak_height_spin = self._double_spinbox(
            minimum=0.1,
            maximum=10.0,
            step=0.1,
            value=3.0,
            decimals=2,
            suffix="std",
            tooltip="Minimum height above baseline in standard deviations",
        )
        spike_layout.addRow("Peak height threshold:", self.peak_height_spin)

        self.peak_prominence_spin = self._double_spinbox(
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            value=0.5,
            decimals=2,
            suffix="std",
            tooltip="Minimum prominence for peak detection",
        )
        spike_layout.addRow("Peak prominence:", self.peak_prominence_spin)

        self.min_interval_spin = self._double_spinbox(
            minimum=0.0001,
            maximum=0.1,
            step=0.001,
            value=0.001,
            decimals=3,
            suffix="s",
            tooltip="Minimum interval between detected spikes",
        )
        spike_layout.addRow("Min spike interval:", self.min_interval_spin)

        self.peak_distance_spin = self._int_spinbox(
            minimum=1,
            maximum=1000,
            step=1,
            value=10,
            suffix="samples",
            tooltip="Minimum horizontal distance between peaks in samples",
        )
        spike_layout.addRow("Peak distance:", self.peak_distance_spin)

        self.peak_width_spin = self._int_spinbox(
            minimum=1,
            maximum=100,
            step=1,
            value=1,
            suffix="samples",
            tooltip="Minimum width of peaks in samples",
        )
        spike_layout.addRow("Peak width:", self.peak_width_spin)

        right_layout.addWidget(self.options_group)
        right_layout.addWidget(self.spike_group)

        self.button_row = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(self.button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.file_status_label = QtWidgets.QLabel("No file selected")
        self.file_status_label.setStyleSheet("QLabel { color: gray; }")
        self.file_status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        button_layout.addWidget(self.file_status_label, stretch=1)
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        button_layout.addStretch()
        right_layout.addWidget(self.button_row)
        layout.addWidget(right_container, stretch=1)
        self._set_file_sections_visible(False)
        self._set_file_selector_visible(True)

    def _populate_cellpose_preview(self) -> None:
        """Fill the left-hand preview with Cellpose data if available."""
        if self.overlay_widget is None or self.overlay_stack is None:
            return

        if (
            self.cellpose_import is None
            or self.cellpose_import.overlay_image is None
            or self.cellpose_import.segmentation_array is None
        ):
            self._show_overlay_placeholder("No cellpose data yet")
            return

        self.overlay_stack.setCurrentWidget(self.overlay_widget)
        self.overlay_widget.set_overlay(
            self.cellpose_import.overlay_image,
            self.cellpose_import.segmentation_array,
            self.cellpose_import.original_cellpose_ids,
            self.cellpose_import.stimulus_to_cellpose_mapping,
        )
        self._apply_overlay_display_options()

    def _show_overlay_placeholder(self, message: str) -> None:
        """Show a text placeholder instead of an empty overlay preview."""
        if self.overlay_stack is None or self.overlay_placeholder is None:
            return

        self.overlay_placeholder.setText(message)
        self.overlay_stack.setCurrentWidget(self.overlay_placeholder)

    def _on_mapping_changed(self) -> None:
        mapping_strategy = self._get_mapping_strategy()
        is_custom = mapping_strategy == "Custom"

        self.custom_mapping_label.setVisible(is_custom)
        self.custom_widget.setVisible(is_custom)
        self._update_custom_mapping_hint()

        if is_custom:
            self.custom_mapping_line.setVisible(True)
            self.custom_neuron_id_line.setVisible(False)
        else:
            self.custom_mapping_line.setVisible(False)
            self.custom_neuron_id_line.setVisible(True)

    def _on_show_stimulus_toggled(self) -> None:
        self._apply_overlay_display_options()

    def _apply_overlay_display_options(self) -> None:
        if self.overlay_widget is None:
            return
        show_stimulus_ids = (
            self.show_stimulus_checkbox.isChecked()
            if self.show_stimulus_checkbox is not None
            else False
        )
        self.overlay_widget.set_display_options(
            show_stimulus_ids=show_stimulus_ids,
            show_cellpose_ids=False,
        )

    def _apply_existing_stim_mapping(self) -> None:
        """Prefill custom mapping when Cellpose data already has stimulus IDs."""
        if (
            self.cellpose_import is None
            or not self.cellpose_import.stimulus_to_cellpose_mapping
        ):
            return

        mapping = self.cellpose_import.stimulus_to_cellpose_mapping
        stim_ids = sorted(mapping.keys())
        mapping_str = ",".join(f"{stim_id}:{mapping[stim_id]}" for stim_id in stim_ids)
        self.custom_mapping_line.setText(mapping_str)
        self.custom_neuron_id_line.setText(",".join(str(stim) for stim in stim_ids))

        self._set_mapping_strategy("Custom")
        if len(stim_ids) > 1:
            self._auto_stimulus_ids = stim_ids
            self._auto_from_cellpose = True
        self._update_custom_mapping_hint()

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
            if self.file_status_label is not None:
                self.file_status_label.setText(self.file_path.name)
                self.file_status_label.setStyleSheet("QLabel { color: black; }")

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
            self._set_file_selector_visible(False)
        except Exception as exc:
            self.logger.exception("Error loading patch clamp file")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{exc}",
            )
            self.import_button.setEnabled(False)
            self._set_file_sections_visible(False)
            self._set_file_selector_visible(True)
            if self.file_status_label is not None:
                self.file_status_label.setText("No file selected")
                self.file_status_label.setStyleSheet("QLabel { color: gray; }")

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
                lines.append(f"Specify {len(target_ids)} IDs to import")
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
            return self._resolve_custom_target_ids(self._safe_parse_stimulus_ids())
        return self._parse_neuron_ids(self.custom_neuron_id_line.text())

    def _resolve_custom_target_ids(self, cellpose_ids: list[int]) -> list[int]:
        """Resolve custom cellpose IDs to stimulus IDs, ensuring uniqueness."""
        mapping = (
            self.cellpose_import.stimulus_to_cellpose_mapping
            if self.cellpose_import is not None
            else {}
        ) or {}
        reverse_mapping = {cellpose: stim for stim, cellpose in mapping.items()}
        assigned_stim_ids: set[int] = set()
        stim_ids: list[int] = []
        for cellpose_id in cellpose_ids:
            stim_id = int(reverse_mapping.get(int(cellpose_id), int(cellpose_id)))
            if stim_id in assigned_stim_ids:
                raise ValueError(
                    "Custom mapping would assign stimulus ID "
                    f"{stim_id} to multiple patchclamp rows."
                )
            assigned_stim_ids.add(stim_id)
            stim_ids.append(stim_id)
        return stim_ids

    def _apply_cellpose_mapping_defaults(self, row_count: int) -> None:
        """Use Cellpose stimulus IDs in ascending order to prefill multi-row targets."""
        stim_ids = self._get_cellpose_stim_ids()
        if len(stim_ids) < 2 or row_count < 2:
            return

        derived_ids = stim_ids[:row_count]
        self.custom_mapping_line.setText(self._format_stimulus_display(derived_ids))
        self.custom_neuron_id_line.setText(",".join(str(stim) for stim in derived_ids))
        self._set_mapping_strategy("Custom")
        self._auto_stimulus_ids = derived_ids
        self._auto_from_cellpose = True
        self._update_custom_mapping_hint()

    def _get_cellpose_stim_ids(self) -> list[int]:
        """Return sorted stimulus IDs from an existing Cellpose mapping."""
        if (
            self.cellpose_import is None
            or not self.cellpose_import.stimulus_to_cellpose_mapping
        ):
            return []
        return sorted(self.cellpose_import.stimulus_to_cellpose_mapping.keys())

    def _cellpose_id_for_stim(self, stim_id: int, stim_index: int) -> int | None:
        """Return the cellpose ID aligned with a given stimulus ID when available."""
        if self.cellpose_import is None:
            return None

        mapping = self.cellpose_import.stimulus_to_cellpose_mapping
        if mapping:
            return mapping.get(stim_id)

        if (
            self.cellpose_import.original_cellpose_ids is not None
            and 0 <= stim_index < len(self.cellpose_import.original_cellpose_ids)
        ):
            return int(self.cellpose_import.original_cellpose_ids[stim_index])

        return None

    def _format_stimulus_display(self, stim_ids: list[int]) -> str:
        """Format stimulus IDs as 'stim:cellpose' when cellpose IDs are known."""
        parts: list[str] = []
        for idx, stim_id in enumerate(stim_ids):
            cellpose_id = self._cellpose_id_for_stim(stim_id, idx)
            if cellpose_id is None:
                parts.append(str(stim_id))
            else:
                parts.append(f"{stim_id}:{cellpose_id}")
        return ",".join(parts)

    def _set_file_sections_visible(self, visible: bool) -> None:
        for group in (self.preview_group, self.options_group, self.spike_group):
            if group is not None:
                group.setVisible(visible)
        if self.button_row is not None:
            self.button_row.setVisible(visible)
        if visible:
            self.resize(
                max(self.width(), getattr(self, "dialog_width", self.width())),
                max(self.height(), getattr(self, "dialog_height", self.height())),
            )

    def _set_file_selector_visible(self, visible: bool) -> None:
        if self.file_group is not None:
            self.file_group.setVisible(visible)

    def _parse_stim_neuron_ids(self, mapping_str: str) -> list[int]:
        """Parse comma-separated cellpose IDs into a list of ints."""
        if not mapping_str.strip():
            raise ValueError("Stimulus IDs cannot be empty")

        try:
            stim_ids: list[int] = []
            for raw_part in mapping_str.split(","):
                part = raw_part.strip()
                if not part:
                    continue
                if ":" in part:
                    part = part.split(":", 1)[1].strip()
                stim_ids.append(int(part))
            return stim_ids
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
            target_ids = [0] if not target_ids else [target_ids[0]]
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
                target_ids = list(range(row_count))

        mapping_str = self._format_stimulus_display(target_ids)
        self.custom_mapping_line.setText(mapping_str)
        self.custom_neuron_id_line.setText(",".join(str(tid) for tid in target_ids))
        self.custom_mapping_label.setText(f"Enter {len(target_ids)} Cellpose ID(s):")
        self._auto_from_cellpose = auto_from_cellpose and len(target_ids) > 1
        self._auto_stimulus_ids = target_ids if self._auto_from_cellpose else []
        self._update_custom_mapping_hint()

    def _update_custom_mapping_hint(self) -> None:
        """Show a hint when custom mapping count mismatches neuron count."""
        if self.custom_mapping_hint is None:
            return

        if self.npy_data is None or self._get_mapping_strategy() != "Custom":
            self.custom_mapping_hint.setVisible(False)
            return

        row_count = self.data_row_count or 0
        stim_ids = self._safe_parse_stimulus_ids()
        if row_count <= 0 or not stim_ids:
            self.custom_mapping_hint.setVisible(False)
            return

        if len(stim_ids) > row_count:
            self.custom_mapping_hint.setText(
                f"Too many mappings for the {row_count} neurons being imported"
            )
            self.custom_mapping_hint.setVisible(True)
        elif len(stim_ids) < row_count:
            self.custom_mapping_hint.setText(
                f"Too few mappings for the {row_count} neurons being imported"
            )
            self.custom_mapping_hint.setVisible(True)
        else:
            self.custom_mapping_hint.setVisible(False)

    def import_data(self) -> None:
        if self.npy_data is None:
            return

        try:
            sampling_rate = self.sampling_rate_spin.value()
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be positive")
            dt = 1.0 / sampling_rate

            neuron_ids = self._target_neuron_ids()
            if self.append_data:
                validate_append_import(
                    self.project,
                    self.experiment_name,
                    neuron_ids,
                )
                self.neurons_to_replace = []
            else:
                self.neurons_to_replace = list(neuron_ids)

            # Generate and store the stimulus-to-cellpose mapping
            mapping_strategy = self._get_mapping_strategy()
            stimulus_mapping = generate_stimulus_mapping(
                self.cellpose_import,
                mapping_strategy,
                self.data_row_count,
                self._safe_parse_stimulus_ids(),
            )

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

            detection_params = PeakDetectionParams(
                peak_height=self.peak_height_spin.value(),
                peak_prominence=self.peak_prominence_spin.value(),
                min_interval=self.min_interval_spin.value(),
                peak_distance=self.peak_distance_spin.value(),
                peak_width=self.peak_width_spin.value(),
            )

            trace_data, spike_data_list = import_patchclamp_data(
                self.npy_data,
                neuron_ids,
                dt=dt,
                detection_params=detection_params,
                logger=self.logger,
            )

            self.imported_spike_data.extend(spike_data_list)
            self.imported_trace_data.append(trace_data)

            total_spikes = sum(len(sd.spike_times) for sd in spike_data_list)
            neuron_list = ", ".join(map(str, neuron_ids))
            duration_seconds = trace_data.traces.shape[1] / sampling_rate
            self.experiment_duration = duration_seconds

            if total_spikes > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Imported {total_spikes} spikes and trace data from neuron(s) {neuron_list}\n"
                    f"Duration: {duration_seconds:.1f} seconds at {sampling_rate:.0f} Hz.",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Spikes Detected",
                    f"No spikes were detected in the recording for neuron(s) {neuron_list}.\n"
                    f"You may need to adjust the detection parameters.",
                )
            self.accept()
        except Exception as exc:
            self.logger.exception("Error importing patch clamp data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing data:\n{exc}",
            )

    @property
    def imported_data(self) -> list:
        """Combined imported spike and trace data."""
        return self.imported_spike_data + self.imported_trace_data
