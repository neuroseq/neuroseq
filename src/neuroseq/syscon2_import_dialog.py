from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.importer_syscon2 import SysconSequence
from neuroseq.importer_syscon2 import get_timed_objects
from neuroseq.importer_syscon2 import group_timed_objects_by_neuron_target
from neuroseq.importer_syscon2 import import_sequence_into_stim_array
from neuroseq.importer_syscon2 import parse_syscon2_seq_file

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray


class Syscon2ImportDialog(QtWidgets.QDialog):
    """Dialog for importing SysCon2 .seq files into an existing StimArray."""

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.sequence: SysconSequence | None = None
        self.file_path: str = ""
        self.selected_stim_name: str = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import SysCon2 .seq File")
        self.setModal(True)
        self.resize(820, 620)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Select a SysCon2 .seq file, preview parsed content, and import timing/intensity "
            "information into a selected stimulation data entity."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        self.file_label = QtWidgets.QLabel("No .seq file selected")
        self.file_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_label, stretch=1)
        file_layout.addWidget(browse_button)
        layout.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("Parsed Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Preview appears here after selecting a .seq file."
        )
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        options_layout.addRow("Target stim data entity:", self.target_combo)

        self.neuron_offset_spin = QtWidgets.QSpinBox()
        self.neuron_offset_spin.setRange(0, 0)
        self.neuron_offset_spin.setValue(0)
        options_layout.addRow("Target neuron offset:", self.neuron_offset_spin)

        self.tick_resolution_spin = QtWidgets.QSpinBox()
        self.tick_resolution_spin.setRange(1, 1_000_000)
        self.tick_resolution_spin.setValue(500)
        self.tick_resolution_spin.setSuffix(" us")
        self.tick_resolution_spin.setToolTip(
            "System tick resolution used to expand repeats. "
            "Typical values: 500us (Holo4D) or 50us (UGA-42)."
        )
        options_layout.addRow("System tick resolution:", self.tick_resolution_spin)

        self.include_non_shape_checkbox = QtWidgets.QCheckBox(
            "Include non-shape timed objects (TTLPulse, WaitForTTL, ...)"
        )
        self.include_non_shape_checkbox.setChecked(False)
        options_layout.addRow(self.include_non_shape_checkbox)

        self.extend_checkbox = QtWidgets.QCheckBox(
            "Extend target stim array to fit imported duration"
        )
        self.extend_checkbox.setChecked(True)
        options_layout.addRow(self.extend_checkbox)

        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing target stim values (instead of merging)"
        )
        self.replace_checkbox.setChecked(True)
        options_layout.addRow(self.replace_checkbox)

        layout.addWidget(options_group)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self.info_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_file)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self._populate_target_stims()

    def _populate_target_stims(self) -> None:
        self.target_combo.clear()
        stim_names = list(self.project.stim_arrays.keys())
        if not stim_names:
            self.target_combo.addItem("(No stim arrays available)")
            self.target_combo.setEnabled(False)
            self.info_label.setText(
                "No stimulation data entities available. Create one before importing."
            )
            return

        self.target_combo.addItems(stim_names)
        self.target_combo.setEnabled(True)
        self._on_target_changed(self.target_combo.currentText())

    def _selected_stim_array(self) -> StimArray | None:
        name = self.target_combo.currentText()
        if not name:
            return None
        return self.project.stim_arrays.get(name)

    def _on_target_changed(self, target_name: str) -> None:
        stim_array = self._selected_stim_array()
        if stim_array is None:
            self.neuron_offset_spin.setRange(0, 0)
            self.neuron_offset_spin.setValue(0)
            self._update_import_button_state()
            return

        max_offset = max(0, stim_array.values.shape[1] - 1)
        self.neuron_offset_spin.setRange(0, max_offset)
        if self.neuron_offset_spin.value() > max_offset:
            self.neuron_offset_spin.setValue(max_offset)
        self.selected_stim_name = target_name
        self._update_import_button_state()

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select SysCon2 .seq file",
            "",
            "SysCon2 sequence (*.seq);;All files (*)",
        )
        if filename:
            self._load_file(filename)

    def _load_file(self, filename: str) -> None:
        try:
            self.sequence = parse_syscon2_seq_file(filename)
            self.file_path = filename
            self.file_label.setText(filename)
            self.file_label.setStyleSheet("QLabel { color: black; }")
            self.preview_text.setPlainText(self._build_preview_text(self.sequence))
            self.info_label.setText(
                "Review options and click Import to apply the sequence."
            )
        except Exception as exc:
            self.logger.exception("Failed to parse SysCon2 file %s", filename)
            self.sequence = None
            self.file_path = ""
            self.file_label.setText("No .seq file selected")
            self.file_label.setStyleSheet("QLabel { color: gray; }")
            self.preview_text.clear()
            self.info_label.setText("")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Could not parse SysCon2 .seq file:\n{exc}",
            )
        self._update_import_button_state()

    def _build_preview_text(self, sequence: SysconSequence) -> str:
        timed_shape_objects = get_timed_objects(sequence, include_non_shape=False)
        timed_all_objects = get_timed_objects(sequence, include_non_shape=True)
        grouped_shape_objects = group_timed_objects_by_neuron_target(
            sequence, include_non_shape=False
        )
        grouped_all_objects = group_timed_objects_by_neuron_target(
            sequence, include_non_shape=True
        )

        lines: list[str] = [
            f"Header version={sequence.header.get('version', '?')}",
            f"Scan mode={sequence.header.get('scanmode', '?')}",
            f"Runs={sequence.header.get('runs', '?')}",
            f"Objects total={len(sequence.objects)}",
            f"Timed shape/entity objects={len(timed_shape_objects)}",
            f"Estimated neuron channels (shape/entity)={len(grouped_shape_objects)}",
            "Center-merged circular entities (<=1um)="
            f"{max(0, len(timed_shape_objects) - len(grouped_shape_objects))}",
            f"Timed objects (all types)={len(timed_all_objects)}",
            f"Estimated neuron channels (all types)={len(grouped_all_objects)}",
            "",
            "First objects:",
        ]

        for obj in sequence.objects[:20]:
            lines.append(
                f"[{obj.block_index}] ID={obj.object_id} type={obj.object_type} "
                f"start_us={obj.start_time_us} repeats={obj.repeats} "
                f"step={obj.step_size} intensity={obj.intensity:.6g}"
            )
        if len(sequence.objects) > 20:
            lines.append(f"... ({len(sequence.objects) - 20} more objects)")
        return "\n".join(lines)

    def _update_import_button_state(self) -> None:
        self.import_button.setEnabled(
            self.sequence is not None and self._selected_stim_array() is not None
        )

    def _import_file(self) -> None:
        if self.sequence is None:
            return
        stim_array = self._selected_stim_array()
        if stim_array is None:
            return

        try:
            result = import_sequence_into_stim_array(
                stim_array,
                self.sequence,
                tick_resolution_us=self.tick_resolution_spin.value(),
                include_non_shape=self.include_non_shape_checkbox.isChecked(),
                replace_existing=self.replace_checkbox.isChecked(),
                extend_to_fit=self.extend_checkbox.isChecked(),
                neuron_offset=self.neuron_offset_spin.value(),
            )
            info_lines = [
                f"Imported into '{stim_array.name}'.",
                f"Eligible timed objects: {result.eligible_objects}",
                "Neuron channels after center-distance grouping: "
                f"{result.grouped_objects}",
                f"Mapped neuron channels into target: {result.mapped_objects}",
                f"Unmapped channels (insufficient target neurons): {result.dropped_objects}",
                f"Generated timepoints: {result.generated_timepoints}",
            ]
            if result.merged_objects > 0:
                info_lines.append(
                    "Merged circular entities by center distance (<=1um): "
                    f"{result.merged_objects}"
                )
            if result.truncated_timepoints > 0:
                info_lines.append(
                    f"Truncated timepoints: {result.truncated_timepoints}"
                )
            if result.truncated_repeats > 0:
                info_lines.append(
                    f"Truncated repeats (safety cap): {result.truncated_repeats}"
                )
            QtWidgets.QMessageBox.information(
                self, "Import Complete", "\n".join(info_lines)
            )
            self.accept()
        except Exception as exc:
            self.logger.exception("Failed to import SysCon2 sequence")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import SysCon2 sequence:\n{exc}",
            )
