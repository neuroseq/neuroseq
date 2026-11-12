from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.importers.import_folders import ImportFolders
from neuroseq.importers.import_folders import remember_folder
from neuroseq.importers.import_folders import start_directory
from neuroseq.importers.importer_syscon2 import SysconSequence
from neuroseq.importers.importer_syscon2 import get_timed_objects
from neuroseq.importers.importer_syscon2 import group_timed_objects_by_neuron_target
from neuroseq.importers.importer_syscon2 import import_sequence_into_stim_array
from neuroseq.importers.importer_syscon2 import parse_syscon2_seq_file
from neuroseq.importers.importer_syscon2 import sequence_to_stim_values
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_timing import format_milliseconds
from neuroseq.stimulation.stim_timing import remap_stim_values_to_dt
from neuroseq.stimulation.stim_timing import stim_dt_matches

if TYPE_CHECKING:
    from neuroseq.project import Project


_BLOCK_START_RE = re.compile(r"^\[(\d+)\]$")


class Syscon2ImportDialog(QtWidgets.QDialog):
    """Dialog for importing SysCon2 .seq files into a new or existing StimArray."""

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
        *,
        import_folders: ImportFolders | None = None,
    ) -> None:
        super().__init__(parent)
        self._import_folders = import_folders
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.sequence: SysconSequence | None = None
        self.file_path: str = ""
        self.selected_stim_name: str = ""
        self._header_template: str | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import SysCon2 .seq File")
        self.setModal(True)
        self.resize(820, 620)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Select a SysCon2 .seq file, preview parsed content, and import timing/intensity "
            "information into a selected input array. Channels are assigned by target order, "
            "not matched to your project neuron IDs. Intensities are file values (zero uses a "
            "nominal 1), not measured optical power. Hardware latency is not reconstructed."
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

        self.create_new_radio = QtWidgets.QRadioButton("Create new input array")
        self.import_existing_radio = QtWidgets.QRadioButton(
            "Import into existing input array"
        )
        self.create_new_radio.toggled.connect(self._update_mode_state)
        options_layout.addRow(self.create_new_radio)

        self.new_name_edit = QtWidgets.QLineEdit(self._default_new_name())
        self.new_name_edit.textChanged.connect(self._update_import_button_state)
        options_layout.addRow("    Input name:", self.new_name_edit)
        self.new_type_combo = QtWidgets.QComboBox()
        self.new_type_combo.addItem("Excitatory", StimType.Excitation)
        self.new_type_combo.addItem("Inhibitory", StimType.Inhibition)
        self.new_type_combo.addItem("External", StimType.External)
        self.new_type_combo.setCurrentIndex(2)
        options_layout.addRow("    Input type:", self.new_type_combo)
        self.new_units_edit = QtWidgets.QLineEdit(self._default_new_units())
        options_layout.addRow("    Units:", self.new_units_edit)

        options_layout.addRow(self.import_existing_radio)
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        options_layout.addRow("    Target input array:", self.target_combo)

        self.neuron_offset_spin = QtWidgets.QSpinBox()
        self.neuron_offset_spin.setRange(0, 0)
        self.neuron_offset_spin.setValue(0)
        options_layout.addRow("    Target neuron offset:", self.neuron_offset_spin)

        self.tick_resolution_spin = QtWidgets.QSpinBox()
        self.tick_resolution_spin.setRange(1, 1_000_000)
        self.tick_resolution_spin.setValue(500)
        self.tick_resolution_spin.setSuffix(" us")
        self.tick_resolution_spin.setToolTip(
            "Legacy repeat spacing only. Holo4D laser TTL pulses always use 520.8125 us ticks."
        )
        options_layout.addRow("Legacy repeat tick:", self.tick_resolution_spin)

        self.extend_checkbox = QtWidgets.QCheckBox(
            "Extend target input array to fit imported duration"
        )
        self.extend_checkbox.setChecked(True)
        options_layout.addRow(self.extend_checkbox)

        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace mapped target channels (preserve other channels)"
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
        # Default to the existing-array workflow when there is something to
        # import into; otherwise creating a new array is the only option.
        has_arrays = bool(self.project.stim_arrays)
        self.import_existing_radio.setEnabled(has_arrays)
        self.import_existing_radio.setChecked(has_arrays)
        self.create_new_radio.setChecked(not has_arrays)
        self._update_mode_state()

    def _default_new_name(self) -> str:
        base = "syscon_input"
        name, index = base, 2
        while name in self.project.stim_arrays:
            name = f"{base}_{index}"
            index += 1
        return name

    def _default_new_units(self) -> str:
        for stim_array in self.project.stim_arrays.values():
            return stim_array.units
        return "nS"

    def _update_mode_state(self) -> None:
        create_new = self.create_new_radio.isChecked()
        self.new_name_edit.setEnabled(create_new)
        self.new_type_combo.setEnabled(create_new)
        self.new_units_edit.setEnabled(create_new)
        self.target_combo.setEnabled(not create_new and bool(self.project.stim_arrays))
        self.neuron_offset_spin.setEnabled(not create_new)
        self.replace_checkbox.setEnabled(not create_new)
        self.extend_checkbox.setEnabled(not create_new)
        if create_new:
            self.selected_stim_name = ""
        else:
            self._on_target_changed(self.target_combo.currentText())
        self._update_import_button_state()

    def _populate_target_stims(self) -> None:
        self.target_combo.clear()
        stim_names = list(self.project.stim_arrays.keys())
        if not stim_names:
            self.target_combo.addItem("(No input arrays available)")
            self.target_combo.setEnabled(False)
            return

        self.target_combo.addItems(stim_names)
        self._on_target_changed(self.target_combo.currentText())

    def _selected_stim_array(self) -> StimArray | None:
        if self.create_new_radio.isChecked():
            return None
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
            start_directory(self._import_folders),
            "SysCon2 sequence (*.seq);;All files (*)",
        )
        if filename:
            remember_folder(self._import_folders, filename)
            self._load_file(filename)

    def load_file(self, filename: str) -> None:
        """Load ``filename`` as if it had been browsed for, e.g. a file
        dropped onto the main window (issue #597)."""
        self._load_file(filename)

    def _load_file(self, filename: str) -> None:
        try:
            self.sequence = parse_syscon2_seq_file(filename)
            self.file_path = filename
            self.file_label.setText(filename)
            self.file_label.setStyleSheet("QLabel { color: black; }")
            self.preview_text.setPlainText(self._build_preview_text(self.sequence))
            self._header_template = self._extract_header_template_text(filename)
            self.info_label.setText(
                "Review options and click Import to apply the sequence."
            )
        except Exception as exc:
            self.logger.exception("Failed to parse SysCon2 file %s", filename)
            self.sequence = None
            self._header_template = None
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

    def _extract_header_template_text(self, filename: str) -> str | None:
        try:
            text = Path(filename).read_text(encoding="utf-8")
        except Exception:
            return None

        header_lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if _BLOCK_START_RE.match(stripped):
                break
            if stripped == "":
                continue
            header_lines.append(stripped)
        if not header_lines:
            return None
        return "\n".join(header_lines) + "\n"

    def _build_preview_text(self, sequence: SysconSequence) -> str:
        timed_shape_objects = get_timed_objects(sequence, include_non_shape=False)
        grouped_shape_objects = group_timed_objects_by_neuron_target(
            sequence, include_non_shape=False
        )

        lines: list[str] = [
            f"Header version={sequence.header.get('version', '?')}",
            f"Scan mode={sequence.header.get('scanmode', '?')}",
            f"Runs={sequence.header.get('runs', '?')}",
            f"Objects total={len(sequence.objects)}",
            f"Timed shape/entity objects={len(timed_shape_objects)}",
            f"Estimated neuron channels (shape/entity)={len(grouped_shape_objects)}",
            "Merged repeated targets="
            f"{max(0, len(timed_shape_objects) - len(grouped_shape_objects))}",
            "",
            "Mapping uses target order plus neuron offset, not project neuron identities.",
            "Repeated point Shapes at identical coordinates share a channel.",
            "Runs is not expanded; one sequence pass is imported.",
            *(
                [
                    "Original input array found next to the file: its exact timings "
                    "and neuron rows are restored on import."
                ]
                if sequence.original_values is not None
                else []
            ),
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
        if self.create_new_radio.isChecked():
            target_ready = self.new_name_edit.text().strip() != ""
        else:
            target_ready = self._selected_stim_array() is not None
        self.import_button.setEnabled(
            self.sequence is not None
            and bool(get_timed_objects(self.sequence))
            and target_ready
        )

    def _build_new_stim_array(self, name: str, dt: float) -> StimArray:
        """Create an empty array with one neuron column per imported channel."""
        assert self.sequence is not None
        tick_us = self.tick_resolution_spin.value()
        # Probing with a single neuron reports how many channels the file has.
        _, probe = sequence_to_stim_values(
            self.sequence, dt_seconds=dt, n_neurons=1, tick_resolution_us=tick_us
        )
        n_neurons = max(1, probe.grouped_objects)
        stim_type = self.new_type_combo.currentData()
        return StimArray(
            name,
            stim_type if isinstance(stim_type, StimType) else StimType.External,
            np.zeros((0, n_neurons)),
            self.new_units_edit.text().strip(),
            dt,
        )

    def _import_file(self) -> None:
        if self.sequence is None:
            return
        create_new = self.create_new_radio.isChecked()
        stim_array = self._selected_stim_array()
        if not create_new and stim_array is None:
            return
        new_name = self.new_name_edit.text().strip()
        if create_new:
            if not new_name:
                return
            if new_name in self.project.stim_arrays:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Name already in use",
                    f"An input array named '{new_name}' already exists.",
                )
                return

        target_dt = _SYSCON2_TTL_PULSE_QUANTUM_S
        if (
            not create_new
            and stim_array is not None
            and not stim_dt_matches(target_dt, stim_array.dt)
        ):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Change input array timestep",
                f"Importing at {format_milliseconds(target_dt * 1000.0)} ms "
                f"will remap input array '{stim_array.name}' from "
                f"{format_milliseconds(stim_array.dt * 1000.0)} ms "
                "using zero-order hold. Timing may shift by up to one sample. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            # Stage the target so failed or cancelled imports leave it untouched.
            if create_new:
                staged_stim = self._build_new_stim_array(new_name, target_dt)
                result = import_sequence_into_stim_array(
                    staged_stim,
                    self.sequence,
                    tick_resolution_us=self.tick_resolution_spin.value(),
                )
            else:
                assert stim_array is not None
                staged_stim = copy.copy(stim_array)
                staged_stim.values = remap_stim_values_to_dt(
                    stim_array.values,
                    old_dt_seconds=stim_array.dt,
                    new_dt_seconds=target_dt,
                )
                staged_stim.dt = target_dt
                result = import_sequence_into_stim_array(
                    staged_stim,
                    self.sequence,
                    tick_resolution_us=self.tick_resolution_spin.value(),
                    replace_existing=self.replace_checkbox.isChecked(),
                    extend_to_fit=self.extend_checkbox.isChecked(),
                    neuron_offset=self.neuron_offset_spin.value(),
                )
            losses = []
            if result.dropped_objects:
                losses.append(
                    f"{result.dropped_objects} target channels will be dropped."
                )
            if result.truncated_timepoints:
                losses.append(
                    f"{result.truncated_timepoints} timepoints will be truncated."
                )
            if result.truncated_repeats:
                losses.append(
                    f"{result.truncated_repeats} repetitions exceed the import limit."
                )
            if (
                losses
                and QtWidgets.QMessageBox.question(
                    self,
                    "Incomplete import",
                    "\n".join(losses) + "\nImport the remaining data?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                != QtWidgets.QMessageBox.StandardButton.Yes
            ):
                return
            if create_new:
                self.project.add_stim_array(staged_stim)
                self.selected_stim_name = new_name
            else:
                assert stim_array is not None
                self.project.replace_stim_values(
                    stim_array.name,
                    staged_stim.values,
                    staged_stim.dt,
                    label=f"Import SysCon2 sequence into {stim_array.name}",
                )
            self.project.set_syscon2_export_template_text(self._header_template)
            self.project.update_limits()
            self.project.mark_dirty()
            info_lines = [
                (
                    f"Imported into new input array '{staged_stim.name}'."
                    if create_new
                    else f"Imported into '{staged_stim.name}'."
                ),
                f"Eligible timed objects: {result.eligible_objects}",
                "Neuron channels after center-distance grouping: "
                f"{result.grouped_objects}",
                f"Mapped neuron channels into target: {result.mapped_objects}",
                f"Unmapped channels (insufficient target neurons): {result.dropped_objects}",
                f"Generated timepoints: {result.generated_timepoints}",
            ]
            if result.merged_objects > 0:
                info_lines.append(f"Merged repeated targets: {result.merged_objects}")
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
