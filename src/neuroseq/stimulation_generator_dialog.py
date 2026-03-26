from __future__ import annotations

import logging
import pathlib
from enum import Enum
from typing import TYPE_CHECKING

import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtWidgets
from qtpy.compat import getexistingdirectory

from neuroseq.exporter_syscon2 import GlobalSlmWindowViolation
from neuroseq.exporter_syscon2 import InitialSlmLoadViolation
from neuroseq.exporter_syscon2 import SimultaneousDurationViolation
from neuroseq.exporter_syscon2 import SlmGapViolation
from neuroseq.exporter_syscon2 import SlmOffOnOverlapViolation
from neuroseq.exporter_syscon2 import SysconExportReport
from neuroseq.exporter_syscon2 import align_stim_values_left_for_slm
from neuroseq.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.exporter_syscon2 import equalize_simultaneous_durations_to_min
from neuroseq.exporter_syscon2 import find_global_slm_window_violations
from neuroseq.exporter_syscon2 import find_initial_slm_load_violation
from neuroseq.exporter_syscon2 import find_simultaneous_duration_violations
from neuroseq.exporter_syscon2 import find_slm_gap_violations
from neuroseq.exporter_syscon2 import find_slm_off_on_overlap_violations
from neuroseq.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.exporter_syscon2 import shift_stim_values_right_for_initial_slm_load
from neuroseq.exporter_syscon2 import snap_stim_values_to_global_slm_windows
from neuroseq.exporter_syscon2 import space_out_stim_values_for_slm
from neuroseq.exporter_syscon2 import space_out_stim_values_for_slm_off_on_overlap

if TYPE_CHECKING:
    from neuroseq.importer_cellpose import CellposeImportResult
    from neuroseq.project import Project
    from neuroseq.stim import StimArray


class StimulationGeneratorDialog(QtWidgets.QDialog):
    """Dialog to generate hardware equipment scripts for stimulation experiments.

    This dialog allows users to:
    - Select stimulation arrays from the project
    - Choose hardware equipment type
    - Configure export settings
    - Generate and save scripts to a specified folder
    """

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.output_folder: pathlib.Path | None = None

        self.setWindowTitle("Stimulation Script Generator")
        self.setModal(True)
        self.resize(600, 500)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Header description
        description_label = QtWidgets.QLabel(
            "Generate SysCon2 stimulation scripts from stim arrays and Cellpose segmentation."
        )
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        # Stimulation selection group
        stim_group = QtWidgets.QGroupBox("Stimulation Selection")
        stim_layout = QtWidgets.QVBoxLayout()

        stim_select_layout = QtWidgets.QHBoxLayout()
        stim_label = QtWidgets.QLabel("Stimulation array:")
        self.stim_combo = QtWidgets.QComboBox()
        self.stim_combo.setToolTip("Select a stimulation array from the project")

        # Populate with stimulation arrays from project
        stim_names = list(self.project.stim_arrays.keys())
        if stim_names:
            for name in stim_names:
                self.stim_combo.addItem(name)
        else:
            self.stim_combo.addItem("(No stimulation arrays available)")
            self.stim_combo.setEnabled(False)

        stim_select_layout.addWidget(stim_label)
        stim_select_layout.addWidget(self.stim_combo)
        stim_layout.addLayout(stim_select_layout)

        # Stimulation info
        self.stim_info_label = QtWidgets.QLabel("")
        self.stim_info_label.setWordWrap(True)
        self.stim_info_label.setStyleSheet("color: gray; font-size: 10pt;")
        stim_layout.addWidget(self.stim_info_label)

        stim_group.setLayout(stim_layout)
        main_layout.addWidget(stim_group)

        # Update stim info when selection changes
        self.stim_combo.currentTextChanged.connect(self._update_stim_info)
        self._update_stim_info()

        # Hardware equipment group
        hardware_group = QtWidgets.QGroupBox("Hardware Equipment")
        hardware_layout = QtWidgets.QVBoxLayout()

        hardware_select_layout = QtWidgets.QHBoxLayout()
        hardware_label = QtWidgets.QLabel("Equipment type:")
        self.hardware_combo = QtWidgets.QComboBox()
        self.hardware_combo.setToolTip(
            "HOLO4D multiphoton holographic stimulation system"
        )

        # Add HOLO4D hardware equipment type
        self.hardware_combo.addItem("HOLO4D")

        hardware_select_layout.addWidget(hardware_label)
        hardware_select_layout.addWidget(self.hardware_combo)
        hardware_layout.addLayout(hardware_select_layout)

        hardware_group.setLayout(hardware_layout)
        main_layout.addWidget(hardware_group)

        # Script configuration group
        config_group = QtWidgets.QGroupBox("Script Configuration")
        config_layout = QtWidgets.QVBoxLayout()

        # Script name
        name_layout = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel("Script name:")
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText("syscon_sequence_example")
        self.name_edit.setToolTip("Base name for the generated script file")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit)
        config_layout.addLayout(name_layout)

        # HOLO4D Configuration data
        config_data_label = QtWidgets.QLabel("HOLO4D Configuration:")
        config_layout.addWidget(config_data_label)

        self.config_data_edit = QtWidgets.QPlainTextEdit()
        self.config_data_edit.setMaximumHeight(150)
        self.config_data_edit.setToolTip("HOLO4D sequence configuration data")
        default_config = self._default_syscon2_config_text()
        self.config_data_edit.setPlainText(default_config)
        config_layout.addWidget(self.config_data_edit)

        self.binary_mode_checkbox = QtWidgets.QCheckBox(
            "Force binary export intensity (all ON samples -> 1.0)"
        )
        self.binary_mode_checkbox.setChecked(True)
        self.binary_mode_checkbox.setToolTip(
            "When unchecked, exported Shape intensities preserve input stim amplitudes "
            "(clipped to [0, 1])."
        )
        config_layout.addWidget(self.binary_mode_checkbox)

        laser_layout = QtWidgets.QHBoxLayout()
        laser_label = QtWidgets.QLabel("Laser timeline:")
        self.laser_combo = QtWidgets.QComboBox()
        self.laser_combo.addItem("940 nm digital", 940)
        self.laser_combo.addItem("1030 nm digital", 1030)
        self.laser_combo.addItem("1140 nm digital", 1140)
        self.laser_combo.setCurrentIndex(self.laser_combo.findData(1140))
        self.laser_combo.setToolTip(
            "Select which fixed laser timeline receives stimulation TTLPulses."
        )
        laser_layout.addWidget(laser_label)
        laser_layout.addWidget(self.laser_combo)
        config_layout.addLayout(laser_layout)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # Output folder selection
        folder_group = QtWidgets.QGroupBox("Output Location")
        folder_layout = QtWidgets.QVBoxLayout()

        folder_select_layout = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Select output folder...")
        self.folder_edit.setReadOnly(True)
        self.folder_button = QtWidgets.QPushButton()
        self.folder_button.setIcon(qta.icon("mdi6.folder-open"))
        self.folder_button.setToolTip("Browse for output folder")
        self.folder_button.clicked.connect(self._select_output_folder)

        folder_select_layout.addWidget(self.folder_edit)
        folder_select_layout.addWidget(self.folder_button)
        folder_layout.addLayout(folder_select_layout)

        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.check_sanity_button = QtWidgets.QPushButton("Check Input Sanity")
        self.check_sanity_button.setIcon(qta.icon("mdi6.clipboard-check-outline"))
        self.check_sanity_button.clicked.connect(self._check_input_sanity)

        button_layout.addWidget(self.check_sanity_button)
        button_layout.addStretch()

        self.generate_button = QtWidgets.QPushButton("Generate Scripts")
        self.generate_button.setIcon(qta.icon("mdi6.file-code"))
        self.generate_button.clicked.connect(self._generate_scripts)
        self.generate_button.setEnabled(
            False
        )  # Disabled until output folder is selected

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.reject)

        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)

    class _SlmFixChoice(Enum):
        SPACE_OUT = "space_out"
        ALIGN_LEFT = "align_left"
        CANCEL = "cancel"

    class _SlmOffOnFixChoice(Enum):
        SPACE_RIGHT = "space_right"
        CANCEL = "cancel"

    class _DurationFixChoice(Enum):
        EQUALIZE_MIN = "equalize_min"
        CANCEL = "cancel"

    class _InitialSlmFixChoice(Enum):
        SHIFT_RIGHT = "shift_right"
        CANCEL = "cancel"

    class _GlobalWindowFixChoice(Enum):
        SNAP = "snap"
        CONTINUE = "continue"
        CANCEL = "cancel"

    def _update_stim_info(self) -> None:
        """Update the stimulation info label based on current selection."""
        stim_name = self.stim_combo.currentText()
        if not stim_name or stim_name == "(No stimulation arrays available)":
            self.stim_info_label.setText("")
            return

        if stim_name in self.project.stim_arrays:
            stim_array = self.project.stim_arrays[stim_name]
            info_text = (
                f"Type: {stim_array.stim_type.name if hasattr(stim_array.stim_type, 'name') else str(stim_array.stim_type)} | "
                f"Shape: {stim_array.values.shape} | "
                f"Units: {stim_array.units} | "
                f"dt: {stim_array.dt}s"
            )
            self.stim_info_label.setText(info_text)
        else:
            self.stim_info_label.setText("")

    def _select_output_folder(self) -> None:
        """Open a dialog to select the output folder."""
        folder = getexistingdirectory(
            self,
            "Select Output Folder",
            "",
        )

        if folder:
            self.output_folder = pathlib.Path(folder)
            self.folder_edit.setText(str(self.output_folder))
            self.generate_button.setEnabled(True)
            self._log_status(f"Output folder set to: {self.output_folder}")

    def _log_status(self, message: str, *args: object) -> None:
        """Log a status message."""
        self.logger.info(message, *args)

    def _generate_scripts(self) -> None:
        """Generate the hardware control scripts."""
        if not self.output_folder:
            QtWidgets.QMessageBox.warning(
                self,
                "No Output Folder",
                "Please select an output folder first.",
            )
            return

        stim_name = self.stim_combo.currentText()
        if not stim_name or stim_name == "(No stimulation arrays available)":
            QtWidgets.QMessageBox.warning(
                self,
                "No Stimulation Selected",
                "Please select a valid stimulation array.",
            )
            return

        try:
            self._log_status(f"Generating scripts for: {stim_name}")

            # Get stimulation array
            if stim_name not in self.project.stim_arrays:
                raise ValueError(
                    f"Stimulation array '{stim_name}' not found in project"
                )

            stim_array = self.project.stim_arrays[stim_name]
            hardware_type = self.hardware_combo.currentText()
            script_name = self.name_edit.text() or "stimulation_script"

            # Generate script based on hardware type
            script_content, export_report = self._generate_script_content(
                stim_array=stim_array,
                stim_name=stim_name,
                hardware_type=hardware_type,
                cellpose_data=self.project.cellpose_data,
            )

            # Determine file extension based on hardware type
            extension = self._get_file_extension(hardware_type)
            script_filename = f"{script_name}{extension}"
            script_path = self.output_folder / script_filename
            overview_filename = f"{script_name}_overview.png"
            overview_path = self.output_folder / overview_filename

            # Write script file
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            generate_syscon2_timeline_overview_png(
                script_content,
                overview_path,
            )

            self._log_status(f"✓ Generated: {script_filename}")
            self._log_status(f"✓ Generated: {overview_filename}")
            self._log_status(
                "Export summary: segments=%d mapped_active_neurons=%d "
                "unmapped_active_neurons=%d entity_lanes=%d objects=%d",
                export_report.exported_segments,
                export_report.mapped_active_neurons,
                len(export_report.unmapped_active_neurons),
                export_report.entity_lanes,
                export_report.generated_objects,
            )

            report_lines = [
                f"Successfully generated script in:\n{self.output_folder}",
                "",
                f"Generated files: {script_filename}, {overview_filename}",
                "",
                f"Exported segments: {export_report.exported_segments}/{export_report.total_segments}",
                "Mapped active neurons: "
                f"{export_report.mapped_active_neurons}/{export_report.active_neurons}",
                f"Entity timeline lanes: {export_report.entity_lanes}",
                f"Generated object blocks: {export_report.generated_objects}",
            ]
            if export_report.unmapped_active_neurons:
                report_lines.append(
                    "Skipped active neurons (unmapped): "
                    + ", ".join(
                        str(idx) for idx in export_report.unmapped_active_neurons
                    )
                )

            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Scripts Generated",
                "\n".join(report_lines),
            )

        except Exception as e:
            self.logger.error(f"Error generating scripts: {e}", exc_info=True)
            self._log_status(f"✗ Error: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Generation Failed",
                f"Failed to generate a script:\n{str(e)}",
            )

    def _check_input_sanity(self) -> None:
        """Run a dry-run sanity check for SysCon2 translation inputs."""
        stim_name = self.stim_combo.currentText()
        if not stim_name or stim_name == "(No stimulation arrays available)":
            QtWidgets.QMessageBox.warning(
                self,
                "No Stimulation Selected",
                "Please select a valid stimulation array.",
            )
            return
        if stim_name not in self.project.stim_arrays:
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                f"Stimulation array '{stim_name}' not found in project.",
            )
            return

        stim_array = self.project.stim_arrays[stim_name]
        max_fix_iterations = max(1, stim_array.values.shape[0] * 2)
        skip_global_window_autofix = False
        for _ in range(max_fix_iterations):
            if not skip_global_window_autofix:
                global_window_violations = find_global_slm_window_violations(
                    stim_array,
                    active_threshold=0.0,
                )
                if global_window_violations:
                    global_window_choice = self._prompt_global_slm_window_fix_choice(
                        global_window_violations,
                    )
                    if global_window_choice == self._GlobalWindowFixChoice.CANCEL:
                        return
                    if global_window_choice == self._GlobalWindowFixChoice.SNAP:
                        (
                            new_values,
                            moved_segments,
                            truncated_tail_timepoints,
                        ) = snap_stim_values_to_global_slm_windows(
                            stim_array,
                            active_threshold=0.0,
                        )
                        changed = self._replace_stim_values_preserving_undo(
                            stim_name,
                            new_values,
                        )
                        if not changed:
                            return
                        self._log_status(
                            "Global 16.66ms window snap moved %d segment(s).",
                            moved_segments,
                        )
                        if truncated_tail_timepoints > 0:
                            self._log_status(
                                "Global-window snap truncated %d tail timepoints.",
                                truncated_tail_timepoints,
                            )
                        stim_array = self.project.stim_arrays[stim_name]
                        continue
                    skip_global_window_autofix = True

            duration_violations = find_simultaneous_duration_violations(
                stim_array,
                active_threshold=0.0,
            )
            if duration_violations:
                duration_choice = self._prompt_duration_fix_choice(
                    duration_violations,
                    dt_seconds=stim_array.dt,
                )
                if duration_choice == self._DurationFixChoice.CANCEL:
                    return

                new_values, trimmed_timepoints = equalize_simultaneous_durations_to_min(
                    stim_array,
                    active_threshold=0.0,
                )
                changed = self._replace_stim_values_preserving_undo(
                    stim_name,
                    new_values,
                )
                if not changed:
                    return
                self._log_status(
                    "Simultaneous-duration equalization trimmed %d sample(s).",
                    trimmed_timepoints,
                )
                stim_array = self.project.stim_arrays[stim_name]
                continue

            initial_slm_load_violation = find_initial_slm_load_violation(
                stim_array,
                active_threshold=0.0,
            )
            if initial_slm_load_violation is not None:
                choice = self._prompt_initial_slm_start_fix_choice(
                    initial_slm_load_violation,
                    dt_seconds=stim_array.dt,
                )
                if choice == self._InitialSlmFixChoice.CANCEL:
                    return

                new_values, truncated_tail_samples = (
                    shift_stim_values_right_for_initial_slm_load(
                        stim_array,
                    )
                )
                changed = self._replace_stim_values_preserving_undo(
                    stim_name,
                    new_values,
                )
                if not changed:
                    return
                if truncated_tail_samples > 0:
                    self._log_status(
                        "Initial SLM-load shift truncated %d tail timepoints.",
                        truncated_tail_samples,
                    )
                stim_array = self.project.stim_arrays[stim_name]
                continue

            off_on_overlap_violations = find_slm_off_on_overlap_violations(
                stim_array,
                active_threshold=0.0,
            )
            if off_on_overlap_violations:
                off_on_choice = self._prompt_slm_off_on_overlap_fix_choice(
                    off_on_overlap_violations,
                    dt_seconds=stim_array.dt,
                )
                if off_on_choice == self._SlmOffOnFixChoice.CANCEL:
                    return

                new_values, truncated_tail_samples = (
                    space_out_stim_values_for_slm_off_on_overlap(
                        stim_array,
                        active_threshold=0.0,
                    )
                )
                changed = self._replace_stim_values_preserving_undo(
                    stim_name,
                    new_values,
                )
                if not changed:
                    return
                if truncated_tail_samples > 0:
                    self._log_status(
                        "SLM off/on-overlap spacing fix truncated %d tail timepoints.",
                        truncated_tail_samples,
                    )
                stim_array = self.project.stim_arrays[stim_name]
                continue

            slm_gap_violations = find_slm_gap_violations(
                stim_array,
                active_threshold=0.0,
            )
            if slm_gap_violations:
                slm_fix_choice = self._prompt_slm_gap_fix_choice(
                    slm_gap_violations,
                    dt_seconds=stim_array.dt,
                )
                if slm_fix_choice == self._SlmFixChoice.CANCEL:
                    return

                if slm_fix_choice == self._SlmFixChoice.SPACE_OUT:
                    new_values, truncated_tail_samples = space_out_stim_values_for_slm(
                        stim_array,
                        active_threshold=0.0,
                    )
                    changed = self._replace_stim_values_preserving_undo(
                        stim_name,
                        new_values,
                    )
                    if not changed:
                        return
                    if truncated_tail_samples > 0:
                        self._log_status(
                            "SLM spacing fix truncated %d tail timepoints.",
                            truncated_tail_samples,
                        )
                else:
                    new_values, moved_runs = align_stim_values_left_for_slm(
                        stim_array,
                        active_threshold=0.0,
                    )
                    changed = self._replace_stim_values_preserving_undo(
                        stim_name,
                        new_values,
                    )
                    if not changed:
                        return
                    self._log_status(
                        "SLM align-left fix moved %d violating run(s).",
                        moved_runs,
                    )

                stim_array = self.project.stim_arrays[stim_name]
                continue

            break
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                "Automatic sanity-fix sequence did not converge.\n"
                "Please adjust the stimulus manually and retry.",
            )
            return

        remaining_duration_violations = find_simultaneous_duration_violations(
            stim_array,
            active_threshold=0.0,
        )
        remaining_initial_violation = find_initial_slm_load_violation(
            stim_array,
            active_threshold=0.0,
        )
        remaining_off_on_overlap_violations = find_slm_off_on_overlap_violations(
            stim_array,
            active_threshold=0.0,
        )
        remaining_slm_gap_violations = find_slm_gap_violations(
            stim_array,
            active_threshold=0.0,
        )
        if (
            remaining_duration_violations
            or remaining_initial_violation is not None
            or remaining_off_on_overlap_violations
            or remaining_slm_gap_violations
        ):
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                "At least one sanity violation remains after applying selected fixes.\n"
                "Please adjust the stimulus manually and retry.",
            )
            return

        force_binary_intensity = self.binary_mode_checkbox.isChecked()
        sanity_report = check_syscon2_input_sanity(
            stim_array,
            self.project.cellpose_data,
            template_text=self.config_data_edit.toPlainText(),
            tick_resolution_us=500,
            active_threshold=0.0,
            binary_intensity=1.0,
            force_binary_intensity=force_binary_intensity,
            active_laser_wavelength_nm=self._selected_laser_wavelength_nm(),
        )

        if not sanity_report.is_valid:
            error_lines = ["Input sanity check failed."]
            if sanity_report.errors:
                error_lines.extend(["", "Errors:", *sanity_report.errors])
            if sanity_report.warnings:
                error_lines.extend(["", "Warnings:", *sanity_report.warnings])
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                "\n".join(error_lines),
            )
            return

        assert sanity_report.export_report is not None
        export_report = sanity_report.export_report
        success_lines = [
            "Input sanity check passed.",
            "",
            f"Exported segments: {export_report.exported_segments}/{export_report.total_segments}",
            "Mapped active neurons: "
            f"{export_report.mapped_active_neurons}/{export_report.active_neurons}",
            f"Entity timeline lanes: {export_report.entity_lanes}",
            f"Generated object blocks: {export_report.generated_objects}",
        ]
        if sanity_report.warnings:
            success_lines.extend(["", "Warnings:", *sanity_report.warnings])
            QtWidgets.QMessageBox.warning(
                self,
                "Sanity Check Passed With Warnings",
                "\n".join(success_lines),
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            "Sanity Check Passed",
            "\n".join(success_lines),
        )

    def _prompt_global_slm_window_fix_choice(
        self,
        violations: list[GlobalSlmWindowViolation],
    ) -> _GlobalWindowFixChoice:
        first_violation = violations[0]
        start_ms = first_violation.start_tick * 520.8125 / 1e3
        snapped_start_ms = first_violation.snapped_start_tick * 520.8125 / 1e3
        shift_ms = first_violation.shift_ticks * 520.8125 / 1e3
        text_lines = [
            (
                "Detected stimulation starts outside global 16.66ms SLM windows "
                "(sub-ms SysCon2 alignment grid)."
            ),
            "",
            f"Violations detected: {len(violations)}",
            (
                "Example: "
                f"neuron {first_violation.neuron_index}, "
                f"start={start_ms:.3f}ms -> nearest allowed={snapped_start_ms:.3f}ms "
                f"(delta {shift_ms:+.3f}ms)"
            ),
            "",
            "Choose how to proceed:",
            "1) Snap starts to nearest allowed global window (optional auto-fix)",
            "2) Continue without this fix",
            "3) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Global 16.66ms Window Misalignment Detected")
        msg_box.setText("\n".join(text_lines))

        snap_button = msg_box.addButton(
            "Snap To Allowed Windows",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        continue_button = msg_box.addButton(
            "Continue Without Fix",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(continue_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == snap_button:
            return self._GlobalWindowFixChoice.SNAP
        if clicked == continue_button:
            return self._GlobalWindowFixChoice.CONTINUE
        return self._GlobalWindowFixChoice.CANCEL

    def _prompt_slm_off_on_overlap_fix_choice(
        self,
        violations: list[SlmOffOnOverlapViolation],
        *,
        dt_seconds: float,
    ) -> _SlmOffOnFixChoice:
        first_violation = violations[0]
        required_shift_ms = first_violation.required_shift_samples * dt_seconds * 1e3
        overlap_start_ms = first_violation.overlap_start_tick * 520.8125 / 1e3
        overlap_end_ms = first_violation.overlap_end_tick * 520.8125 / 1e3
        text_lines = [
            (
                "Detected SLM off/on overlap: the OFF phase of a newly started neuron "
                "object overlaps the ON phase of another neuron object."
            ),
            "",
            f"Violations detected: {len(violations)}",
            (
                "Example: "
                f"off-neuron {first_violation.off_neuron_index} overlaps "
                f"ON-neuron {first_violation.on_neuron_index} in "
                f"[{overlap_start_ms:.2f}, {overlap_end_ms:.2f})ms"
            ),
            f"Proposed right spacing: {required_shift_ms:.2f}ms",
            "",
            "Choose how to resolve:",
            f"1) Space input right by {required_shift_ms:.2f} ms",
            "2) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("SLM OFF/ON Overlap Detected")
        msg_box.setText("\n".join(text_lines))

        space_button = msg_box.addButton(
            "Space Input Right",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == space_button:
            return self._SlmOffOnFixChoice.SPACE_RIGHT
        return self._SlmOffOnFixChoice.CANCEL

    def _prompt_slm_gap_fix_choice(
        self,
        violations: list[SlmGapViolation],
        *,
        dt_seconds: float,
    ) -> _SlmFixChoice:
        first_violation = violations[0]
        gap_ms = first_violation.gap_samples * dt_seconds * 1e3
        required_gap_ms = first_violation.required_gap_samples * dt_seconds * 1e3
        text_lines = [
            (
                "Detected stimulation transitions that violate the SLM load-time rule "
                "(minimum 16.66 ms whenever a new neuron target appears)."
            ),
            "",
            f"Violations detected: {len(violations)}",
            (
                "Example: "
                f"{first_violation.previous_neuron_indices} -> "
                f"{first_violation.current_neuron_indices}, "
                f"gap={gap_ms:.2f}ms (required {required_gap_ms:.2f}ms)"
            ),
            "",
            "Choose how to resolve:",
            "1) Space input out (by 16.66 ms)",
            "2) Align troublesome stimuli (to the left)",
            "3) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("SLM Gap Violations Detected")
        msg_box.setText("\n".join(text_lines))

        space_button = msg_box.addButton(
            "Space Input Out",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        align_button = msg_box.addButton(
            "Align Troublesome Stimuli (Left)",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == space_button:
            return self._SlmFixChoice.SPACE_OUT
        if clicked == align_button:
            return self._SlmFixChoice.ALIGN_LEFT
        return self._SlmFixChoice.CANCEL

    def _prompt_duration_fix_choice(
        self,
        violations: list[SimultaneousDurationViolation],
        *,
        dt_seconds: float,
    ) -> _DurationFixChoice:
        first_violation = violations[0]
        start_ms = first_violation.start_idx * dt_seconds * 1e3
        duration_summary = ", ".join(
            f"neuron {neuron_idx}: {duration_idx * dt_seconds * 1e3:.2f}ms"
            for neuron_idx, duration_idx in zip(
                first_violation.neuron_indices,
                first_violation.durations_idx,
                strict=False,
            )
        )
        min_duration_ms = first_violation.min_duration_idx * dt_seconds * 1e3

        text_lines = [
            (
                "Detected simultaneous stimulation events with unequal durations. "
                "All neurons stimulated together must have equal duration."
            ),
            "",
            f"Violations detected: {len(violations)}",
            f"Example at t={start_ms:.2f}ms: {duration_summary}",
            f"Minimum duration in example: {min_duration_ms:.2f}ms",
            "",
            "Choose how to resolve:",
            "1) Equalize (to lowest value)",
            "2) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Unequal Simultaneous Durations Detected")
        msg_box.setText("\n".join(text_lines))

        equalize_button = msg_box.addButton(
            "Equalize (to lowest value)",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == equalize_button:
            return self._DurationFixChoice.EQUALIZE_MIN
        return self._DurationFixChoice.CANCEL

    def _prompt_initial_slm_start_fix_choice(
        self,
        violation: InitialSlmLoadViolation,
        *,
        dt_seconds: float,
    ) -> _InitialSlmFixChoice:
        first_start_ms = violation.first_active_idx * dt_seconds * 1e3
        required_start_ms = violation.required_start_idx * dt_seconds * 1e3
        shift_ms = required_start_ms
        text_lines = [
            (
                "Detected stimulation too close to experiment start. "
                "SysCon2 requires at least 16.66 ms before first stimulation "
                "for initial SLM load-up."
            ),
            "",
            f"First stimulation starts at: {first_start_ms:.2f}ms",
            f"Required minimum start: {required_start_ms:.2f}ms",
            "",
            "Choose how to resolve:",
            f"1) Translate all inputs right by {shift_ms:.2f} ms",
            "2) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Initial SLM Load-Up Violation Detected")
        msg_box.setText("\n".join(text_lines))

        shift_button = msg_box.addButton(
            "Translate All Inputs Right",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == shift_button:
            return self._InitialSlmFixChoice.SHIFT_RIGHT
        return self._InitialSlmFixChoice.CANCEL

    def _replace_stim_values_preserving_undo(
        self,
        stim_name: str,
        new_values: np.ndarray,
    ) -> bool:
        if stim_name not in self.project.stim_arrays:
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                f"Stimulation array '{stim_name}' not found in project.",
            )
            return False

        try:
            parent_widget = self.parentWidget()
            stim_editor = (
                getattr(parent_widget, "stim_editor", None)
                if parent_widget is not None
                else None
            )
            if stim_editor is not None and hasattr(
                stim_editor, "replace_stim_values_with_undo"
            ):
                stim_editor.replace_stim_values_with_undo(stim_name, new_values)
            else:
                target = self.project.stim_arrays[stim_name]
                if target.values.shape != new_values.shape:
                    raise ValueError(
                        "Automatic sanity fixes require matching stim-array shape."
                    )
                target.values[:, :] = new_values
            return True
        except Exception as exc:
            self.logger.error("Failed to apply sanity fix: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                f"Could not apply automatic sanity fix:\n{exc}",
            )
            return False

    def _get_file_extension(self, hardware_type: str) -> str:
        """Get the appropriate file extension for the hardware type."""
        return ".seq"

    def _generate_script_content(
        self,
        stim_array: StimArray,
        stim_name: str,
        hardware_type: str,
        cellpose_data: CellposeImportResult | None,
    ) -> tuple[str, SysconExportReport]:
        """Generate the script content based on hardware type."""
        if hardware_type != "HOLO4D":
            raise ValueError(f"Unsupported hardware type: {hardware_type}")
        return self._generate_holo4d_script(stim_array, stim_name, cellpose_data)

    def _generate_holo4d_script(
        self,
        stim_array: StimArray,
        stim_name: str,
        cellpose_data: CellposeImportResult | None,
    ) -> tuple[str, SysconExportReport]:
        """Generate HOLO4D SysCon2 sequence from stim-array/cellpose data."""
        del (
            stim_name
        )  # currently used only for UI; exporter works on selected StimArray
        config_data = self.config_data_edit.toPlainText()
        return generate_syscon2_sequence_text(
            stim_array,
            cellpose_data,
            template_text=config_data,
            tick_resolution_us=500,
            active_threshold=0.0,
            binary_intensity=1.0,
            force_binary_intensity=self.binary_mode_checkbox.isChecked(),
            active_laser_wavelength_nm=self._selected_laser_wavelength_nm(),
        )

    def _selected_laser_wavelength_nm(self) -> int:
        current_data = self.laser_combo.currentData()
        if current_data is None:
            return 1140
        return int(current_data)

    def _default_syscon2_config_text(self) -> str:
        imported_template = getattr(
            self.project,
            "syscon2_export_template_text",
            None,
        )
        if imported_template:
            return imported_template
        return """nohash
version=3
scanmode=Accurate
runs=1
invertedportcnt=1
inverted0=dummy_1_0 RMI In
AutoTriggerRulesEnabled=False
AutoTriggerRulesCount=1
SourceConnectorPort0=dummy_1_0 RMI In
TargetConnectorPort0=Holo4D TTL Out 2
PreTiming_us0=20000
PostTiming_us0=20000
timelinezoomfactor=10
timelineviewposx=0
timelineviewposy=0
"""
