from __future__ import annotations

import contextlib
import datetime
import logging
import pathlib
import re
from collections.abc import Callable
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getexistingdirectory

from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.export.exporter_syscon2 import InitialSlmLoadViolation
from neuroseq.export.exporter_syscon2 import SimultaneousDurationViolation
from neuroseq.export.exporter_syscon2 import SlmGapViolation
from neuroseq.export.exporter_syscon2 import SlmOffOnOverlapViolation
from neuroseq.export.exporter_syscon2 import SysconExportReport
from neuroseq.export.exporter_syscon2 import SysconInputSanityReport
from neuroseq.export.exporter_syscon2 import align_overlapping_pulses_to_shortest
from neuroseq.export.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.export.exporter_syscon2 import equalize_simultaneous_durations_to_min
from neuroseq.export.exporter_syscon2 import find_initial_slm_load_violation
from neuroseq.export.exporter_syscon2 import find_overlapping_pulse_group_violations
from neuroseq.export.exporter_syscon2 import find_simultaneous_duration_violations
from neuroseq.export.exporter_syscon2 import find_slm_gap_violations
from neuroseq.export.exporter_syscon2 import find_slm_off_on_overlap_violations
from neuroseq.export.exporter_syscon2 import resample_stim_values_to_dt
from neuroseq.export.exporter_syscon2 import (
    shift_stim_values_right_for_initial_slm_load,
)
from neuroseq.export.exporter_syscon2 import space_out_stim_values_for_slm
from neuroseq.export.exporter_syscon2 import (
    space_out_stim_values_for_slm_off_on_overlap,
)
from neuroseq.export.power_lut import PowerLut
from neuroseq.export.power_lut import parse_power_lut_text
from neuroseq.export.power_lut import resolve_power_lut
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.export.syscon2_export_service import default_syscon2_config_text
from neuroseq.export.syscon2_export_service import export_stim_array
from neuroseq.export.syscon2_export_service import generate_sequence_text
from neuroseq.export.syscon2_export_service import validate_export_name
from neuroseq.icon_manager import icons

if TYPE_CHECKING:
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray


_EXPORT_SUFFIXES = (".seq", "_original_stim.npz", "_overview.png")


def next_default_export_name(
    folder: pathlib.Path | None,
    today: datetime.date | None = None,
    reserved: Iterable[str] = (),
) -> str:
    """``export_seq_<YYYY-MM-DD>_<n>``, with ``n`` one past the highest number already
    used today.

    Any file of an export (``.seq``, ``_overview.png``, ``_original_stim.npz``) in
    ``folder`` counts, as do ``reserved`` file names (e.g. sequences already in the
    stimulation setup's sequence manager, with or without a folder part). Matching is
    case-insensitive, as on Windows, so a default name never collides.
    """
    day = (today or datetime.date.today()).isoformat()
    prefix = f"export_seq_{day}_"
    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d+)(?:\.seq|_export_debug\.csv|_original_stim\.npz|_overview\.png)$",
        re.IGNORECASE,
    )
    names: list[str] = [
        str(name).replace("\\", "/").rsplit("/", 1)[-1] for name in reserved
    ]
    if folder is not None:
        with contextlib.suppress(OSError):
            names.extend(path.name for path in folder.iterdir())
    highest = 0
    for name in names:
        match = pattern.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1}"


class StimulationGeneratorDialog(QtWidgets.QDialog):
    """Dialog to generate hardware equipment scripts for stimulation experiments.

    This dialog allows users to:
    - Select stimulation arrays from the project
    - Choose hardware equipment type
    - Configure export settings
    - Generate and save scripts to a specified folder
    """

    #: Emitted after a script is successfully written to disk, with the generated
    #: ``.seq`` file path, its timeline overview PNG path, and the frozenset[int] of
    #: stim ids actually included in the exported sequence (excluding skipped neurons) -- lets a caller (e.g. the SysCon2 panel) act on the
    #: freshly generated files, and know who they target, without polling the
    #: filesystem or re-deriving it from the array itself.
    sequence_generated = QtCore.Signal(str, str, object)

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
        current_stim_name: str | None = None,
        output_folder: pathlib.Path | None = None,
    ) -> None:
        """`current_stim_name` is the input array the user has open in the
        editor; it is preselected so exporting doesn't default to the first
        array created (#588). Ignored if it isn't one of the project's.
        `output_folder` is where the scripts go unless the user picks another
        folder (issue #614)."""
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.output_folder: pathlib.Path | None = None
        #: Extra file names a default export name must avoid (e.g. remote sequences).
        self.reserved_names: Callable[[], Iterable[str]] = lambda: ()

        self.setWindowTitle("Input Script Generator")
        self.setModal(True)
        self.resize(600, 500)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Header description
        description_label = QtWidgets.QLabel(
            "Generate SysCon2 input scripts from input arrays and Cellpose segmentation."
        )
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        # Stimulation selection group
        stim_group = QtWidgets.QGroupBox("Input Selection")
        stim_layout = QtWidgets.QVBoxLayout()

        stim_select_layout = QtWidgets.QHBoxLayout()
        stim_label = QtWidgets.QLabel("Input array:")
        self.stim_combo = QtWidgets.QComboBox()
        self.stim_combo.setToolTip("Select an input array from the project")

        # Populate with stimulation arrays from project
        stim_names = list(self.project.stim_arrays.keys())
        if stim_names:
            for name in stim_names:
                self.stim_combo.addItem(name)
            if current_stim_name in stim_names:
                self.stim_combo.setCurrentText(current_stim_name)
        else:
            self.stim_combo.addItem("(No input arrays available)")
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
        self._auto_name = next_default_export_name(None)
        self.name_edit.setText(self._auto_name)
        self.name_edit.setToolTip("Base name for the generated script file")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit)
        config_layout.addLayout(name_layout)

        self.binary_mode_checkbox = QtWidgets.QCheckBox(
            "Force binary export intensity (all ON samples -> 1.0)"
        )
        self.binary_mode_checkbox.setChecked(False)
        self.binary_mode_checkbox.setToolTip(
            "When unchecked, export preprocessing preserves the array's original "
            "amplitudes (clipped to [0, 1]) instead of normalizing ON samples to 1.0."
        )
        config_layout.addWidget(self.binary_mode_checkbox)

        self.spiralling_checkbox = QtWidgets.QCheckBox(
            "Spiralling (TTL fires 1 tick before SLM load completes)"
        )
        self.spiralling_checkbox.setChecked(False)
        self.spiralling_checkbox.setToolTip(
            "When checked, every laser TTL pulse is shifted 1 tick earlier so the "
            "laser edge aligns with the SLM-ready edge in spiralling mode. Uncheck "
            "for legacy non-spiralling hardware: TTL fires exactly at SLM-ready "
            "(start tick + 16.666 ms)."
        )
        config_layout.addWidget(self.spiralling_checkbox)

        self.exact_timing_checkbox = QtWidgets.QCheckBox(
            "Exact stimulus timing (shapes are not snapped to 16.7 ms windows)"
        )
        self.exact_timing_checkbox.setChecked(True)
        self.exact_timing_checkbox.setToolTip(
            "Place each hologram shape one SLM load window before its first pulse "
            "and delay stimuli only where the load would not fit. Uncheck to snap "
            "shapes to the 16.7 ms window grid (stimuli may be delayed by up to a "
            "window per hologram)."
        )
        config_layout.addWidget(self.exact_timing_checkbox)

        self.ten_spots_per_roi_checkbox = QtWidgets.QCheckBox(
            "Use 10 input spots per ROI"
        )
        self.ten_spots_per_roi_checkbox.setChecked(True)
        self.ten_spots_per_roi_checkbox.setToolTip(
            "Place 10 evenly distributed input spots inside each Cellpose mask."
        )
        config_layout.addWidget(self.ten_spots_per_roi_checkbox)

        spot_algorithm_layout = QtWidgets.QHBoxLayout()
        spot_algorithm_layout.addWidget(QtWidgets.QLabel("10-spot placement:"))
        self.spot_algorithm_combo = QtWidgets.QComboBox()
        self.spot_algorithm_combo.addItem(
            "Farthest apart (may use ROI border)", "farthest_point"
        )
        self.spot_algorithm_combo.addItem(
            "Equal spacing from ROI border", "edge_balanced"
        )
        self.spot_algorithm_combo.setEnabled(
            self.ten_spots_per_roi_checkbox.isChecked()
        )
        self.spot_algorithm_combo.setToolTip(
            "Choose whether spacing is maximized only between spots or also "
            "balanced against the ROI border."
        )
        self.ten_spots_per_roi_checkbox.toggled.connect(
            self.spot_algorithm_combo.setEnabled
        )
        spot_algorithm_layout.addWidget(self.spot_algorithm_combo)
        config_layout.addLayout(spot_algorithm_layout)

        laser_layout = QtWidgets.QHBoxLayout()
        laser_label = QtWidgets.QLabel("Laser timeline:")
        self.laser_combo = QtWidgets.QComboBox()
        self.laser_combo.addItem("940 nm digital", 940)
        self.laser_combo.addItem("1030 nm digital", 1030)
        self.laser_combo.addItem("1100 nm digital", 1100)
        self.laser_combo.setCurrentIndex(self.laser_combo.findData(940))
        self.laser_combo.setToolTip(
            "Select which fixed laser timeline receives input TTLPulses."
        )
        laser_layout.addWidget(laser_label)
        laser_layout.addWidget(self.laser_combo)
        config_layout.addLayout(laser_layout)

        power_layout = QtWidgets.QHBoxLayout()
        self.target_power_spin = QtWidgets.QDoubleSpinBox()
        self.target_power_spin.setRange(0.0, 1000.0)
        self.target_power_spin.setDecimals(2)
        self.target_power_spin.setSuffix(" mW")
        self.target_power_spin.setSingleStep(0.5)
        self.target_power_spin.setValue(1.0)
        self.target_power_spin.setToolTip(
            "Desired delivered power. Set to 0 to disable power control "
            "(Intensity_0 fields written as 0 — current default behavior)."
        )
        self.load_lut_button = QtWidgets.QPushButton("Load LUT…")
        self.load_lut_button.setToolTip(
            "Load a SysCon .lut file for the currently selected laser wavelength. "
            "The LUT is bundled with the project on save."
        )
        self.load_lut_button.clicked.connect(self._on_load_lut_clicked)
        self.lut_status_label = QtWidgets.QLabel(self._format_lut_status())
        self.lut_status_label.setStyleSheet("color: gray; font-style: italic;")
        power_layout.addWidget(QtWidgets.QLabel("Target power:"))
        power_layout.addWidget(self.target_power_spin)
        power_layout.addWidget(self.load_lut_button)
        power_layout.addWidget(self.lut_status_label, stretch=1)
        config_layout.addLayout(power_layout)
        self.laser_combo.currentIndexChanged.connect(
            lambda _index: self.lut_status_label.setText(self._format_lut_status())
        )

        # HOLO4D Configuration data
        config_data_label = QtWidgets.QLabel("HOLO4D Configuration:")
        config_layout.addWidget(config_data_label)

        self.config_data_edit = QtWidgets.QPlainTextEdit()
        self.config_data_edit.setMaximumHeight(150)
        self.config_data_edit.setToolTip(
            "HOLO4D sequence configuration data. Note: the laser/lightsource "
            "ID is always recomputed from the laser selected above, and "
            "invertedportcnt is always forced to 0 (inverted pulses are "
            "never wanted, so inverted0 is dropped) - overriding whatever "
            "is typed here for those specific fields."
        )
        default_config = self._default_syscon2_config_text()
        self.config_data_edit.setPlainText(default_config)
        config_layout.addWidget(self.config_data_edit)

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
        icons.set_widget_icon(self.folder_button, "mdi6.folder-open")
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
        icons.set_widget_icon(self.check_sanity_button, "mdi6.clipboard-check-outline")
        self.check_sanity_button.clicked.connect(self._check_input_sanity)

        button_layout.addWidget(self.check_sanity_button)
        button_layout.addStretch()

        self.generate_button = QtWidgets.QPushButton("Generate Scripts")
        icons.set_widget_icon(self.generate_button, "mdi6.file-code")
        self.generate_button.clicked.connect(self._generate_scripts)
        self.generate_button.setEnabled(
            False
        )  # Disabled until output folder is selected

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.reject)

        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)
        if output_folder is not None:
            self._set_output_folder(output_folder)

    class _SlmFixChoice(Enum):
        SPACE_OUT = "space_out"
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

    class _DtFixChoice(Enum):
        CLEAR = "clear"
        RESAMPLE = "resample"
        CANCEL = "cancel"

    def _update_stim_info(self) -> None:
        """Update the stimulation info label based on current selection."""
        stim_name = self.stim_combo.currentText()
        if not stim_name or stim_name == "(No input arrays available)":
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

    def _refresh_default_name(self) -> None:
        """Advance the default script name, unless the user typed their own."""
        if self.name_edit.text().strip() in ("", self._auto_name):
            self._auto_name = next_default_export_name(
                self.output_folder, reserved=self.reserved_names()
            )
            self.name_edit.setText(self._auto_name)

    def _select_output_folder(self) -> None:
        """Open a dialog to select the output folder."""
        folder = getexistingdirectory(
            self,
            "Select Output Folder",
            str(self.output_folder or ""),
        )

        if folder:
            self._set_output_folder(pathlib.Path(folder))

    def _set_output_folder(self, folder: pathlib.Path) -> None:
        self.output_folder = folder
        self.folder_edit.setText(str(self.output_folder))
        self.generate_button.setEnabled(True)
        self._refresh_default_name()
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
        if not stim_name or stim_name == "(No input arrays available)":
            QtWidgets.QMessageBox.warning(
                self,
                "No Input Selected",
                "Please select a valid input array.",
            )
            return

        try:
            self._log_status(f"Generating scripts for: {stim_name}")

            # Get stimulation array
            if stim_name not in self.project.stim_arrays:
                raise ValueError(f"Input array '{stim_name}' not found in project")

            stim_array = self.project.stim_arrays[stim_name]
            hardware_type = self.hardware_combo.currentText()
            # Re-resolve an untouched default name, so files created since the dialog
            # opened (another export, the stimulation setup) are never overwritten.
            self._refresh_default_name()
            script_name = self.name_edit.text() or "stimulation_script"

            if hardware_type != "HOLO4D":
                raise ValueError(f"Unsupported hardware: {hardware_type}")
            validate_export_name(script_name)
            filenames = [
                f"{script_name}.seq",
                f"{script_name}_original_stim.npz",
                f"{script_name}_overview.png",
            ]
            if any(
                (self.output_folder / name).exists() for name in filenames
            ) and not self._confirm_export(
                "Replace existing export files?\n" + "\n".join(filenames)
            ):
                return
            result = export_stim_array(
                self.project,
                stim_array,
                self.export_settings(),
                self.output_folder,
                script_name,
                confirm_warnings=lambda warnings: self._confirm_export(
                    "Review export changes before generating or importing:\n\n"
                    + "\n".join(warnings)
                ),
            )
            export_report = result.report
            script_path, overview_path = result.seq_path, result.overview_png_path
            script_filename, original_filename, overview_filename = filenames

            self._log_status(f"✓ Generated: {script_filename}")
            self._log_status(f"✓ Generated: {original_filename}")
            self._log_status(f"✓ Generated: {overview_filename}")
            self._refresh_default_name()
            self._log_status(
                "Export summary: segments=%d mapped_active_neurons=%d "
                "unmapped_active_neurons=%d entity_lanes=%d objects=%d",
                export_report.exported_segments,
                export_report.mapped_active_neurons,
                len(export_report.unmapped_active_neurons),
                export_report.entity_lanes,
                export_report.generated_objects,
            )
            self.sequence_generated.emit(
                str(script_path), str(overview_path), result.targeted_stim_ids
            )

            report_lines = [
                f"Successfully generated script in:\n{self.output_folder}",
                "",
                "Generated files: "
                f"{script_filename}, {original_filename}, {overview_filename}",
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

    def _confirm_export(self, message: str) -> bool:
        return (
            QtWidgets.QMessageBox.question(
                self,
                "Review export",
                message,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            == QtWidgets.QMessageBox.StandardButton.Yes
        )

    def _input_sanity_report(self, stim_array: StimArray) -> SysconInputSanityReport:
        force_binary_intensity = self.binary_mode_checkbox.isChecked()
        return check_syscon2_input_sanity(
            stim_array,
            self.project.cellpose_data,
            template_text=self.config_data_edit.toPlainText(),
            tick_resolution_us=500,
            active_threshold=0.0,
            binary_intensity=1.0,
            force_binary_intensity=force_binary_intensity,
            active_laser_wavelength_nm=self._selected_laser_wavelength_nm(),
            spiralling=self.spiralling_checkbox.isChecked(),
            target_power_mw=self.target_power_spin.value(),
            power_lut=self._selected_power_lut(),
            stimulation_spots_per_roi=(
                10 if self.ten_spots_per_roi_checkbox.isChecked() else 1
            ),
            stimulation_spot_algorithm=str(self.spot_algorithm_combo.currentData()),
            exact_slm_timing=self.exact_timing_checkbox.isChecked(),
        )

    def _check_input_sanity(self) -> None:
        """Run sanity check using the fixed 17ms->16.666ms export model."""
        stim_name = self.stim_combo.currentText()
        if not stim_name or stim_name == "(No input arrays available)":
            QtWidgets.QMessageBox.warning(
                self,
                "No Input Selected",
                "Please select a valid input array.",
            )
            return
        if stim_name not in self.project.stim_arrays:
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                f"Input array '{stim_name}' not found in project.",
            )
            return

        stim_array = self.project.stim_arrays[stim_name]

        # 0. dt check — always first; wrong dt makes all other fixes meaningless.
        if abs(stim_array.dt - _SYSCON2_TTL_PULSE_QUANTUM_S) > 1e-9:
            dt_choice = self._prompt_dt_fix_choice(
                current_dt=stim_array.dt, required_dt=_SYSCON2_TTL_PULSE_QUANTUM_S
            )
            if dt_choice == self._DtFixChoice.CANCEL:
                return

            if dt_choice == self._DtFixChoice.CLEAR:
                self.project.replace_stim_values(
                    stim_name,
                    np.zeros_like(stim_array.values),
                    _SYSCON2_TTL_PULSE_QUANTUM_S,
                    label=f"Clear {stim_name} for SysCon2 timing",
                )
                parent_widget = self.parentWidget()
                stim_editor = (
                    getattr(parent_widget, "stim_editor", None)
                    if parent_widget is not None
                    else None
                )
                if stim_editor is not None and hasattr(stim_editor, "select_stim"):
                    stim_editor.select_stim(stim_name)
                QtWidgets.QMessageBox.information(
                    self,
                    "dt Updated — Array Cleared",
                    f"dt has been set to {_SYSCON2_TTL_PULSE_QUANTUM_S} s "
                    f"({_SYSCON2_TTL_PULSE_QUANTUM_S * 1e6:.4f} µs per tick).\n\n"
                    "The input array has been cleared.\n"
                    "Please redraw your inputs.",
                )
                self._log_status(
                    "Set dt of '%s' to %s s and cleared all stimuli — please redraw.",
                    stim_name,
                    _SYSCON2_TTL_PULSE_QUANTUM_S,
                )
                return

            # RESAMPLE: unlike every other fix here, this changes the array's timepoint
            # count, so it replaces the whole sample grid through the project (one
            # undoable edit) -- deliberately NOT returning here, so the checks below
            # re-validate the resampled pattern rather than trusting it's still clean.
            old_dt = stim_array.dt
            n_old = stim_array.values.shape[0]
            new_values, n_new = resample_stim_values_to_dt(
                stim_array, new_dt=_SYSCON2_TTL_PULSE_QUANTUM_S
            )
            self.project.replace_stim_values(
                stim_name,
                new_values,
                _SYSCON2_TTL_PULSE_QUANTUM_S,
                label=f"Resample {stim_name} to SysCon2 timing",
            )
            stim_array = self.project.stim_arrays[stim_name]
            parent_widget = self.parentWidget()
            stim_editor = (
                getattr(parent_widget, "stim_editor", None)
                if parent_widget is not None
                else None
            )
            if stim_editor is not None and hasattr(stim_editor, "select_stim"):
                stim_editor.select_stim(stim_name)
            self._log_status(
                "Resampled stim '%s' from dt=%s s (%d samples) to dt=%s s "
                "(%d samples) via nearest-neighbor.",
                stim_name,
                old_dt,
                n_old,
                _SYSCON2_TTL_PULSE_QUANTUM_S,
                n_new,
            )

        # --- Pre-sanity SLM violation checks ---
        # 1. Initial SLM load: stim starts before SLM has had time to load.
        slm_load_violation = find_initial_slm_load_violation(stim_array)
        if slm_load_violation is not None:
            load_choice = self._prompt_initial_slm_start_fix_choice(
                slm_load_violation, dt_seconds=stim_array.dt
            )
            if load_choice == self._InitialSlmFixChoice.CANCEL:
                return
            fixed_values, extended_samples = (
                shift_stim_values_right_for_initial_slm_load(stim_array)
            )
            if not self._replace_stim_values_preserving_undo(stim_name, fixed_values):
                return
            stim_array = self.project.stim_arrays[stim_name]
            self._warn_input_extended(stim_name, extended_samples, stim_array.dt)
            self._log_status(
                "Shifted stim '%s' right for initial SLM load-up.", stim_name
            )

        # 2. SLM gap: new neuron appears without enough gap for hologram reload.
        slm_gap_violations = find_slm_gap_violations(stim_array)
        if slm_gap_violations:
            gap_choice = self._prompt_slm_gap_fix_choice(
                slm_gap_violations, dt_seconds=stim_array.dt
            )
            if gap_choice == self._SlmFixChoice.CANCEL:
                return
            fixed_values, extended_samples = space_out_stim_values_for_slm(stim_array)
            if not self._replace_stim_values_preserving_undo(stim_name, fixed_values):
                return
            stim_array = self.project.stim_arrays[stim_name]
            self._warn_input_extended(stim_name, extended_samples, stim_array.dt)
            self._log_status(
                "Spaced out stim '%s' to resolve SLM gap violations.", stim_name
            )

        # 3. SLM off/on overlap: new-phase OFF window overlaps another neuron's ON.
        slm_offon_violations = find_slm_off_on_overlap_violations(stim_array)
        if slm_offon_violations:
            offon_choice = self._prompt_slm_off_on_overlap_fix_choice(
                slm_offon_violations, dt_seconds=stim_array.dt
            )
            if offon_choice == self._SlmOffOnFixChoice.CANCEL:
                return
            fixed_values, extended_samples = (
                space_out_stim_values_for_slm_off_on_overlap(stim_array)
            )
            if not self._replace_stim_values_preserving_undo(stim_name, fixed_values):
                return
            stim_array = self.project.stim_arrays[stim_name]
            self._warn_input_extended(stim_name, extended_samples, stim_array.dt)
            self._log_status(
                "Spaced out stim '%s' to resolve SLM off/on overlap.", stim_name
            )

        # 4. Simultaneous duration mismatch: neurons fired together with unequal durations.
        duration_violations = find_simultaneous_duration_violations(stim_array)
        if duration_violations:
            dur_choice = self._prompt_duration_fix_choice(
                duration_violations, dt_seconds=stim_array.dt
            )
            if dur_choice == self._DurationFixChoice.CANCEL:
                return
            fixed_values, _ = equalize_simultaneous_durations_to_min(stim_array)
            if not self._replace_stim_values_preserving_undo(stim_name, fixed_values):
                return
            stim_array = self.project.stim_arrays[stim_name]
            self._log_status(
                "Equalized simultaneous durations in stim '%s'.", stim_name
            )

        # 4b. Review overlap alignment before changing the project timeline.
        overlap_violations = find_overlapping_pulse_group_violations(stim_array)
        if overlap_violations:
            if not self._confirm_export(
                "Align overlapping pulses to the shortest pulse? This changes the selected input array."
            ):
                return
            fixed_values, _ = align_overlapping_pulses_to_shortest(stim_array)
            if not self._replace_stim_values_preserving_undo(stim_name, fixed_values):
                return
            stim_array = self.project.stim_arrays[stim_name]
            self._log_status(
                "Aligned overlapping pulse groups in stim '%s'.", stim_name
            )

        sanity_report = self._input_sanity_report(stim_array)

        sanitized_values = sanity_report.sanitized_values
        # The fixes may lengthen the array (pulses shifted past its end), never
        # shorten it or change the neuron count.
        has_sanitized_update = (
            sanitized_values is not None
            and sanitized_values.ndim == 2
            and sanitized_values.shape[1] == stim_array.values.shape[1]
            and sanitized_values.shape[0] >= stim_array.values.shape[0]
            and not np.array_equal(sanitized_values, stim_array.values)
        )
        if has_sanitized_update:
            apply_choice = QtWidgets.QMessageBox.question(
                self,
                "Apply Sanity Fixes",
                "Sanity checking proposes changes to pulse timing or amplitudes.\n"
                "Apply these changes to the selected input array now?\n\n"
                + "\n".join(sanity_report.fix_messages + sanity_report.warnings),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if apply_choice == QtWidgets.QMessageBox.StandardButton.Yes:
                changed = self._replace_stim_values_preserving_undo(
                    stim_name,
                    sanitized_values,
                )
                if changed:
                    self._log_status(
                        "Applied %d sanity fix(es) to '%s'.",
                        len(sanity_report.fix_messages),
                        stim_name,
                    )
                    stim_array = self.project.stim_arrays[stim_name]
                    sanity_report = self._input_sanity_report(stim_array)
                else:
                    return

        if not sanity_report.is_valid:
            error_lines = ["Input sanity check failed."]
            if sanity_report.errors:
                error_lines.extend(["", "Errors:", *sanity_report.errors])
            if sanity_report.warnings:
                error_lines.extend(["", "Warnings:", *sanity_report.warnings])
            is_pulse_overrun = sanity_report.pulse_overrun_fix_values is not None
            fix_clicked = self._show_scrollable_message(
                "Sanity Check Failed",
                "\n".join(error_lines),
                QtWidgets.QMessageBox.Icon.Critical,
                extra_button_text="Fix pulse overrun" if is_pulse_overrun else None,
            )
            if fix_clicked and is_pulse_overrun:
                fix_values = sanity_report.pulse_overrun_fix_values
                self._replace_stim_values_preserving_undo(stim_name, fix_values)
                self._log_status(
                    "Applied pulse overrun fix to '%s'. Re-run sanity check to verify.",
                    stim_name,
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
            f"Sanity fixes applied: {len(sanity_report.fix_messages)}",
        ]
        if sanity_report.warnings:
            success_lines.extend(["", "Warnings:", *sanity_report.warnings])
            self._show_scrollable_message(
                "Sanity Check Passed With Warnings",
                "\n".join(success_lines),
                QtWidgets.QMessageBox.Icon.Warning,
            )
            return
        self._show_scrollable_message(
            "Sanity Check Passed",
            "\n".join(success_lines),
            QtWidgets.QMessageBox.Icon.Information,
        )

    def _show_scrollable_message(
        self,
        title: str,
        text: str,
        icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Icon.Information,
        extra_button_text: str | None = None,
    ) -> bool:
        """Show a message dialog with a scrollable text area.

        Returns True if the extra button (if any) was clicked, False otherwise.
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(600, 400)

        text_edit = QtWidgets.QPlainTextEdit(dlg)
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)

        button_box = QtWidgets.QDialogButtonBox(dlg)
        ok_button = cast(
            QtWidgets.QPushButton,
            button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Ok),
        )
        ok_button.clicked.connect(dlg.accept)

        extra_clicked = [False]
        if extra_button_text is not None:
            extra_btn = cast(
                QtWidgets.QPushButton,
                button_box.addButton(
                    extra_button_text,
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )

            def _on_extra() -> None:
                extra_clicked[0] = True
                dlg.accept()

            extra_btn.clicked.connect(_on_extra)

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(text_edit)
        layout.addWidget(button_box)
        dlg.exec()
        return extra_clicked[0]

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

    def _prompt_dt_fix_choice(
        self,
        *,
        current_dt: float,
        required_dt: float,
    ) -> _DtFixChoice:
        text_lines = [
            f"SysCon2 export requires stim_array.dt == {required_dt} s "
            f"({required_dt * 1e6:.4f} µs per tick).",
            f"Got {current_dt} s ({current_dt * 1e6:.4f} µs per tick).",
            "",
            "Choose how to resolve:",
            "1) Clear array: set dt and blank all stimuli — you redraw from scratch.",
            "2) Resample: nearest-neighbor resample the existing pattern onto the new "
            "tick grid, preserving it as closely as the new grid allows (undoable).",
            "3) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("dt Mismatch")
        msg_box.setText("\n".join(text_lines))

        clear_button = msg_box.addButton(
            "Clear Array", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        resample_button = msg_box.addButton(
            "Resample (Nearest-Neighbor)", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == clear_button:
            return self._DtFixChoice.CLEAR
        if clicked == resample_button:
            return self._DtFixChoice.RESAMPLE
        return self._DtFixChoice.CANCEL

    def _prompt_slm_gap_fix_choice(
        self,
        violations: list[SlmGapViolation],
        *,
        dt_seconds: float,
    ) -> _SlmFixChoice:
        first_violation = violations[0]
        gap_ms = first_violation.gap_samples * dt_seconds * 1e3
        required_gap_ms = first_violation.required_gap_samples * dt_seconds * 1e3
        shift_ms = (
            (first_violation.required_gap_samples - first_violation.gap_samples)
            * dt_seconds
            * 1e3
        )
        text_lines = [
            (
                "Detected input transitions that violate the SLM load-time rule "
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
            f"Proposed fix: translate inputs right by {shift_ms:.2f} ms from the violation point.",
            "",
            "Choose how to resolve:",
            f"1) Translate inputs right by {shift_ms:.2f} ms",
            "2) Cancel",
        ]
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("SLM Gap Violations Detected")
        msg_box.setText("\n".join(text_lines))

        space_button = msg_box.addButton(
            "Translate Inputs Right",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == space_button:
            return self._SlmFixChoice.SPACE_OUT
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
                "Detected simultaneous input events with unequal durations. "
                "All neurons activated together must have equal duration."
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
                "Detected input too close to experiment start. "
                "SysCon2 requires at least 16.66 ms before first input "
                "for initial SLM load-up."
            ),
            "",
            f"First input starts at: {first_start_ms:.2f}ms",
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

    def _warn_input_extended(
        self, stim_name: str, extended_samples: int, dt_seconds: float
    ) -> None:
        """Tell the user a sanity fix lengthened the array instead of cutting
        pulses shifted past its end."""
        if extended_samples <= 0:
            return
        message = (
            f"Shifting pulses extended input array '{stim_name}' by "
            f"{extended_samples} sample(s) ({extended_samples * dt_seconds * 1e3:.3f} ms) "
            "so no pulse is cut off."
        )
        self._log_status("%s", message)
        QtWidgets.QMessageBox.warning(self, "Input Array Extended", message)

    def _replace_stim_values_preserving_undo(
        self,
        stim_name: str,
        new_values: np.ndarray,
    ) -> bool:
        if stim_name not in self.project.stim_arrays:
            QtWidgets.QMessageBox.critical(
                self,
                "Sanity Check Failed",
                f"Input array '{stim_name}' not found in project.",
            )
            return False

        try:
            parent_widget = self.parentWidget()
            stim_editor = (
                getattr(parent_widget, "stim_editor", None)
                if parent_widget is not None
                else None
            )
            target_shape = self.project.stim_arrays[stim_name].values.shape
            if new_values.shape != target_shape:
                # A longer sample grid (sanity fixes shifted pulses past the end)
                # replaces the whole grid through the project as one undoable edit.
                self.project.replace_stim_values(
                    stim_name,
                    np.array(new_values, copy=True),
                    label=f"Apply SysCon2 sanity fixes to {stim_name}",
                )
                if stim_editor is not None and hasattr(stim_editor, "select_stim"):
                    stim_editor.select_stim(stim_name)
            elif stim_editor is not None and hasattr(
                stim_editor, "replace_stim_values_with_undo"
            ):
                stim_editor.replace_stim_values_with_undo(stim_name, new_values)
            else:
                self.project.replace_stim_values(
                    stim_name, np.array(new_values, copy=True)
                )
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
        return generate_sequence_text(
            self.project, stim_array, self.export_settings(), cellpose_data
        )

    def export_settings(self) -> SysconExportSettings:
        """Snapshot of the export options currently shown in the dialog."""
        return SysconExportSettings(
            config_text=self.config_data_edit.toPlainText(),
            force_binary_intensity=self.binary_mode_checkbox.isChecked(),
            spiralling=self.spiralling_checkbox.isChecked(),
            ten_spots_per_roi=self.ten_spots_per_roi_checkbox.isChecked(),
            spot_algorithm=str(self.spot_algorithm_combo.currentData()),
            exact_slm_timing=self.exact_timing_checkbox.isChecked(),
            laser_wavelength_nm=self._selected_laser_wavelength_nm(),
            target_power_mw=self.target_power_spin.value(),
        )

    def apply_settings(self, settings: SysconExportSettings) -> None:
        """Preload the dialog with previously used export options."""
        self.config_data_edit.setPlainText(settings.config_text)
        self.binary_mode_checkbox.setChecked(settings.force_binary_intensity)
        self.spiralling_checkbox.setChecked(settings.spiralling)
        self.exact_timing_checkbox.setChecked(settings.exact_slm_timing)
        self.ten_spots_per_roi_checkbox.setChecked(settings.ten_spots_per_roi)
        index = self.spot_algorithm_combo.findData(settings.spot_algorithm)
        if index >= 0:
            self.spot_algorithm_combo.setCurrentIndex(index)
        index = self.laser_combo.findData(settings.laser_wavelength_nm)
        if index >= 0:
            self.laser_combo.setCurrentIndex(index)
        self.target_power_spin.setValue(settings.target_power_mw)

    def _selected_laser_wavelength_nm(self) -> int:
        current_data = self.laser_combo.currentData()
        if current_data is None:
            return 1100
        return int(current_data)

    def _selected_power_lut(self) -> PowerLut | None:
        wavelength_nm = self._selected_laser_wavelength_nm()
        lut, _ = resolve_power_lut(self.project.power_luts, wavelength_nm)
        return lut

    def _format_lut_status(self) -> str:
        wavelength_nm = self._selected_laser_wavelength_nm()
        lut, is_override = resolve_power_lut(self.project.power_luts, wavelength_nm)
        if lut is None:
            return f"No LUT loaded for {wavelength_nm} nm"
        tag = "" if is_override else " (default)"
        return f"LUT: {lut.description} (max {lut.max_mw:.1f} mW){tag}"

    def _on_load_lut_clicked(self) -> None:
        wavelength_nm = self._selected_laser_wavelength_nm()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Load SysCon LUT for {wavelength_nm} nm",
            "",
            "SysCon LUT (*.lut);;All files (*)",
        )
        if not path:
            return
        self._apply_lut_from_path(path, wavelength_nm)

    def _apply_lut_from_path(self, path: str, wavelength_nm: int) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                lut = parse_power_lut_text(fh.read())
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "LUT load failed",
                f"Could not parse LUT file:\n{exc}",
            )
            return
        self.project.set_power_lut(wavelength_nm, lut)
        self.lut_status_label.setText(self._format_lut_status())

    def _default_syscon2_config_text(self) -> str:
        return default_syscon2_config_text(self.project)
