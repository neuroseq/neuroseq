from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import qtawesome as qta
from qtpy import QtWidgets
from qtpy.compat import getexistingdirectory

if TYPE_CHECKING:
    from neuroseq.project import Project


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
            "Generate stimulation scripts for your experiments."
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
        default_config = """CC-D0-82-61-A8-44-C9-77-DC-C1-F4-75-86-F5-E8-7B
version=2
scanmode=Accurate
runs=5
DeviceSequenceID=0
invertedportcnt=2
inverted0=dummy_0_0 RMI In
inverted1=dummy_2_0 RMI In
AutoTriggerRulesEnabled=True
AutoTriggerRulesCount=2
SourceConnectorPort0=dummy_1_0 RMI In
TargetConnectorPort0=Holo4D TTL Out 2
PreTiming_us0=20000
PostTiming_us0=20000
SourceConnectorPort1=dummy_2_0 RMI In
TargetConnectorPort1=Holo4D TTL Out 2
PreTiming_us1=20000
PostTiming_us1=20000
timelinezoomfactor=23
timelineviewposx=0
timelineviewposy=0
TotalEditorGroups=0
[0]
ID=3
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=4999800
TimelineInfo0_description=
TimelineInfo0_repeats=599
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.246
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=0
Entity0_Filled=False
Entity0_CenterX=214.0635
Entity0_CenterY=318.5917
Entity0_CenterZ=0
Entity0_Rotation=0
Entity0_ScaleX=1
Entity0_ScaleY=1
Entity0_ScaleZ=1
Entity0_VertexCount=0
Entity0_reversed=False
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=4999800
TimelineInfo0_description=
TimelineInfo0_repeats=1800
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.246
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=0
Entity1_Filled=False
Entity1_CenterX=270.3961
Entity1_CenterY=334.2396
Entity1_CenterZ=0
Entity1_Rotation=0
Entity1_ScaleX=1
Entity1_ScaleY=1
Entity1_ScaleZ=1
Entity1_VertexCount=0
Entity1_reversed=False
[/1]
[2]
ID=4
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=14999400
TimelineInfo0_description=
TimelineInfo0_repeats=600
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.246
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity2_Type=0
Entity2_Filled=False
Entity2_CenterX=163.3643
Entity2_CenterY=179.6381
Entity2_CenterZ=0
Entity2_Rotation=0
Entity2_ScaleX=1
Entity2_ScaleY=1
Entity2_ScaleZ=1
Entity2_VertexCount=0
Entity2_reversed=False
[/2]
[3]
ID=5
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=25015666
TimelineInfo0_description=
TimelineInfo0_repeats=600
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.246
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity3_Type=0
Entity3_Filled=False
Entity3_CenterX=313.5843
Entity3_CenterY=157.1051
Entity3_CenterZ=0
Entity3_Rotation=0
Entity3_ScaleX=1
Entity3_ScaleY=1
Entity3_ScaleZ=1
Entity3_VertexCount=0
Entity3_reversed=False
[/3]
[4]
ID=6
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=35031932
TimelineInfo0_description=
TimelineInfo0_repeats=600
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.246
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity4_Type=0
Entity4_Filled=False
Entity4_CenterX=112.665
Entity4_CenterY=340.4988
Entity4_CenterZ=0
Entity4_Rotation=0
Entity4_ScaleX=1
Entity4_ScaleY=1
Entity4_ScaleZ=1
Entity4_VertexCount=0
Entity4_reversed=False
[/4]
[5]
ID=1
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=845278.6875
TimelineInfo0_description=
TimelineInfo0_repeats=958
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/5]
[6]
ID=7
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=2
TimelineInfo0_starttime=4937302.5
TimelineInfo0_description=
TimelineInfo0_repeats=77357
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/6]
[7]
ID=8
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=6
TimelineInfo0_starttime=4917511.625
TimelineInfo0_description=
TimelineInfo0_repeats=77092
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 2
[/7]
"""
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

    def _log_status(self, message: str) -> None:
        """Log a status message."""
        self.logger.info(message)

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
            script_content = self._generate_script_content(
                stim_array=stim_array,
                stim_name=stim_name,
                hardware_type=hardware_type,
                cellpose_data=self.project.cellpose_data,
            )

            # Determine file extension based on hardware type
            extension = self._get_file_extension(hardware_type)
            script_filename = f"{script_name}{extension}"
            script_path = self.output_folder / script_filename

            # Write script file
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            self._log_status(f"✓ Generated: {script_filename}")

            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Scripts Generated",
                f"Successfully generated script in:\n{self.output_folder}",
            )

        except Exception as e:
            self.logger.error(f"Error generating scripts: {e}", exc_info=True)
            self._log_status(f"✗ Error: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Generation Failed",
                f"Failed to generate a script:\n{str(e)}",
            )

    def _get_file_extension(self, hardware_type: str) -> str:
        """Get the appropriate file extension for the hardware type."""
        return ".seq"

    def _generate_script_content(
        self,
        stim_array,
        stim_name: str,
        hardware_type: str,
        cellpose_data,
    ) -> str:
        """Generate the script content based on hardware type."""
        return self._generate_holo4d_script(stim_array, stim_name, cellpose_data)

    def _generate_holo4d_script(self, stim_array, stim_name: str, cellpose_data) -> str:
        """Generate HOLO4D holographic stimulation script."""
        # Start with HOLO4D configuration data
        config_data = self.config_data_edit.toPlainText()
        script = config_data + "\n\n"

        # Append cellpose segmentation data if available
        # if cellpose_data is not None:
        #     script += "\n\n# ========== CELLPOSE SEGMENTATION DATA ==========\n"
        #     script += f"# Image path: {cellpose_data.image_path}\n"
        #     script += f"# Segmentation array shape: {cellpose_data.segmentation_array.shape}\n"
        #     if cellpose_data.original_cellpose_ids is not None:
        #         script += f"# Number of cells: {len(cellpose_data.original_cellpose_ids)}\n"
        #         script += "# Original Cellpose IDs: "
        #         script += ",".join(str(id) for id in cellpose_data.original_cellpose_ids)
        #         script += "\n"
        #     if cellpose_data.stimulus_to_cellpose_mapping is not None:
        #         script += "# Stimulus to Cellpose mapping:\n"
        #         for stim_id, cellpose_id in cellpose_data.stimulus_to_cellpose_mapping.items():
        #             script += f"#   Stimulus {stim_id} -> Cellpose {cellpose_id}\n"

        # Append stimulation array data at the end
        # script += "\n\n# ========== STIMULATION DATA ==========\n"
        # script += f"# Stimulation Array: {stim_name}\n"
        # script += f"# Shape: {stim_array.values.shape} (timesteps x neurons)\n"
        # script += f"# Units: {stim_array.units}\n"
        # script += f"# Timestep (dt): {stim_array.dt}s\n"
        # script += (
        #     "# Data format: Each row represents a timestep, each column a neuron\n\n"
        # )

        # Convert array data to string representation
        # script += "# BEGIN DATA\n"
        # for _t_idx, timestep_values in enumerate(stim_array.values):
        #     # Format each timestep as a comma-separated line
        #     values_str = ",".join(f"{val:.6f}" for val in timestep_values)
        #     script += f"{values_str}\n"
        # script += "# END DATA\n"

        return script
