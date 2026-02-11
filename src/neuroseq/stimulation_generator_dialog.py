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

        self.setWindowTitle("HOLO4D Stimulation Script Generator")
        self.setModal(True)
        self.resize(600, 500)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Header description
        description_label = QtWidgets.QLabel(
            "Generate HOLO4D holographic stimulation scripts for your experiments."
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
        self.hardware_combo.setToolTip("HOLO4D holographic stimulation system")

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
        self.name_edit.setText("stimulation_script")
        self.name_edit.setToolTip("Base name for the generated script file(s)")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit)
        config_layout.addLayout(name_layout)

        # Include timing checkbox
        self.include_timing_checkbox = QtWidgets.QCheckBox("Include timing information")
        self.include_timing_checkbox.setChecked(True)
        self.include_timing_checkbox.setToolTip(
            "Include timestep and duration information in the script"
        )
        config_layout.addWidget(self.include_timing_checkbox)

        # Include metadata checkbox
        self.include_metadata_checkbox = QtWidgets.QCheckBox(
            "Include metadata comments"
        )
        self.include_metadata_checkbox.setChecked(True)
        self.include_metadata_checkbox.setToolTip(
            "Add comments with information about the stimulation array"
        )
        config_layout.addWidget(self.include_metadata_checkbox)

        # Generate visualization checkbox
        self.generate_viz_checkbox = QtWidgets.QCheckBox("Generate visualization plot")
        self.generate_viz_checkbox.setChecked(False)
        self.generate_viz_checkbox.setToolTip(
            "Create a separate script to visualize the stimulation pattern"
        )
        config_layout.addWidget(self.generate_viz_checkbox)

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
TotalEditorGroups=0"""
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

        # Status/log area
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("Status messages will appear here...")
        main_layout.addWidget(self.status_text)

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
        """Add a status message to the status text area."""
        self.status_text.append(message)
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
            )

            # Determine file extension based on hardware type
            extension = self._get_file_extension(hardware_type)
            script_filename = f"{script_name}{extension}"
            script_path = self.output_folder / script_filename

            # Write script file
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            self._log_status(f"✓ Generated: {script_filename}")

            # Generate visualization script if requested
            if self.generate_viz_checkbox.isChecked():
                viz_content = self._generate_visualization_script(stim_array, stim_name)
                viz_filename = f"{script_name}_visualization.py"
                viz_path = self.output_folder / viz_filename

                with open(viz_path, "w", encoding="utf-8") as f:
                    f.write(viz_content)

                self._log_status(f"✓ Generated: {viz_filename}")

            # Generate data file (CSV or NPY)
            data_filename = f"{script_name}_data.npy"
            data_path = self.output_folder / data_filename

            import numpy as np

            np.save(data_path, stim_array.values)
            self._log_status(f"✓ Generated: {data_filename}")

            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Scripts Generated",
                f"Successfully generated scripts in:\n{self.output_folder}",
            )

        except Exception as e:
            self.logger.error(f"Error generating scripts: {e}", exc_info=True)
            self._log_status(f"✗ Error: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Generation Failed",
                f"Failed to generate scripts:\n{str(e)}",
            )

    def _get_file_extension(self, hardware_type: str) -> str:
        """Get the appropriate file extension for the hardware type."""
        return ".seq"

    def _generate_script_content(
        self,
        stim_array,
        stim_name: str,
        hardware_type: str,
    ) -> str:
        """Generate the script content based on hardware type."""
        return self._generate_holo4d_script(stim_array, stim_name)

    def _generate_header(self, stim_array, stim_name: str) -> str:
        """Generate common header comment block."""
        if not self.include_metadata_checkbox.isChecked():
            return ""

        header = f"""\"\"\"
HOLO4D Holographic Stimulation Script
Generated by NeuroSeq - Stimulation Generator

Stimulation: {stim_name}
Type: {stim_array.stim_type.name if hasattr(stim_array.stim_type, "name") else str(stim_array.stim_type)}
Shape: {stim_array.values.shape} (time x neurons)
Units: {stim_array.units}
Timestep: {stim_array.dt}s
Duration: {stim_array.values.shape[0] * stim_array.dt}s
\"\"\"

"""
        return header

    def _generate_holo4d_script(self, stim_array, stim_name: str) -> str:
        """Generate HOLO4D holographic stimulation script."""
        # Start with HOLO4D configuration data
        config_data = self.config_data_edit.toPlainText()
        script = config_data + "\n\n"

        # Add header comment if enabled
        script += self._generate_header(stim_array, stim_name)

        script += (
            """import numpy as np
import time

# Load stimulation data
data = np.load('"""
            + self.name_edit.text()
            + """_data.npy')

"""
        )

        if self.include_timing_checkbox.isChecked():
            script += f"""# Timing parameters
dt = {stim_array.dt}  # seconds
n_timesteps = {stim_array.values.shape[0]}
n_neurons = {stim_array.values.shape[1]}

"""

        script += """# HOLO4D Configuration
class HOLO4DController:
    \"\"\"Controller for HOLO4D holographic stimulation system.\"\"\"

    def __init__(self):
        # TODO: Initialize HOLO4D hardware connection
        # Example: self.device = holo4d.connect()
        self.initialized = False
        print("Initializing HOLO4D system...")

    def initialize(self):
        \"\"\"Initialize the HOLO4D hardware system.\"\"\"
        # TODO: Add your HOLO4D initialization code here
        # Example:
        # self.device.calibrate()
        # self.device.set_laser_power(power_mw)
        self.initialized = True
        print("HOLO4D system initialized")

    def set_holographic_pattern(self, neuron_idx, intensity_values):
        \"\"\"
        Set holographic stimulation pattern for specified neurons.

        Parameters:
        -----------
        neuron_idx : array-like
            Indices of neurons to stimulate
        intensity_values : array-like
            Stimulation intensity for each neuron
        \"\"\"
        # TODO: Implement HOLO4D-specific pattern generation
        # Example:
        # active_neurons = neuron_idx[intensity_values > 0]
        # intensities = intensity_values[intensity_values > 0]
        # self.device.generate_hologram(active_neurons, intensities)
        pass

    def apply_stimulation(self, stim_values):
        \"\"\"Apply stimulation pattern for current timestep.\"\"\"
        active_idx = np.where(stim_values > 0)[0]
        if len(active_idx) > 0:
            self.set_holographic_pattern(active_idx, stim_values[active_idx])

    def shutdown(self):
        \"\"\"Safely shutdown the HOLO4D system.\"\"\"
        # TODO: Add cleanup code
        # Example: self.device.disconnect()
        print("HOLO4D system shutdown")

def run_stimulation():
    \"\"\"Execute the HOLO4D stimulation protocol.\"\"\"
    print(f"Starting HOLO4D stimulation protocol...")
    print(f"Data shape: {data.shape}")

    # Initialize controller
    controller = HOLO4DController()
    controller.initialize()

    try:
        for t_idx in range(data.shape[0]):
            # Get stimulation values for this timestep
            stim_values = data[t_idx, :]

            # Apply holographic stimulation
            controller.apply_stimulation(stim_values)

            # Progress update
            if t_idx % 100 == 0:
                print(f"Timestep {t_idx}/{data.shape[0]}")

"""

        if self.include_timing_checkbox.isChecked():
            script += """            # Wait for next timestep
            time.sleep(dt)
"""

        script += """
        print("Stimulation complete!")

    finally:
        # Ensure proper cleanup
        controller.shutdown()

if __name__ == "__main__":
    run_stimulation()
"""

        # Append stimulation array data at the end
        script += "\n\n# ========== STIMULATION DATA ==========\n"
        script += f"# Stimulation Array: {stim_name}\n"
        script += f"# Shape: {stim_array.values.shape} (timesteps x neurons)\n"
        script += f"# Units: {stim_array.units}\n"
        script += f"# Timestep (dt): {stim_array.dt}s\n"
        script += (
            "# Data format: Each row represents a timestep, each column a neuron\n\n"
        )

        # Convert array data to string representation
        script += "# BEGIN DATA\n"
        for _t_idx, timestep_values in enumerate(stim_array.values):
            # Format each timestep as a comma-separated line
            values_str = ",".join(f"{val:.6f}" for val in timestep_values)
            script += f"{values_str}\n"
        script += "# END DATA\n"

        return script

    def _generate_generic_python_script(self, stim_array, stim_name: str) -> str:
        """Generate a generic Python script."""
        script = self._generate_header(stim_array, stim_name)

        script += (
            """import numpy as np
import time

# Load stimulation data
data = np.load('"""
            + self.name_edit.text()
            + """_data.npy')

"""
        )

        if self.include_timing_checkbox.isChecked():
            script += f"""# Timing parameters
dt = {stim_array.dt}  # seconds
n_timesteps = {stim_array.values.shape[0]}
n_neurons = {stim_array.values.shape[1]}

"""

        script += """# Main stimulation loop
def run_stimulation():
    print(f"Starting stimulation protocol...")
    print(f"Data shape: {data.shape}")

    for t_idx in range(data.shape[0]):
        # Get stimulation values for this timestep
        stim_values = data[t_idx, :]

        # TODO: Send stimulation values to hardware
        # Example: your_hardware_function(stim_values)

        if t_idx % 100 == 0:
            print(f"Timestep {t_idx}/{data.shape[0]}")

"""

        if self.include_timing_checkbox.isChecked():
            script += """        # Wait for next timestep
        time.sleep(dt)
"""

        script += """
    print("Stimulation complete!")

if __name__ == "__main__":
    run_stimulation()
"""

        return script

    def _generate_arduino_script(self, stim_array, stim_name: str) -> str:
        """Generate Arduino/C++ script."""
        script = f"""/*
 * Hardware Stimulation Script - Arduino
 * Generated by NeuroSeq
 *
 * Stimulation: {stim_name}
 * Shape: {stim_array.values.shape}
 * Units: {stim_array.units}
 * Timestep: {stim_array.dt}s
 */

// Configuration
const int N_NEURONS = {stim_array.values.shape[1]};
const int N_TIMESTEPS = {stim_array.values.shape[0]};
const long DT_MICROS = {int(stim_array.dt * 1000000)};

// Pin assignments (adjust as needed)
const int outputPins[] = {{2, 3, 4, 5, 6, 7, 8, 9}};  // Example pins

void setup() {{
  Serial.begin(9600);

  // Initialize output pins
  for (int i = 0; i < N_NEURONS && i < 8; i++) {{
    pinMode(outputPins[i], OUTPUT);
  }}

  Serial.println("Arduino stimulation controller ready");
}}

void loop() {{
  // TODO: Read stimulation data from Serial or SD card
  // TODO: Apply stimulation pattern to output pins

  // Example: waiting for serial data
  if (Serial.available() > 0) {{
    // Process incoming commands
  }}
}}
"""
        return script

    def _generate_nidaq_script(self, stim_array, stim_name: str) -> str:
        """Generate NI-DAQ Python script."""
        script = self._generate_header(stim_array, stim_name)

        script += (
            """import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping

# Load stimulation data
data = np.load('"""
            + self.name_edit.text()
            + """_data.npy')

# Configuration
"""
        )

        if self.include_timing_checkbox.isChecked():
            script += f"""dt = {stim_array.dt}  # seconds
sample_rate = {1.0 / stim_array.dt}  # Hz

"""

        script += """# DAQ device name (change to match your system)
device_name = "Dev1"
channel_start = 0
channel_end = data.shape[1] - 1

def run_stimulation():
    with nidaqmx.Task() as task:
        # Add analog output channels
        task.ao_channels.add_ao_voltage_chan(
            f"{device_name}/ao{channel_start}:{channel_end}"
        )

        # Configure timing
"""

        if self.include_timing_checkbox.isChecked():
            script += """        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=data.shape[0]
        )

"""

        script += """        print("Starting NI-DAQ stimulation...")

        # Write data to DAQ
        task.write(data.T, auto_start=True)
        task.wait_until_done()

        print("Stimulation complete!")

if __name__ == "__main__":
    run_stimulation()
"""

        return script

    def _generate_laser_script(self, stim_array, stim_name: str) -> str:
        """Generate laser control Python script."""
        script = self._generate_header(stim_array, stim_name)

        script += (
            """import numpy as np
import time

# Load stimulation data
data = np.load('"""
            + self.name_edit.text()
            + """_data.npy')

# Laser control configuration
"""
        )

        if self.include_timing_checkbox.isChecked():
            script += f"""dt = {stim_array.dt}
n_locations = {stim_array.values.shape[1]}

"""

        script += """class LaserController:
    def __init__(self):
        # TODO: Initialize laser hardware connection
        pass

    def set_power(self, location_idx, power_value):
        \"\"\"Set laser power for a specific location.\"\"\"
        # TODO: Implement hardware-specific power control
        pass

    def run_stimulation_protocol(self, data):
        print(f"Running laser stimulation protocol...")
        print(f"Data shape: {data.shape}")

        for t_idx in range(data.shape[0]):
            stim_values = data[t_idx, :]

            # Apply stimulation to each location
            for loc_idx, power in enumerate(stim_values):
                if power > 0:  # Only activate laser if power > 0
                    self.set_power(loc_idx, power)

            if t_idx % 100 == 0:
                print(f"Timestep {t_idx}/{data.shape[0]}")

"""

        if self.include_timing_checkbox.isChecked():
            script += """            time.sleep(dt)
"""

        script += """
        print("Stimulation complete!")

if __name__ == "__main__":
    controller = LaserController()
    controller.run_stimulation_protocol(data)
"""

        return script

    def _generate_pyboard_script(self, stim_array, stim_name: str) -> str:
        """Generate PyBoard/MicroPython script."""
        script = f"""\"\"\"
PyBoard Stimulation Script
Generated by NeuroSeq

Stimulation: {stim_name}
Shape: {stim_array.values.shape}
\"\"\"

import pyb
import time

# Configuration
N_NEURONS = {stim_array.values.shape[1]}
DT = {stim_array.dt}

# Pin assignments (adjust for your setup)
output_pins = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
pins = [pyb.Pin(pin, pyb.Pin.OUT_PP) for pin in output_pins[:N_NEURONS]]

def run_stimulation():
    # TODO: Load stimulation data
    # Note: MicroPython has limited storage, consider loading incrementally

    print("PyBoard stimulation controller ready")

    # Example stimulation loop
    while True:
        # TODO: Implement your stimulation protocol
        time.sleep(DT)

if __name__ == "__main__":
    run_stimulation()
"""

        return script

    def _generate_visualization_script(self, stim_array, stim_name: str) -> str:
        """Generate a visualization script for the stimulation pattern."""
        script = (
            f"""\"\"\"
Stimulation Visualization Script
Generated by NeuroSeq

Stimulation: {stim_name}
\"\"\"

import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('"""
            + self.name_edit.text()
            + """_data.npy')

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Heatmap
im = axes[0].imshow(data.T, aspect='auto', interpolation='nearest', cmap='viridis')
axes[0].set_xlabel('Time (steps)')
axes[0].set_ylabel('Neuron Index')
axes[0].set_title(f'Stimulation Pattern: {stim_name}')
plt.colorbar(im, ax=axes[0], label='{stim_array.units}')

# Average activity over time
avg_activity = data.mean(axis=1)
time_points = np.arange(len(avg_activity)) * {stim_array.dt}
axes[1].plot(time_points, avg_activity)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel(f'Average Activity ({stim_array.units})')
axes[1].set_title('Average Stimulation Intensity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('"""
            + self.name_edit.text()
            + """_visualization.png', dpi=150)
print("Visualization saved as """
            + self.name_edit.text()
            + """_visualization.png")
plt.show()
"""
        )

        return script
