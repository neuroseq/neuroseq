from __future__ import annotations

import logging
import pathlib
import runpy
from collections.abc import Callable
from dataclasses import dataclass

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq import __version__
from neuroseq.b2_model import B2Model
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.fluorescent_suite2p_import_dialog import FluorescentSuite2pImportDialog
from neuroseq.fluorescent_suite2p_contrasting_traces_import_dialog import FluorescentSuite2pContrastingTracesImportDialog
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.optostim_import_dialog import OptoStimReadingsImportDialog
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim_console import StimConsole
from neuroseq.stim_editor import StimEditor
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import add_action
from neuroseq.utils import construct_synapse_matrix
from neuroseq.utils import example_models_path
from neuroseq.utils import with_wait_cursor


@dataclass
class GuiSettings:
    max_ram_synapse_plot_matrix: int = 256
    brian2_backend: str = "cython"
    stim_padding_secs: int = 5


class Gui(QtWidgets.QMainWindow):
    def __init__(self, filename: pathlib.Path | str | None = None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("starting Gui")
        self.settings = GuiSettings()
        self.model = B2Model()
        self.external_simulation_function: Callable | None = None
        self.external_simulation_data: (
            dict[str, dict[str, str | int | float | np.ndarray]] | None
        ) = None
        self.neurons: dict[str, tuple[b2.StateMonitor, str]] = {}
        self.synapses: dict[str, tuple[b2.StateMonitor, str]] = {}
        self.spikes: dict[str, b2.SpikeMonitor] = {}
        self.setWindowTitle("NeuroSeq")
        self.cmb_neuron = QtWidgets.QComboBox()
        self.cmb_spike = QtWidgets.QComboBox()
        self.cmb_synapse = QtWidgets.QComboBox()
        self._create_menus_and_toolbars()
        self.setWindowIcon(qta.icon("mdi6.brain"))
        main_horizontal_layout = QtWidgets.QHBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_horizontal_layout)
        # vertical toolbar for stim editing on the left
        self.toolbar_stim_editor = QtWidgets.QToolBar()
        self.toolbar_stim_editor.setOrientation(QtCore.Qt.Orientation.Vertical)
        main_horizontal_layout.addWidget(self.toolbar_stim_editor)
        # left/right splitters between heatmap and console
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.heatmap_widget = HeatmapWidget()
        split_left_right.addWidget(self.heatmap_widget)
        self.custom_analysis_widget = CustomAnalysisWidget()
        split_left_right.addWidget(self.custom_analysis_widget)
        self.stim_console = StimConsole()
        split_left_right.addWidget(self.stim_console)
        split_left_right.setSizes([1000, 1000, 1000])
        # main vertical splitter layout
        split_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.stim_editor = StimEditor(
            cmb_stim=self.cmb_stim, toolbar=self.toolbar_stim_editor
        )
        split_vertical.addWidget(self.stim_editor)
        self.trace_widget = TraceWidget()
        split_vertical.addWidget(self.trace_widget)
        self.spike_widget = SpikeWidget()
        split_vertical.addWidget(self.spike_widget)
        # heatmap / custom analysis widget / console
        split_vertical.addWidget(split_left_right)
        main_horizontal_layout.addWidget(split_vertical)
        self.setCentralWidget(central_widget)
        self._create_actions()
        self.showMaximized()
        self._create_progressbar()
        self.link_x_axes(True)
        self.link_y_axes(True)
        # connect widgets
        self.heatmap_widget.connect_timeseries_widget_timeline(
            self.stim_editor.stim_widget
        )
        self.stim_editor.stim_widget.connect_heatmap_widget(self.heatmap_widget)
        self.heatmap_widget.connect_timeseries_widget_timeline(self.trace_widget)
        self.trace_widget.connect_heatmap_widget(self.heatmap_widget)
        self.heatmap_widget.connect_timeseries_widget_timeline(self.spike_widget)
        self.spike_widget.connect_heatmap_widget(self.heatmap_widget)
        self.stim_console.sig_stims_values_dict_updated.connect(
            self.stim_editor.update_stim_values_dict
        )
        self.trace_widget.custom_analysis_requested.connect(
            self.custom_analysis_widget.set_data
        )
        self.spike_widget.custom_analysis_requested.connect(
            self.custom_analysis_widget.set_data
        )
        self.cmb_neuron.currentTextChanged.connect(self.set_neuron)
        self.cmb_synapse.currentTextChanged.connect(self.set_synapse)
        self.cmb_spike.currentTextChanged.connect(self.set_spike)
        if filename is None:
            filename = example_models_path / "default_model.json"
        self.load_model(str(filename))

    def _update_data(self) -> None:
        if self.model.network.t > 0:
            self.continue_simulation_action.setText("&Continue simulation")
        else:
            self.continue_simulation_action.setText("&Run simulation")
        self.stim_editor.set_model(self.model)
        self.set_neuron(self.cmb_neuron.currentText())
        self.set_synapse(self.cmb_synapse.currentText())
        self.set_spike(self.cmb_spike.currentText())

    def _update_simulation_progress(
        self,
        elapsed: b2.Quantity,
        completed: b2.Quantity,
        start: b2.Quantity,
        duration: b2.Quantity,
    ) -> None:
        completed_percent = int(100 * completed)
        self.logger.info(
            f"simulating {start} -> {duration} :: {completed_percent}% complete in {elapsed}"
        )
        self.progress_dialog.setLabelText("Simulating...")
        self.progress_dialog.setValue(completed_percent)

    def _update_load_progress(
        self,
        completed_percent: int,
    ) -> None:
        self.logger.info(f"loading :: {completed_percent}% complete")
        self.progress_dialog.setLabelText("Loading...")
        self.progress_dialog.setValue(completed_percent)

    def btn_new_clicked(self) -> None:
        self.load_model(str(example_models_path / "default_model.json"))

    def btn_open_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load model",
            "model.json",
            "json file (*.json)",
        )
        if filename != "":
            self.load_model(filename)

    def btn_save_clicked(self) -> None:
        filename, _ = getsavefilename(
            self,
            "Save model",
            "model.json",
            "json file (*.json)",
        )
        if filename == "":
            return
        try:
            self.model.save_model(filename)
        except Exception as e:
            self.logger.warning(f"Failed to save file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "File save error",
                f"Could not save file '{filename}': {e}",
            )
            return

    def btn_export_clicked(self) -> None:
        filename, _ = getsavefilename(
            self,
            "Export results",
            "data.h5",
            "hdf5 file (*.h5)",
        )
        if filename == "":
            return
        try:
            self.model.save_to_h5(filename)
        except Exception as e:
            self.logger.warning(f"Failed to export to file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export to file '{filename}': {e}",
            )
            return

    def btn_edit_clicked(self) -> None:
        if not self.model.neurongroups:
            message = "There is no model to be edited, first open an existing one or define a new one"
            self.logger.critical(message)
            QtWidgets.QMessageBox.critical(
                self,
                "Model edit error",
                message,
            )
        else:
            mdiag = ModelEditorDialog(self.model, parent=self)

            if mdiag.exec():
                self.update_model(mdiag.b2model)

    @with_wait_cursor
    def load_model(self, filename: str) -> None:
        try:
            self.progress_dialog.setLabelText("Loading model...")
            model = B2Model.load_model(filename, report=self._update_load_progress)
            self.logger.info(f"Model {self.model.name} loaded from {filename}")
            self.update_model(model)

        except Exception as e:
            self.logger.warning(f"Failed to load model from file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Model construction error",
                f"Could not load model from file '{filename}': {e}",
            )
            self.progress_dialog.reset()
            return

    @with_wait_cursor
    def update_model(self, model: B2Model) -> None:
        self.model = model
        self.stim_console.set_model(self.model)
        self.continue_simulation_action.setText("&Run simulation")
        self.heatmap_widget.clear_data()
        self.trace_widget.clear_data()
        self.stim_editor.set_model(self.model)
        self.stim_editor.action_toggle_draw.setChecked(False)
        self.cmb_neuron.clear()
        self.neurons.clear()
        self.cmb_synapse.clear()
        self.synapses.clear()
        self.cmb_spike.clear()
        self.spikes.clear()
        if self.model is not None:
            self.stim_console.set_stims(self.stim_editor.stims_dict)
            for m in self.model.network.objects:
                if isinstance(m, b2.StateMonitor):
                    dct = (
                        self.neurons
                        if isinstance(m.source, b2.NeuronGroup)
                        else self.synapses
                        if isinstance(m.source, b2.Synapses)
                        else None
                    )
                    if dct is not None:
                        for var in m.record_variables:
                            dct[f"{m.source.name} / {var}"] = (m, var)
                elif isinstance(m, b2.SpikeMonitor):
                    self.spikes[f"{m.source.name}"] = m
            self.logger.debug(f"neurons: {self.neurons}")
            self.logger.debug(f"synapses: {self.synapses}")
            self.logger.debug(f"spikes: {self.spikes}")
            self.cmb_neuron.addItems(list(self.neurons.keys()))
            self.cmb_synapse.addItems(list(self.synapses.keys()))
            self.cmb_spike.addItems(list(self.spikes.keys()))

    @with_wait_cursor
    def set_neuron(self, neuron_name: str) -> None:
        if (
            self.external_simulation_data is not None
            and neuron_name in self.external_simulation_data
        ):
            self.set_neuron_external(neuron_name)
        if neuron_name not in self.neurons:
            return
        monitor, variable = self.neurons[neuron_name]
        timestamps = monitor.t / b2.msecond
        if timestamps.shape[0] == 0:
            self.trace_widget.clear_data()
            return
        traces = monitor.variables[variable].get_value().T
        self.logger.info(
            f"Displaying {neuron_name} in TraceWidget: {timestamps.shape} timestamps, {traces.shape} data"
        )
        self.trace_widget.set_data(traces, timestamps)

    def set_neuron_external(self, neuron_name: str) -> None:
        try:
            assert self.external_simulation_data is not None
            data = self.external_simulation_data[neuron_name]
            timestamps = data["timestamps"]
            assert isinstance(timestamps, np.ndarray)
            timestamps / 0.001  # convert to millisecondss
            traces = data["traces"]
            assert isinstance(traces, np.ndarray)
            self.logger.info(
                f"Displaying external TraceWidget data {neuron_name}: {timestamps.shape} timestamps, {traces.shape} data"
            )
            self.trace_widget.set_data(traces, timestamps)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation data error",
                f"Could not load external trace simulation data {neuron_name}: {e}",
            )
            self.logger.exception(e)

    @with_wait_cursor
    def set_synapse(self, synapse_name: str) -> None:
        if (
            self.external_simulation_data is not None
            and synapse_name in self.external_simulation_data
        ):
            return self.set_synapse_external(synapse_name)
        if synapse_name not in self.synapses:
            return
        monitor, variable = self.synapses[synapse_name]
        matrix, timestamps = construct_synapse_matrix(
            monitor, variable, self.settings.max_ram_synapse_plot_matrix
        )
        if matrix.size == 0:
            self.heatmap_widget.clear_data()
            return
        self.logger.info(
            f"Displaying {synapse_name} in HeatmapWidget: {timestamps.shape} timestamps, {matrix.shape} data"
        )
        self.heatmap_widget.set_data(matrix, timestamps)

    def set_synapse_external(self, synapse_name: str) -> None:
        try:
            assert self.external_simulation_data is not None
            data = self.external_simulation_data[synapse_name]
            timestamps = data["timestamps"]
            assert isinstance(timestamps, np.ndarray)
            matrix = data["matrix"]
            assert isinstance(matrix, np.ndarray)
            self.logger.info(
                f"Displaying external HeatmapWidget data {synapse_name}: {timestamps.shape} timestamps, {matrix.shape} data"
            )
            self.heatmap_widget.set_data(
                matrix, timestamps / 0.001
            )  # convert to milliseconds
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation data error",
                f"Could not load external heatmap simulation data {synapse_name}: {e}",
            )
            self.logger.exception(e)

    @with_wait_cursor
    def set_spike(self, spike_name: str) -> None:
        if (
            self.external_simulation_data is not None
            and spike_name in self.external_simulation_data
        ):
            return self.set_spike_external(spike_name)
        if spike_name not in self.spikes:
            return
        monitor = self.spikes[spike_name]
        spikes = monitor.variables["i"].get_value().T
        timestamps = monitor.t / b2.msecond
        dt = monitor.clock.dt / b2.msecond
        n_neurons = monitor.source.N
        n_timesteps = int(monitor.clock.t / monitor.clock.dt)
        if timestamps.shape[0] == 0:
            self.spike_widget.clear_data()
            return
        self.logger.info(
            f"Displaying {spike_name} in SpikeWidget: {timestamps.shape} timestamps, {spikes.shape} data"
        )
        self.spike_widget.set_data(spikes, timestamps, dt, n_neurons, n_timesteps)

    def set_spike_external(self, spike_name: str) -> None:
        try:
            assert self.external_simulation_data is not None
            data = self.external_simulation_data[spike_name]
            timestamps = data["timestamps"]
            assert isinstance(timestamps, np.ndarray)
            timestamps /= 0.001  # convert to milliseconds
            spikes = data["spikes"]
            assert isinstance(spikes, np.ndarray)
            dt = float(data["dt"]) / 0.001  # convert to milliseconds
            assert isinstance(dt, float)
            n_neurons = data["n_neurons"]
            assert isinstance(n_neurons, int)
            n_timesteps = data["n_timesteps"]
            assert isinstance(n_timesteps, int)
            self.logger.info(
                f"Displaying external SpikeWidget data {spike_name}: {timestamps.shape} timestamps, {spikes.shape} data, dt {dt}, n_neurons {n_neurons}, n_timesteps {n_timesteps}"
            )
            self.spike_widget.set_data(spikes, timestamps, dt, n_neurons, n_timesteps)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation data error",
                f"Could not load external spike simulation data {spike_name}: {e}",
            )
            self.logger.exception(e)

    def btn_exit_clicked(self) -> None:
        self.close()

    def btn_store_checkpoint_clicked(self) -> None:
        checkpoint, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Store simulation checkpoint",
            "Store the current simulation state as a named checkpoint.\nEnter a new name, or select an existing checkpoint to overwrite:",
            self.model.checkpoints,
            current=0,
            editable=True,
        )
        if not ok:
            return
        self.model.store(checkpoint)

    def btn_restore_checkpoint_clicked(self) -> None:
        if len(self.model.checkpoints) == 0:
            QtWidgets.QMessageBox.information(
                self,
                "No simulation checkpoints",
                "No simulation checkpoints available to restore - try running a simulation first.",
            )
            return
        checkpoint, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Restore simulation checkpoint",
            "Restore simulation checkpoint",
            self.model.checkpoints,
            current=0,
            editable=False,
        )
        if not ok:
            return
        try:
            self.model.restore(checkpoint)
            self._update_data()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Error", f"{error}")
            return

    def btn_continue_simulation_clicked(self) -> None:
        simulation_time_str, ok = QtWidgets.QInputDialog.getText(
            self,
            "Simulate",
            "Simulation time (s)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            "1.0",
        )
        if not ok:
            return
        try:
            simulation_time = float(simulation_time_str)
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse simulation time '{simulation_time_str}' as a number",
            )
            return
        if self.external_simulation_function is not None:
            return self.do_external_simulation(simulation_time)
        try:
            self.progress_dialog.setLabelText("Compiling...")
            self.progress_dialog.setValue(0)
            self.progress_dialog.setVisible(True)
            QtCore.QCoreApplication.processEvents()
            self.model.run_simulation(
                simulation_time * b2.second,
                report=self._update_simulation_progress,
                timed_array_padding_seconds=self.settings.stim_padding_secs,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Simulation error",
                f"Could not run simulation: {e}",
            )
            self.logger.exception(e)
            return
        self._update_data()

    @with_wait_cursor
    def do_external_simulation(self, simulation_time: float) -> None:
        try:
            # pre-emptively pad timed arrays before simulating
            self.model._pad_timed_arrays(
                simulation_time, self.settings.stim_padding_secs
            )
            self.stim_editor.set_model(self.model)
            stims_dict = {
                k: {"array": v.timed_array.values, "dt": v.timed_array.dt}
                for k, v in self.stim_editor.stims_dict.items()
            }
            assert self.external_simulation_function is not None
            self.external_simulation_data = self.external_simulation_function(
                stims_dict,
                simulation_time,
            )
            assert isinstance(self.external_simulation_data, dict)
            self.cmb_neuron.clear()
            self.cmb_spike.clear()
            self.cmb_synapse.clear()
            for name, data in self.external_simulation_data.items():
                widget = data.get("widget", "")
                if widget == "trace":
                    self.cmb_neuron.addItem(name)
                elif widget == "spike":
                    self.cmb_spike.addItem(name)
                elif widget == "heatmap":
                    self.cmb_synapse.addItem(name)
                else:
                    self.logger.info(
                        f"Ignoring unknown widget type {widget} for external simulation data {name}"
                    )
                self.logger.info(f"External simulation data: {name}: {data.keys()}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation error",
                f"Could not run external simulation: {e}",
            )
            self.logger.exception(e)

    def btn_skip_to_start_clicked(self) -> None:
        self.heatmap_widget.setCurrentIndex(0)

    def btn_stop_clicked(self) -> None:
        self.heatmap_widget.play(0)
        self.btn_skip_to_start_clicked()

    def btn_play_clicked(self) -> None:
        self.heatmap_widget.togglePause()

    def btn_playback_speed_clicked(self) -> None:
        new_speed, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set playback speed",
            "Playback speed (1.0=realtime, 0.5=half speed, 2.0=double speed, etc.)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.heatmap_widget.relative_playback_speed}",
        )
        if not ok:
            return
        try:
            new_speed_as_float = float(new_speed)
            if new_speed_as_float <= 0:
                raise ValueError("Playback speed must be a positive number")
            self.heatmap_widget.set_relative_playback_speed(new_speed_as_float)
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse playback speed '{new_speed}' as a valid positive number",
            )
            return

    def btn_skip_to_end_clicked(self) -> None:
        self.heatmap_widget.setCurrentIndex(self.heatmap_widget.nframes() - 1)

    def link_x_axes(self, checked: bool) -> None:
        link_target = self.stim_editor.stim_widget if checked else None
        for widget in [self.trace_widget, self.spike_widget]:
            widget.setXLink(link_target)

    def link_y_axes(self, checked: bool) -> None:
        link_target = self.stim_editor.stim_widget if checked else None
        for widget in [self.trace_widget, self.spike_widget]:
            widget.setYLink(link_target)

    def choose_max_ram_synapse_plot_matrix(self) -> None:
        new_max_ram, ok = QtWidgets.QInputDialog.getText(
            self,
            "RAM limit",
            "RAM limit in MB for synapse matrix plot (0 for no limit)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.settings.max_ram_synapse_plot_matrix}",
        )
        if not ok:
            return
        try:
            self.settings.max_ram_synapse_plot_matrix = max(0, int(new_max_ram))
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse RAM limit '{new_max_ram}' as a number",
            )
            return
        self._update_data()

    def connect_external_simulation_function(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load external simulate.py",
            "simulate.py",
            "Python file (*.py)",
        )
        if filename != "":
            try:
                path = pathlib.Path(filename).resolve()
                self.external_simulation_function = runpy.run_path(filename)["simulate"]
                self.logger.info(f"Loaded custom simulate function from {path}")
                QtWidgets.QMessageBox.information(
                    self,
                    "Custom simulate function loaded",
                    f"Custom simulate function loaded from {path} - pressing simulate will now call this function until the GUI is restarted",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error loading simulate function",
                    f"Could not load simulate function from {path}: {e}",
                )
                self.logger.exception(e)

    def import_patchclamp_data(self) -> None:
        """Open dialog to import patch clamp data and integrate with current model"""
        dialog = OptoStimReadingsImportDialog(self)
        dialog.data_imported.connect(self.handle_general_import)
        dialog.exec_()

    def import_suite2p_spikes(self) -> None:
        """Open dialog to import Suite2p spike data"""
        dialog = FluorescentSuite2pImportDialog(self)
        dialog.data_imported.connect(self.handle_general_import)  # Use same handler
        dialog.exec_()

    def import_suite2p_traces(self) -> None:
        """Open dialog to import Suite2p Fneu trace data"""
        dialog = FluorescentSuite2pContrastingTracesImportDialog(self)
        dialog.data_imported.connect(self.handle_general_import)  # Use same handler
        dialog.exec_()

    def handle_general_import(self, imported_data: dict) -> None:
        """Handle imported patch clamp data and expand stims if needed"""
        try:
            # Get metadata
            metadata = imported_data.get("_metadata", {})
            n_timepoints = metadata.get("n_timepoints", 0)
            dt = metadata.get("dt", 1.0/30.0)
            
            # Expand stims to match imported data length if needed
            if self.model.parameters and n_timepoints > 0:
                self.expand_stims_to_length(n_timepoints, dt)
            
            # Store the imported data for display, preserving existing data
            if self.external_simulation_data is None:
                self.external_simulation_data = {}
            
            # Update with new data (preserves existing traces/spikes from other imports)
            for key, value in imported_data.items():
                if not key.startswith("_"):
                    self.external_simulation_data[key] = value
            
            # Update the GUI to show imported data
            self._update_external_data_widgets()
            
            self.logger.info(f"Successfully imported patch clamp data with {n_timepoints} timepoints")
            
        except Exception as e:
            self.logger.exception("Error handling patch clamp import")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error", 
                f"Error processing imported data:\n{str(e)}"
            )

    def expand_stims_to_length(self, target_length: int, target_dt: float) -> None:
        """Expand stimulation arrays to match imported data length"""
        if not self.model.parameters:
            return
            
        for param_name, param_value in self.model.parameters.items():
            if hasattr(param_value, 'values') and hasattr(param_value, 'dt'):
                # This is a TimedArray
                current_values = param_value.values
                current_length = current_values.shape[0] if current_values.ndim > 0 else 0
                
                if current_length < target_length:
                    # Expand with zeros
                    if current_values.ndim == 1:
                        # 1D array
                        padding = np.zeros(target_length - current_length)
                        new_values = np.concatenate([current_values, padding])
                    elif current_values.ndim == 2:
                        # 2D array (timepoints x neurons)
                        n_neurons = current_values.shape[1]
                        padding = np.zeros((target_length - current_length, n_neurons))
                        new_values = np.concatenate([current_values, padding], axis=0)
                    else:
                        # Skip arrays with unexpected dimensions
                        continue
                        
                    # Create new TimedArray with expanded values
                    import brian2 as b2
                    new_timed_array = b2.TimedArray(new_values, dt=target_dt*b2.second)
                    self.model.parameters[param_name] = new_timed_array
                    
                    self.logger.info(f"Expanded {param_name} from {current_length} to {target_length} timepoints")

    def _update_external_data_widgets(self) -> None:
        """Update widgets to display external simulation data"""
        if not self.external_simulation_data:
            return
            
        # Update comboboxes with new data
        if hasattr(self, 'cmb_neuron'):
            self.cmb_neuron.clear()
            trace_datasets = [key for key, value in self.external_simulation_data.items() 
                            if value.get("widget") == "trace"]
            self.logger.info(f"Available trace datasets: {trace_datasets}")
            if trace_datasets:
                self.cmb_neuron.addItems(trace_datasets)
                
        if hasattr(self, 'cmb_spike'):
            self.cmb_spike.clear() 
            spike_datasets = [key for key, value in self.external_simulation_data.items()
                            if value.get("widget") == "spike"]
            self.logger.info(f"Available spike datasets: {spike_datasets}")
            if spike_datasets:
                self.cmb_spike.addItems(spike_datasets)

    def choose_brian2_backend(self) -> None:
        backends = ["cython", "numpy"]
        try:
            current_backend_index = backends.index(self.settings.brian2_backend)
        except ValueError:
            current_backend_index = 0
        new_backend, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Brian2 backend",
            "Choose the Brian2 backend for simulations",
            backends,
            current_backend_index,
            editable=False,
        )
        if not ok:
            return
        self.logger.info(f"Setting Brian2 backend to {new_backend}")
        b2.prefs.codegen.target = new_backend
        self.settings.brian2_backend = new_backend

    def choose_stim_padding_secs(self) -> None:
        new_padding_secs, ok = QtWidgets.QInputDialog.getText(
            self,
            "Stimulus padding",
            "Number of seconds of padding for stimuli beyond simulation time",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.settings.stim_padding_secs}",
        )
        if not ok:
            return
        try:
            self.settings.stim_padding_secs = max(0, int(new_padding_secs))
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse padding '{new_padding_secs}' as a number",
            )
            return
        self.logger.info(
            f"Setting stimulus padding to {self.settings.stim_padding_secs}s"
        )

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About neuroseq",
            f"NeuroSeq\n\nVersion {__version__}\n\n"
            + "https://github.com/ssciwr/neuroseq",
        )

    def _create_progressbar(self) -> None:
        self.progress_dialog = QtWidgets.QProgressDialog(
            "Simulating...", "", 0, 100, self
        )
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setMinimumDuration(10)
        self.progress_dialog.reset()

    def _create_menus_and_toolbars(self) -> None:
        menu = self.menuBar()
        self.menu_file = menu.addMenu("&File")
        self.toolbar_file = QtWidgets.QToolBar("File")
        self.addToolBar(self.toolbar_file)
        self.menu_simulate = menu.addMenu("&Simulate")
        self.toolbar_simulate = QtWidgets.QToolBar("Simulate")
        self.addToolBar(self.toolbar_simulate)
        self.menu_playback = menu.addMenu("&Playback")
        self.toolbar_playback = QtWidgets.QToolBar("Playback")
        self.addToolBar(self.toolbar_playback)
        self.menu_view = menu.addMenu("&View")
        self.toolbar_view = QtWidgets.QToolBar("View")
        self.addToolBar(self.toolbar_view)
        self.toolbar_data = QtWidgets.QToolBar("Data")
        self.addToolBar(self.toolbar_data)
        self.menu_settings = menu.addMenu("&Settings")
        self.menu_help = menu.addMenu("&Help")
        self._create_actions_data(toolbar=self.toolbar_data, menu=None)

    def _create_actions(self) -> None:
        self._create_actions_file(self.menu_file, self.toolbar_file)
        self._create_actions_simulate(self.menu_simulate, self.toolbar_simulate)
        self._create_actions_playback(self.menu_playback, self.toolbar_playback)
        self._create_actions_view(self.menu_view, self.toolbar_view)
        self._create_actions_settings(menu=self.menu_settings, toolbar=None)
        self._create_actions_help(menu=self.menu_help, toolbar=None)

    def _create_actions_file(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&New",
            self.btn_new_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+N",
            icon=qta.icon("mdi6.file"),
        )
        add_action(
            "&Open model",
            self.btn_open_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+O",
            icon=qta.icon("mdi6.folder-open"),
        )
        add_action(
            "&Save model",
            self.btn_save_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+S",
            icon=qta.icon("mdi6.content-save"),
        )
        add_action(
            "&Export data",
            self.btn_export_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+E",
            icon=qta.icon("mdi6.content-save-move"),
        )
        add_action(
            "&Edit model",
            self.btn_edit_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+E",
            icon=qta.icon("mdi6.cog"),
        )
        menu.addSeparator()
        add_action(
            "Import &Patchclamp",
            self.import_patchclamp_data,
            menu,
            None,
            shortcut="Ctrl+I",
            icon=qta.icon("mdi6.import"),
        )
        add_action(
            "Import Suite2p &Spikes",
            self.import_suite2p_spikes,
            menu,
            None,
            shortcut="Ctrl+Shift+I",
            icon=qta.icon("mdi6.import"),
        )
        add_action(
            "Import Suite2p &Traces (Fneu)",
            self.import_suite2p_traces,
            menu,
            None,
            shortcut="Ctrl+Shift+T",
            icon=qta.icon("mdi6.import"),
        )
        menu.addSeparator()
        add_action(
            "E&xit",
            self.btn_exit_clicked,
            menu,
            None,
            shortcut=None,
            icon=qta.icon("mdi6.exit-to-app"),
        )

    def _create_actions_simulate(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        self.continue_simulation_action = add_action(
            "&Run simulation",
            self.btn_continue_simulation_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+R",
            icon=qta.icon("mdi6.head"),
        )
        add_action(
            "&Store simulation checkpoint",
            self.btn_store_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+S",
            icon=qta.icon("mdi6.head-check"),
        )
        add_action(
            "&Restore simulation checkpoint",
            self.btn_restore_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+R",
            icon=qta.icon("mdi6.head-sync"),
        )

    def _create_actions_playback(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&Set playback speed",
            self.btn_playback_speed_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.speedometer"),
        )
        add_action(
            "&Skip to start",
            self.btn_skip_to_start_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.skip-previous"),
        )
        add_action(
            "&Stop",
            self.btn_stop_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.stop"),
        )
        add_action(
            "&Play / Pause",
            self.btn_play_clicked,
            menu,
            toolbar,
            shortcut="Space",
            icon=qta.icon("mdi6.play-pause"),
        )
        add_action(
            "&Skip to end",
            self.btn_skip_to_end_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.skip-next"),
        )

    def _create_actions_view(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        for name, widget, icon in [
            ("&Stimuli", self.stim_editor, "mdi6.chart-timeline"),
            ("&Traces", self.trace_widget, "mdi.waves"),
            ("&Spikes", self.spike_widget, "ph.barcode-light"),
            ("&Heatmap", self.heatmap_widget, "mdi6.data-matrix"),
            ("&Custom analysis", self.custom_analysis_widget, "mdi6.chart-histogram"),
            ("&Console", self.stim_console, "mdi6.console"),
        ]:
            add_action(
                f"Show {name}",
                lambda toggled, w=widget: w.setVisible(toggled),
                menu,
                toolbar,
                icon=qta.icon(f"{icon}"),
                checkable=True,
                is_checked=True,
            )
        add_action(
            "Link X axes",
            self.link_x_axes,
            menu,
            toolbar,
            icon=qta.icon("mdi6.arrow-horizontal-lock"),
            checkable=True,
            is_checked=True,
        )
        add_action(
            "Link Y axes",
            self.link_y_axes,
            menu,
            toolbar,
            icon=qta.icon("mdi6.arrow-vertical-lock"),
            checkable=True,
            is_checked=True,
        )

    def _create_actions_settings(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action("&Brian2 simulation backend", self.choose_brian2_backend, menu)
        add_action("&Stimulus automatic padding", self.choose_stim_padding_secs, menu)
        add_action(
            "&Synapse matrix RAM limit", self.choose_max_ram_synapse_plot_matrix, menu
        )
        add_action(
            "&Connect external simulation function",
            self.connect_external_simulation_function,
            menu,
        )

    def _create_actions_help(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action("&About", self.about, menu)

    def _create_actions_data(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        self.cmb_stim = QtWidgets.QComboBox()
        toolbar.addWidget(QtWidgets.QLabel("  Stim data: "))
        toolbar.addWidget(self.cmb_stim)
        toolbar.addWidget(QtWidgets.QLabel("  Neuron data: "))
        toolbar.addWidget(self.cmb_neuron)
        toolbar.addWidget(QtWidgets.QLabel("  Spike data: "))
        toolbar.addWidget(self.cmb_spike)
        toolbar.addWidget(QtWidgets.QLabel("  Synapse data: "))
        toolbar.addWidget(self.cmb_synapse)
