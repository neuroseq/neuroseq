from __future__ import annotations

import logging
import pathlib
import runpy
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import cast

import brian2 as b2
import brian2.numpy_ as np
import numpy.typing as npt
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq import __version__
from neuroseq.b2_model import B2Model
from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.gui_settings import GuiSettings
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.import_dialog import Suite2pImportDialog
from neuroseq.import_dialog import Suite2pImportResult
from neuroseq.importer import CellposeImportResult
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.resource_monitor_label import ResourceMonitorLabel
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim_console import StimConsole
from neuroseq.stim_editor import StimEditor
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import add_action
from neuroseq.utils import add_items_to_combobox
from neuroseq.utils import construct_synapse_matrix
from neuroseq.utils import example_models_path
from neuroseq.utils import with_wait_cursor

PayloadDict = dict[str, object]


class Gui(QtWidgets.QMainWindow):
    def __init__(self, filename: pathlib.Path | str | None = None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("starting Gui")
        self.settings = GuiSettings()
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(ResourceMonitorLabel())
        self.model = B2Model()
        self.external_simulation_function: Callable | None = None
        self.external_simulation_data: dict[str, PayloadDict] | None = None
        self.neurons: dict[str, tuple[b2.StateMonitor, str]] = {}
        self.synapses: dict[str, tuple[b2.StateMonitor, str]] = {}
        self.spikes: dict[str, b2.SpikeMonitor] = {}
        self.cellpose_data: CellposeImportResult | None = None
        self.setWindowTitle("NeuroSeq")
        self.cmb_neuron = QtWidgets.QComboBox()
        self.cmb_spike = QtWidgets.QComboBox()
        self.cmb_synapse = QtWidgets.QComboBox()
        self._create_menus_and_toolbars()
        self.setWindowIcon(qta.icon("fa6s.brain"))
        self.controller = ConstrainedPanZoomController()
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
        split_vertical = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical, handleWidth=0
        )
        self.stim_editor = StimEditor(
            controller=self.controller,
            cmb_stim=self.cmb_stim,
            toolbar=self.toolbar_stim_editor,
        )
        split_vertical.addWidget(self.stim_editor)
        self.trace_widget = TraceWidget(self.controller)
        split_vertical.addWidget(self.trace_widget)
        self.spike_widget = SpikeWidget(self.controller)
        split_vertical.addWidget(self.spike_widget)
        # heatmap / custom analysis widget / console
        split_vertical.addWidget(split_left_right)
        main_horizontal_layout.addWidget(split_vertical)
        self.setCentralWidget(central_widget)
        self._create_actions()
        self.showMaximized()
        self._create_progressbar()
        # connect widgets
        self.heatmap_widget.connect_timeseries_widget_timeline(
            self.stim_editor.stim_widget
        )
        self.stim_editor.stim_widget.connect_heatmap_widget(self.heatmap_widget)
        self.heatmap_widget.connect_timeseries_widget_timeline(self.trace_widget)
        self.trace_widget.connect_heatmap_widget(self.heatmap_widget)
        # re-enable when fastplotlib scatter plots support timeline
        # self.heatmap_widget.connect_timeseries_widget_timeline(self.spike_widget)
        self.spike_widget.connect_heatmap_widget(self.heatmap_widget)
        self.stim_console.sig_stims_values_dict_updated.connect(
            self.stim_editor.update_stim_values_dict
        )
        self.trace_widget.custom_analysis_requested.connect(self._custom_analysis)
        self.spike_widget.custom_analysis_requested.connect(self._custom_analysis)
        self.cmb_neuron.currentTextChanged.connect(self.set_neuron)
        self.cmb_synapse.currentTextChanged.connect(self.set_synapse)
        self.cmb_spike.currentTextChanged.connect(self.set_spike)
        if filename is None:
            filename = example_models_path / "default_model.nseq"
        self.load_model(str(filename))

    def _custom_analysis(
        self,
        data_type: CustomAnalysisDataType | None,
        analysis_type: CustomAnalysisType | None,
        new_window: bool,
        neuron_slice: slice | None,
        time_slice: slice | None,
        data: dict[str, int | float | npt.NDArray],
    ):
        if new_window:
            analysis_window = CustomAnalysisWindow(parent=self)
            analysis_window.analysis_widget.set_data(
                data_type,
                analysis_type,
                neuron_slice,
                time_slice,
                data,
            )
            analysis_window.setWindowTitle(
                analysis_window.analysis_widget.subplot.get_title()
            )
            analysis_window.show()
        else:
            self.custom_analysis_widget.set_data(
                data_type, analysis_type, neuron_slice, time_slice, data
            )

    def _update_data(self, keep_undo_history: bool) -> None:
        if self.model.network.t > 0:
            self.continue_simulation_action.setText("&Continue simulation")
        else:
            self.continue_simulation_action.setText("&Run simulation")
        self.stim_editor.set_model(self.model, keep_undo_history)
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
        self.load_model(str(example_models_path / "default_model.nseq"))

    def btn_open_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load model",
            "model.nseq",
            "neuroseq model (*.nseq)",
        )
        if filename != "":
            self.load_model(filename)

    def btn_save_clicked(self) -> None:
        filename, _ = getsavefilename(
            self,
            "Save model",
            "model.nseq",
            "neuroseq model (*.nseq)",
        )
        if filename == "":
            return
        filename = pathlib.Path(filename)
        if filename.suffix != ".nseq":
            filename = filename.with_suffix(".nseq")
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.model.save_model(filename)
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
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

        except Exception as e:
            self.logger.warning(f"Failed to load model from file {filename}: {e}")
            self.logger.exception(e)
            QtWidgets.QMessageBox.critical(
                self,
                "Model construction error",
                f"Could not load model from file '{filename}': {e}",
            )
            self.progress_dialog.reset()
            return
        self.update_model(model)

    @with_wait_cursor
    def update_model(self, model: B2Model) -> None:
        self.model = model
        self.controller.max_height = 1
        self.controller.max_width = 1
        self.stim_console.set_model(self.model)
        self.continue_simulation_action.setText("&Run simulation")
        self.heatmap_widget.clear_data()
        self.trace_widget.clear_data()
        self.stim_editor.set_model(self.model)
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
            add_items_to_combobox(self.cmb_neuron, list(self.neurons.keys()))
            add_items_to_combobox(self.cmb_synapse, list(self.synapses.keys()))
            add_items_to_combobox(self.cmb_spike, list(self.spikes.keys()))

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
        timestamps = monitor.t / b2.second
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
            self.heatmap_widget.set_data(matrix, timestamps)
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
        timestamps = monitor.t / b2.second
        dt = monitor.clock.dt / b2.second
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
            spikes = data["spikes"]
            assert isinstance(spikes, np.ndarray)
            dt = self._safe_float(data.get("dt"), 0.0)
            n_neurons, n_timesteps = self._ensure_spike_metadata(data, dt)
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
            self._update_data(keep_undo_history=False)
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
        self._update_data(keep_undo_history=True)

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
            for name, data in self.external_simulation_data.items():
                self.logger.info(f"External simulation data: {name}: {data.keys()}")
            self._update_external_data_widgets()
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

    def toggle_selection_text(self):
        for widget in [
            self.stim_editor.stim_widget,
            self.trace_widget,
            self.spike_widget,
        ]:
            widget.toggle_selection_text()

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
        self._update_data(keep_undo_history=True)

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

    def import_suite2p_data(self) -> None:
        """Open dialog to import Suite2p traces (and optional spikes)"""
        dialog = Suite2pImportDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            imported_data = dialog.get_imported_data()
            if imported_data:
                self.handle_general_import(imported_data)

    def import_cellpose_data(self) -> None:
        """Open dialog to import Cellpose segmentation data"""
        dialog = CellposeImportDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            imported_data = dialog.get_imported_data()
            if imported_data:
                self.cellpose_data = imported_data
                QtWidgets.QMessageBox.information(
                    self,
                    "Cellpose Data Loaded",
                    "Cellpose data has been loaded.",
                )

    # todo: Refactor this out of this function. It's not particularly/relaly the GUI except the widget interaction.
    # Certainly the code about clearing spks etc, this management of state, when we have multiple experiments, needs to be elsewhere.
    def handle_general_import(
        self, imported_data: Suite2pImportResult | Mapping[str, object]
    ) -> None:
        """Handle imported data with timeline concatenation"""
        try:
            dt_for_expand: float | None = None
            replace_mode = bool(imported_data.get("_replace_existing"))
            cleared_widget_types: set[str] = set()
            imported_widget_types: set[str] = set()

            # Initialize external data if needed
            if self.external_simulation_data is None:
                self.external_simulation_data = {}

            # Process each dataset and concatenate to existing timeline
            for key, value in imported_data.items():
                if key.startswith("_"):
                    continue

                if not isinstance(value, Mapping):
                    self.logger.warning(
                        f"Ignoring imported dataset '{key}' because it is not a mapping"
                    )
                    continue

                payload: PayloadDict = dict(value)

                # Use the key as-is since we're no longer using timestamps
                base_key = key

                widget_type_obj = payload.get("widget")
                widget_type = (
                    widget_type_obj if isinstance(widget_type_obj, str) else ""
                )

                if widget_type:
                    imported_widget_types.add(widget_type)

                if (
                    replace_mode
                    and widget_type in {"trace", "spike"}
                    and widget_type not in cleared_widget_types
                ):
                    self._clear_external_widget_data(widget_type)
                    cleared_widget_types.add(widget_type)

                if widget_type == "trace":
                    dataset_dt = self._safe_float(payload.get("dt"), 1.0 / 30.0)
                    dt_for_expand = dataset_dt
                    self._concatenate_trace_data(base_key, payload, dataset_dt)
                elif widget_type == "spike":
                    dataset_dt = self._safe_float(payload.get("dt"), 1.0 / 30.0)
                    if dt_for_expand is None:
                        dt_for_expand = dataset_dt
                    self._concatenate_spike_data(base_key, payload, dataset_dt)
                else:
                    # For other data types, just store with base key
                    self.external_simulation_data[base_key] = payload

            # extend not replace mode
            if (
                not replace_mode
                and "trace" in imported_widget_types
                and "spike" not in imported_widget_types
                and "spike" not in cleared_widget_types
            ):
                self._clear_external_widget_data("spike")

            # Calculate total timepoints across all concatenated data for stimulus expansion
            total_timepoints = self._get_max_timepoints()
            if (
                self.model.parameters
                and total_timepoints > 0
                and dt_for_expand is not None
            ):
                self.expand_stims_to_length(total_timepoints, dt_for_expand)

            # Update the GUI to show imported data
            self._update_external_data_widgets()

            self.logger.info(
                f"Successfully imported and concatenated data (total timepoints: {total_timepoints})"
            )

        except Exception as e:
            self.logger.exception("Error handling import")
            QtWidgets.QMessageBox.critical(
                self, "Import Error", f"Error processing imported data:\n{str(e)}"
            )

    @staticmethod
    def _safe_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.shape == ():
                try:
                    return int(value.item())
                except (TypeError, ValueError, OverflowError):
                    return None
            return None
        try:
            return int(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        if value is None:
            return default
        try:
            return float(cast(Any, value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _infer_spike_neuron_count(spikes: object) -> int:
        if isinstance(spikes, np.ndarray) and spikes.size > 0:
            return int(np.max(spikes)) + 1
        return 0

    @staticmethod
    def _infer_spike_timestep_count(timestamps: object, dt: float) -> int:
        if isinstance(timestamps, np.ndarray) and timestamps.size > 0 and dt > 0:
            max_timestamp = float(np.max(timestamps))
            approx_steps = int(np.round(max_timestamp / dt))
            return max(approx_steps + 1, 1)
        return 0

    def _ensure_spike_metadata(
        self, payload: PayloadDict, dt: float
    ) -> tuple[int, int]:
        n_neurons = self._safe_int(payload.get("n_neurons"))
        n_timesteps = self._safe_int(payload.get("n_timesteps"))
        spikes = payload.get("spikes")
        timestamps = payload.get("timestamps")

        if n_neurons is None:
            n_neurons = self._infer_spike_neuron_count(spikes)
            payload["n_neurons"] = n_neurons
        if n_timesteps is None:
            n_timesteps = self._infer_spike_timestep_count(timestamps, dt)
            payload["n_timesteps"] = n_timesteps
        return n_neurons, n_timesteps

    def _concatenate_trace_data(
        self, base_key: str, new_value: PayloadDict, dt: float
    ) -> None:
        """Concatenate trace data to existing timeline"""
        new_timestamps = new_value.get("timestamps")
        new_traces = new_value.get("traces")

        if self.external_simulation_data and base_key in self.external_simulation_data:
            # Existing data - concatenate along time axis
            existing_data = self.external_simulation_data[base_key]
            existing_timestamps = existing_data.get("timestamps")
            existing_traces = existing_data.get("traces")

            if (
                isinstance(existing_timestamps, np.ndarray)
                and isinstance(existing_traces, np.ndarray)
                and isinstance(new_timestamps, np.ndarray)
                and isinstance(new_traces, np.ndarray)
            ):
                # Calculate time offset (continue from where last data ended)
                time_offset = float(existing_timestamps[-1]) + dt
                adjusted_new_timestamps = new_timestamps + time_offset

                # Concatenate along time axis (axis=1 for traces)
                combined_timestamps = np.concatenate(
                    [existing_timestamps, adjusted_new_timestamps]
                )
                combined_traces = np.concatenate([existing_traces, new_traces], axis=1)

                if self.external_simulation_data:
                    self.external_simulation_data[base_key] = {
                        "widget": "trace",
                        "timestamps": combined_timestamps,
                        "traces": combined_traces,
                    }

                self.logger.info(
                    f"Concatenated trace data for {base_key}: {existing_traces.shape} + {new_traces.shape} = {combined_traces.shape}"
                )
                return

        # No existing data - store as new
        if self.external_simulation_data is None:
            self.external_simulation_data = {}
        self.external_simulation_data[base_key] = new_value
        if isinstance(new_traces, np.ndarray):
            self.logger.info(
                f"Stored new trace data for {base_key}: {new_traces.shape}"
            )

    def _concatenate_spike_data(
        self, base_key: str, new_value: PayloadDict, dt: float
    ) -> None:
        """Concatenate spike data to existing timeline"""
        new_timestamps = new_value.get("timestamps")
        new_spikes = new_value.get("spikes")
        dataset_dt = self._safe_float(new_value.get("dt"), dt)
        new_n_neurons, new_n_timesteps = self._ensure_spike_metadata(
            new_value, dataset_dt
        )

        if self.external_simulation_data and base_key in self.external_simulation_data:
            # Existing data - concatenate spikes
            existing_data = self.external_simulation_data[base_key]
            existing_timestamps = existing_data.get("timestamps")
            existing_spikes = existing_data.get("spikes")
            existing_dt = self._safe_float(existing_data.get("dt"), dataset_dt)
            existing_n_neurons, existing_n_timesteps = self._ensure_spike_metadata(
                existing_data, existing_dt
            )

            if (
                isinstance(existing_timestamps, np.ndarray)
                and isinstance(existing_spikes, np.ndarray)
                and isinstance(new_timestamps, np.ndarray)
                and isinstance(new_spikes, np.ndarray)
            ):
                # Calculate time offset (continue from where last data ended)
                time_offset = float(existing_n_timesteps) * existing_dt
                adjusted_new_timestamps = new_timestamps + time_offset

                # Concatenate spike arrays
                combined_timestamps = np.concatenate(
                    [existing_timestamps, adjusted_new_timestamps]
                )
                combined_spikes = np.concatenate([existing_spikes, new_spikes])
                combined_n_timesteps = int(existing_n_timesteps) + int(new_n_timesteps)

                if self.external_simulation_data:
                    self.external_simulation_data[base_key] = {
                        "widget": "spike",
                        "timestamps": combined_timestamps,
                        "spikes": combined_spikes,
                        "dt": dt,
                        "n_neurons": max(existing_n_neurons, new_n_neurons),
                        "n_timesteps": combined_n_timesteps,
                    }

                self.logger.info(
                    f"Concatenated spike data for {base_key}: {len(existing_spikes)} + {len(new_spikes)} = {len(combined_spikes)} spikes"
                )
                return

        # No existing data - store as new
        if self.external_simulation_data is None:
            self.external_simulation_data = {}
        self.external_simulation_data[base_key] = new_value
        if isinstance(new_spikes, np.ndarray):
            self.logger.info(
                f"Stored new spike data for {base_key}: {len(new_spikes)} spikes"
            )

    def _clear_external_widget_data(self, widget_type: str) -> None:
        """Remove existing external data for the specified widget type."""
        if not self.external_simulation_data:
            return

        keys_to_remove = [
            key
            for key, value in self.external_simulation_data.items()
            if value.get("widget") == widget_type
        ]
        if not keys_to_remove:
            return

        for key in keys_to_remove:
            del self.external_simulation_data[key]

        if widget_type == "trace":
            self.trace_widget.clear_data()
        elif widget_type == "spike":
            self.spike_widget.clear_data()

        self.logger.info(
            f"Cleared existing {widget_type} data before replacement (removed: {keys_to_remove})"
        )

    def _get_max_timepoints(self) -> int:
        """Get maximum timepoints across all external data for stimulus expansion"""
        max_timepoints = 0
        if self.external_simulation_data:
            for data in self.external_simulation_data.values():
                if data.get("widget") == "trace":
                    timestamps = data.get("timestamps")
                    if isinstance(timestamps, np.ndarray):
                        max_timepoints = max(max_timepoints, len(timestamps))
                elif data.get("widget") == "spike":
                    n_timesteps_value = data.get("n_timesteps")
                    if isinstance(n_timesteps_value, int | float):
                        inferred_timesteps = int(n_timesteps_value)
                    else:
                        dt = self._safe_float(data.get("dt"), 0.0)
                        _, inferred_timesteps = self._ensure_spike_metadata(data, dt)
                    max_timepoints = max(max_timepoints, max(0, inferred_timesteps))
        return max_timepoints

    def expand_stims_to_length(self, target_length: int, target_dt: float) -> None:
        """Expand stimulation arrays to match imported data length"""
        if not self.model.parameters:
            return

        for param_name, param_value in self.model.parameters.items():
            if hasattr(param_value, "values") and hasattr(param_value, "dt"):
                # This is a TimedArray
                current_values = param_value.values
                current_length = (
                    current_values.shape[0] if current_values.ndim > 0 else 0
                )

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

                    new_timed_array = b2.TimedArray(
                        new_values, dt=target_dt * b2.second
                    )
                    self.model.parameters[param_name] = new_timed_array

                    self.logger.info(
                        f"Expanded {param_name} from {current_length} to {target_length} timepoints"
                    )

    def _update_external_data_widgets(self) -> None:
        """Update widgets to display external simulation data"""
        if not self.external_simulation_data:
            return

        widget_map = {
            "trace": getattr(self, "cmb_neuron", None),
            "spike": getattr(self, "cmb_spike", None),
            "heatmap": getattr(self, "cmb_synapse", None),
        }

        known_types = set(widget_map.keys())
        for name, data in self.external_simulation_data.items():
            widget_type = data.get("widget", "")
            if widget_type not in known_types:
                self.logger.info(
                    f"Ignoring unknown widget type {widget_type} for external simulation data {name}"
                )

        for widget_type, combo in widget_map.items():
            if combo is None:
                continue

            current_selection = combo.currentText()
            combo.clear()
            datasets = [
                key
                for key, value in self.external_simulation_data.items()
                if value.get("widget") == widget_type
            ]
            self.logger.info(f"Available {widget_type} datasets: {datasets}")
            if datasets:
                datasets.sort(reverse=True)
                combo.addItems(datasets)
                if current_selection in datasets:
                    combo.setCurrentText(current_selection)
                else:
                    combo.setCurrentIndex(0)

    def connect_external_simulation_function_with_brian2(self) -> None:
        filename = "./external_simulation_example/run_brian2_externally.py"
        if filename != "":
            try:
                path = pathlib.Path(filename).resolve()
                self.external_simulation_function = runpy.run_path(filename)["simulate"]
                self.logger.info(f"Loaded custom simulate_brian2 function from {path}")
                QtWidgets.QMessageBox.information(
                    self,
                    "Custom simulate_brian2 function loaded",
                    f"Custom simulate_brian2 function loaded from {path} - pressing simulate will now call this function until the GUI is restarted",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error loading simulate function",
                    f"Could not load simulate function from {path}: {e}",
                )
                self.logger.exception(e)

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

        self.menu_experiment = menu.addMenu("&Experiment")
        self.toolbar_experiment = QtWidgets.QToolBar("Experiment")
        self.addToolBar(self.toolbar_experiment)

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
        self._create_action_experiment(self.menu_experiment, self.toolbar_experiment)

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
        # add "open example model" item to the file menu which opens a sub-menu with all of the example models from src/neuroseq/example_models
        example_models_menu = menu.addMenu("&Open example model")
        for example_model_file in sorted(example_models_path.glob("*.nseq")):
            add_action(
                example_model_file.stem,
                lambda checked, f=example_model_file: self.load_model(str(f)),
                example_models_menu,
                None,
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
            icon=qta.icon("ph.polygon-light"),
        )
        menu.addSeparator()
        add_action(
            "Import Suite2p &Data",
            self.import_suite2p_data,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+I",
            icon=qta.icon("mdi6.import"),
        )
        add_action(
            "Import &Cellpose Data",
            self.import_cellpose_data,
            menu,
            None,
            shortcut="Ctrl+Shift+C",
            icon=qta.icon("mdi6.brain"),
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
            icon=qta.icon("mdi6.rocket-launch-outline"),
        )
        add_action(
            "&Store simulation checkpoint",
            self.btn_store_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+S",
            icon=qta.icon("mdi6.timeline-check-outline"),
        )
        add_action(
            "&Restore simulation checkpoint",
            self.btn_restore_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+R",
            icon=qta.icon("mdi.ray-start-arrow"),
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
            "Toggle selection text",
            self.toggle_selection_text,
            menu,
            toolbar,
            icon=qta.icon("mdi6.format-text"),
            checkable=True,
            is_checked=True,
        )

    def _create_actions_settings(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
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
        add_action(
            "&Run brian2 with external simulation function",
            self.connect_external_simulation_function_with_brian2,
            menu,
        )

    def _create_actions_help(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        add_action("&About", self.about, menu)

    def _create_actions_data(
        self, menu: QtWidgets.QMenu | None, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        if toolbar is None:
            return
        self.cmb_stim = QtWidgets.QComboBox()
        toolbar.addWidget(QtWidgets.QLabel("  Stim data: "))
        toolbar.addWidget(self.cmb_stim)
        toolbar.addWidget(QtWidgets.QLabel("  Neuron data: "))
        toolbar.addWidget(self.cmb_neuron)
        toolbar.addWidget(QtWidgets.QLabel("  Spike data: "))
        toolbar.addWidget(self.cmb_spike)
        toolbar.addWidget(QtWidgets.QLabel("  Synapse data: "))
        toolbar.addWidget(self.cmb_synapse)

    def _create_action_experiment(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&Run external stimulation function",
            self.connect_external_simulation_function_with_brian2,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+E",
            icon=qta.icon("mdi6.function"),
        )
