from __future__ import annotations

import logging
import pathlib
import runpy
from collections.abc import Callable

import brian2 as b2
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq import __version__
from neuroseq.b2_model import B2Model
from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.data_widget import DataWidget
from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData
from neuroseq.gui_settings import GuiSettings
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.import_dialog import Suite2pImportDialog
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog
from neuroseq.resource_monitor_label import ResourceMonitorLabel
from neuroseq.save_options_dialog import SaveOptionsDialog
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim_console import StimConsole
from neuroseq.stim_editor import StimEditor
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import add_action
from neuroseq.utils import add_items_to_combobox
from neuroseq.utils import example_models_path
from neuroseq.utils import getSimulationTime
from neuroseq.utils import with_wait_cursor


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
        self.setWindowTitle("NeuroSeq")
        self.cmb_trace = QtWidgets.QComboBox()
        self.cmb_trace.setMinimumWidth(120)
        self.cmb_spike = QtWidgets.QComboBox()
        self.cmb_spike.setMinimumWidth(120)
        self.cmb_synapse = QtWidgets.QComboBox()
        self.cmb_synapse.setMinimumWidth(120)
        self.cmb_data = QtWidgets.QComboBox()
        self.cmb_data.setMinimumWidth(120)
        self.controller = ConstrainedPanZoomController()
        self.trace_widget = TraceWidget(self.controller)
        self._create_menus_and_toolbars()
        self.setWindowIcon(qta.icon("fa6s.brain"))
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
            QtCore.Qt.Orientation.Vertical, handleWidth=0, childrenCollapsible=False
        )
        self.stim_editor = StimEditor(
            controller=self.controller,
            cmb_stim=self.cmb_stim,
            toolbar=self.toolbar_stim_editor,
        )
        split_vertical.addWidget(self.stim_editor)
        split_vertical.addWidget(self.trace_widget)
        self.spike_widget = SpikeWidget(self.controller)
        split_vertical.addWidget(self.spike_widget)
        self.data_widget = DataWidget(self.controller)
        split_vertical.addWidget(self.data_widget)
        self.data_widget.setVisible(False)
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
        self.data_widget.trace_widget.custom_analysis_requested.connect(
            self._custom_analysis
        )
        self.spike_widget.custom_analysis_requested.connect(self._custom_analysis)
        self.data_widget.spike_widget.custom_analysis_requested.connect(
            self._custom_analysis
        )
        self.cmb_trace.currentTextChanged.connect(self.set_neuron)
        self.cmb_synapse.currentTextChanged.connect(self.set_synapse)
        self.cmb_spike.currentTextChanged.connect(self.set_spike)
        self.cmb_data.currentTextChanged.connect(self.set_data)
        self.last_cellpose_import: CellposeImportResult | None = None
        if filename is None:
            filename = example_models_path / "default_model.nseq"
        self.load_model(str(filename))

    def _custom_analysis(
        self,
        data: TraceData | SpikeData | SynapseData,
        analysis_type: CustomAnalysisType | None,
        new_window: bool,
        neuron_slice: slice | None,
        time_slice: slice | None,
    ):
        if new_window:
            analysis_window = CustomAnalysisWindow(parent=self)
            analysis_window.analysis_widget.set_data(
                data,
                analysis_type,
                neuron_slice,
                time_slice,
            )
            analysis_window.setWindowTitle(
                analysis_window.analysis_widget.subplot.get_title()
            )
            analysis_window.show()
        else:
            self.custom_analysis_widget.set_data(
                data, analysis_type, neuron_slice, time_slice
            )

    def _update_data(self, keep_undo_history: bool) -> None:
        if self.model.network.t > 0:
            self.continue_simulation_action.setText("&Continue simulation")
        else:
            self.continue_simulation_action.setText("&Run simulation")
        self.stim_editor.set_model(self.model, keep_undo_history)
        self.set_neuron(self.cmb_trace.currentText())
        self.set_synapse(self.cmb_synapse.currentText())
        self.set_spike(self.cmb_spike.currentText())
        self.set_data(self.cmb_data.currentText())

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
        save_options_dialog = SaveOptionsDialog()
        if not save_options_dialog.exec():
            return
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
            self.model.save_model(
                filename,
                include_simulation_data=save_options_dialog.include_simulation_data,
                include_experimental_data=save_options_dialog.include_experimental_data,
            )
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
        self.trace_widget.clear_data()
        self.spike_widget.clear_data()
        self.heatmap_widget.clear_data()
        self.controller.clear_grid_lines()
        self.stim_editor.set_model(self.model)
        cmb_trace_text = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        cmb_synapse_text = self.cmb_synapse.currentText()
        self.cmb_synapse.clear()
        cmb_spike_text = self.cmb_spike.currentText()
        self.cmb_spike.clear()
        cmb_data_text = self.cmb_data.currentText()
        self.cmb_data.clear()
        if self.model is not None:
            self.stim_console.set_stims(self.stim_editor.stims_dict)
            add_items_to_combobox(self.cmb_trace, self.model.get_data_names(TraceData))
            self.cmb_trace.setCurrentText(cmb_trace_text)
            add_items_to_combobox(self.cmb_spike, self.model.get_data_names(SpikeData))
            self.cmb_spike.setCurrentText(cmb_spike_text)
            add_items_to_combobox(
                self.cmb_synapse, self.model.get_data_names(SynapseData)
            )
            self.cmb_synapse.setCurrentText(cmb_synapse_text)
            add_items_to_combobox(
                self.cmb_data,
                [f"{d} [trace]" for d in self.model.get_data_names(TraceData)]
                + [f"{d} [spike]" for d in self.model.get_data_names(SpikeData)]
                + [f"{d} [synapse]" for d in self.model.get_data_names(SynapseData)],
            )
            self.cmb_data.setCurrentText(cmb_data_text)

    @with_wait_cursor
    def set_neuron(self, cmb_name: str) -> None:
        data = self.model.get_data(cmb_name, TraceData)
        self.trace_widget.set_data(data)

    @with_wait_cursor
    def set_synapse(self, cmb_name: str) -> None:
        self.model.max_ram_synapse_plot_matrix = (
            self.settings.max_ram_synapse_plot_matrix
        )
        data = self.model.get_data(cmb_name, SynapseData)
        self.heatmap_widget.set_data(data)

    @with_wait_cursor
    def set_spike(self, cmb_name: str) -> None:
        data = self.model.get_data(cmb_name, SpikeData)
        self.spike_widget.set_data(data)

    @with_wait_cursor
    def set_data(self, cmb_name: str) -> None:
        data: TraceData | SpikeData | SynapseData | None = None
        if cmb_name.endswith(" [trace]"):
            data = self.model.get_data(cmb_name[:-8], TraceData)
        elif cmb_name.endswith(" [spike]"):
            data = self.model.get_data(cmb_name[:-8], SpikeData)
        elif cmb_name.endswith(" [synapse]"):
            data = self.model.get_data(cmb_name[:-10], SynapseData)
        self.data_widget.set_data(data)

    def btn_exit_clicked(self) -> None:
        self.close()

    def btn_store_checkpoint_clicked(self) -> None:
        checkpoint, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Store simulation checkpoint",
            "Store the current simulation state as a named checkpoint.\nEnter a new name, or select an existing checkpoint to overwrite:",
            [f"Checkpoint {len(self.model.checkpoints) + 1}"] + self.model.checkpoints,
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
        simulation_time = getSimulationTime(self)
        if simulation_time is None:
            return
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
    def _do_external_simulation(self, simulation_time: float):
        if self.external_simulation_function is None:
            return
        try:
            # pre-emptively pad timed arrays before simulating
            self.model.pad_timed_arrays(total_time_seconds=simulation_time)
            self.stim_editor.set_model(self.model)
            stims_dict = {
                k: {"array": v.timed_array.values, "dt": v.timed_array.dt}
                for k, v in self.stim_editor.stims_dict.items()
            }
            self.logger.info(
                f"Running external simulation function for {simulation_time} seconds"
            )
            data = self.external_simulation_function(
                stims_dict,
                simulation_time,
            )
            self.logger.info(
                f"External simulation complete, returned {len(data)} data objects"
            )
            self.model.add_experimental_data("External Simulation", data, False)
            self.update_model(self.model)
            self._select_experiment_data("External Simulation")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation error",
                f"Could not run external simulation: {e}",
            )
            self.logger.exception(e)

    def btn_do_external_simulation_clicked(self) -> None:
        if self.external_simulation_function is None:
            QtWidgets.QMessageBox.information(
                self,
                "No external simulation function",
                "No external simulation function loaded - please load one first.",
            )
            return
        simulation_time = getSimulationTime(self)
        if simulation_time is None:
            return
        self._do_external_simulation(simulation_time)

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
            self.data_widget.trace_widget,
            self.data_widget.spike_widget,
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
        self._load_external_simulation_function(filename)

    def _select_experiment_data(self, experiment_name):
        for cmb in (self.cmb_trace, self.cmb_spike, self.cmb_synapse, self.cmb_data):
            for i in range(cmb.count()):
                if cmb.itemText(i).startswith(f"{experiment_name} ::"):
                    cmb.setCurrentIndex(i)
                    break

    def btn_import_suite2p_data_clicked(self) -> None:
        """Open dialog to import experimental traces/spikes from Suite2p/CSV data"""
        # possibly: pass in existing experiment data to the dialog to let it do sanity checks & auto-fill the offset?
        dialog = Suite2pImportDialog(
            self,
            import_handler=self._apply_imported_experimental_data,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.update_model(self.model)
            self._select_experiment_data(dialog.experiment_name)

    def _apply_imported_experimental_data(
        self,
        experiment_name: str,
        data: list[TraceData | SpikeData],
        append: bool,
    ) -> None:
        self.model.add_experimental_data(experiment_name, data, append)

    def btn_import_cellpose_data_clicked(self) -> None:
        dialog = CellposeImportDialog(self, existing_data=self.last_cellpose_import)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.last_cellpose_import = dialog.get_imported_data()
            QtWidgets.QMessageBox.information(
                self,
                "Cellpose import",
                "Cellpose segmentation loaded. Now ready for Patchclamp import",
            )

    def btn_import_patchclamp_data_clicked(self) -> None:
        dialog = PatchClampImportDialog(
            cellpose_import=self.last_cellpose_import, parent=self
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if not dialog.imported_data:
                return
            try:
                self.model.add_experimental_data(
                    dialog.experiment_name, dialog.imported_data, dialog.append_data
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Patch clamp import error",
                    f"Could not import patch clamp data: {exc}",
                )
                self.logger.exception(exc)
                return
            self.update_model(self.model)
            self._select_experiment_data(dialog.experiment_name)

    def connect_external_simulation_function_with_brian2(self) -> None:
        self._load_external_simulation_function(
            "./external_simulation_example/run_brian2_externally.py"
        )

    def _load_external_simulation_function(self, filename: str) -> None:
        if filename == "":
            return
        try:
            path = pathlib.Path(filename).resolve()
            self.external_simulation_function = runpy.run_path(filename)["simulate"]
            self.logger.info(f"Loaded custom simulate function from {path}")
            QtWidgets.QMessageBox.information(
                self,
                "Custom simulate function loaded",
                f"Custom simulate function loaded from {path}. You can run this simulation using the Experiment -> Run external simulation function menu option.",
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
            self.btn_import_suite2p_data_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+I",
            icon=qta.icon("mdi6.import"),
        )
        add_action(
            "Import Cellpose &Segmentation",
            self.btn_import_cellpose_data_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.image-edit-outline"),
        )
        add_action(
            "Import &Patch Clamp Data",
            self.btn_import_patchclamp_data_clicked,
            menu,
            toolbar,
            shortcut=None,
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
        for name, widget, icon, is_checked in [
            ("&Stimuli", self.stim_editor, "mdi6.chart-timeline", True),
            ("&Traces", self.trace_widget, "mdi.waves", True),
            ("&Spikes", self.spike_widget, "ph.barcode-light", True),
            ("&Data", self.data_widget, "mdi6.chart-donut", False),
            ("&Heatmap", self.heatmap_widget, "mdi6.data-matrix", True),
            (
                "&Custom analysis",
                self.custom_analysis_widget,
                "mdi6.chart-histogram",
                True,
            ),
            ("&Console", self.stim_console, "mdi6.console", True),
        ]:
            add_action(
                f"Show {name}",
                lambda toggled, w=widget: w.setVisible(toggled),
                menu,
                toolbar,
                icon=qta.icon(f"{icon}"),
                checkable=True,
                is_checked=is_checked,
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
        toolbar.addWidget(self.cmb_trace)
        add_action(
            "&Trace normalization",
            self.trace_widget.trace_norm_dialog,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.arrow-expand-vertical"),
        )
        toolbar.addWidget(QtWidgets.QLabel("  Spike data: "))
        toolbar.addWidget(self.cmb_spike)
        toolbar.addWidget(QtWidgets.QLabel("  Synapse data: "))
        toolbar.addWidget(self.cmb_synapse)
        toolbar.addWidget(QtWidgets.QLabel("  All data: "))
        toolbar.addWidget(self.cmb_data)

    def _create_action_experiment(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&Connect external stimulation function",
            self.connect_external_simulation_function,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.code-braces"),
        )
        add_action(
            "&Connect brian2 external stimulation function",
            self.connect_external_simulation_function_with_brian2,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+E",
            icon=qta.icon("mdi6.function"),
        )
        add_action(
            "&Run external stimulation function",
            self.btn_do_external_simulation_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.rocket-launch-outline"),
        )

    def closeEvent(self, event):
        # explicitly close all fastplotlib figures to avoid renderer trying to access them after they have been destroyed
        self.spike_widget.close()
        self.trace_widget.close()
        self.stim_editor.stim_widget.close()
        self.data_widget.trace_widget.close()
        self.data_widget.spike_widget.close()
        super().closeEvent(event)
