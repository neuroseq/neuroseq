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
from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.console_widget import ConsoleWidget
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.data_widget import DataWidget
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui_settings import GuiSettings
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.import_dialog import Suite2pImportDialog
from neuroseq.import_experiment_dialog import ImportExperimentDialog
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import clone_cellpose_result
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog
from neuroseq.project import Project
from neuroseq.resource_monitor_label import ResourceMonitorLabel
from neuroseq.save_load_options import SaveLoadOptionsDialog
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim_editor import StimEditor
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.synthetic_data_dialog import SyntheticDataDialog
from neuroseq.time_slider_widget import TimeSliderWidget
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import add_action
from neuroseq.utils import add_items_to_combobox
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
        self.project = Project()
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
        self.trace_widget = TraceWidget(self.controller, parent=self)
        self.heatmap_widget = HeatmapWidget(parent=self)
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
        split_left_right.addWidget(self.heatmap_widget)
        self.custom_analysis_widget = CustomAnalysisWidget(parent=self)
        split_left_right.addWidget(self.custom_analysis_widget)
        self.console_widget = ConsoleWidget(parent=self)
        split_left_right.addWidget(self.console_widget)
        split_left_right.setSizes([1000, 1000, 1000])
        # main vertical splitter layout
        split_vertical = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical, handleWidth=0, childrenCollapsible=False
        )
        self.stim_editor = StimEditor(
            controller=self.controller,
            cmb_stim=self.cmb_stim,
            toolbar=self.toolbar_stim_editor,
            parent=self,
        )
        split_vertical.addWidget(self.stim_editor)
        split_vertical.addWidget(self.trace_widget)
        self.spike_widget = SpikeWidget(self.controller, parent=self)
        split_vertical.addWidget(self.spike_widget)
        self.data_widget = DataWidget(self.controller, parent=self)
        split_vertical.addWidget(self.data_widget)
        self.time_slider = TimeSliderWidget(self)
        self.plot_widgets: list[TimeseriesWidget | HeatmapWidget] = [
            self.stim_editor.stim_widget,
            self.trace_widget,
            self.spike_widget,
            self.data_widget,
            self.heatmap_widget,
        ]
        for widget in self.plot_widgets:
            self.time_slider.connect_widget(widget)
        split_vertical.addWidget(self.time_slider)
        self.data_widget.setVisible(False)
        # heatmap / custom analysis widget / console
        split_vertical.addWidget(split_left_right)
        main_horizontal_layout.addWidget(split_vertical)
        self.setCentralWidget(central_widget)
        self._create_actions()
        self.showMaximized()
        self._create_progressbar()
        # connect widgets
        self.console_widget.console_command_entered.connect(self._update_data)
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
        self.stim_editor.limits_modified.connect(self.update_limits)
        self.last_cellpose_import: CellposeImportResult | None = None
        if filename is not None:
            self.load_project(filename)
        else:
            self.load_project(default_project())

    def update_limits(self) -> None:
        self.controller.set_limits(self.project.max_time_secs, self.project.max_neurons)
        self.time_slider.set_max_time_secs(self.project.max_time_secs)

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

    def _update_data(self, keep_undo_history: bool = False) -> None:
        if self.project.b2_model.network.t > 0:
            self.continue_simulation_action.setText("&Continue simulation")
        else:
            self.continue_simulation_action.setText("&Run simulation")
        self.stim_editor.set_project(self.project, keep_undo_history)
        self.update_limits()
        self._update_combo_boxes()

    def _update_combo_box(
        self,
        cmb: QtWidgets.QComboBox,
        data_type: type[TraceData | SpikeData | SynapseData],
    ) -> None:
        cmb_text = cmb.currentText()
        cmb.clear()
        if self.project is not None:
            add_items_to_combobox(cmb, self.project.get_data_names(data_type))
            cmb.setCurrentText(cmb_text)

    def _update_combo_boxes(self) -> None:
        self._update_combo_box(self.cmb_trace, TraceData)
        self._update_combo_box(self.cmb_spike, SpikeData)
        self._update_combo_box(self.cmb_synapse, SynapseData)
        cmb_data_text = self.cmb_data.currentText()
        self.cmb_data.clear()
        if self.project is not None:
            add_items_to_combobox(
                self.cmb_data,
                [f"{d} [trace]" for d in self.project.get_data_names(TraceData)]
                + [f"{d} [spike]" for d in self.project.get_data_names(SpikeData)]
                + [f"{d} [synapse]" for d in self.project.get_data_names(SynapseData)],
            )
            self.cmb_data.setCurrentText(cmb_data_text)

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
        self.load_project(default_project())

    def btn_open_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load project",
            "project.nseq",
            "neuroseq project (*.nseq)",
        )
        if filename != "":
            self.load_project(filename)

    def btn_import_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "import from project",
            "project.nseq",
            "neuroseq project (*.nseq)",
        )
        if filename == "":
            return
        options_dialog = SaveLoadOptionsDialog(action="Import")
        if not options_dialog.exec():
            return
        try:
            self.progress_dialog.setLabelText("Importing from project...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.project.import_project(
                filename, options_dialog.options, report=self._update_load_progress
            )
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.logger.warning(f"Failed to import from project {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Import error",
                f"Could not import from file '{filename}': {e}",
            )
            self.progress_dialog.reset()
            return
        self.update_project(self.project)

    def btn_generate_synthetic_data_clicked(self) -> None:
        dialog = SyntheticDataDialog(self.project, parent=self)
        dialog.exec()
        # Refresh combo boxes if new data was added
        if dialog.data_added:
            self._update_combo_boxes()

    def btn_save_clicked(self) -> None:
        options_dialog = SaveLoadOptionsDialog(action="Save")
        if not options_dialog.exec():
            return
        filename, _ = getsavefilename(
            self,
            "Save project",
            str(self.project.filepath) if self.project.filepath else "project.nseq",
            "neuroseq project (*.nseq)",
        )
        if filename == "":
            return
        filename = pathlib.Path(filename)
        if filename.suffix != ".nseq":
            filename = filename.with_suffix(".nseq")
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.project.save(filename, options_dialog.options)
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
        self._update_window_title()

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
            self.project.b2_model.save_to_h5(filename)
        except Exception as e:
            self.logger.warning(f"Failed to export to file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export to file '{filename}': {e}",
            )
            return

    def btn_edit_clicked(self) -> None:
        mdiag = ModelEditorDialog(self.project.b2_model, parent=self)
        if mdiag.exec():
            self.project.b2_model = mdiag.b2model
            self.project.update_limits()
            self.update_project(self.project)

    @with_wait_cursor
    def load_project(self, project: str | pathlib.Path | Project) -> None:
        try:
            self.progress_dialog.setLabelText("Loading project...")
            if not isinstance(project, Project):
                filename = pathlib.Path(project)
                project = Project(filename, report=self._update_load_progress)
                self.logger.info(f"Project loaded from {filename}")
            else:
                self.logger.info("Project loaded from Project object")
        except Exception as e:
            self.logger.warning(f"Failed to load project from file {filename}: {e}")
            self.logger.exception(e)
            QtWidgets.QMessageBox.critical(
                self,
                "Project load error",
                f"Could not load project from file '{filename}': {e}",
            )
            self.progress_dialog.reset()
            return
        self.last_cellpose_import = clone_cellpose_result(project.cellpose_data)
        if project is not None:
            project.cellpose_data = self.last_cellpose_import
        self.update_project(project, reset_controller=True)

    def _update_window_title(self) -> None:
        if self.project.filepath is not None:
            self.setWindowTitle(f"{self.project.filepath} - Neuroseq")
        else:
            self.setWindowTitle("Neuroseq")

    @with_wait_cursor
    def update_project(self, project: Project, reset_controller=False) -> None:
        self.project = project
        self._update_window_title()
        if reset_controller:
            self.controller.reset()
        self.console_widget.set_project(self.project)
        self.continue_simulation_action.setText("&Run simulation")
        for widget in self.plot_widgets:
            widget.clear_data()
        if self.project is not None:
            self.last_cellpose_import = clone_cellpose_result(
                self.project.cellpose_data
            )
            self.project.cellpose_data = self.last_cellpose_import

        self.stim_editor.set_project(self.project)
        self.console_widget.set_project(self.project)
        self._update_combo_boxes()

    @with_wait_cursor
    def set_neuron(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, TraceData)
        self.trace_widget.set_data(data)

    @with_wait_cursor
    def set_synapse(self, cmb_name: str) -> None:
        self.project.max_ram_synapse_plot_matrix = (
            self.settings.max_ram_synapse_plot_matrix
        )
        data = self.project.get_data(cmb_name, SynapseData)
        self.heatmap_widget.set_data(data)

    @with_wait_cursor
    def set_spike(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, SpikeData)
        self.spike_widget.set_data(data)

    @with_wait_cursor
    def set_data(self, cmb_name: str) -> None:
        data: TraceData | SpikeData | SynapseData | None = None
        if cmb_name.endswith(" [trace]"):
            data = self.project.get_data(cmb_name[:-8], TraceData)
        elif cmb_name.endswith(" [spike]"):
            data = self.project.get_data(cmb_name[:-8], SpikeData)
        elif cmb_name.endswith(" [synapse]"):
            data = self.project.get_data(cmb_name[:-10], SynapseData)
        self.data_widget.set_data(data)

    def btn_exit_clicked(self) -> None:
        self.close()

    def btn_store_checkpoint_clicked(self) -> None:
        checkpoint, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Store simulation checkpoint",
            "Store the current simulation state as a named checkpoint.\nEnter a new name, or select an existing checkpoint to overwrite:",
            [f"Checkpoint {len(self.project.b2_model.checkpoints) + 1}"]
            + self.project.b2_model.checkpoints,
            current=0,
            editable=True,
        )
        if not ok:
            return
        self.project.b2_model.store(checkpoint)

    def btn_restore_checkpoint_clicked(self) -> None:
        if len(self.project.b2_model.checkpoints) == 0:
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
            self.project.b2_model.checkpoints,
            current=0,
            editable=False,
        )
        if not ok:
            return
        try:
            self.project.b2_model.restore(checkpoint)
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
            self.project.run_simulation(
                simulation_time,
                report=self._update_simulation_progress,
                stim_array_padding_seconds=self.settings.stim_padding_secs,
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

    def _append_experimental_data(self, experiment: Experiment) -> None:
        try:
            dialog = ImportExperimentDialog(experiment, self.project, parent=self)
            if not dialog.exec():
                return
            if dialog.append_to_experiment:
                self.project.append_experimental_data(
                    dialog.append_experiment_name,
                    dialog.experiment.duration,
                    dialog.experiment.data,
                    dialog.offset,
                )
            else:
                self.project.add_experiment(experiment)
            self.update_project(self.project)
            self._select_experiment_data(experiment.name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Data import error",
                f"Could not import data: {e}",
            )
            self.logger.exception(e)
            return

    @with_wait_cursor
    def _do_external_simulation(self, simulation_time: float):
        if self.external_simulation_function is None:
            return
        try:
            # pre-emptively pad timed arrays before simulating
            self.project.pad_stim_arrays(total_time_seconds=simulation_time)
            self.stim_editor.set_project(self.project)
            stims_dict = {
                k: {"array": v.stim_array.values, "dt": v.stim_array.dt}
                for k, v in self.stim_editor.stims_dict.items()
            }
            self.logger.info(
                f"Running external simulation function for {simulation_time} seconds"
            )
            experiment = self.external_simulation_function(
                stims_dict,
                simulation_time,
            )
            self.logger.info(
                f"External simulation complete, returned {experiment.duration}s experiment '{experiment.name}' with {len(experiment.data)} data objects"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "External Simulation error",
                f"Could not run external simulation: {e}",
            )
            self.logger.exception(e)
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._append_experimental_data(experiment)

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

    def btn_playback_speed_clicked(self) -> None:
        new_speed, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set playback speed",
            "Playback speed (1.0=realtime, 0.5=half speed, 2.0=double speed, etc.)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.time_slider.relative_playback_speed}",
        )
        if not ok:
            return
        try:
            new_speed_as_float = float(new_speed)
            if new_speed_as_float <= 0:
                raise ValueError("Playback speed must be a positive number")
            self.time_slider.set_relative_playback_speed(new_speed_as_float)
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse playback speed '{new_speed}' as a valid positive number",
            )
            return

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
        """Select the experiment within the GUI, for example after an import."""
        for cmb in (self.cmb_trace, self.cmb_spike, self.cmb_synapse, self.cmb_data):
            for i in range(cmb.count()):
                if cmb.itemText(i).startswith(f"{experiment_name} ::"):
                    cmb.setCurrentIndex(i)
                    break

    def btn_import_suite2p_data_clicked(self) -> None:
        """Open dialog to import experimental traces/spikes from Suite2p/CSV data"""
        dialog = Suite2pImportDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._append_experimental_data(
                Experiment(
                    dialog.experiment_name,
                    dialog.experiment_duration,
                    [],
                    dialog.imported_data,
                )
            )

    def btn_import_cellpose_data_clicked(self) -> None:
        dialog = CellposeImportDialog(
            self,
            existing_data=self.last_cellpose_import,
            delete_handler=self._delete_cellpose_data,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            imported = clone_cellpose_result(dialog.get_imported_data())
            self.last_cellpose_import = imported
            if self.project is not None:
                self.project.cellpose_data = imported
            QtWidgets.QMessageBox.information(
                self,
                "Cellpose import",
                "Cellpose segmentation loaded. Now ready for Patchclamp import",
            )

    def _delete_cellpose_data(self) -> None:
        """Remove stored cellpose data from both GUI and project."""
        self.last_cellpose_import = None
        if self.project is not None:
            self.project.cellpose_data = None

    def btn_import_patchclamp_data_clicked(self) -> None:
        dialog = PatchClampImportDialog(
            cellpose_import=self.last_cellpose_import, parent=self
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if not dialog.imported_data:
                return
            try:
                if dialog.cellpose_import is not None:
                    # keep GUI/project copies in sync with any updated stimulus mapping
                    self.last_cellpose_import = clone_cellpose_result(
                        dialog.cellpose_import
                    )
                    if self.project is not None:
                        self.project.cellpose_data = self.last_cellpose_import
                self._append_experimental_data(
                    Experiment(
                        dialog.experiment_name,
                        dialog.experiment_duration,
                        [],
                        dialog.imported_data,
                    )
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Patch clamp import error",
                    f"Could not import patch clamp data: {exc}",
                )
                self.logger.exception(exc)
                return

    def connect_external_simulation_function_with_brian2(self) -> None:
        self._load_external_simulation_function(
            "./external_simulation_example/run_brian2_externally.py"
        )

    def _load_external_simulation_function(self, filename: str) -> None:
        if filename == "":
            return
        path = pathlib.Path(filename).resolve()
        try:
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

    def open_stimulation_generator_dialog(self) -> None:
        """Open the stimulation script generator dialog."""
        dialog = StimulationGeneratorDialog(self.project, parent=self)
        dialog.exec()

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
        # add "open example project" item to the file menu which opens a sub-menu containing the example projects from src/neuroseq/example_projects
        example_projects_menu = menu.addMenu("&Open example project")
        for name, generator in example_projects.items():
            add_action(
                name,
                lambda checked, _generator=generator: self.load_project(_generator()),
                example_projects_menu,
                None,
            )
        add_action(
            "&Open project",
            self.btn_open_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+O",
            icon=qta.icon("mdi6.folder-open"),
        )
        add_action(
            "&Import from project",
            self.btn_import_clicked,
            menu,
            toolbar,
            shortcut="None",
            icon=qta.icon("mdi6.file-import"),
        )
        add_action(
            "&Save project",
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
            "Generate Synthetic &Data...",
            self.btn_generate_synthetic_data_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+G",
            icon=qta.icon("mdi6.microsoft-xbox-controller-view")
            if hasattr(qta, "icon")
            else None,
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
            self.time_slider.skip_to_start,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.skip-previous"),
        )
        add_action(
            "&Stop",
            self.time_slider.stop,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.stop"),
        )
        add_action(
            "&Play / Pause",
            self.time_slider.toggle_pause,
            menu,
            toolbar,
            shortcut="Space",
            icon=qta.icon("mdi6.play-pause"),
        )
        add_action(
            "&Skip to end",
            self.time_slider.skip_to_end,
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
            ("&Console", self.console_widget, "mdi6.console", True),
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
        add_action(
            "&Synapse normalization",
            self.heatmap_widget.heatmap_norm_dialog,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.arrow-expand-vertical"),
        )
        toolbar.addWidget(QtWidgets.QLabel("  All data: "))
        toolbar.addWidget(self.cmb_data)

    def _create_action_experiment(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&Generate stimulation scripts",
            self.open_stimulation_generator_dialog,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.file-code"),
        )
        menu.addSeparator()
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
        for widget in self.plot_widgets:
            widget.close()
        super().closeEvent(event)
