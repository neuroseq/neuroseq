from __future__ import annotations

import logging
import pathlib
import runpy
from collections.abc import Callable

import brian2 as b2
import brian2.numpy_ as np
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
from neuroseq.experiments_manager_dialog import ExperimentsManagerDialog
from neuroseq.exporter_igor2 import export_igor2_patchclamp_csv
from neuroseq.gui_settings import GuiSettings
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.igor2_patchclamp_export_dialog import Igor2PatchClampExportDialog
from neuroseq.igor_import_dialog import IgorImportDialog
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
from neuroseq.syscon2_import_dialog import Syscon2ImportDialog
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
        # Keep strong Python references to these menus/actions to avoid premature
        # PySide object deletion on some platforms (notably macOS CI).
        self.open_example_project_menu: QtWidgets.QMenu | None = None
        self.open_example_project_section_menus: dict[str, QtWidgets.QMenu] = {}
        self.open_example_project_actions: list[QtWidgets.QAction] = []
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
        split_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split_vertical.setHandleWidth(0)
        split_vertical.setChildrenCollapsible(False)
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
        self.plot_widgets: list[TimeseriesWidget | HeatmapWidget | DataWidget] = [
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
        max_time_secs = float(self.project.max_time_secs)
        max_neurons = int(self.project.max_neurons)
        self.controller.set_limits(max_time_secs, max_neurons)
        self.time_slider.set_max_time_secs(max_time_secs)

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

    def btn_export_simulation_results_clicked(self) -> None:
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

    def btn_export_igor2_patchclamp_stim_clicked(self) -> None:
        if not self.project.stim_arrays:
            QtWidgets.QMessageBox.information(
                self,
                "No stimulation entities",
                "No stimulation data entities available. Create one before exporting to Igor2 patch clamp CSV.",
            )
            return

        stim_name = self.cmb_stim.currentText()
        if not stim_name or stim_name not in self.project.stim_arrays:
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                "No stimulation array is currently selected for export.",
            )
            return

        stim_array = self.project.stim_arrays[stim_name]
        if int(stim_array.values.shape[1]) <= 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                "The selected stimulation array has no neurons to export.",
            )
            return

        dialog = Igor2PatchClampExportDialog(stim_name, stim_array, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        suggested_filename = f"{dialog.experiment_name}.csv"
        filename, _ = getsavefilename(
            self,
            "Export Igor2 Patch Clamp CSV",
            suggested_filename,
            "CSV file (*.csv)",
        )
        if filename == "":
            return

        output_path = pathlib.Path(filename)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        try:
            export_report = export_igor2_patchclamp_csv(
                output_path,
                stim_array,
                experiment_name=dialog.experiment_name,
                neuron_index=dialog.selected_neuron_index,
            )
        except Exception as e:
            self.logger.warning(
                "Failed to export Igor2 patch clamp CSV %s: %s",
                output_path,
                e,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export Igor2 patch clamp CSV to '{output_path}': {e}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Export Successful",
            "Exported Igor2 patch clamp CSV for "
            f"stimulus '{stim_name}', neuron {export_report.neuron_index}, "
            f"{export_report.sample_count} samples.",
        )

    def btn_edit_clicked(self) -> None:
        mdiag = ModelEditorDialog(self.project.b2_model, parent=self)
        if mdiag.exec():
            self.project.b2_model = mdiag.b2model
            self.project.update_limits()
            self.update_project(self.project)

    def btn_manage_experiments_clicked(self) -> None:
        dialog = ExperimentsManagerDialog(self.project, parent=self)
        dialog.experiments_changed.connect(lambda: self.update_project(self.project))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Update the GUI with changes made in the dialog
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

    def _cellpose_neuron_count(self, imported: CellposeImportResult | None) -> int:
        if imported is None:
            return 0
        if imported.original_cellpose_ids is not None:
            return int(len(imported.original_cellpose_ids))
        if imported.segmentation_array is None:
            return 0
        unique_ids = np.unique(imported.segmentation_array)
        return int(np.sum(unique_ids >= 0))

    def _expand_stim_arrays_to_neuron_count(
        self, target_neuron_count: int
    ) -> list[str]:
        if self.project is None:
            return []
        resized_names: list[str] = []
        for name, stim_array in self.project.stim_arrays.items():
            if int(stim_array.values.shape[1]) >= target_neuron_count:
                continue
            self.project.update_stim_array_properties(
                name,
                n_neurons=target_neuron_count,
                stim_type=stim_array.stim_type,
                units=stim_array.units,
                dt=stim_array.dt,
            )
            resized_names.append(name)
        return resized_names

    def _apply_cellpose_import_result(
        self,
        imported: CellposeImportResult,
    ) -> list[str]:
        imported_copy = clone_cellpose_result(imported)
        if imported_copy is None:
            return []
        self.last_cellpose_import = imported_copy
        if self.project is None:
            return []

        self.project.cellpose_data = imported_copy
        detected_neurons = self._cellpose_neuron_count(imported_copy)
        resized_names: list[str] = []
        if detected_neurons > 0 and self.project.stim_arrays:
            current_max_neurons = max(
                int(stim_array.values.shape[1])
                for stim_array in self.project.stim_arrays.values()
            )
            if detected_neurons > current_max_neurons:
                resize_reply = QtWidgets.QMessageBox.question(
                    self,
                    "Resize Stim Arrays",
                    "The imported Cellpose segmentation contains "
                    f"{detected_neurons} cells, but the largest existing stimulation "
                    f"array has {current_max_neurons} neurons.\n\n"
                    "Expand all existing stimulation arrays to match the Cellpose "
                    "cell count?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.Yes,
                )
                if resize_reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    resized_names = self._expand_stim_arrays_to_neuron_count(
                        detected_neurons
                    )
                    self.update_project(self.project)
                    return resized_names

        self.project.update_limits()
        self.update_limits()
        return resized_names

    def _focus_neuron_range(self, n_neurons: int) -> None:
        if int(n_neurons) <= 0:
            return
        self.controller.focus_neuron_range(0, int(n_neurons))

    def _apply_trace_selection(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, TraceData)
        self.trace_widget.set_data(data)
        self._focus_neuron_range(self.trace_widget.n_neurons)

    @with_wait_cursor
    def set_neuron(self, cmb_name: str) -> None:
        self._apply_trace_selection(cmb_name)

    @with_wait_cursor
    def set_synapse(self, cmb_name: str) -> None:
        self.project.max_ram_synapse_plot_matrix = (
            self.settings.max_ram_synapse_plot_matrix
        )
        data = self.project.get_data(cmb_name, SynapseData)
        self.heatmap_widget.set_data(data)

    def _apply_spike_selection(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, SpikeData)
        self.spike_widget.set_data(data)
        self._focus_neuron_range(self.spike_widget.n_neurons)

    @with_wait_cursor
    def set_spike(self, cmb_name: str) -> None:
        self._apply_spike_selection(cmb_name)

    def _apply_data_selection(self, cmb_name: str) -> None:
        data: TraceData | SpikeData | SynapseData | None = None
        if cmb_name.endswith(" [trace]") and not self.data_widget.isVisible():
            return
        if cmb_name.endswith(" [trace]"):
            data = self.project.get_data(cmb_name[:-8], TraceData)
        elif cmb_name.endswith(" [spike]"):
            data = self.project.get_data(cmb_name[:-8], SpikeData)
        elif cmb_name.endswith(" [synapse]"):
            data = self.project.get_data(cmb_name[:-10], SynapseData)
        self.data_widget.set_data(data)
        if isinstance(data, TraceData):
            self._focus_neuron_range(self.data_widget.trace_widget.n_neurons)
        elif isinstance(data, SpikeData):
            self._focus_neuron_range(self.data_widget.spike_widget.n_neurons)

    @with_wait_cursor
    def set_data(self, cmb_name: str) -> None:
        self._apply_data_selection(cmb_name)

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
                f"Could not import data to Stim IDs: {e}",
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

    def toggle_grid(self, is_visible: bool) -> None:
        for widget in [
            self.stim_editor.stim_widget,
            self.trace_widget,
            self.spike_widget,
            self.data_widget.trace_widget,
            self.data_widget.spike_widget,
        ]:
            widget.set_grid_visible(is_visible)

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
            imported = dialog.get_imported_data()
            if imported is None:
                return
            resized_names = self._apply_cellpose_import_result(imported)
            message = "Cellpose segmentation loaded."
            if resized_names:
                detected_neurons = self._cellpose_neuron_count(
                    self.last_cellpose_import
                )
                message += (
                    "\n\nExpanded "
                    f"{len(resized_names)} stimulation array(s) to {detected_neurons} neurons."
                )
            QtWidgets.QMessageBox.information(
                self,
                "Cellpose import",
                message,
            )

    def btn_import_syscon2_seq_clicked(self) -> None:
        if not self.project.stim_arrays:
            QtWidgets.QMessageBox.information(
                self,
                "No stimulation entities",
                "No stimulation data entities available. Create one before importing a SysCon2 sequence.",
            )
            return

        dialog = Syscon2ImportDialog(self.project, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.update_project(self.project)
            if dialog.selected_stim_name:
                self.cmb_stim.setCurrentText(dialog.selected_stim_name)

    def btn_import_igor2_wave_clicked(self) -> None:
        dialog = IgorImportDialog(self.project, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.update_project(self.project)
            if dialog.selected_stim_name:
                self.cmb_stim.setCurrentText(dialog.selected_stim_name)

    def btn_import_igor2_trace_clicked(self) -> None:
        from neuroseq.igor_trace_import_dialog import IgorTraceImportDialog

        dialog = IgorTraceImportDialog(self.project, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.update_project(self.project)
            if dialog.selected_trace_name:
                combo_label = (
                    f"{dialog.selected_experiment_name} :: {dialog.selected_trace_name}"
                )
                self.cmb_trace.setCurrentText(combo_label)
                # update_project() clears the trace widget; if the combo label did
                # not change (for example importing into the currently selected
                # TraceData), currentTextChanged will not fire, so refresh explicitly.
                self._apply_trace_selection(combo_label)

    def _delete_cellpose_data(self) -> None:
        """Remove stored cellpose data from both GUI and project."""
        self.last_cellpose_import = None
        if self.project is not None:
            self.project.cellpose_data = None

    def btn_import_patchclamp_data_clicked(self) -> None:
        dialog = PatchClampImportDialog(
            cellpose_import=self.last_cellpose_import,
            project=self.project,
            parent=self,
        )
        try:
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
        finally:
            dialog.close()
            dialog.deleteLater()

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

    # Import stimulus wrappers

    def btn_import_stimulus_light_clicked(self) -> None:
        """Pre-dialog: choose software, then import a light stimulus waveform."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Import light stimulus",
            "Import from:",
            ["SysCon2"],
            0,
            False,
        )
        if not ok:
            return
        if software == "SysCon2":
            self.btn_import_syscon2_seq_clicked()

    def btn_import_stimulus_patchclamp_clicked(self) -> None:
        """Pre-dialog: choose software, then import a patch-clamp stimulus waveform."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Import patch clamp stimulus",
            (
                "Patch-clamp convention:\n"
                "  • Current Clamp (CC): the injected CURRENT is the stimulus\n"
                "  • Voltage Clamp (VC): the commanded VOLTAGE is the stimulus\n"
                "\n"
                "Import from:"
            ),
            ["Igor2"],
            0,
            False,
        )
        if not ok:
            return
        if software == "Igor2":
            self.btn_import_igor2_wave_clicked()

    def btn_import_trace_patchclamp_clicked(self) -> None:
        """Pre-dialog: choose software, then import recorded patch-clamp traces.

        Convention:
          Current Clamp (CC) -> recorded signal is voltage (mV)
          Voltage Clamp (VC) -> recorded signal is current (pA)
        Igor2 routes to the Igor trace importer; NumPy routes to the existing
        PatchClampImportDialog.
        """
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Import patch clamp traces",
            (
                "Patch-clamp convention:\n"
                "  • Current Clamp (CC): the recorded VOLTAGE is the trace\n"
                "  • Voltage Clamp (VC): the recorded CURRENT is the trace\n"
                "\n"
                "Import from:"
            ),
            ["Igor2", "NumPy (.npy)"],
            0,
            False,
        )
        if not ok:
            return
        if software == "Igor2":
            self.btn_import_igor2_trace_clicked()
        elif software == "NumPy (.npy)":
            self.btn_import_patchclamp_data_clicked()

    # Export wrappers

    def btn_export_patch_clamp_stim_clicked(self) -> None:
        """Pre-dialog: choose software, then export patch-clamp stimulus CSV."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Export patch clamp stimulus",
            "Export for:",
            ["Igor2"],
            0,
            False,
        )
        if not ok:
            return
        if software == "Igor2":
            self.btn_export_igor2_patchclamp_stim_clicked()

    def btn_export_light_stimulus_clicked(self) -> None:
        """Pre-dialog: choose software, then open the stimulation script generator."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Export light stimulus",
            "Export for:",
            ["SysCon2"],
            0,
            False,
        )
        if not ok:
            return
        if software == "SysCon2":
            self.open_stimulation_generator_dialog()

    def btn_export_traces_clicked(self) -> None:
        """Export the currently displayed trace data as a publication-quality figure.

        Exports all neuron traces for the selected TraceData as a single matplotlib
        figure. The Y-axis label uses the units stored in the TraceData (e.g. 'mV'
        for current-clamp recordings, 'pA' for voltage-clamp recordings). DPI is 300.
        """
        import matplotlib.pyplot as plt

        trace_name = self.cmb_trace.currentText()
        if not trace_name:
            QtWidgets.QMessageBox.information(
                self,
                "No trace data",
                "No trace data is currently selected in the Traces panel.",
            )
            return
        trace_data = self.project.get_data(trace_name, TraceData)
        if trace_data is None:
            QtWidgets.QMessageBox.information(
                self,
                "No trace data",
                f"Could not retrieve trace data '{trace_name}'.",
            )
            return

        filename, _ = getsavefilename(
            self,
            "Export traces",
            f"{trace_name.replace(' :: ', '_')}_traces.png",
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf)",
        )
        if not filename:
            return

        fig = None
        try:
            n_neurons, n_timepoints = trace_data.traces.shape
            times = np.arange(n_timepoints) * trace_data.dt
            fig, ax = plt.subplots(figsize=(12, max(4, n_neurons * 0.5 + 2)))
            for i in range(n_neurons):
                ax.plot(times, trace_data.traces[i], linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel(trace_data.units or "Amplitude", fontsize=12)
            ax.set_title(trace_name, fontsize=13)
            ax.tick_params(labelsize=10)
            fig.tight_layout()
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(
                self,
                "Export successful",
                f"Traces exported to '{filename}'.",
            )
        except Exception as exc:
            self.logger.exception("Failed to export traces")
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export traces: {exc}",
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def btn_export_spikes_clicked(self) -> None:
        """Export the currently displayed spike data as a publication-quality raster.

        Produces a scatter-plot raster (spike time vs neuron index) saved at 300 DPI.
        """
        import matplotlib.pyplot as plt

        spike_name = self.cmb_spike.currentText()
        if not spike_name:
            QtWidgets.QMessageBox.information(
                self,
                "No spike data",
                "No spike data is currently selected in the Spikes panel.",
            )
            return
        spike_data = self.project.get_data(spike_name, SpikeData)
        if spike_data is None:
            QtWidgets.QMessageBox.information(
                self,
                "No spike data",
                f"Could not retrieve spike data '{spike_name}'.",
            )
            return

        filename, _ = getsavefilename(
            self,
            "Export spikes",
            f"{spike_name.replace(' :: ', '_')}_spikes.png",
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf)",
        )
        if not filename:
            return

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(
                spike_data.spike_times,
                spike_data.spike_indices,
                s=1,
                c="black",
                alpha=0.7,
                linewidths=0,
            )
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Neuron index", fontsize=12)
            ax.set_title(spike_name, fontsize=13)
            ax.tick_params(labelsize=10)
            fig.tight_layout()
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(
                self,
                "Export successful",
                f"Spikes exported to '{filename}'.",
            )
        except Exception as exc:
            self.logger.exception("Failed to export spikes")
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export spikes: {exc}",
            )
        finally:
            if fig is not None:
                plt.close(fig)

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
        assert menu is not None
        self.menu_file = menu.addMenu("&File")
        assert self.menu_file is not None
        self.toolbar_file = QtWidgets.QToolBar("File")
        self.addToolBar(self.toolbar_file)
        self.menu_simulate = menu.addMenu("&Simulate")
        assert self.menu_simulate is not None
        self.toolbar_simulate = QtWidgets.QToolBar("Simulate")
        self.addToolBar(self.toolbar_simulate)

        self.menu_playback = menu.addMenu("&Playback")
        assert self.menu_playback is not None
        self.toolbar_playback = QtWidgets.QToolBar("Playback")
        self.addToolBar(self.toolbar_playback)
        self.menu_view = menu.addMenu("&View")
        assert self.menu_view is not None
        self.toolbar_view = QtWidgets.QToolBar("View")
        self.toolbar_data = QtWidgets.QToolBar("Data")
        self.addToolBar(self.toolbar_data)

        self.menu_experiment = menu.addMenu("&Experiment")
        assert self.menu_experiment is not None
        self.toolbar_experiment = QtWidgets.QToolBar("Experiment")
        self.addToolBar(self.toolbar_experiment)
        self.addToolBarBreak()
        self.addToolBar(self.toolbar_view)

        self.menu_settings = menu.addMenu("&Settings")
        assert self.menu_settings is not None
        self.menu_help = menu.addMenu("&Help")
        assert self.menu_help is not None
        self._create_actions_data(toolbar=self.toolbar_data, menu=None)

    def _create_actions(self) -> None:
        assert self.menu_file is not None
        assert self.menu_simulate is not None
        assert self.menu_playback is not None
        assert self.menu_view is not None
        assert self.menu_settings is not None
        assert self.menu_help is not None
        assert self.menu_experiment is not None
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
        self.open_example_project_menu = menu.addMenu("&Open example project")
        assert self.open_example_project_menu is not None
        self.open_example_project_section_menus = {}
        self.open_example_project_actions = []
        for section, section_projects in example_projects.items():
            section_menu = self.open_example_project_menu.addMenu(section)
            assert section_menu is not None
            self.open_example_project_section_menus[section] = section_menu
            for name, generator in section_projects.items():
                action = add_action(
                    name,
                    lambda checked=False, _generator=generator: self.load_project(
                        _generator()
                    ),
                    section_menu,
                    None,
                )
                self.open_example_project_actions.append(action)
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

        import_data_menu = menu.addMenu("&Import data")
        assert import_data_menu is not None

        import_stimulus_menu = import_data_menu.addMenu("Import &stimulus")
        assert import_stimulus_menu is not None
        add_action(
            "Import &light stimulus",
            self.btn_import_stimulus_light_clicked,
            import_stimulus_menu,
            None,
            icon=qta.icon("mdi6.file-import-outline"),
        )
        add_action(
            "Import &patch clamp",
            self.btn_import_stimulus_patchclamp_clicked,
            import_stimulus_menu,
            None,
            icon=qta.icon("mdi6.waveform"),
        )

        import_trace_menu = import_data_menu.addMenu("Import &trace")
        assert import_trace_menu is not None
        add_action(
            "Import &calcium imaging",
            self.btn_import_suite2p_data_clicked,
            import_trace_menu,
            None,
            icon=qta.icon("mdi6.import"),
        )
        add_action(
            "Import &patch clamp",
            self.btn_import_trace_patchclamp_clicked,
            import_trace_menu,
            None,
            icon=qta.icon("mdi6.import"),
        )

        add_action(
            "Import cell &segmentation",
            self.btn_import_cellpose_data_clicked,
            import_data_menu,
            None,
            icon=qta.icon("mdi6.image-edit-outline"),
        )

        export_data_menu = menu.addMenu("&Export data")
        assert export_data_menu is not None

        add_action(
            "Export &patch clamp stimulus",
            self.btn_export_patch_clamp_stim_clicked,
            export_data_menu,
            None,
            icon=qta.icon("mdi6.waveform"),
        )
        add_action(
            "Export &light stimulus",
            self.btn_export_light_stimulus_clicked,
            export_data_menu,
            None,
            icon=qta.icon("mdi6.file-code"),
        )
        add_action(
            "Export &traces",
            self.btn_export_traces_clicked,
            export_data_menu,
            None,
            icon=qta.icon("mdi.waves"),
        )
        add_action(
            "Export &spikes",
            self.btn_export_spikes_clicked,
            export_data_menu,
            None,
            icon=qta.icon("ph.barcode-light"),
        )
        add_action(
            "Export &simulation results",
            self.btn_export_simulation_results_clicked,
            export_data_menu,
            None,
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
            "Generate Synthetic &Data...",
            self.btn_generate_synthetic_data_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+G",
            icon=qta.icon("mdi6.microsoft-xbox-controller-view")
            if hasattr(qta, "icon")
            else None,
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
                lambda toggled, w=widget: self._set_widget_visible(w, toggled),
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
        add_action(
            "Toggle Grid",
            self.toggle_grid,
            menu,
            toolbar,
            icon=qta.icon("mdi6.grid"),
            checkable=True,
            is_checked=True,
        )

    def _set_widget_visible(self, widget: QtWidgets.QWidget, toggled: bool) -> None:
        widget.setVisible(toggled)
        if toggled and widget is self.data_widget:
            self.set_data(self.cmb_data.currentText())

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
            "&Manage Experiments",
            self.btn_manage_experiments_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.beaker"),
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
            shortcut=None,
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
        # Stop timers and queued widget updates before tearing down owned widgets.
        self.time_slider.close()
        # explicitly close all fastplotlib figures to avoid renderer trying to access them after they have been destroyed
        for widget in self.plot_widgets:
            widget.close()
        # Close any detached analysis windows that are parented to this GUI.
        for analysis_window in self.findChildren(CustomAnalysisWindow):
            analysis_window.close()
        # Drop strong menu refs so closed GUI instances do not keep dangling
        # QMenu/QAction wrappers alive across tests and CI runs.
        self.menuBar().clear()
        self.open_example_project_menu = None
        self.open_example_project_section_menus.clear()
        self.open_example_project_actions.clear()
        super().closeEvent(event)
