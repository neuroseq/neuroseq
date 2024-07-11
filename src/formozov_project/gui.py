from __future__ import annotations

import logging
from importlib import resources

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from formozov_project import __version__
from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import B2ModelWidget
from formozov_project.heatmap_widget import HeatmapWidget
from formozov_project.model_editor import ModelEditorDialog
from formozov_project.stim_console import StimConsole
from formozov_project.stim_editor import StimEditor
from formozov_project.trace_widget import TraceWidget
from formozov_project.utils import add_action
from formozov_project.utils import with_wait_cursor


class Gui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("starting Gui")
        self.model = B2Model()
        self.neurons: dict[str, b2.StateMonitor] = {}
        self.synapses: dict[str, b2.StateMonitor] = {}
        self.setWindowTitle("Formozov Project")
        self._create_menus_and_toolbar()
        self.addToolBar(self.toolbar)
        self.setWindowIcon(qta.icon("mdi6.brain"))
        grid_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(grid_layout)
        # left/right split between traces and heatmap
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # traces on left
        left_parent_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_parent_widget.setLayout(left_layout)
        self.cmb_neuron = QtWidgets.QComboBox()
        left_layout.addWidget(self.cmb_neuron)
        self.trace_widget = TraceWidget()
        left_layout.addWidget(self.trace_widget)
        split_left_right.addWidget(left_parent_widget)
        # heatmap on right
        right_parent_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_parent_widget.setLayout(right_layout)
        self.cmb_synapse = QtWidgets.QComboBox()
        right_layout.addWidget(self.cmb_synapse)
        self.heatmap_widget = HeatmapWidget()
        right_layout.addWidget(self.heatmap_widget)
        split_left_right.addWidget(right_parent_widget)
        split_left_right.setSizes([3000, 2000])
        # main vertical splitter layout
        split_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.stim_editor = StimEditor(menu=self.menu_stimuli)
        split_vertical.addWidget(self.stim_editor)
        split_vertical.addWidget(split_left_right)
        self.stim_console = StimConsole()
        split_vertical.addWidget(self.stim_console)
        grid_layout.addWidget(split_vertical)
        self.setCentralWidget(central_widget)
        self.resize(800, 600)
        # add a modal progress bar dialog while running simulations
        self.progress_dialog = QtWidgets.QProgressDialog(
            "Simulating...", "", 0, 100, self
        )
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setMinimumDuration(10)
        self.progress_dialog.reset()
        # connect widgets
        self.heatmap_widget.connect_trace_widget(self.trace_widget)
        self.trace_widget.connect_heatmap_widget(self.heatmap_widget)
        self.stim_console.sig_stims_values_dict_updated.connect(
            self.stim_editor.update_stims_values
        )
        self.cmb_neuron.currentTextChanged.connect(self.set_neuron)
        self.cmb_synapse.currentTextChanged.connect(self.set_synapse)

    def _update_data(self) -> None:
        self.set_neuron(self.cmb_neuron.currentText())
        self.set_synapse(self.cmb_synapse.currentText())

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
        b2m = B2ModelWidget(B2Model(), self)
        if b2m.exec():
            self.model = b2m.model
            self.update_model()

    def btn_open_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load model",
            "model.yaml",
            "yaml file (*.yaml)",
        )
        if filename != "":
            self.load_model(filename)

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
                self.update_model()

    def btn_open_example_model_clicked(self) -> None:
        self.load_model(
            str(
                resources.files("formozov_project")
                / "example_models"
                / "default_model.yaml"
            )
        )

    @with_wait_cursor
    def load_model(self, filename: str) -> None:
        try:
            self.progress_dialog.setLabelText("Loading model...")
            model = B2Model.load_model(filename, report=self._update_load_progress)
            self.model = model
            self.logger.info(f"Model {self.model.name} loaded from {filename}")

        except Exception as e:
            self.logger.warning(f"Failed to load model from file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Model construction error",
                f"Could not load model from file '{filename}': {e}",
            )
            return
        self.update_model()

    def update_model(self) -> None:
        self.heatmap_widget.clear_data()
        self.trace_widget.clear_data()
        self.stim_editor.set_model(self.model)
        self.cmb_neuron.clear()
        self.cmb_synapse.clear()
        if self.model is not None:
            self.stim_editor.set_model(self.model)
            self.stim_console.set_stims(self.stim_editor.stims_dict)
            self.neurons = {
                f"{m.source.name} / {m.record_variables[0]}": m
                for m in self.model.network.objects
                if isinstance(m, b2.StateMonitor)
                and isinstance(m.source, b2.NeuronGroup)
            }
            self.logger.info(f"traces: {self.neurons}")
            self.synapses = {
                f"{m.source.name} / {m.record_variables[0]}": m
                for m in self.model.network.objects
                if isinstance(m, b2.StateMonitor) and isinstance(m.source, b2.Synapses)
            }
            self.logger.info(f"synapses: {self.synapses}")
            self.cmb_neuron.addItems(list(self.neurons.keys()))
            self.cmb_synapse.addItems(list(self.synapses.keys()))

    @with_wait_cursor
    def set_neuron(self, neuron_name: str) -> None:
        if neuron_name not in self.neurons:
            return
        monitor = self.neurons[neuron_name]
        variable = monitor.record_variables[0]
        timestamps = monitor.t / b2.second
        if timestamps.shape[0] == 0:
            self.trace_widget.clear_data()
            return
        traces = getattr(monitor, variable)
        self.logger.info(
            f"Displaying {neuron_name} in TraceWidget: {timestamps.shape} timestamps, {traces.shape} data"
        )
        self.trace_widget.set_data(traces, timestamps)

    @with_wait_cursor
    def set_synapse(self, synapse_name: str) -> None:
        if synapse_name not in self.synapses:
            return
        monitor = self.synapses[synapse_name]
        synapse_group = monitor.source
        neuron_group = synapse_group.source
        variable = monitor.record_variables[0]
        values = getattr(monitor, variable)
        matrix = np.zeros((neuron_group.N, neuron_group.N, values.shape[1]))
        n_timestamps = values.shape[1]
        if n_timestamps == 0:
            self.heatmap_widget.clear_data()
            return
        for i, j, value in zip(synapse_group.i, synapse_group.j, values, strict=True):
            matrix[i, j, :] = value
        timestamps = monitor.t / b2.second
        self.logger.info(
            f"Displaying {synapse_name} in HeatmapWidget: {timestamps.shape} timestamps, {matrix.shape} data"
        )
        self.heatmap_widget.set_data(matrix, timestamps)

    def btn_save_clicked(self) -> None:
        filename, _ = getsavefilename(
            self,
            "Save results",
            "data.h5",
            "hdf5 file (*.h5)",
        )
        if filename == "":
            return
        try:
            self.model.save_to_h5(filename)
        except Exception as e:
            self.logger.warning(f"Failed to save file {filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "File save error",
                f"Could not save file '{filename}': {e}",
            )
            return

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
        try:
            self.model.run_simulation(
                simulation_time * b2.second, report=self._update_simulation_progress
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

    def btn_skip_to_start_clicked(self) -> None:
        self.heatmap_widget.setCurrentIndex(0)

    def btn_stop_clicked(self) -> None:
        self.heatmap_widget.play(0)
        self.btn_skip_to_start_clicked()

    def btn_play_clicked(self) -> None:
        self.heatmap_widget.togglePause()

    def btn_skip_to_end_clicked(self) -> None:
        self.heatmap_widget.setCurrentIndex(self.heatmap_widget.nframes() - 1)

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About formozov_project",
            f"Formozov Project\n\nVersion {__version__}\n\n"
            + "https://github.com/ssciwr/formozov-project",
        )

    def _create_menus_and_toolbar(self) -> None:
        menu = self.menuBar()
        self.toolbar = QtWidgets.QToolBar()
        file_menu = menu.addMenu("&File")
        add_action(
            "&New",
            self.btn_new_clicked,
            file_menu,
            self.toolbar,
            shortcut="Ctrl+N",
            icon=qta.icon("mdi6.file"),
        )
        add_action(
            "Open &example model",
            self.btn_open_example_model_clicked,
            file_menu,
            self.toolbar,
            shortcut="Ctrl+Shift+O",
            icon=qta.icon("mdi6.file-document"),
        )
        add_action(
            "&Open",
            self.btn_open_clicked,
            file_menu,
            self.toolbar,
            shortcut="Ctrl+O",
            icon=qta.icon("mdi6.folder-open"),
        )
        add_action(
            "&Save",
            self.btn_save_clicked,
            file_menu,
            self.toolbar,
            shortcut="Ctrl+S",
            icon=qta.icon("mdi6.content-save"),
        )
        add_action(
            "&Edit",
            self.btn_edit_clicked,
            file_menu,
            self.toolbar,
            shortcut="Ctrl+E",
            icon=qta.icon("mdi6.cog"),
        )
        self.toolbar.addSeparator()
        add_action(
            "E&xit",
            self.btn_exit_clicked,
            file_menu,
            None,
            shortcut=None,
            icon=qta.icon("mdi6.exit-to-app"),
        )
        self.menu_stimuli = menu.addMenu("&Stimuli")
        experiment_model = menu.addMenu("&Simulate")
        add_action(
            "&Continue simulation",
            self.btn_continue_simulation_clicked,
            experiment_model,
            self.toolbar,
            shortcut="Ctrl+R",
            icon=qta.icon("mdi6.head"),
        )
        add_action(
            "&Store simulation checkpoint",
            self.btn_store_checkpoint_clicked,
            experiment_model,
            self.toolbar,
            shortcut="Ctrl+Shift+S",
            icon=qta.icon("mdi6.head-check"),
        )
        add_action(
            "&Restore simulation checkpoint",
            self.btn_restore_checkpoint_clicked,
            experiment_model,
            self.toolbar,
            shortcut="Ctrl+Shift+R",
            icon=qta.icon("mdi6.head-sync"),
        )
        self.toolbar.addSeparator()
        experiment_results = menu.addMenu("&Playback")
        add_action(
            "&Skip to start",
            self.btn_skip_to_start_clicked,
            experiment_results,
            self.toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.skip-previous"),
        )
        add_action(
            "&Stop",
            self.btn_stop_clicked,
            experiment_results,
            self.toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.stop"),
        )
        add_action(
            "&Play / Pause",
            self.btn_play_clicked,
            experiment_results,
            self.toolbar,
            shortcut="Space",
            icon=qta.icon("mdi6.play-pause"),
        )
        add_action(
            "&Skip to end",
            self.btn_skip_to_end_clicked,
            experiment_results,
            self.toolbar,
            shortcut=None,
            icon=qta.icon("mdi6.skip-next"),
        )
        self.toolbar.addSeparator()
        help_menu = menu.addMenu("&Help")
        add_action("&About", self.about, help_menu)
