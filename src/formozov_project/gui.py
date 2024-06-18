from __future__ import annotations

import logging
from typing import Callable

import brian2 as b2
import numpy as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getsavefilename

from formozov_project import __version__
from formozov_project.b2_model import B2Model
from formozov_project.heatmap_widget import HeatmapWidget
from formozov_project.trace_widget import TraceWidget
from formozov_project.b2_model_widget import B2ModelWidget


def temp_new_model() -> B2Model:
    """Placeholder built-in model based on test_b2model/test_ornstein_uhlenbeck to demo GUI features."""
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """
    n_neurons = 64
    model = B2Model(seed=4321, logpath=None)
    model.define_neuronnumber("neurongroup", n_neurons)
    model.define_dt(0.5 * b2.ms)
    model.define_solver_method("neurongroup", "euler")
    model.define_equation("neurongroup", eqs)
    model.assemble_neurongroup(name="neurongroup")
    model.define_monitor("M", b2.StateMonitor, "neurongroup", "y")

    # init
    model.network["neurongroup"].tau = np.random.rand(n_neurons) * b2.second
    model.network["neurongroup"].sigma = 0.5 * np.random.rand(n_neurons)
    model.network["neurongroup"].y = 0
    return model


class Gui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.models = {"ornstein_uhlenbeck": temp_new_model(),}
        self.setWindowTitle("Formozov Project")
        grid_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(grid_layout)
        # main GUI is left/right split between traces and heatmap
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.trace_widget = TraceWidget()
        split_left_right.addWidget(self.trace_widget)
        self.heatmap_widget = HeatmapWidget()
        split_left_right.addWidget(self.heatmap_widget)
        grid_layout.addWidget(split_left_right)
        split_left_right.setSizes([3000, 2000])
        self.setCentralWidget(central_widget)
        self.resize(800, 600)
        # add toolbar and menu
        self.toolbar = _create_menu_and_toolbar(self)
        self.addToolBar(self.toolbar)
        self.setWindowIcon(qta.icon("mdi6.brain"))

        # connect widgets
        self.heatmap_widget.connect_trace_widget(self.trace_widget)
        self.trace_widget.connect_heatmap_widget(self.heatmap_widget)

        # self.btn_new_clicked()

    def _update_data(self) -> None:
        if len(self.models["ornstein_uhlenbeck"].network["M"].t) == 0:
            self.trace_widget.set_data()
            self.heatmap_widget.set_data()
            return
        self.trace_widget.set_data(
            self.models["ornstein_uhlenbeck"].network["M"].y, self.models["ornstein_uhlenbeck"].network["M"].t / b2.second
        )
        # for now just display outer product of traces as heatmap data
        outer = np.array(
            [np.outer(trace, trace) for trace in self.models["ornstein_uhlenbeck"].network["M"].y.T]
        )
        self.heatmap_widget.set_data(outer, self.models["ornstein_uhlenbeck"].network["M"].t / b2.second)

    def btn_new_clicked(self) -> None:
        self.model = temp_new_model()
        self.trace_widget.set_data()
        self.heatmap_widget.set_data()
        self.b2m = B2ModelWidget(self.model, self, self.models)
        self.b2m.exec()

    def btn_open_clicked(self) -> None:
        self.btn_new_clicked()

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
            self.models["ornstein_uhlenbeck"].save_to_h5(filename)
        except Exception as e:
            logging.warning(f"Failed to save file {filename}: {e}")
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
            self.models["ornstein_uhlenbeck"].checkpoints,
            current=0,
            editable=True,
        )
        if not ok:
            return
        self.models["ornstein_uhlenbeck"].store(checkpoint)

    def btn_restore_checkpoint_clicked(self) -> None:
        if len(self.models["ornstein_uhlenbeck"].checkpoints) == 0:
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
            self.models["ornstein_uhlenbeck"].checkpoints,
            current=0,
            editable=False,
        )
        if not ok:
            return
        try:
            self.models["ornstein_uhlenbeck"].restore(checkpoint)
            self._update_data()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Error", f"{error}")
            return

    def btn_continue_simulation_clicked(self) -> None:
        simulation_time_str, ok = QtWidgets.QInputDialog.getText(
            self, "Simulate", "Simulation time (s)", QtWidgets.QLineEdit.Normal, "1.0"
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
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.models["ornstein_uhlenbeck"].run_simulation(simulation_time * b2.second)
        self._update_data()
        QtWidgets.QApplication.restoreOverrideCursor()

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


def _add_action(
    name: str,
    callback: Callable,
    menu: QtWidgets.QMenu,
    toolbar: QtWidgets.QToolBar | None = None,
    shortcut: str | None = None,
    icon: QtGui.QIcon | None = None,
) -> None:
    action = QtWidgets.QAction(name, menu)
    action.triggered.connect(callback)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(QtGui.QKeySequence(shortcut))
    menu.addAction(action)
    if toolbar is not None:
        toolbar.addAction(action)


def _create_menu_and_toolbar(gui: Gui) -> QtWidgets.QToolBar:
    menu = gui.menuBar()
    toolbar = QtWidgets.QToolBar()
    file_menu = menu.addMenu("&File")
    _add_action(
        "&New",
        gui.btn_new_clicked,
        file_menu,
        toolbar,
        "Ctrl+N",
        qta.icon("mdi6.file"),
    )
    _add_action(
        "&Open",
        gui.btn_open_clicked,
        file_menu,
        toolbar,
        "Ctrl+O",
        qta.icon("mdi6.folder-open"),
    )
    _add_action(
        "&Save",
        gui.btn_save_clicked,
        file_menu,
        toolbar,
        "Ctrl+S",
        qta.icon("mdi6.content-save"),
    )
    toolbar.addSeparator()
    _add_action(
        "E&xit",
        gui.btn_exit_clicked,
        file_menu,
        None,
        None,
        qta.icon("mdi6.exit-to-app"),
    )
    experiment_model = menu.addMenu("&Simulate")
    _add_action(
        "&Continue simulation",
        gui.btn_continue_simulation_clicked,
        experiment_model,
        toolbar,
        "Ctrl+R",
        qta.icon("mdi6.head"),
    )
    _add_action(
        "&Store simulation checkpoint",
        gui.btn_store_checkpoint_clicked,
        experiment_model,
        toolbar,
        "Ctrl+Shift+S",
        qta.icon("mdi6.head-check"),
    )
    _add_action(
        "&Restore simulation checkpoint",
        gui.btn_restore_checkpoint_clicked,
        experiment_model,
        toolbar,
        "Ctrl+Shift+R",
        qta.icon("mdi6.head-sync"),
    )
    toolbar.addSeparator()
    experiment_results = menu.addMenu("&Playback")
    _add_action(
        "&Skip to start",
        gui.btn_skip_to_start_clicked,
        experiment_results,
        toolbar,
        None,
        qta.icon("mdi6.skip-previous"),
    )
    _add_action(
        "&Stop",
        gui.btn_stop_clicked,
        experiment_results,
        toolbar,
        None,
        qta.icon("mdi6.stop"),
    )
    _add_action(
        "&Play / Pause",
        gui.btn_play_clicked,
        experiment_results,
        toolbar,
        "Space",
        qta.icon("mdi6.play-pause"),
    )
    _add_action(
        "&Skip to end",
        gui.btn_skip_to_end_clicked,
        experiment_results,
        toolbar,
        None,
        qta.icon("mdi6.skip-next"),
    )
    toolbar.addSeparator()
    help_menu = menu.addMenu("&Help")
    _add_action("&About", gui.about, help_menu)
    return toolbar
