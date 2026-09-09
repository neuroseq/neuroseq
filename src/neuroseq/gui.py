from __future__ import annotations

import contextlib
import gc
import inspect
import logging
import pathlib
import threading
from collections.abc import Callable

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq import __version__
from neuroseq import theme as theme_module
from neuroseq.analysis_plugins import AnalysisPlugin
from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.chat_widget import ChatWidget
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
from neuroseq.detached_panel_window import DetachedPanelWindow
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.experiments_manager_dialog import ExperimentsManagerDialog
from neuroseq.exporter_igor2 import export_igor2_csv
from neuroseq.exporter_igor2 import export_igor2_ibw
from neuroseq.exporter_igor2 import export_igor2_pxp
from neuroseq.gui_settings import GuiSettings
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.icon_manager import icons
from neuroseq.igor2_pxp_stim_export_dialog import Igor2PxpStimExportDialog
from neuroseq.igor2_single_wave_export_dialog import Igor2SingleWaveExportDialog
from neuroseq.igor_import_dialog import IgorImportDialog
from neuroseq.import_dialog import Suite2pImportDialog
from neuroseq.import_experiment_dialog import ImportExperimentDialog
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import clone_cellpose_result
from neuroseq.llm_settings_dialog import LLMSettingsDialog
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog
from neuroseq.project import Project
from neuroseq.resource_monitor_label import ResourceMonitorLabel
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.save_load_options import SaveLoadOptionsDialog
from neuroseq.spatial_footprint_state import SpatialFootprintState
from neuroseq.spatial_footprint_widget import SpatialFootprintWidget
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim import StimArray
from neuroseq.stim_editor import StimEditor
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.synthetic_data_dialog import SyntheticDataDialog
from neuroseq.syscon2_import_dialog import Syscon2ImportDialog
from neuroseq.time_slider_widget import TimeSliderWidget
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.trace_widget import TraceWidget
from neuroseq.trace_widget_state import TraceWidgetState
from neuroseq.utils import add_action
from neuroseq.utils import add_items_to_combobox
from neuroseq.utils import getSimulationTime
from neuroseq.utils import with_wait_cursor


class SimulationCancelledError(Exception):
    """Raised from a progress callback when the user clicks Cancel."""


class ToolbarProgressWidget(QtWidgets.QWidget):
    """Compact progress and cancellation controls embedded in a toolbar."""

    canceled = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._toolbar_action: QtWidgets.QAction | None = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 2, 0)
        layout.setSpacing(4)

        self.label = QtWidgets.QLabel("Simulating...", self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(100)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.setFixedWidth(58)
        self.cancel_button.clicked.connect(self._cancel)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.label)
        self.cancel_button.hide()
        self.hide()

    def set_toolbar_action(self, action: QtWidgets.QAction) -> None:
        self._toolbar_action = action
        action.setVisible(False)

    def show(self) -> None:
        if self._toolbar_action is not None:
            self._toolbar_action.setVisible(True)
        super().show()

    def hide(self) -> None:
        if self._toolbar_action is not None:
            self._toolbar_action.setVisible(False)
        super().hide()

    @QtCore.Slot()
    def _cancel(self) -> None:
        self.hide()
        self.canceled.emit()

    @QtCore.Slot(str)
    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def labelText(self) -> str:
        """Return the progress label, matching ``QProgressDialog``'s API."""
        return self.label.text()

    def setRange(self, minimum: int, maximum: int) -> None:
        self.progress_bar.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:
        # Do not auto-hide when the bar reaches its maximum: progress callbacks
        # can legitimately report 100% before an operation is fully done, and
        # every operation hides the bar explicitly via reset() at teardown.
        self.progress_bar.setValue(value)

    def value(self) -> int:
        return self.progress_bar.value()

    def setCancelButtonText(self, text: str) -> None:
        self.cancel_button.setText(text)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()

    def setCancelButton(self, button: QtWidgets.QPushButton | None) -> None:
        if button is None:
            self.cancel_button.hide()

    def setAutoClose(self, _enabled: bool) -> None:
        # No-op: the bar is never auto-hidden on reaching its maximum; teardown
        # paths hide it explicitly via reset(). Kept for QProgressDialog API
        # compatibility.
        pass

    def setAutoReset(self, _enabled: bool) -> None:
        # Resetting is explicitly managed by the operation teardown paths.
        pass

    def setMinimumDuration(self, _milliseconds: int) -> None:
        # Toolbar progress is shown immediately and never spawns a dialog.
        pass

    def reset(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.cancel_button.hide()
        self.hide()


class BackgroundIOWorker(QtCore.QObject):
    """Runs a blocking IO task on a worker QThread.

    The task is a callable taking a single argument: a
    ``report(percent: int)`` callback that the worker translates into the
    ``progress(int)`` signal. Tasks that don't natively report progress can
    ignore the callback. The task's return value is delivered to the main
    thread via ``finished(result)``; any exception is stringified and
    delivered via ``failed(str)``.
    """

    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, task: Callable[[Callable[[int], None]], object]) -> None:
        super().__init__()
        self._task = task

    @QtCore.Slot()
    def run(self) -> None:
        def _report(percent: int) -> None:
            with contextlib.suppress(Exception):
                self.progress.emit(int(percent))

        try:
            result = self._task(_report)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class ExternalSimWorker(QtCore.QObject):
    """Runs an external simulation on a worker QThread."""

    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(
        self,
        simulate_fn: Callable,
        stims_dict: dict,
        simulation_time: float,
        params: dict,
        accepts_progress: bool,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._simulate_fn = simulate_fn
        self._stims_dict = stims_dict
        self._simulation_time = float(simulation_time)
        self._params = params
        self._accepts_progress = accepts_progress
        self._cancel_event = cancel_event

    @QtCore.Slot()
    def run(self) -> None:
        def _progress_cb(fraction: float) -> None:
            if self._cancel_event.is_set():
                raise SimulationCancelledError("user cancelled")
            with contextlib.suppress(Exception):
                self.progress.emit(float(fraction))

        try:
            if self._accepts_progress:
                experiment = self._simulate_fn(
                    self._stims_dict,
                    self._simulation_time,
                    params=self._params,
                    progress=_progress_cb,
                )
            else:
                experiment = self._simulate_fn(
                    self._stims_dict,
                    self._simulation_time,
                    params=self._params,
                )
        except SimulationCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(experiment)


class EliseProtocolWorker(QtCore.QObject):
    """Runs the ELiSe train/validation/replay protocol on a worker QThread."""

    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(
        self,
        run_fn: Callable,
        kwargs: dict,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._run_fn = run_fn
        self._kwargs = kwargs
        self._cancel_event = cancel_event

    @QtCore.Slot()
    def run(self) -> None:
        def _progress_cb(fraction: float) -> None:
            if self._cancel_event.is_set():
                raise SimulationCancelledError("user cancelled")
            with contextlib.suppress(Exception):
                self.progress.emit(float(fraction))

        try:
            result = self._run_fn(progress=_progress_cb, **self._kwargs)
        except SimulationCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


def _external_simulator_accepts_progress(simulate_fn: Callable) -> bool:
    """True if ``simulate_fn`` advertises a ``progress`` keyword argument.

    Used so old-style third-party simulators (``simulate(stims_dict,
    simulation_time, params=None)``) keep working unchanged, while the
    bundled :class:`NN`-based backends receive the progress callback the
    main window's progress dialog needs.
    """
    try:
        sig = inspect.signature(simulate_fn)
    except (TypeError, ValueError):
        return False
    return "progress" in sig.parameters


class Gui(QtWidgets.QMainWindow):
    TOOLBAR_ICON_SIZE = QtCore.QSize(20, 20)

    def _configure_toolbar_icon_style(self, toolbar: QtWidgets.QToolBar) -> None:
        toolbar.setIconSize(self.TOOLBAR_ICON_SIZE)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setStyleSheet("QToolButton { margin: 0px; padding: 2px 1px; }")

    def __init__(self, filename: pathlib.Path | str | None = None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("starting Gui")
        self.settings = GuiSettings()
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(ResourceMonitorLabel())
        # Each open project lives in its own browser-style tab; ``self.project``
        # (see the property below) always refers to the active tab's project.
        self.tabs: list[Project] = [Project()]
        self._active_tab_index = 0
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
        self.trace_controller = ConstrainedPanZoomController()
        self.spike_controller = ConstrainedPanZoomController()
        self._trace_state = TraceWidgetState(parent=self)
        self._spatial_footprint_state = SpatialFootprintState(parent=self)
        self.trace_widget = TraceWidget(
            self.trace_controller,
            parent=self,
            default_controller=self.controller,
            show_axis_link_controls=True,
            state=self._trace_state,
        )
        self.heatmap_widget = HeatmapWidget(parent=self)
        self.spatial_footprint_widget = SpatialFootprintWidget(
            parent=self, state=self._spatial_footprint_state
        )
        self.spatial_footprint_widget.import_requested.connect(
            self.btn_import_cellpose_data_clicked
        )
        self.spatial_footprint_widget.trim_to_mapped_neurons_requested.connect(
            self.btn_trim_to_mapped_neurons_clicked
        )
        self.spatial_footprint_widget.neuron_greyed_out_changed.connect(
            self._on_neuron_greyed_out_changed
        )
        self._create_menus_and_toolbars()
        self.setWindowIcon(qta.icon("fa6s.brain"))
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.tab_bar.setMovable(True)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDocumentMode(True)
        self.tab_bar.addTab(self._tab_label_for(self.tabs[0]))
        self.tab_bar.currentChanged.connect(self._on_tab_bar_current_changed)
        self.tab_bar.tabCloseRequested.connect(self._close_tab)
        main_horizontal_layout = QtWidgets.QHBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_horizontal_layout)
        # vertical toolbar for stim editing on the left
        self.toolbar_stim_editor = QtWidgets.QToolBar()
        self.toolbar_stim_editor.setOrientation(QtCore.Qt.Orientation.Vertical)
        self._configure_toolbar_icon_style(self.toolbar_stim_editor)
        main_horizontal_layout.addWidget(self.toolbar_stim_editor)
        # left/right splitters between heatmap and console
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split_left_right.addWidget(self.heatmap_widget)
        split_left_right.addWidget(self.spatial_footprint_widget)
        self.custom_analysis_widget = CustomAnalysisWidget(parent=self)
        split_left_right.addWidget(self.custom_analysis_widget)
        self.console_widget = ConsoleWidget(parent=self)
        split_left_right.addWidget(self.console_widget)
        self.chat_widget = ChatWidget(parent=self)
        self.chat_widget.setVisible(False)
        self.chat_widget.request_settings_dialog.connect(self.open_llm_settings)
        split_left_right.addWidget(self.chat_widget)
        split_left_right.setSizes([1000, 1000, 1000, 1000, 1000])
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
        self.stim_editor.simulation_requested.connect(
            self._do_brian2_simulation_threaded
        )
        self.stim_editor.time_range_focus_requested.connect(
            self.trace_controller.focus_time_range
        )
        self.stim_editor.current_stim_changed.connect(self._on_current_stim_changed)
        split_vertical.addWidget(self.stim_editor)
        self.trace_widget.set_default_scroll_bars(
            self.stim_editor.x_scrollbar,
            self.stim_editor.y_scrollbar,
        )
        split_vertical.addWidget(self.trace_widget)
        self.spike_widget = SpikeWidget(
            self.spike_controller,
            parent=self,
            default_controller=self.controller,
            show_axis_link_controls=True,
        )
        self.spike_widget.set_default_scroll_bars(
            self.stim_editor.x_scrollbar,
            self.stim_editor.y_scrollbar,
        )
        split_vertical.addWidget(self.spike_widget)
        self.data_widget = DataWidget(self.controller, parent=self)
        self.data_widget.set_default_scroll_bars(
            self.stim_editor.x_scrollbar,
            self.stim_editor.y_scrollbar,
        )
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
        self.trace_widget.setVisible(False)
        self.spike_widget.setVisible(False)
        self.heatmap_widget.setVisible(False)
        self.spatial_footprint_widget.setVisible(True)
        self.custom_analysis_widget.setVisible(False)
        # heatmap / custom analysis widget / console
        split_vertical.addWidget(split_left_right)
        main_horizontal_layout.addWidget(split_vertical)
        outer_widget = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self.tab_bar)
        outer_layout.addWidget(central_widget)
        self.setCentralWidget(outer_widget)
        self._create_actions()
        self.showMaximized()
        self._create_progressbar()
        # connect widgets
        self.console_widget.console_command_entered.connect(self._update_data)
        self.console_widget.close_requested.connect(
            lambda: (
                self._view_actions[self.console_widget].trigger()
                if self._view_actions.get(self.console_widget, None) is not None
                and self._view_actions[self.console_widget].isChecked()
                else None
            )
        )
        self.trace_widget.custom_analysis_requested.connect(self._custom_analysis)
        self.trace_widget.vector_export_requested.connect(
            self.btn_export_traces_vector_clicked
        )
        self.data_widget.trace_widget.custom_analysis_requested.connect(
            self._custom_analysis
        )
        self.spike_widget.custom_analysis_requested.connect(self._custom_analysis)
        self.data_widget.spike_widget.custom_analysis_requested.connect(
            self._custom_analysis
        )
        self._install_detach_context_menus()
        self.cmb_trace.currentTextChanged.connect(self.set_neuron)
        self.cmb_synapse.currentTextChanged.connect(self.set_synapse)
        self.cmb_spike.currentTextChanged.connect(self.set_spike)
        self.cmb_data.currentTextChanged.connect(self.set_data)
        self.stim_editor.limits_modified.connect(self.update_limits)
        self.stim_editor.limits_modified.connect(self._rewindow_simulation_views)
        # Longest stim length last used to window simulated data; lets
        # ``_rewindow_simulation_views`` re-fetch traces only when a stim
        # edit actually changed the timeline (e.g. trim/undo), not on every
        # pixel drawn.
        self._last_sim_timeline_secs: float | None = None
        self.last_cellpose_import: CellposeImportResult | None = None
        self._last_custom_analysis: (
            tuple[
                TraceData | SpikeData | SynapseData,
                CustomAnalysisType | AnalysisPlugin | None,
                slice | None,
                slice | None,
            ]
            | None
        ) = None
        self._detached_panel_windows: list[DetachedPanelWindow] = []
        self._llm_settings_dialog: LLMSettingsDialog | None = None
        self._external_sim_thread: QtCore.QThread | None = None
        self._external_sim_worker: ExternalSimWorker | None = None
        self._external_sim_cancel: threading.Event | None = None
        self._external_sim_dispatch_project: Project | None = None
        self._elise_protocol_thread: QtCore.QThread | None = None
        self._elise_protocol_worker: EliseProtocolWorker | None = None
        self._elise_protocol_cancel: threading.Event | None = None
        self._io_thread: QtCore.QThread | None = None
        self._io_worker: BackgroundIOWorker | None = None
        self._io_on_success: Callable[[object], None] | None = None
        self._io_on_failure: Callable[[str], None] | None = None
        self._io_failure_title = "File operation failed"
        self._io_failure_prefix = "The file operation could not complete:"
        self._close_confirmed = False
        if filename is not None:
            self.load_project(filename, new_tab=False)
        else:
            self.load_project(default_project(), new_tab=False)
        self.apply_theme(self.settings.theme)

    @property
    def project(self) -> Project:
        return self.tabs[self._active_tab_index]

    @project.setter
    def project(self, value: Project) -> None:
        self.tabs[self._active_tab_index] = value

    def _tab_label_for(self, project: Project) -> str:
        if project.filepath is not None:
            return pathlib.Path(project.filepath).name
        if project.project_section and project.project_name:
            return f"{project.project_section} / {project.project_name}"
        return project.b2_model.name or "Untitled"

    def _popup_title(self, title: str) -> str:
        return f"{title} - {self._tab_label_for(self.project)}"

    def _activate_tab_index(self, index: int, reset_controller: bool = False) -> None:
        if 0 <= self._active_tab_index < len(self.tabs):
            # Remember where the outgoing tab's playback position was, so
            # switching back to it later restores this instead of showing
            # whatever the (shared) time slider happens to display.
            self.tabs[self._active_tab_index].last_time_ms = self.time_slider.t_ms
        self._active_tab_index = index
        self.update_project(self.tabs[index], reset_controller=reset_controller)

    def _on_tab_bar_current_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.tabs) or index == self._active_tab_index:
            return
        self._activate_tab_index(index, reset_controller=True)

    def _open_project_in_new_tab(
        self, project: Project, reset_controller: bool = True
    ) -> None:
        self.tabs.append(project)
        index = len(self.tabs) - 1
        self.tab_bar.blockSignals(True)
        self.tab_bar.addTab(self._tab_label_for(project))
        self.tab_bar.setCurrentIndex(index)
        self.tab_bar.blockSignals(False)
        self._activate_tab_index(index, reset_controller=reset_controller)

    def _close_tab(self, index: int) -> None:
        if index < 0 or index >= len(self.tabs):
            return
        if index != self._active_tab_index:
            self.tab_bar.setCurrentIndex(index)
        self._prompt_save_before_action(
            "Close tab",
            "Closing this tab will discard any unsaved changes to its "
            "project.\n\nSave the project first?",
            lambda: self._remove_tab(index),
        )

    def _remove_tab(self, index: int) -> None:
        if len(self.tabs) == 1:
            self.update_project(default_project(), reset_controller=True)
            return
        del self.tabs[index]
        self.tab_bar.removeTab(index)

    def update_limits(self) -> None:
        max_time_secs = float(self.project.max_time_secs)
        max_neurons = int(self.project.max_neurons)
        self.controller.set_limits(max_time_secs, max_neurons)
        self.trace_controller.set_limits(max_time_secs, max_neurons)
        self.spike_controller.set_limits(max_time_secs, max_neurons)
        self.data_widget.set_limits(max_time_secs, max_neurons)
        self.time_slider.set_max_time_secs(max_time_secs)

    def _custom_analysis(
        self,
        data: TraceData | SpikeData | SynapseData,
        analysis_type: CustomAnalysisType | AnalysisPlugin | None,
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
                self._popup_title(analysis_window.analysis_widget.subplot.get_title())
            )
            analysis_window.show()
        else:
            self._last_custom_analysis = (
                data,
                analysis_type,
                neuron_slice,
                time_slice,
            )
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
        self._refresh_elise_replay_action()
        if hasattr(self, "spatial_footprint_widget"):
            self.spatial_footprint_widget.set_data(self.project.cellpose_data)

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

    def _update_load_progress(
        self,
        completed_percent: int,
    ) -> None:
        self.logger.info(f"loading :: {completed_percent}% complete")
        self.progress_dialog.setLabelText("Loading...")
        self.progress_dialog.setValue(completed_percent)

    def _prompt_save_before_action(
        self, title: str, message: str, proceed: Callable[[], None]
    ) -> None:
        """Warn that an action may discard unsaved changes, offering to save first.

        Shows a Save / Discard / Cancel prompt. Cancel aborts. Discard runs
        ``proceed`` immediately. Save persists the current project first and
        runs ``proceed`` only after the save completes; if the user backs out
        of the save dialog, ``proceed`` never runs and the current project is
        left untouched.
        """
        answer = QtWidgets.QMessageBox.warning(
            self,
            title,
            message,
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self._save_project(on_success=lambda _result: proceed())
            return
        proceed()

    def btn_new_clicked(self) -> None:
        # Opens in a new tab (see load_project), so there is nothing to
        # discard here -- the current tab's project is left untouched.
        self._create_new_project()

    def _prompt_save_before_close(self) -> bool:
        """Return True when the current close event may continue immediately."""
        message = (
            "Closing NeuroSeq will discard any unsaved changes to the "
            "current project.\n\nSave the current project first?"
        )
        if len(self.tabs) > 1:
            message = (
                "Closing NeuroSeq will discard any unsaved changes in "
                f"{len(self.tabs)} open project tabs.\n\nOnly the active tab "
                "can be saved here -- switch to another tab first if you "
                "need to save its changes.\n\nSave the active tab's project "
                "first?"
            )
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Close NeuroSeq",
            message,
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.StandardButton.Save:

            def _close_after_save(_result: object) -> None:
                self._close_confirmed = True
                self.close()

            self._save_project(on_success=_close_after_save)
            return False
        self._close_confirmed = True
        return True

    def _create_new_project(self) -> None:
        gc.collect()
        self.load_project(default_project())

    def _load_example_project(self, generator: Callable[[], Project]) -> None:
        gc.collect()
        self.load_project(generator())

    def btn_open_clicked(self) -> None:
        # Opens in a new tab (see load_project), so there is nothing to
        # discard here -- the current tab's project is left untouched.
        self._open_project_dialog()

    def _open_project_dialog(self) -> None:
        filename, _ = getopenfilename(
            self,
            "load project",
            "project.nseq",
            "neuroseq project (*.nseq)",
        )
        if filename != "":
            self.load_project(filename)

    def btn_import_clicked(self) -> None:
        self._prompt_save_before_action(
            "Import from a project",
            "Importing may overwrite parts of the current project.\n\nSave the "
            "current project first?",
            self._import_project_dialog,
        )

    def _import_project_dialog(self) -> None:
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
        if self._io_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "A file operation is already in progress.",
            )
            return

        project = self.project
        options = options_dialog.options
        filename_path = pathlib.Path(filename)

        def _import_task(report: Callable[[int], None]) -> None:
            project.import_project(filename_path, options, report=report)
            return None

        def _on_imported(_: object) -> None:
            self.update_project(self.project)

        self._dispatch_io_task(
            _import_task,
            f"Importing from {filename_path.name}…",
            _on_imported,
        )

    def btn_generate_synthetic_data_clicked(self) -> None:
        dialog = SyntheticDataDialog(self.project, parent=self)
        dialog.exec()
        # Refresh combo boxes if new data was added
        if dialog.data_added:
            self._update_combo_boxes()

    def btn_save_clicked(self) -> None:
        self._save_project()

    def btn_save_as_clicked(self) -> None:
        self._save_project(force_dialog=True)

    def _save_project(
        self,
        on_success: Callable[[object], None] | None = None,
        force_dialog: bool = False,
    ) -> bool:
        """Run the interactive save flow.

        If the project already has a file path and ``force_dialog`` is False
        (a plain "Save"), it is written straight back to that path with no
        prompts. Otherwise (a "Save As", or a first-ever save with no existing
        path), a file picker is shown. All four project components are always
        included; the options dialog remains available for import workflows.

        Returns True if a save task was dispatched, False if the user cancelled
        at the file dialog (or a file operation is already running).
        ``on_success`` runs after the save completes successfully; it lets a
        caller (e.g. New) chain an action once the current project is safely
        persisted.
        """
        if self._io_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "A file operation is already in progress.",
            )
            return False

        options = SaveLoadOptions(True, True, True, True)
        if not force_dialog and self.project.filepath is not None:
            filename = self.project.filepath
        else:
            picked_filename, _ = getsavefilename(
                self,
                "Save project",
                str(self.project.filepath) if self.project.filepath else "project.nseq",
                "neuroseq project (*.nseq)",
            )
            if picked_filename == "":
                return False
            filename = pathlib.Path(picked_filename)
            if filename.suffix != ".nseq":
                filename = filename.with_suffix(".nseq")

        project = self.project

        def _save_task(report: Callable[[int], None]) -> pathlib.Path:
            # project.save has no native progress callback; bracket it with
            # 0% / 100% so the dialog at least shows start/end.
            report(0)
            project.save(filename, options)
            report(100)
            return filename

        def _on_saved(result: object) -> None:
            self.logger.info(f"Project saved to '{result}'.")
            self._update_window_title()
            if on_success is not None:
                on_success(result)

        self._dispatch_io_task(_save_task, f"Saving to {filename.name}…", _on_saved)
        return True

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

    def btn_export_igor2_single_wave_clicked(self) -> None:
        if not self.project.stim_arrays:
            QtWidgets.QMessageBox.information(
                self,
                "No stimulation entities",
                "No stimulation data entities available. Create one before exporting.",
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

        dialog = Igor2SingleWaveExportDialog(stim_name, stim_array, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        suggested_filename = f"{dialog.experiment_name}.ibw"
        filename, _ = getsavefilename(
            self,
            "Export single wave to Igor2",
            suggested_filename,
            "Igor Binary Wave (*.ibw);;CSV file (*.csv)",
        )
        if filename == "":
            return

        output_path = pathlib.Path(filename)
        suffix = output_path.suffix.lower()
        if suffix not in {".ibw", ".csv"}:
            output_path = output_path.with_suffix(".ibw")
            suffix = ".ibw"

        try:
            if suffix == ".ibw":
                export_report = export_igor2_ibw(
                    output_path,
                    stim_array,
                    wave_name=dialog.experiment_name,
                    neuron_index=dialog.selected_neuron_index,
                    unit_override=dialog.unit_override,
                    scaling_factor=dialog.scaling_factor,
                    as_sweeps=dialog.as_sweeps,
                )
                if dialog.as_sweeps and export_report.n_sweeps == 1:
                    self.logger.info(
                        "Sweep export requested for '%s' but no period was detected; "
                        "exported as single sweep.",
                        stim_array.name,
                    )
                fmt_label = f"Igor Binary Wave '{export_report.experiment_name}'"
            else:
                if dialog.as_sweeps:
                    self.logger.info(
                        "Sweep export requested for '%s' but CSV was selected; "
                        "exported as single sweep.",
                        stim_array.name,
                    )
                export_report = export_igor2_csv(
                    output_path,
                    stim_array,
                    experiment_name=dialog.experiment_name,
                    neuron_index=dialog.selected_neuron_index,
                    scaling_factor=dialog.scaling_factor,
                )
                fmt_label = f"CSV experiment '{export_report.experiment_name}'"
        except Exception as e:
            self.logger.warning(
                "Failed to export Igor2 single wave %s: %s",
                output_path,
                e,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"Could not export Igor2 single wave to '{output_path}': {e}",
            )
            return

        if export_report.n_sweeps > 1:
            sweep_note = (
                f" Split into {export_report.n_sweeps} sweeps × "
                f"{export_report.sample_count // export_report.n_sweeps} samples."
            )
        else:
            sweep_note = ""
        QtWidgets.QMessageBox.information(
            self,
            "Export Successful",
            f"Exported {fmt_label} for "
            f"stimulus '{stim_name}', neuron {export_report.neuron_index}, "
            f"{export_report.sample_count} samples.{sweep_note}",
        )

    def btn_export_igor2_pxp_stim_clicked(self) -> None:
        if not self.project.stim_arrays:
            QtWidgets.QMessageBox.information(
                self,
                "No stimulation entities",
                "No stimulation data entities available. Create one before exporting.",
            )
            return

        dialog = Igor2PxpStimExportDialog(self.project, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected = dialog.selected_array_names
        if not selected:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing selected",
                "No stim arrays were selected for export.",
            )
            return

        filename, _ = getsavefilename(
            self,
            "Export Igor2 Packed Experiment",
            "stim_arrays.pxp",
            "Igor Packed Experiment (*.pxp)",
        )
        if filename == "":
            return

        output_path = pathlib.Path(filename)
        if output_path.suffix.lower() != ".pxp":
            output_path = output_path.with_suffix(".pxp")

        try:
            export_report = export_igor2_pxp(
                output_path,
                self.project.stim_arrays,
                array_names=selected,
                unit_override=dialog.unit_override,
                scaling_factor=dialog.scaling_factor,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Export error",
                f"PXP export failed:\n{e}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Exported {len(export_report.wave_names)} wave(s) to:\n{export_report.path}",
        )

    def btn_edit_clicked(self) -> None:
        old_neuron_count = self.project.b2_model.get_defined_neuron_count()
        mdiag = ModelEditorDialog(
            self.project.b2_model, parent=self, project=self.project
        )
        if mdiag.exec():
            if mdiag.external_simulate_params is not None:
                self.project.external_simulate_params = mdiag.external_simulate_params
            else:
                self.project.b2_model = mdiag.b2model
                # keep the stim arrays in step with the (possibly changed)
                # neuron count: grow to cover more neurons, or trim back when
                # the count is reduced. Grow-only on increase/unchanged edits
                # so a deliberately wider stim (e.g. from Cellpose) isn't lost.
                new_neuron_count = self.project.b2_model.get_defined_neuron_count()
                if 0 < new_neuron_count < old_neuron_count:
                    self.project.resize_stim_arrays_to_neuron_count(new_neuron_count)
                else:
                    self.project.pad_stim_arrays_to_neuron_count(new_neuron_count)
                self._warn_if_cellpose_model_mismatch()
            self.project.update_limits()
            self.update_project(self.project)

    def btn_manage_experiments_clicked(self) -> None:
        dialog = ExperimentsManagerDialog(self.project, parent=self)
        dialog.experiments_changed.connect(lambda: self.update_project(self.project))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Update the GUI with changes made in the dialog
            self.update_project(self.project)

    def load_project(
        self, project: str | pathlib.Path | Project, new_tab: bool = True
    ) -> None:
        """Load a project. File paths run on a worker thread; in-memory
        ``Project`` instances (test fixtures, example projects) load
        synchronously since there is no file IO to thread off.

        By default the project opens in a new tab, leaving any other open
        tabs untouched. Pass ``new_tab=False`` to instead replace the active
        tab's project in place (used only for the initial tab at startup).
        """

        def _place(loaded: Project) -> None:
            if new_tab:
                self._open_project_in_new_tab(loaded, reset_controller=True)
            else:
                self.update_project(loaded, reset_controller=True)

        if isinstance(project, Project):
            self.logger.info("Project loaded from Project object")
            self.last_cellpose_import = clone_cellpose_result(project.cellpose_data)
            project.cellpose_data = self.last_cellpose_import
            _place(project)
            return

        if self._io_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "A file operation is already in progress.",
            )
            return
        filename = pathlib.Path(project)

        def _load_task(report: Callable[[int], None]) -> Project:
            return Project(filename, report=report)

        def _on_loaded(result: object) -> None:
            if not isinstance(result, Project):
                return
            self.logger.info(f"Project loaded from {filename}")
            self.last_cellpose_import = clone_cellpose_result(result.cellpose_data)
            result.cellpose_data = self.last_cellpose_import
            _place(result)

        self._dispatch_io_task(_load_task, f"Loading {filename.name}…", _on_loaded)

    def _update_window_title(self) -> None:
        title_context: str | None = None
        if self.project is not None:
            if self.project.project_section and self.project.project_name:
                title_context = (
                    f"{self.project.project_section} / {self.project.project_name}"
                )
            else:
                title_context = self.project.b2_model.name
        model_suffix = f" [{title_context}]" if title_context else ""
        if self.project.filepath is not None:
            self.setWindowTitle(f"{self.project.filepath}{model_suffix} - Neuroseq")
        else:
            self.setWindowTitle(f"{model_suffix} - Neuroseq")

    @with_wait_cursor
    def update_project(
        self,
        project: Project,
        reset_controller: bool = False,
    ) -> None:
        self.project = project
        self._update_window_title()
        if hasattr(self, "tab_bar"):
            self.tab_bar.setTabText(
                self._active_tab_index, self._tab_label_for(project)
            )
        # Refresh the shared controllers/time-slider range for the newly
        # active project unconditionally — StimEditor.set_project() below
        # only does this itself when the project has stim arrays, which
        # would otherwise leave a stale range (and a stale time-slider
        # position) for any project with none.
        self.update_limits()
        if reset_controller:
            self.controller.reset()
            self.trace_controller.reset()
            self.spike_controller.reset()
        self.console_widget.set_project(self.project)
        self.chat_widget.set_project(self.project)
        self.continue_simulation_action.setText("&Run simulation")
        for widget in self.plot_widgets:
            widget.clear_data()
        self.custom_analysis_widget.clear_data()
        self._last_custom_analysis = None
        if self.project is not None:
            self.last_cellpose_import = clone_cellpose_result(
                self.project.cellpose_data
            )
            self.project.cellpose_data = self.last_cellpose_import

        self.stim_editor.set_project(self.project)
        # A replaced project must not inherit the previous run's Design dynamic
        # input state (plots, log, EPSP history); drop the cached dialog so it
        # is rebuilt fresh on next open.
        self.stim_editor.discard_input_designer()
        self.console_widget.set_project(self.project)
        self.chat_widget.set_project(self.project)
        self._update_combo_boxes()
        self._refresh_elise_replay_action()
        if hasattr(self, "spatial_footprint_widget"):
            self.spatial_footprint_widget.set_data(self.project.cellpose_data)
        # Restore this project's own remembered playback position now that
        # its data is loaded (rather than leaving whatever the previously
        # active project's position happened to be).
        self.time_slider.set_current_time_ms(project.last_time_ms)

    def _cellpose_neuron_count(self, imported: CellposeImportResult | None) -> int:
        if imported is None:
            return 0
        if imported.original_cellpose_ids is not None:
            return int(len(imported.original_cellpose_ids))
        if imported.segmentation_array is None:
            return 0
        unique_ids = np.unique(imported.segmentation_array)
        return int(np.sum(unique_ids >= 0))

    def _warn_if_cellpose_model_mismatch(self) -> None:
        """Warn when Cellpose data is present but its cell count differs from the
        model's neuron count (the two must agree for neuron-to-cell mapping,
        e.g. holographic stimulation export)."""
        if self.project is None or self.project.cellpose_data is None:
            return
        cellpose_count = self._cellpose_neuron_count(self.project.cellpose_data)
        model_count = self.project.b2_model.get_defined_neuron_count()
        if cellpose_count > 0 and model_count > 0 and cellpose_count != model_count:
            QtWidgets.QMessageBox.warning(
                self,
                "Cellpose / model size mismatch",
                f"The model has {model_count} neuron(s) but the Cellpose "
                f"segmentation has {cellpose_count} cell(s).\n\n"
                "These counts should match so that model neurons map one-to-one "
                "onto segmented cells (e.g. for holographic stimulation export).",
            )

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
        failed_remaps = self.project.remap_cellpose_indices()
        if failed_remaps:
            QtWidgets.QMessageBox.warning(
                self,
                "Trace mapping conflict",
                "Could not update the neuron mapping for previously imported "
                "trace/spike data in: "
                + ", ".join(sorted(set(failed_remaps)))
                + ".\n\nThe new Cellpose mapping conflicts with this data (e.g. "
                "duplicate or out-of-range IDs); their displayed indices were left "
                "unchanged.",
            )
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
        if hasattr(self, "spatial_footprint_widget"):
            self.spatial_footprint_widget.set_data(self.project.cellpose_data)
        return resized_names

    def _focus_neuron_range(self, n_neurons: int) -> None:
        if int(n_neurons) <= 0:
            return
        self.controller.focus_neuron_range(0, int(n_neurons))

    def _rewindow_simulation_views(self) -> None:
        """Re-fetch the shown simulated data when the stim timeline changes.

        Simulated traces/spikes/synapses are windowed to the longest stim
        array (see ``Project.get_data``), so trimming a stim (or undoing the
        trim) must re-pull the currently displayed data. The combo selections
        do not change, so ``currentTextChanged`` never fires; refresh here
        instead -- but only when the timeline length actually changed, so
        ordinary drawing edits don't trigger a costly re-fetch.
        """
        if self.project is None:
            return
        timeline = self.project.simulation_timeline_seconds()
        if (
            self._last_sim_timeline_secs is not None
            and timeline == self._last_sim_timeline_secs
        ):
            return
        self._last_sim_timeline_secs = timeline
        # Refresh in place without auto-showing hidden widgets or changing the
        # focused neuron range -- this is a re-window of already-shown data,
        # not a new user selection.
        if self.cmb_trace.currentText():
            self.project.max_ram_trace_plot_matrix = (
                self.settings.max_ram_trace_plot_matrix
            )
            self.trace_widget.set_data(
                self.project.get_data(self.cmb_trace.currentText(), TraceData)
            )
        if self.cmb_spike.currentText():
            self.spike_widget.set_data(
                self.project.get_data(self.cmb_spike.currentText(), SpikeData)
            )
        if self.cmb_synapse.currentText():
            self.project.max_ram_synapse_plot_matrix = (
                self.settings.max_ram_synapse_plot_matrix
            )
            self.heatmap_widget.set_data(
                self.project.get_data(self.cmb_synapse.currentText(), SynapseData)
            )
        if self.cmb_data.currentText():
            self._apply_data_selection(self.cmb_data.currentText())

    def _apply_trace_selection(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, TraceData)
        self.trace_widget.set_data(data)
        if data is not None and hasattr(self, "_auto_show_widget"):
            self._auto_show_widget(self.trace_widget)
        self._focus_neuron_range(self.trace_widget.n_neurons)

    @with_wait_cursor
    def set_neuron(self, cmb_name: str) -> None:
        self.project.max_ram_trace_plot_matrix = self.settings.max_ram_trace_plot_matrix
        self._apply_trace_selection(cmb_name)

    @with_wait_cursor
    def set_synapse(self, cmb_name: str) -> None:
        self.project.max_ram_synapse_plot_matrix = (
            self.settings.max_ram_synapse_plot_matrix
        )
        data = self.project.get_data(cmb_name, SynapseData)
        self.heatmap_widget.set_data(data)
        if data is not None and hasattr(self, "_auto_show_widget"):
            self._auto_show_widget(self.heatmap_widget)

    def _on_neuron_greyed_out_changed(
        self,
        neuron_index: int,
        greyed: bool,
    ) -> None:
        self._trace_state.set_neuron_greyed(neuron_index, greyed)
        self.stim_editor.stim_widget.set_neuron_greyed(neuron_index, greyed)

    def _current_stim_array(self) -> StimArray | None:
        modifier = self.stim_editor.current_stim_modifier
        return modifier.stim_array if modifier is not None else None

    def _on_current_stim_changed(self, stim_array: StimArray | None) -> None:
        if self.action_overlay_stim_on_traces.isChecked():
            self.trace_widget.set_stim_overlay_data(stim_array)

    def _toggle_stim_trace_overlay(self) -> None:
        """Toggle between the separated Stimuli+Traces panels and a single
        combined view where the stim heatmap renders behind the trace lines
        (issue #530). Only overlays a neuron's stim row onto its own trace
        row for now - overlaying a different neuron's stim/trace pair is
        deferred."""
        enabled = self.action_overlay_stim_on_traces.isChecked()
        self.trace_widget.set_stim_overlay_enabled(enabled)
        self.trace_widget.set_stim_overlay_data(self._current_stim_array())
        stim_panel_action = self._view_actions[self.stim_editor]
        if enabled:
            self._stim_panel_visible_before_overlay = stim_panel_action.isChecked()
            if stim_panel_action.isChecked():
                stim_panel_action.trigger()
        elif (
            getattr(self, "_stim_panel_visible_before_overlay", True)
            and not stim_panel_action.isChecked()
        ):
            stim_panel_action.trigger()

    def _apply_spike_selection(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, SpikeData)
        self.spike_widget.set_data(data)
        if data is not None and hasattr(self, "_auto_show_widget"):
            self._auto_show_widget(self.spike_widget)
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
        simulation_time = getSimulationTime(
            self,
            default_seconds=self.project.suggested_simulation_time_seconds(),
        )
        if simulation_time is None:
            return
        if self.project.external_simulate_path is not None:
            self._do_external_simulation(simulation_time)
            return
        self._do_brian2_simulation_threaded(simulation_time)

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

    def _do_external_simulation(self, simulation_time: float) -> None:
        if self.project.external_simulate_function is None:
            return
        if self._external_sim_worker is not None:
            self.logger.warning(
                "External simulation dispatch ignored - a sim is already running."
            )
            return

        cumulative_so_far = 0.0
        sim = getattr(self.project, "external_simulator", None)
        if sim is not None and hasattr(sim, "simulated_time_seconds"):
            cumulative_so_far = float(sim.simulated_time_seconds)
        target_seconds = cumulative_so_far + simulation_time
        self.project.tile_stim_arrays_to_length(target_seconds)
        self.stim_editor.set_project(self.project)
        stims_dict = {
            k: {"array": v.stim_array.values, "dt": v.stim_array.dt}
            for k, v in self.stim_editor.stims_dict.items()
        }
        params = self.project.external_simulate_params_flat()
        accepts_progress = _external_simulator_accepts_progress(
            self.project.external_simulate_function
        )
        self._external_sim_dispatch_project = self.project

        self.progress_dialog.setLabelText("Simulating...")
        self.progress_dialog.setValue(0)
        self.progress_dialog.setCancelButtonText("Cancel")
        # Threaded path: make the dialog non-modal so the rest of the
        # GUI stays interactive while the worker runs. Other entry
        # points (project load, import) keep the default WindowModal
        # via _teardown_external_sim restoring it.
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.progress_dialog.show()

        self._external_sim_cancel = threading.Event()
        self.progress_dialog.canceled.connect(self._external_sim_cancel.set)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._set_simulation_actions_enabled(False)

        thread = QtCore.QThread(self)
        worker = ExternalSimWorker(
            simulate_fn=self.project.external_simulate_function,
            stims_dict=stims_dict,
            simulation_time=simulation_time,
            params=params,
            accepts_progress=accepts_progress,
            cancel_event=self._external_sim_cancel,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_external_sim_progress)
        worker.finished.connect(self._on_external_sim_finished)
        worker.cancelled.connect(self._on_external_sim_cancelled)
        worker.failed.connect(self._on_external_sim_failed)
        for signal in (worker.finished, worker.cancelled, worker.failed):
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._external_sim_thread = thread
        self._external_sim_worker = worker
        self.logger.info(
            f"Dispatching external simulation: {simulation_time} s "
            f"(accepts_progress={accepts_progress})"
        )
        thread.start()

    def _teardown_external_sim(self) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
        self.progress_dialog.setCancelButton(None)
        # Restore the default modality used by other long-running paths
        # (project load, import) that block the main thread.
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        with contextlib.suppress(TypeError, RuntimeError):
            self.progress_dialog.canceled.disconnect()
        self._external_sim_worker = None
        self._external_sim_thread = None
        self._external_sim_cancel = None
        self._external_sim_dispatch_project = None
        self._set_simulation_actions_enabled(True)

    @QtCore.Slot(float)
    def _on_external_sim_progress(self, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.progress_dialog.setValue(int(round(clamped * 100)))

    @QtCore.Slot(object)
    def _on_external_sim_finished(self, experiment: Experiment) -> None:
        dispatch_project = self._external_sim_dispatch_project
        self._teardown_external_sim()
        self.progress_dialog.setValue(100)
        self.progress_dialog.reset()
        self.logger.info(
            f"External simulation complete: '{experiment.name}' "
            f"{experiment.duration}s, {len(experiment.data)} data objects"
        )
        if dispatch_project is None:
            self.logger.warning(
                "External simulation finished without a dispatch-time project; "
                "using the current project."
            )
            self._append_experimental_data(experiment)
            return
        if self.project is dispatch_project:
            self._append_experimental_data(experiment)
            return

        dispatch_project.add_experiment(experiment)
        self.logger.warning(
            f"Project swapped during external simulation. Experiment "
            f"'{experiment.name}' was appended to the dispatch-time project, "
            "not the currently active one."
        )
        QtWidgets.QMessageBox.information(
            self,
            "Simulation finished",
            "The external simulation finished after a project switch.\n"
            f"Experiment '{experiment.name}' was added to the project that "
            f"started it ('{dispatch_project.project_name or '(unnamed)'}'), "
            "not the project currently open.",
        )

    @QtCore.Slot()
    def _on_external_sim_cancelled(self) -> None:
        self._teardown_external_sim()
        self.progress_dialog.reset()
        self.logger.info("External simulation cancelled by user.")
        QtWidgets.QMessageBox.information(
            self,
            "Simulation cancelled",
            "Simulation cancelled - no data was added to the project.",
        )

    @QtCore.Slot(str)
    def _on_external_sim_failed(self, message: str) -> None:
        self._teardown_external_sim()
        self.progress_dialog.reset()
        self.logger.error(f"External simulation failed: {message}")
        QtWidgets.QMessageBox.critical(
            self,
            "External Simulation error",
            f"Could not run external simulation: {message}",
        )

    def _set_simulation_actions_enabled(self, enabled: bool) -> None:
        for attr in ("continue_simulation_action", "_do_external_simulation_action"):
            action = getattr(self, attr, None)
            if action is not None:
                action.setEnabled(bool(enabled))

    def _set_io_actions_enabled(self, enabled: bool) -> None:
        """Gate file-IO entry points while a save/load/simulation worker is running.

        Disables the File menu and toolbar plus the simulation entry
        points (so a long-running sim can't start mid-save and race the
        worker reading the project state), and the tab bar itself -- a
        background IO task (in particular a running Brian2 simulation)
        always targets the tab that was active when it was dispatched, so
        switching/closing/opening tabs mid-task would let its completion
        callback land on the wrong tab's now-active widgets.
        """
        for attr in ("menu_file", "toolbar_file", "tab_bar"):
            target = getattr(self, attr, None)
            if target is not None:
                target.setEnabled(bool(enabled))
        self._set_simulation_actions_enabled(bool(enabled))

    def _dispatch_io_task(
        self,
        task: Callable[[Callable[[int], None]], object],
        label: str,
        on_success: Callable[[object], None],
        failure_title: str = "File operation failed",
        failure_prefix: str = "The file operation could not complete:",
        cancel_callback: Callable[[], None] | None = None,
        on_failure: Callable[[str], None] | None = None,
    ) -> bool:
        """Run a blocking IO task on a worker thread.

        Returns ``True`` if the task was dispatched, ``False`` if another
        IO task is already in flight.

        If ``cancel_callback`` is given, the progress dialog shows a Cancel
        button wired to it that calls the callback on the GUI thread.
        """
        if self._io_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another file operation is in progress. Please wait for it "
                "to finish before starting a new one.",
            )
            return False

        self.progress_dialog.setLabelText(label)
        self.progress_dialog.setRange(0, 100)
        # Suppress the dialog's "smart" hiding for IO: the load callback
        # rapidly drives the bar 0 → 100 during the model phase, after
        # which the rest of the load (experiments, stim arrays, cellpose)
        # has no native progress. With the defaults (autoClose +
        # autoReset + minimumDuration=10) the dialog would either flash
        # briefly and disappear, or never show at all for fast tasks.
        # We close it ourselves in _teardown_io_task.
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        if cancel_callback is not None:
            self.progress_dialog.setCancelButtonText("Cancel")

            def _on_cancel() -> None:
                # The dialog hides on cancel, but the worker can keep running
                # (e.g. uninterruptible Brian2 compilation). Re-show a busy
                # "Cancelling..." bar until the worker actually finishes.
                cancel_callback()
                self.progress_dialog.setLabelText("Cancelling...")
                self.progress_dialog.setRange(0, 0)
                self.progress_dialog.setCancelButton(None)
                self.progress_dialog.show()

            self.progress_dialog.canceled.connect(_on_cancel)
        else:
            self.progress_dialog.setCancelButton(None)
        # Non-modal so the rest of the GUI stays interactive while the
        # worker thread does the IO. File-related actions are still
        # gated via _set_io_actions_enabled below.
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.progress_dialog.show()
        self.progress_dialog.raise_()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._set_io_actions_enabled(False)
        self._io_on_success = on_success
        self._io_failure_title = failure_title
        self._io_failure_prefix = failure_prefix
        self._io_on_failure = on_failure

        thread = QtCore.QThread(self)
        worker = BackgroundIOWorker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_io_progress)
        worker.finished.connect(self._on_io_finished)
        worker.failed.connect(self._on_io_failed)
        for signal in (worker.finished, worker.failed):
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._io_thread = thread
        self._io_worker = worker
        self.logger.info(f"Dispatching IO task: {label}")
        thread.start()
        return True

    def _teardown_io_task(self) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
        # Restore the dialog's default behaviour for synchronous users
        # (project save options dialog, Brian2 sim, etc.).
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setMinimumDuration(10)
        with contextlib.suppress(TypeError, RuntimeError):
            self.progress_dialog.canceled.disconnect()
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.reset()
        self._io_thread = None
        self._io_worker = None
        self._io_on_success = None
        self._io_on_failure = None
        self._io_failure_title = "File operation failed"
        self._io_failure_prefix = "The file operation could not complete:"
        self._set_io_actions_enabled(True)

    @QtCore.Slot(int)
    def _on_io_progress(self, percent: int) -> None:
        clamped = max(0, min(100, int(percent)))
        self.progress_dialog.setValue(clamped)

    @QtCore.Slot(object)
    def _on_io_finished(self, result: object) -> None:
        on_success = self._io_on_success
        # _teardown_io_task resets the dialog and restores its default
        # auto-close/auto-reset/min-duration behaviour for the next user.
        self._teardown_io_task()
        if on_success is not None:
            try:
                on_success(result)
            except Exception as exc:
                self.logger.exception(exc)
                QtWidgets.QMessageBox.critical(
                    self,
                    "Post-IO error",
                    f"Loading completed but a follow-up step failed: {exc}",
                )

    @QtCore.Slot(str)
    def _on_io_failed(self, message: str) -> None:
        failure_title = self._io_failure_title
        failure_prefix = self._io_failure_prefix
        on_failure = getattr(self, "_io_on_failure", None)
        # _teardown_io_task resets the dialog and restores its defaults.
        self._teardown_io_task()
        self.logger.error(f"IO task failed: {message}")
        QtWidgets.QMessageBox.critical(
            self,
            failure_title,
            f"{failure_prefix} {message}",
        )
        if on_failure is not None:
            on_failure(message)

    def _do_brian2_simulation_threaded(self, simulation_time: float) -> None:
        if self._io_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another operation is already in progress.",
            )
            # If Design dynamic input requested this run, tell it the request
            # was dropped so its closed loop stops instead of stalling forever
            # waiting for a completion callback that will never arrive. This is
            # a no-op when no routine is awaiting a simulation.
            self.stim_editor.input_designer_simulation_failed(
                "another operation is already in progress"
            )
            return

        project = self.project
        padding_secs = self.settings.stim_padding_secs

        def _request_brian2_cancel() -> None:
            self.logger.info("Brian2 simulation cancellation requested by user.")
            project.request_cancel_simulation()

        def _brian2_task(report: Callable[[int], None]) -> None:
            first_report_seen: list[bool] = []

            def _brian2_report(
                elapsed: b2.Quantity,
                completed: b2.Quantity,
                start: b2.Quantity,
                duration: b2.Quantity,
            ) -> None:
                completed_percent = int(100 * completed)
                self.logger.info(
                    f"simulating {start} -> {duration} :: "
                    f"{completed_percent}% complete in {elapsed}"
                )
                if not first_report_seen:
                    first_report_seen.append(True)
                    QtCore.QMetaObject.invokeMethod(
                        self.progress_dialog,
                        "setLabelText",
                        QtCore.Qt.ConnectionType.QueuedConnection,
                        QtCore.Q_ARG(str, "Simulating Brian2..."),
                    )
                report(int(100 * float(completed)))

            project.run_simulation(
                simulation_time,
                report=_brian2_report,
                stim_array_padding_seconds=padding_secs,
            )
            return None

        def _on_brian2_finished(_result: object) -> None:
            self.logger.info("Brian2 simulation finished.")
            visible_time_range = self.controller.current_time_range()
            self._update_data(keep_undo_history=True)
            if visible_time_range is not None:
                self.controller.focus_time_range(*visible_time_range)
                self.trace_controller.focus_time_range(*visible_time_range)
            # Let Qt paint the newly refreshed trace before Design dynamic input
            # appends and focuses the next stimulation block.
            QtCore.QTimer.singleShot(
                0, self.stim_editor.input_designer_simulation_finished
            )

        self._dispatch_io_task(
            _brian2_task,
            "Compiling Brian2 model...",
            _on_brian2_finished,
            failure_title="Simulation error",
            failure_prefix="Could not run simulation:",
            cancel_callback=_request_brian2_cancel,
            on_failure=self.stim_editor.input_designer_simulation_failed,
        )

    def btn_do_external_simulation_clicked(self) -> None:
        if self.project.external_simulate_function is None:
            QtWidgets.QMessageBox.information(
                self,
                "No external simulation function",
                "No external simulation function loaded - please load one first.",
            )
            return
        simulation_time = getSimulationTime(
            self,
            default_seconds=self.project.suggested_simulation_time_seconds(),
        )
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

    def toggle_minimum_stimulus_size(self, is_visible: bool) -> None:
        self.stim_editor.stim_widget.set_keep_short_stimuli_visible(is_visible)

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

    def choose_max_ram_trace_plot_matrix(self) -> None:
        new_max_ram, ok = QtWidgets.QInputDialog.getText(
            self,
            "RAM limit",
            "RAM limit in MB for trace plot (0 for no limit)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.settings.max_ram_trace_plot_matrix}",
        )
        if not ok:
            return
        try:
            self.settings.max_ram_trace_plot_matrix = max(0, int(new_max_ram))
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

    def btn_import_tif_clicked(self) -> None:
        """Open dialog to run the Suite2p pipeline on raw 2-photon TIFFs."""
        from neuroseq.tif_import_dialog import TifImportDialog

        dialog = TifImportDialog(
            self,
            cellpose_import=self.last_cellpose_import,
            stim_arrays=self.project.stim_arrays,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._append_experimental_data(
                Experiment(
                    dialog.experiment_name,
                    dialog.experiment_duration,
                    [],
                    dialog.imported_data,
                )
            )
            if dialog.new_cellpose_result is not None:
                self._apply_cellpose_import_result(dialog.new_cellpose_result)
                if hasattr(self, "_auto_show_widget"):
                    self._auto_show_widget(self.spatial_footprint_widget)

    def btn_open_fiji_clicked(self) -> None:
        from neuroseq.external_tools import launch_fiji

        launch_fiji(self)

    def btn_open_cellpose_clicked(self) -> None:
        from neuroseq.external_tools import launch_cellpose

        launch_cellpose(self)

    def choose_fiji_path(self) -> None:
        """Let the user pick/correct the Fiji executable path from Settings."""
        from neuroseq.fiji import resolve_fiji_path

        resolve_fiji_path(self, force_prompt=True)

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
            if hasattr(self, "_auto_show_widget"):
                self._auto_show_widget(self.spatial_footprint_widget)
            message = "Cell spatial footprints loaded."
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
                "Cell spatial footprint import",
                message,
            )
            self._warn_if_cellpose_model_mismatch()

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
        if hasattr(self, "spatial_footprint_widget"):
            self.spatial_footprint_widget.set_data(None)

    def btn_trim_to_mapped_neurons_clicked(self) -> None:
        """Permanently keep only Cellpose-mapped neurons across the whole
        project, then require the result to be saved under a new file (never
        silently overwriting the currently-open one)."""
        if self.project is None or self.project.cellpose_data is None:
            return
        mapping = self.project.cellpose_data.stimulus_to_cellpose_mapping
        if not mapping:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing to trim",
                "No Cellpose stimulus mapping is set, so there is nothing to trim to.",
            )
            return
        n_kept = len(mapping)
        n_removed = max(0, int(self.project.max_neurons) - n_kept)
        if n_removed == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing to trim",
                f"All {n_kept} neuron(s) already have a Cellpose cell mapped "
                "to them - there is nothing to remove.",
            )
            return
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Trim to Cellpose-mapped neurons",
            f"This will permanently remove {n_removed} neuron(s) with no "
            f"Cellpose cell mapped to them, keeping the {n_kept} that do, "
            "from every stim array and every experiment's trace/spike/synapse "
            "data in this project.\n\n"
            "This cannot be undone. You will be asked to save the result "
            "under a new file name.\n\n"
            "Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.project.trim_to_mapped_neurons()
        self.update_project(self.project)
        if not self._save_project(force_dialog=True):
            QtWidgets.QMessageBox.information(
                self,
                "Not saved",
                "The trim was applied in this open project, but hasn't been "
                "saved yet - use Save Project As to keep it.",
            )

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
            self.project.external_simulate_path = str(path)
            self.project.load_NN()
            self.logger.info(f"Loaded custom simulate function from {path}")
            QtWidgets.QMessageBox.information(
                self,
                "Custom simulate function loaded",
                f"Custom simulate function loaded from {path}. You can run this simulation using the Experiment -> Run external simulation function menu option.",
            )
        except Exception as e:
            self.project.external_simulate_path = None
            self.project.external_simulator = None
            self.project.external_simulate_function = None
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
            ["Igor2 - single array (.ibw / .csv)", "Igor2 - all arrays (.pxp)"],
            0,
            False,
        )
        if not ok:
            return
        if software.startswith("Igor2 - single"):
            self.btn_export_igor2_single_wave_clicked()
        elif software.startswith("Igor2 - all"):
            self.btn_export_igor2_pxp_stim_clicked()

    def btn_export_light_stimulus_clicked(self) -> None:
        """Pre-dialog: choose software, then open the corresponding export flow."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Export light stimulus",
            "Export for:",
            [
                "SysCon2",
                "Igor2 - all arrays (.pxp)",
                "Igor2 - single neuron (.ibw / .csv)",
            ],
            0,
            False,
        )
        if not ok:
            return
        if software == "SysCon2":
            self.open_stimulation_generator_dialog()
        elif software.startswith("Igor2 - all"):
            self.btn_export_igor2_pxp_stim_clicked()
        elif software.startswith("Igor2 - single"):
            self.btn_export_igor2_single_wave_clicked()

    def btn_export_traces_clicked(self) -> None:
        """Export the currently displayed trace data as a publication-quality figure.

        Exports all neuron traces for the selected TraceData as a single matplotlib
        figure. The Y-axis label uses the units stored in the TraceData (e.g. 'mV'
        for current-clamp recordings, 'pA' for voltage-clamp recordings). DPI is 300.
        """
        Gui._export_traces(self, vector_only=False)

    def btn_export_traces_vector_clicked(self) -> None:
        """Export the selected traces to an SVG or PDF vector graphic."""
        Gui._export_traces(self, vector_only=True)

    def _export_traces(self, *, vector_only: bool) -> None:
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

        default_suffix = "svg" if vector_only else "png"
        file_filter = (
            "SVG vector (*.svg);;PDF document (*.pdf)"
            if vector_only
            else "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf)"
        )
        filename, _ = getsavefilename(
            self,
            "Export traces as vector graphic" if vector_only else "Export traces",
            f"{trace_name.replace(' :: ', '_')}_traces.{default_suffix}",
            file_filter,
        )
        if not filename:
            return

        fig = None
        try:
            n_neurons, n_timepoints = trace_data.traces.shape
            times = np.arange(n_timepoints) * trace_data.dt
            fig, ax = plt.subplots(figsize=(12, max(4, n_neurons * 0.5 + 2)))
            for i in range(n_neurons):
                (line,) = ax.plot(
                    times, trace_data.traces[i], linewidth=0.8, alpha=0.85
                )
                line.set_gid(f"trace_{int(trace_data.indices[i])}")
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel(trace_data.units or "Amplitude", fontsize=12)
            ax.set_title(trace_name, fontsize=13)
            ax.tick_params(labelsize=10)
            fig.tight_layout()
            fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                transparent=vector_only,
            )
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

    def apply_theme(self, theme: str) -> None:
        """Apply a light/dark theme to the whole application and all plots."""
        theme = theme_module.normalize_theme(theme)
        self.logger.info(f"Applying {theme} theme")
        theme_module.apply_app_theme(QtWidgets.QApplication.instance(), theme)
        icons.apply_theme(theme)
        for widget in self.plot_widgets:
            widget.apply_theme(theme)
        self.custom_analysis_widget.apply_theme(theme)
        self.stim_editor.apply_input_designer_theme(theme)
        action = getattr(self, "_theme_actions", {}).get(theme)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self.settings.theme = theme

    def open_llm_settings(self) -> None:
        if self._llm_settings_dialog is None:
            self._llm_settings_dialog = LLMSettingsDialog(self)
            self._llm_settings_dialog.settings_changed.connect(
                self.chat_widget.reload_backend
            )
        self._llm_settings_dialog.show()
        self._llm_settings_dialog.raise_()
        self._llm_settings_dialog.activateWindow()

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About neuroseq",
            f"NeuroSeq\n\nVersion {__version__}\n\n"
            + "https://github.com/ssciwr/neuroseq",
        )

    def _create_progressbar(self) -> None:
        self.progress_dialog = ToolbarProgressWidget(self.toolbar_view)
        self.toolbar_view.addSeparator()
        action = self.toolbar_view.addWidget(self.progress_dialog)
        assert action is not None
        self.progress_dialog.set_toolbar_action(action)

    def _create_menus_and_toolbars(self) -> None:
        menu = self.menuBar()
        assert menu is not None
        menu.setStyleSheet("QMenuBar::item { padding: 3px 6px; margin: 0px; }")
        self.menu_file = menu.addMenu("&File")
        assert self.menu_file is not None
        self.toolbar_file = QtWidgets.QToolBar("File")
        self._configure_toolbar_icon_style(self.toolbar_file)
        self.addToolBar(self.toolbar_file)
        self.menu_simulate = menu.addMenu("&Simulate")
        assert self.menu_simulate is not None
        self.toolbar_simulate = QtWidgets.QToolBar("Simulate")
        self._configure_toolbar_icon_style(self.toolbar_simulate)
        self.addToolBar(self.toolbar_simulate)

        self.menu_playback = menu.addMenu("&Playback")
        assert self.menu_playback is not None
        self.toolbar_playback = QtWidgets.QToolBar("Playback")
        self._configure_toolbar_icon_style(self.toolbar_playback)
        self.addToolBar(self.toolbar_playback)
        self.menu_view = menu.addMenu("&View")
        assert self.menu_view is not None
        self.toolbar_view = QtWidgets.QToolBar("View")
        self.toolbar_data = QtWidgets.QToolBar("Data")
        self._configure_toolbar_icon_style(self.toolbar_view)
        self._configure_toolbar_icon_style(self.toolbar_data)
        self.addToolBar(self.toolbar_data)

        self.menu_experiment = menu.addMenu("&Experiment")
        assert self.menu_experiment is not None
        self.toolbar_experiment = QtWidgets.QToolBar("Experiment")
        self._configure_toolbar_icon_style(self.toolbar_experiment)
        self.addToolBar(self.toolbar_experiment)
        self.addToolBarBreak()
        self.addToolBar(self.toolbar_view)

        self.menu_settings = menu.addMenu("Se&ttings")
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
        self._create_actions_replay_analysis()

    def _create_actions_file(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&New",
            self.btn_new_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+N",
            icon="mdi6.file",
        )
        # add "open example project" item to the file menu which opens a sub-menu containing the example projects from src/neuroseq/example_projects
        open_example_project_menu = menu.addMenu("Open example &project")
        assert open_example_project_menu is not None
        for section, section_projects in example_projects.items():
            section_menu = open_example_project_menu.addMenu(section)
            assert section_menu is not None
            for name, generator in section_projects.items():
                add_action(
                    name,
                    lambda checked=False, _generator=generator: (
                        self._load_example_project(_generator)
                    ),
                    section_menu,
                    None,
                )
        add_action(
            "&Open Project",
            self.btn_open_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+O",
            icon="mdi6.folder-open",
        )
        add_action(
            "Import &from Project",
            self.btn_import_clicked,
            menu,
            toolbar,
            shortcut="None",
            icon="mdi6.file-import",
        )
        add_action(
            "Cell &segmentation",
            self.btn_import_cellpose_data_clicked,
            menu,
            None,
            icon="mdi6.image-edit-outline",
        )
        add_action(
            "&Save Project",
            self.btn_save_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+S",
            icon="mdi6.content-save",
        )
        add_action(
            "Save Project &As...",
            self.btn_save_as_clicked,
            menu,
            None,
            icon="mdi6.content-save-edit",
        )

        import_data_menu = menu.addMenu("&Import data")
        assert import_data_menu is not None

        import_stimulus_menu = import_data_menu.addMenu("&Stimulus")
        assert import_stimulus_menu is not None
        add_action(
            "&Optogenetics",
            self.btn_import_stimulus_light_clicked,
            import_stimulus_menu,
            None,
            icon="mdi6.file-import-outline",
        )
        add_action(
            "&Patch Clamp",
            self.btn_import_stimulus_patchclamp_clicked,
            import_stimulus_menu,
            None,
            icon="mdi6.waveform",
        )

        import_trace_menu = import_data_menu.addMenu("&Trace")
        assert import_trace_menu is not None
        add_action(
            "&Calcium Imaging (raw .tif)",
            self.btn_import_tif_clicked,
            import_trace_menu,
            None,
            icon="mdi6.import",
        )
        add_action(
            "Calcium Imaging (&Suite2p outputs)",
            self.btn_import_suite2p_data_clicked,
            import_trace_menu,
            None,
            icon="mdi6.import",
        )
        add_action(
            "&Patch Clamp",
            self.btn_import_trace_patchclamp_clicked,
            import_trace_menu,
            None,
            icon="mdi6.import",
        )

        export_data_menu = menu.addMenu("E&xport")
        assert export_data_menu is not None

        add_action(
            "&Patch Clamp sStimulus",
            self.btn_export_patch_clamp_stim_clicked,
            export_data_menu,
            None,
            icon="mdi6.waveform",
        )
        add_action(
            "&Optogenetic Stimulus",
            self.btn_export_light_stimulus_clicked,
            export_data_menu,
            None,
            icon="mdi6.file-code",
        )

        export_graphics_menu = menu.addMenu("Export &graphics")
        assert export_graphics_menu is not None

        add_action(
            "&Traces",
            self.btn_export_traces_clicked,
            export_graphics_menu,
            None,
            icon="mdi.waves",
        )
        add_action(
            "&Spikes",
            self.btn_export_spikes_clicked,
            export_graphics_menu,
            None,
            icon="ph.barcode-light",
        )
        add_action(
            "Export &simulation results",
            self.btn_export_simulation_results_clicked,
            export_data_menu,
            None,
            shortcut="Ctrl+Shift+E",
            icon="mdi6.content-save-move",
        )

        add_action(
            "&Edit Model",
            self.btn_edit_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+E",
            icon="ph.polygon-light",
        )
        menu.addSeparator()
        add_action(
            "Generate Synthetic &Data...",
            self.btn_generate_synthetic_data_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+G",
            icon="mdi6.microsoft-xbox-controller-view"
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
            icon="mdi6.exit-to-app",
        )

    def _create_actions_simulate(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        self.continue_simulation_action = add_action(
            "&Run Simulation",
            self.btn_continue_simulation_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+R",
            icon="mdi6.rocket-launch-outline",
        )
        add_action(
            "&Store Simulation Checkpoint",
            self.btn_store_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+S",
            icon="mdi6.timeline-check-outline",
        )
        add_action(
            "&Restore Simulation Checkpoint",
            self.btn_restore_checkpoint_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+Shift+R",
            icon="mdi.ray-start-arrow",
        )

    def _create_actions_playback(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        add_action(
            "&Set Playback Speed",
            self.btn_playback_speed_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.speedometer",
        )
        add_action(
            "&Skip to Start",
            self.time_slider.skip_to_start,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.skip-previous",
        )
        add_action(
            "&Stop",
            self.time_slider.stop,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.stop",
        )
        add_action(
            "&Play / Pause",
            self.time_slider.toggle_pause,
            menu,
            toolbar,
            shortcut="Space",
            icon="mdi6.play-pause",
        )
        add_action(
            "&Skip to End",
            self.time_slider.skip_to_end,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.skip-next",
        )

    def _create_actions_view(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar
    ) -> None:
        self._view_actions: dict[QtWidgets.QWidget, QtWidgets.QAction] = {}
        for name, widget, icon, is_checked in [
            ("&Stimuli", self.stim_editor, "mdi6.chart-timeline", True),
            ("&Traces", self.trace_widget, "mdi.waves", False),
            ("&Spikes", self.spike_widget, "ph.barcode-light", False),
            ("&Data", self.data_widget, "mdi6.chart-donut", False),
            ("&Heatmap", self.heatmap_widget, "mdi6.data-matrix", False),
            (
                "&Spatial Footprints",
                self.spatial_footprint_widget,
                "mdi6.map-marker-radius",
                True,
            ),
            (
                "&Custom Analysis",
                self.custom_analysis_widget,
                "mdi6.chart-histogram",
                False,
            ),
            ("&Console", self.console_widget, "mdi6.console", True),
            ("&Chat", self.chat_widget, "mdi6.robot-happy-outline", False),
        ]:
            w: QtWidgets.QWidget = widget  # type: ignore[assignment]
            self._view_actions[w] = add_action(
                f"Show {name}",
                lambda toggled, w=w: self._set_widget_visible(w, toggled),
                menu,
                toolbar,
                icon=icon,
                checkable=True,
                is_checked=is_checked,
            )
        add_action(
            "Toggle Selection Text",
            self.toggle_selection_text,
            menu,
            toolbar,
            icon="mdi6.format-text",
            checkable=True,
            is_checked=True,
        )
        add_action(
            "Toggle Grid",
            self.toggle_grid,
            menu,
            toolbar,
            icon="mdi6.grid",
            checkable=True,
            is_checked=True,
        )
        add_action(
            "Keep Short Stimuli Visible (4 px)",
            self.toggle_minimum_stimulus_size,
            menu,
            toolbar,
            icon="mdi6.square-medium",
            checkable=True,
            is_checked=True,
        )

    def _set_widget_visible(self, widget: QtWidgets.QWidget, toggled: bool) -> None:
        widget.setVisible(toggled)
        if toggled and widget is self.data_widget:
            self.set_data(self.cmb_data.currentText())

    def _auto_show_widget(self, widget: QtWidgets.QWidget) -> None:
        action = getattr(self, "_view_actions", {}).get(widget)
        if action is not None and not action.isChecked():
            action.trigger()

    def _install_detach_context_menus(self) -> None:
        for source, targets in {
            self.data_widget: (
                self.data_widget,
                self.data_widget.stacked_widget,
                self.data_widget.heatmap_widget,
                self.data_widget.heatmap_widget.widget,
            ),
            self.heatmap_widget: (self.heatmap_widget, self.heatmap_widget.widget),
            self.spatial_footprint_widget: (
                self.spatial_footprint_widget,
                self.spatial_footprint_widget.empty_state_widget,
                self.spatial_footprint_widget.overlay_widget,
                self.spatial_footprint_widget.overlay_widget.overlay_view,
                self.spatial_footprint_widget.overlay_widget.overlay_view.widget,
            ),
            self.custom_analysis_widget: (
                self.custom_analysis_widget,
                self.custom_analysis_widget.canvas,
            ),
            self.console_widget: (self.console_widget,),
            self.chat_widget: (
                self.chat_widget,
                self.chat_widget.history_view,
            ),
        }.items():
            for target in targets:
                target.setContextMenuPolicy(
                    QtCore.Qt.ContextMenuPolicy.CustomContextMenu
                )
                target.customContextMenuRequested.connect(
                    lambda pos, s=source, t=target: self._show_detach_context_menu(
                        s,
                        t.mapToGlobal(pos),
                    )
                )

    def _show_detach_context_menu(
        self,
        widget: QtWidgets.QWidget,
        global_position: QtCore.QPoint,
    ) -> None:
        menu = QtWidgets.QMenu(widget)
        detach_action = menu.addAction("Detach panel")
        assert detach_action is not None
        detach_action.triggered.connect(
            lambda: self.detach_panel(widget, keep_original=False)
        )
        copy_action = menu.addAction("Detach copy of panel")
        assert copy_action is not None
        copy_action.triggered.connect(
            lambda: self.detach_panel(widget, keep_original=True)
        )
        menu.exec(global_position)

    def _panel_title(self, widget: QtWidgets.QWidget) -> str:
        source = self._detachable_panel_source(widget)
        action = self._view_actions.get(source)
        if action is None:
            return "Panel"
        return action.text().split(" ", 1)[-1].replace("&", "")

    def _detachable_panel_source(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        if widget in (
            self.data_widget.trace_widget,
            self.data_widget.spike_widget,
            self.data_widget.heatmap_widget,
        ):
            return self.data_widget
        return widget

    def _current_data_selection(self) -> TraceData | SpikeData | SynapseData | None:
        cmb_name = self.cmb_data.currentText()
        if cmb_name.endswith(" [trace]"):
            return self.project.get_data(cmb_name[:-8], TraceData)
        if cmb_name.endswith(" [spike]"):
            return self.project.get_data(cmb_name[:-8], SpikeData)
        if cmb_name.endswith(" [synapse]"):
            return self.project.get_data(cmb_name[:-10], SynapseData)
        return None

    def _build_detached_widget(
        self,
        source: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget | None:
        source = self._detachable_panel_source(source)
        if source is self.trace_widget:
            return self._build_fresh_trace_widget()
        if source is self.spike_widget:
            return self._build_fresh_spike_widget()
        if source is self.data_widget:
            return self._build_fresh_data_widget()
        if source is self.heatmap_widget:
            return self._build_fresh_heatmap_widget()
        if source is self.spatial_footprint_widget:
            return self._build_fresh_spatial_footprint_widget()
        if source is self.custom_analysis_widget:
            return self._build_fresh_custom_analysis_widget()
        if source is self.console_widget:
            return self._build_fresh_console_widget()
        if source is self.chat_widget:
            return self._build_fresh_chat_widget()
        return None

    def detach_panel(self, source: QtWidgets.QWidget, *, keep_original: bool) -> None:
        source = self._detachable_panel_source(source)
        widget = self._build_detached_widget(source)
        if widget is None:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing to detach",
                "This panel has nothing to show yet.",
            )
            return

        window: DetachedPanelWindow | None = None

        def _forget_window() -> None:
            if window in self._detached_panel_windows:
                self._detached_panel_windows.remove(window)

        window = DetachedPanelWindow(
            widget,
            self._popup_title(self._panel_title(source)),
            on_close=_forget_window,
            parent=self,
        )
        # A widget's own embedded "close" control (e.g. ConsoleWidget's
        # close_requested) should close the detached window it's actually
        # in, not reach back and toggle the original docked panel.
        close_requested = getattr(widget, "close_requested", None)
        if close_requested is not None:
            close_requested.connect(window.close)
        self._detached_panel_windows.append(window)
        window.show()
        if not keep_original:
            action = self._view_actions.get(source)
            if action is not None and action.isChecked():
                action.trigger()

    def detach_panel_copy(self, source: QtWidgets.QWidget) -> None:
        self.detach_panel(source, keep_original=True)

    def _build_fresh_trace_widget(self) -> TraceWidget:
        widget = TraceWidget(
            ConstrainedPanZoomController(),
            parent=None,
            default_controller=self.controller,
            show_axis_link_controls=True,
            state=self._trace_state,
        )
        widget.custom_analysis_requested.connect(self._custom_analysis)
        widget.vector_export_requested.connect(self.btn_export_traces_vector_clicked)
        return widget

    def _build_fresh_spike_widget(self) -> SpikeWidget:
        widget = SpikeWidget(
            ConstrainedPanZoomController(),
            parent=None,
            default_controller=self.controller,
            show_axis_link_controls=True,
        )
        widget.set_data(self.project.get_data(self.cmb_spike.currentText(), SpikeData))
        widget.custom_analysis_requested.connect(self._custom_analysis)
        return widget

    def _build_fresh_data_widget(self) -> DataWidget:
        widget = DataWidget(ConstrainedPanZoomController(), parent=None)
        widget.set_data(self._current_data_selection())
        widget.trace_widget.custom_analysis_requested.connect(self._custom_analysis)
        widget.spike_widget.custom_analysis_requested.connect(self._custom_analysis)
        return widget

    def _build_fresh_heatmap_widget(self) -> HeatmapWidget:
        widget = HeatmapWidget(parent=None)
        widget.set_data(
            self.project.get_data(self.cmb_synapse.currentText(), SynapseData)
        )
        return widget

    def _build_fresh_spatial_footprint_widget(self) -> SpatialFootprintWidget:
        widget = SpatialFootprintWidget(
            parent=None, state=self._spatial_footprint_state
        )
        widget.import_requested.connect(self.btn_import_cellpose_data_clicked)
        widget.trim_to_mapped_neurons_requested.connect(
            self.btn_trim_to_mapped_neurons_clicked
        )
        return widget

    def _build_fresh_custom_analysis_widget(self) -> CustomAnalysisWidget | None:
        if self._last_custom_analysis is None:
            return None
        widget = CustomAnalysisWidget(parent=None)
        data, analysis_type, neuron_slice, time_slice = self._last_custom_analysis
        widget.set_data(data, analysis_type, neuron_slice, time_slice)
        return widget

    def _build_fresh_console_widget(self) -> ConsoleWidget:
        widget = ConsoleWidget(parent=None)
        widget.set_project(self.project)
        widget.output.setPlainText(self.console_widget.output.toPlainText())
        widget.input.history = list(self.console_widget.input.history)
        widget.console_command_entered.connect(self._update_data)
        # close_requested is wired generically in detach_panel() to close
        # this widget's own detached window, not the original docked panel.
        return widget

    def _build_fresh_chat_widget(self) -> ChatWidget:
        widget = ChatWidget(parent=None)
        widget.set_project(self.project)
        widget._history = list(self.chat_widget._history)
        widget._render_history()
        widget.request_settings_dialog.connect(self.open_llm_settings)
        return widget

    def _create_actions_settings(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        add_action("&Brian2 Simulation Backend", self.choose_brian2_backend, menu)
        add_action("&Stimulus Automatic Padding", self.choose_stim_padding_secs, menu)
        add_action("&LLM Assistant", self.open_llm_settings, menu)
        add_action(
            "&Synapse Matrix RAM Limit", self.choose_max_ram_synapse_plot_matrix, menu
        )
        add_action("&Trace Plot RAM Limit", self.choose_max_ram_trace_plot_matrix, menu)
        add_action("&Change Fiji Path…", self.choose_fiji_path, menu)
        self._create_theme_menu(menu)

    def _create_theme_menu(self, menu: QtWidgets.QMenu) -> None:
        theme_menu = menu.addMenu("&Theme")
        assert theme_menu is not None
        self._theme_action_group = QtGui.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        current = theme_module.normalize_theme(self.settings.theme)
        self._theme_actions: dict[str, QtWidgets.QAction] = {}
        for label, name in (
            ("&Light", theme_module.LIGHT),
            ("&Dark", theme_module.DARK),
        ):
            action = add_action(
                label,
                lambda _checked=False, theme=name: self.apply_theme(theme),
                theme_menu,
                checkable=True,
                is_checked=(name == current),
            )
            self._theme_action_group.addAction(action)
            self._theme_actions[name] = action

    def _create_actions_help(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        add_action("&About", self.about, menu)

    def _create_actions_data(
        self, menu: QtWidgets.QMenu | None, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        if toolbar is None:
            return
        self.btn_open_fiji = QtWidgets.QPushButton("Open Fiji")
        self.btn_open_fiji.setToolTip("Launch Fiji/ImageJ")
        icons.set_widget_icon(self.btn_open_fiji, "mdi6.microscope")
        self.btn_open_fiji.clicked.connect(self.btn_open_fiji_clicked)
        toolbar.addWidget(self.btn_open_fiji)
        self.btn_open_cellpose = QtWidgets.QPushButton("Open Cellpose")
        self.btn_open_cellpose.setToolTip("Launch the Cellpose GUI")
        icons.set_widget_icon(self.btn_open_cellpose, "mdi6.dots-hexagon")
        self.btn_open_cellpose.clicked.connect(self.btn_open_cellpose_clicked)
        toolbar.addWidget(self.btn_open_cellpose)
        toolbar.addSeparator()
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
            icon="mdi6.arrow-expand-vertical",
        )
        self.action_overlay_stim_on_traces = add_action(
            "Overlay stim on traces",
            self._toggle_stim_trace_overlay,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.layers-outline",
            checkable=True,
            is_checked=False,
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
            icon="mdi6.arrow-expand-vertical",
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
            icon="mdi6.beaker",
        )
        menu.addSeparator()
        add_action(
            "&Connect External Stimulation Function",
            self.connect_external_simulation_function,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.code-braces",
        )
        add_action(
            "&Connect Brian2 External Stimulation Function",
            self.connect_external_simulation_function_with_brian2,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.function",
        )
        self._do_external_simulation_action = add_action(
            "&Run External Stimulation Function",
            self.btn_do_external_simulation_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.rocket-launch-outline",
        )

    def _create_actions_replay_analysis(self) -> None:
        """Left-toolbar one-click ELiSe target MSE."""
        self.elise_replay_action = add_action(
            "Analyse ELiSe Replay (target MSE)",
            self._analyse_elise_replay,
            menu=None,
            toolbar=self.toolbar_stim_editor,
            shortcut=None,
            icon="mdi6.delta",
        )
        self.elise_replay_action.setToolTip(
            "Compute the paper-style replay MSE and correlation against "
            "the binary teaching pattern, on the output-neuron firing "
            "rate. Requires an ELiSe project with a completed simulation."
        )
        self.elise_protocol_action = add_action(
            "Run ELiSe Protocol (learning curves)",
            self._run_elise_protocol,
            menu=None,
            toolbar=self.toolbar_stim_editor,
            shortcut=None,
            icon="mdi6.school-outline",
        )
        self.elise_protocol_action.setToolTip(
            "Train the ELiSe network on the teaching pattern and measure the "
            "paper's train + replay learning curves (Kriener et al. 2024, "
            "Fig. 2b/c). Requires an ELiSe project with 'opsin_exc' and "
            "'opsin_inh' stims."
        )
        self._refresh_elise_replay_action()

    def _refresh_elise_replay_action(self) -> None:
        """Enable the ELiSe toolbar actions when the project supports them."""
        replay_action = getattr(self, "elise_replay_action", None)
        if replay_action is not None:
            replay_action.setEnabled(self._elise_replay_inputs() is not None)
        protocol_action = getattr(self, "elise_protocol_action", None)
        if protocol_action is not None:
            protocol_action.setEnabled(
                self._elise_protocol_inputs() is not None
                and self._elise_protocol_worker is None
            )

    def _elise_replay_inputs(
        self,
    ) -> tuple[TraceData, np.ndarray, float] | None:
        """Return (firing_rate_r, opsin_exc_values, opsin_dt_s) if available."""
        if self.project.external_simulate_path != "neuroseq:elise":
            return None
        trace = None
        for experiment in self.project.experiments.values():
            for datum in experiment.data:
                if isinstance(datum, TraceData) and datum.name == "firing_rate_r":
                    trace = datum
        if trace is None or np.asarray(trace.traces).size == 0:
            return None
        stim = self.project.stim_arrays.get("opsin_exc")
        if stim is None:
            return None
        return trace, np.asarray(stim.values), float(stim.dt)

    def _analyse_elise_replay(self) -> None:
        import neuroseq.resources.analysis_plugins.elise_target_mse as plugin_mod
        from neuroseq.example_models.elise import build_target_rate_pattern
        from neuroseq.replay_metrics import compute_performance_against_target
        from neuroseq.replay_metrics import save_replay_metrics_csv

        inputs = self._elise_replay_inputs()
        if inputs is None:
            QtWidgets.QMessageBox.information(
                self,
                "ELiSe replay analysis",
                "This action requires an ELiSe project with a completed "
                "simulation (needs the 'firing_rate_r' trace and an "
                "'opsin_exc' stim).",
            )
            return
        trace, opsin_exc, stim_dt_s = inputs
        params_dict = self.project.external_simulate_params or {}

        def _param(key: str, default: float) -> float:
            entry = params_dict.get(key)
            if isinstance(entry, dict) and "value" in entry:
                try:
                    return float(entry["value"])
                except (TypeError, ValueError):
                    return default
            return default

        target_rate, cycle_stim_samples = build_target_rate_pattern(
            opsin_exc,
            E_l=_param("E_l", -70.0),
            a=_param("a", 0.3),
            b=_param("b", -58.0),
        )

        trace_dt_s = float(trace.dt)
        if trace_dt_s <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                "Firing-rate trace has non-positive dt; cannot align target.",
            )
            return
        upsample = int(round(stim_dt_s / trace_dt_s))
        if upsample < 1 or abs(stim_dt_s - upsample * trace_dt_s) > 1e-9:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                f"Stim dt ({stim_dt_s * 1000:.4g} ms) is not an integer "
                f"multiple of trace dt ({trace_dt_s * 1000:.4g} ms); "
                "cannot align target.",
            )
            return
        target_rate_resampled = np.repeat(target_rate, upsample, axis=1)
        num_vis = target_rate_resampled.shape[0]

        n_total_neurons = int(np.asarray(trace.traces).shape[0])
        if num_vis > n_total_neurons:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                f"Opsin stim has {num_vis} pitches but the trace only "
                f"covers {n_total_neurons} neurons.",
            )
            return

        # Ask the user where to save the per-cycle CSV before running.
        default_stem = "elise_replay_mse"
        if self.project.filepath is not None:
            default_stem = pathlib.Path(self.project.filepath).stem + "_replay_mse"
        csv_filename, _ = getsavefilename(
            self,
            "Save replay MSE analysis as CSV",
            default_stem + ".csv",
            "CSV files (*.csv)",
        )
        if not csv_filename:
            return
        csv_path = pathlib.Path(csv_filename)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")

        output_rates = np.asarray(trace.traces)[:num_vis, :].astype(np.float64)
        try:
            mses, corrs, shifts = compute_performance_against_target(
                output_rates, target_rate_resampled
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                f"Cannot compute replay metrics: {exc}",
            )
            return

        try:
            save_replay_metrics_csv(
                csv_path,
                mses,
                corrs,
                shifts,
                cycle_dt_ms=trace_dt_s * 1000.0,
                cycle_length_samples=int(target_rate_resampled.shape[1]),
                n_output_neurons=num_vis,
                source=str(self.project.filepath) if self.project.filepath else None,
            )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                f"Could not write CSV to {csv_path}: {exc}",
            )
            return

        analysis_window = CustomAnalysisWindow(parent=self)
        analysis_widget = analysis_window.analysis_widget
        setattr(analysis_widget, plugin_mod.TARGET_ATTR, target_rate_resampled)
        setattr(analysis_widget, plugin_mod.TARGET_DT_ATTR, trace_dt_s)

        plugin = next(
            (p for p in discover_analysis_plugins() if p.id == plugin_mod.PLUGIN_ID),
            None,
        )
        if plugin is None:
            QtWidgets.QMessageBox.warning(
                self,
                "ELiSe replay analysis",
                "Could not find the ELiSe target MSE analysis plugin.",
            )
            return

        n_samples = int(np.asarray(trace.traces).shape[1])
        analysis_widget.set_data(
            trace,
            plugin,
            slice(0, num_vis),
            slice(0, n_samples),
        )
        analysis_window.setWindowTitle(
            self._popup_title(f"ELiSe replay MSE — saved to {csv_path.name}")
        )
        analysis_window.show()

    def _elise_protocol_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Return (opsin_exc, opsin_inh, stim_dt_s) if this is an ELiSe project."""
        if self.project.external_simulate_path != "neuroseq:elise":
            return None
        exc = self.project.stim_arrays.get("opsin_exc")
        inh = self.project.stim_arrays.get("opsin_inh")
        if exc is None or inh is None:
            return None
        exc_values = np.asarray(exc.values)
        inh_values = np.asarray(inh.values)
        if exc_values.size == 0 or exc_values.shape != inh_values.shape:
            return None
        return exc_values, inh_values, float(exc.dt)

    def _run_elise_protocol(self) -> None:
        from neuroseq.elise_protocol_dialog import EliseProtocolDialog
        from neuroseq.example_models.elise import run_protocol as elise_run_protocol

        inputs = self._elise_protocol_inputs()
        if inputs is None:
            QtWidgets.QMessageBox.information(
                self,
                "Run ELiSe protocol",
                "This action requires an ELiSe project with 'opsin_exc' and "
                "'opsin_inh' stim arrays.",
            )
            return
        if self._elise_protocol_worker is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Run ELiSe protocol",
                "An ELiSe protocol run is already in progress.",
            )
            return

        dialog = EliseProtocolDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        opsin_exc, opsin_inh, stim_dt_s = inputs
        kwargs = {
            "opsin_exc": opsin_exc.astype(float),
            "opsin_inh": opsin_inh.astype(float),
            "stim_dt_s": stim_dt_s,
            "base_params": self.project.external_simulate_params_flat(),
            **dialog.get_values(),
        }

        self.progress_dialog.setLabelText("Running ELiSe protocol...")
        self.progress_dialog.setValue(0)
        self.progress_dialog.setCancelButtonText("Cancel")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.progress_dialog.show()

        self._elise_protocol_cancel = threading.Event()
        self.progress_dialog.canceled.connect(self._elise_protocol_cancel.set)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        if getattr(self, "elise_protocol_action", None) is not None:
            self.elise_protocol_action.setEnabled(False)

        thread = QtCore.QThread(self)
        worker = EliseProtocolWorker(
            run_fn=elise_run_protocol,
            kwargs=kwargs,
            cancel_event=self._elise_protocol_cancel,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_elise_protocol_progress)
        worker.finished.connect(self._on_elise_protocol_finished)
        worker.cancelled.connect(self._on_elise_protocol_cancelled)
        worker.failed.connect(self._on_elise_protocol_failed)
        for signal in (worker.finished, worker.cancelled, worker.failed):
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._elise_protocol_thread = thread
        self._elise_protocol_worker = worker
        thread.start()

    def _teardown_elise_protocol(self) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        with contextlib.suppress(TypeError, RuntimeError):
            self.progress_dialog.canceled.disconnect()
        self._elise_protocol_worker = None
        self._elise_protocol_thread = None
        self._elise_protocol_cancel = None
        self._refresh_elise_replay_action()

    @QtCore.Slot(float)
    def _on_elise_protocol_progress(self, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.progress_dialog.setValue(int(round(clamped * 100)))

    def _update_elise_stim_arrays(self, result: dict) -> None:
        """Replace opsin_exc/opsin_inh with the protocol's full-timeline stims.

        The teaching pattern is present during training cycles and blank during
        teacher-off validation/replay, so the stim array reflects the true
        timeline of what was simulated and lines up with the imported traces.
        """
        from neuroseq.stim import StimArray
        from neuroseq.stim import StimType

        exc_full = result.get("stim_exc_full")
        inh_full = result.get("stim_inh_full")
        if exc_full is None or inh_full is None:
            return
        dt = float(result.get("stim_dt_s") or 0.0)
        if dt <= 0:
            return
        defaults = {
            "opsin_exc": StimType.Excitation,
            "opsin_inh": StimType.Inhibition,
        }
        for name, values in (("opsin_exc", exc_full), ("opsin_inh", inh_full)):
            existing = self.project.stim_arrays.get(name)
            stim_type = existing.stim_type if existing is not None else defaults[name]
            units = existing.units if existing is not None else "mm**-2 / second"
            self.project.add_stim_array(
                StimArray(
                    name=name,
                    stim_type=stim_type,
                    values=np.asarray(values),
                    units=units,
                    dt=dt,
                )
            )
        self.stim_editor.set_project(self.project)

    @QtCore.Slot(object)
    def _on_elise_protocol_finished(self, result: dict) -> None:
        import neuroseq.resources.analysis_plugins.elise_learning_curves as plugin_mod
        from neuroseq.replay_metrics import save_learning_curves_csv

        self._teardown_elise_protocol()
        self.progress_dialog.setValue(100)
        self.progress_dialog.reset()

        # Rebuild the opsin stim arrays to the full protocol timeline (teaching
        # pattern during training, blank during teacher-off validation/replay)
        # so the stim array shows the true timeline of what was simulated, then
        # import the aligned activity Experiment.
        self._update_elise_stim_arrays(result)
        experiment = result.get("experiment")
        if experiment is not None:
            self._append_experimental_data(experiment)

        default_stem = "elise_learning_curves"
        if self.project.filepath is not None:
            default_stem = pathlib.Path(self.project.filepath).stem + "_learning_curves"
        csv_filename, _ = getsavefilename(
            self,
            "Save ELiSe learning curves as CSV",
            default_stem + ".csv",
            "CSV files (*.csv)",
        )
        csv_path: pathlib.Path | None = None
        if csv_filename:
            csv_path = pathlib.Path(csv_filename)
            if csv_path.suffix.lower() != ".csv":
                csv_path = csv_path.with_suffix(".csv")
            try:
                save_learning_curves_csv(
                    csv_path,
                    result,
                    source=(
                        str(self.project.filepath) if self.project.filepath else None
                    ),
                )
            except OSError as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Run ELiSe protocol",
                    f"Could not write CSV to {csv_path}: {exc}",
                )
                csv_path = None

        num_vis = int(result["num_vis"])
        final_rate = result.get("final_replay_rate")
        analysis_window = CustomAnalysisWindow(parent=self)
        analysis_widget = analysis_window.analysis_widget
        setattr(analysis_widget, plugin_mod.CURVES_ATTR, result)
        plugin = next(
            (p for p in discover_analysis_plugins() if p.id == plugin_mod.PLUGIN_ID),
            None,
        )
        if plugin is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Run ELiSe protocol",
                "Could not find the ELiSe learning-curves analysis plugin.",
            )
            return
        # The plugin reads the curves off the widget; set_data still needs a
        # data object, so hand it the final replay trace.
        final_arr = np.asarray(final_rate, dtype=np.float64)
        data = TraceData(
            name="firing_rate_r (final replay)",
            traces=final_arr,
            indices=np.arange(final_arr.shape[0], dtype=np.uint32),
            units="1",
            dt=float(result["trace_dt_s"]),
        )
        analysis_widget.set_data(
            data, plugin, slice(0, num_vis), slice(0, final_arr.shape[1])
        )
        title = "ELiSe learning curves"
        if csv_path is not None:
            title += f" — saved to {csv_path.name}"
        analysis_window.setWindowTitle(self._popup_title(title))
        analysis_window.show()

    @QtCore.Slot()
    def _on_elise_protocol_cancelled(self) -> None:
        self._teardown_elise_protocol()
        self.progress_dialog.reset()
        QtWidgets.QMessageBox.information(
            self,
            "Run ELiSe protocol",
            "ELiSe protocol cancelled - no learning curves were produced.",
        )

    @QtCore.Slot(str)
    def _on_elise_protocol_failed(self, message: str) -> None:
        self._teardown_elise_protocol()
        self.progress_dialog.reset()
        self.logger.error(f"ELiSe protocol failed: {message}")
        QtWidgets.QMessageBox.critical(
            self,
            "Run ELiSe protocol",
            f"Could not run the ELiSe protocol: {message}",
        )

    def _stop_worker_thread(
        self, thread: QtCore.QThread | None, timeout_ms: int, label: str
    ) -> None:
        """Stop a worker thread during shutdown."""
        if thread is None or not thread.isRunning():
            return
        thread.quit()
        if not thread.wait(timeout_ms):
            self.logger.warning(
                f"{label} did not stop in {timeout_ms} ms during shutdown."
            )

    def closeEvent(self, event):
        if not self._close_confirmed and not self._prompt_save_before_close():
            event.ignore()
            return
        if (
            self._external_sim_thread is not None
            and self._external_sim_cancel is not None
        ):
            self._external_sim_cancel.set()
        self._stop_worker_thread(
            self._external_sim_thread, 2000, "External simulation worker"
        )
        self._stop_worker_thread(self._io_thread, 10000, "IO worker")
        # Stop timers and queued widget updates before tearing down owned widgets.
        self.time_slider.close()
        # explicitly close all fastplotlib figures to avoid renderer trying to access them after they have been destroyed
        for widget in self.plot_widgets:
            widget.close()
        self.spatial_footprint_widget.close()
        # Close any detached analysis windows that are parented to this GUI.
        for analysis_window in self.findChildren(CustomAnalysisWindow):
            analysis_window.close()
        for detached_window in self.findChildren(DetachedPanelWindow):
            detached_window.close()
        super().closeEvent(event)
