from __future__ import annotations

import contextlib
import gc
import inspect
import logging
import pathlib
import threading
from collections.abc import Callable
from collections.abc import Sequence

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

import neuroseq.elise_support.workflows as elise_workflows
from neuroseq import __version__
from neuroseq import theme as theme_module
from neuroseq.add_experiment_data_dialog import AddExperimentDataDialog
from neuroseq.analysis_plugins import AnalysisPlugin
from neuroseq.background_jobs import BackgroundIOWorker
from neuroseq.background_jobs import BackgroundJob
from neuroseq.background_jobs import EliseProtocolWorker
from neuroseq.background_jobs import ExternalSimWorker
from neuroseq.background_jobs import SimulationGuard
from neuroseq.background_jobs import wait_for_jobs
from neuroseq.console_widget import ConsoleWidget
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.data_manager_dialog import DataManagerDialog
from neuroseq.device_control.syscon2_panel import SysCon2Panel
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.export.stimulus_export_dialog import show_stimulus_export_dialog
from neuroseq.gui_settings import GuiSettings
from neuroseq.icon_manager import icons
from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import clone_cellpose_result
from neuroseq.importers.importer_tif import DEFAULT_FS
from neuroseq.importers.patch_clamp_data_input_import_dialog import (
    PatchClampDataInputImportDialog,
)
from neuroseq.importers.suite2p_import_dialog import Suite2pImportDialog
from neuroseq.importers.syscon2_import_dialog import Syscon2ImportDialog
from neuroseq.llm.chat_widget import ChatWidget
from neuroseq.llm.llm_settings_dialog import LLMSettingsDialog
from neuroseq.main_scroll import WorkspaceZoom
from neuroseq.main_scroll import ZoomControl
from neuroseq.main_scroll import make_main_scroll_area
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.save_load_options import SaveLoadOptionsDialog
from neuroseq.stimulation.dynamic_input.dynamic_input_controller import (
    DynamicInputController,
)
from neuroseq.stimulation.static_input.static_input_controller import (
    StaticInputController,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim_editor import StimEditor
from neuroseq.synthetic_data_dialog import SyntheticDataDialog
from neuroseq.utils import add_action
from neuroseq.utils import add_items_to_combobox
from neuroseq.utils import getSimulationTime
from neuroseq.utils import with_wait_cursor
from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisType
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.visualization.data_widget import DataWidget
from neuroseq.visualization.heatmap_widget import HeatmapWidget
from neuroseq.visualization.spatial_footprint_widget import SpatialFootprintWidget
from neuroseq.visualization.spike_widget import SpikeWidget
from neuroseq.visualization.timeseries_widget import TimeseriesWidget
from neuroseq.visualization.trace_widget import TraceWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from neuroseq.visualization_support.spatial_footprint_state import SpatialFootprintState
from neuroseq.visualization_support.time_slider_widget import TimeSliderWidget
from neuroseq.visualization_support.trace_widget_state import TraceWidgetState
from neuroseq.workspace_acquisition_controller import WorkspaceAcquisitionController
from neuroseq.workspace_devices import WorkspaceDevices


class DetachedPanelWindow(QtWidgets.QWidget):
    """Top-level window wrapping one detached panel instance."""

    def __init__(
        self,
        widget: QtWidgets.QWidget,
        title: str,
        *,
        on_close: Callable[[], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.Tool)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self._widget = widget
        self._on_close = on_close

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        layout.addWidget(scroll_area)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._widget.close()
        if self._on_close is not None:
            self._on_close()
        super().closeEvent(event)


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


# File drops (issue #597): imaging files and what can be done with them.
DROP_IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}
DROP_SEGMENT = "Segment cells with Cellpose"
DROP_TRACES = "Extract calcium traces"
DROP_VIDEO = "Add as imaging data (video)"


#: How long closing waits for cancelled background jobs to wind down.
CLOSE_JOB_TIMEOUT_MS = 10000


class WorkspaceCloseGate(QtCore.QObject):
    """Decide, without stopping anything, whether workspaces may close now.

    Closing first collects every workspace's close blockers (work that cannot
    be cancelled safely) and refuses with one message listing them all. Only
    when there are none does it cancel running jobs, wait for them to wind
    down without blocking the GUI, and retry the close.
    """

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._window = parent
        self._waiter: QtCore.QObject | None = None

    @property
    def waiting(self) -> bool:
        return self._waiter is not None

    def ready(
        self,
        workspaces: Sequence[ProjectWorkspace],
        title: str,
        retry: Callable[[], object],
    ) -> bool:
        """Return True when ``workspaces`` can be stopped and disposed now.

        Otherwise the close is refused: either blockers are reported, or
        cancellable jobs are cancelled and ``retry`` runs once they have ended.
        """
        if self._waiter is not None:
            return False
        blockers = [
            (workspace, reason)
            for workspace in workspaces
            for reason in workspace.close_blockers()
        ]
        if blockers:
            if len(workspaces) > 1:
                lines = [
                    f"- {workspace._tab_label_for(workspace.project)}: {reason}"
                    for workspace, reason in blockers
                ]
            else:
                lines = [f"- {reason}" for _workspace, reason in blockers]
            QtWidgets.QMessageBox.information(
                self._window,
                title,
                "This cannot be closed yet:\n\n"
                + "\n".join(lines)
                + "\n\nNothing was stopped.",
            )
            return False
        busy = [w for w in workspaces if w._background_job.busy]
        if not busy:
            return True
        for workspace in busy:
            workspace._cancel_background_job()

        def _idle() -> None:
            self._waiter = None
            retry()

        def _timed_out() -> None:
            self._waiter = None
            names = ", ".join(
                f"'{w._tab_label_for(w.project)}'"
                for w in busy
                if w._background_job.busy
            )
            QtWidgets.QMessageBox.warning(
                self._window,
                title,
                f"Background work in {names} is still stopping, so it stays "
                "open. Close again once the work has finished.",
            )

        self._waiter = wait_for_jobs(
            [w._background_job for w in busy],
            on_idle=_idle,
            on_timeout=_timed_out,
            timeout_ms=CLOSE_JOB_TIMEOUT_MS,
            parent=self,
        )
        return False


class ProjectWorkspace(QtWidgets.QMainWindow):
    """Own one project's widgets, controllers, dialogs and background jobs.

    QMainWindow provides the workspace's menu/toolbar layout; when embedded
    in Gui it is an ordinary child widget, not another application window.
    """

    open_project_requested = QtCore.Signal(object)
    exit_requested = QtCore.Signal()
    theme_changed = QtCore.Signal(str)

    TOOLBAR_ICON_SIZE = QtCore.QSize(20, 20)

    def _configure_toolbar_icon_style(self, toolbar: QtWidgets.QToolBar) -> None:
        toolbar.setIconSize(self.TOOLBAR_ICON_SIZE)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setStyleSheet("QToolButton { margin: 0px; padding: 2px 1px; }")

    def __init__(
        self,
        filename: pathlib.Path | str | Project | None = None,
        *,
        parent: QtWidgets.QWidget | None = None,
        settings: GuiSettings | None = None,
        devices: WorkspaceDevices | None = None,
    ) -> None:
        super().__init__(parent)
        if parent is not None:
            self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self._embedded = parent is not None
        self._owns_devices = devices is None
        self.devices = devices if devices is not None else WorkspaceDevices(self)
        self._disposed = False
        self._background_job = BackgroundJob(self)
        self._background_job_label = ""
        self._close_gate = WorkspaceCloseGate(self)
        self.logger = logging.getLogger(__name__)
        self.logger.info("starting ProjectWorkspace")
        self.settings = settings if settings is not None else GuiSettings()
        self.project = Project()
        self._unsubscribe_project_history: Callable[[], None] | None = None
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
        # Vertical views of the plots per set of hidden neurons; see
        # _on_neurons_greyed_out_changed().
        self._saved_vertical_views: dict[
            frozenset[int], list[tuple[float, float] | None]
        ] = {}
        self._trace_state = TraceWidgetState(parent=self)
        self._spatial_footprint_state = SpatialFootprintState(parent=self)
        self.trace_widget = TraceWidget(
            self.trace_controller,
            parent=self,
            default_controller=self.controller,
            show_axis_link_controls=True,
            state=self._trace_state,
        )
        self.trace_widget.neuron_hovered.connect(
            self._spatial_footprint_state.set_hovered_stim_id
        )
        self.heatmap_widget = HeatmapWidget(parent=self)
        self.spatial_footprint_widget = SpatialFootprintWidget(
            parent=self, state=self._spatial_footprint_state
        )
        self.spatial_footprint_widget.import_requested.connect(
            self.btn_import_cellpose_data_clicked
        )
        self.spatial_footprint_widget.neurons_greyed_out_changed.connect(
            self._on_neurons_greyed_out_changed
        )
        self.spatial_footprint_widget.neuron_focus_requested.connect(
            self._focus_traces_on_neuron
        )
        # The calcium video now lives in the second tab of the imaging panel;
        # keep this attribute as a compatibility handle for timeline and
        # project synchronization code.
        self.calcium_video_widget = self.spatial_footprint_widget.calcium_video_widget
        self.calcium_video_widget.import_requested.connect(
            self._on_calcium_video_import_requested
        )
        self.calcium_video_widget.remove_requested.connect(
            self._on_calcium_video_remove_requested
        )
        self.calcium_video_widget.channels_changed.connect(
            self._on_calcium_video_channels_changed
        )
        self.acquisition_controller = WorkspaceAcquisitionController(
            self.devices.scanimage,
            self.settings,
            self.calcium_video_widget,
            project=lambda: self.project,
            current_stim=self._current_stim_array,
            export_settings=lambda: self.syscon2_panel.export_settings,
            video_widgets=lambda: self.findChildren(CalciumVideoWidget),
            show_video=lambda: self._auto_show_widget(self.calcium_video_widget),
            parent=self,
        )
        self.acquisition_controller.recording_added.connect(self._on_recording_added)
        # Compatibility handles for integrations using the workspace directly.
        self.scanimage_connection = self.acquisition_controller.connection
        self.scanimage_dialog = self.acquisition_controller.dialog
        self.scanimage_controls = self.acquisition_controller.controls
        self._scanimage_run_ids = self.acquisition_controller.run_ids
        self._create_menus_and_toolbars()
        self.setWindowIcon(qta.icon("fa6s.brain"))
        main_horizontal_layout = QtWidgets.QHBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_horizontal_layout)
        # vertical toolbar for stim editing on the left
        self.toolbar_stim_editor = QtWidgets.QToolBar()
        self.toolbar_stim_editor.setOrientation(QtCore.Qt.Orientation.Vertical)
        self._configure_toolbar_icon_style(self.toolbar_stim_editor)
        main_horizontal_layout.addWidget(self.toolbar_stim_editor)
        # left/right splitters between custom analysis and console
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.custom_analysis_widget = CustomAnalysisWidget(parent=self)
        split_left_right.addWidget(self.custom_analysis_widget)
        self.console_widget = ConsoleWidget(parent=self)
        split_left_right.addWidget(self.console_widget)
        self.chat_widget = ChatWidget(parent=self)
        self.chat_widget.setVisible(False)
        self.chat_widget.request_settings_dialog.connect(self.open_llm_settings)
        self.chat_widget.set_busy_check(self._background_job_busy)
        split_left_right.addWidget(self.chat_widget)
        self.syscon2_panel = SysCon2Panel(
            parent=self,
            settings=self.settings,
            manager=self.devices.syscon_session(self),
        )
        self.syscon2_panel.setVisible(False)
        # Mark, with a cross in the Spatial Footprints panel, every neuron
        # targeted by the preview whose import SysCon2 most recently confirmed -
        # see SysCon2Panel.targets_changed's docstring for why this can't be
        # a live hardware readout.
        self.syscon2_panel.targets_changed.connect(
            self._spatial_footprint_state.set_targeted_stim_ids
        )
        self.syscon2_panel.set_current_stim_provider(self._current_stim_array)
        split_left_right.addWidget(self.syscon2_panel)
        split_left_right.setSizes([1000, 1000, 1000, 1000])
        # synapses (heatmap) and cell segmentation (spatial footprint) stacked
        # to the right of the stim editor
        self.split_stim_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.split_stim_right.addWidget(self.heatmap_widget)
        self.split_stim_right.addWidget(self.spatial_footprint_widget)
        self.split_stim_right.setSizes([1000, 1400])
        self.split_stim_right.setHandleWidth(3)
        self._apply_right_splitter_style(self.settings.theme)
        # main vertical splitter layout
        split_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split_vertical.setHandleWidth(0)
        split_vertical.setChildrenCollapsible(False)
        # The timeseries panels and the time slider all share one column, so
        # they are always exactly as wide as each other and their time axes
        # line up: a sample at a given x in the stim editor sits at the same x
        # in the traces, the spikes and under the slider handle.
        split_timeseries = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split_timeseries.setHandleWidth(0)
        split_timeseries.setChildrenCollapsible(False)
        self.stim_editor = StimEditor(
            controller=self.controller,
            cmb_stim=self.cmb_stim,
            toolbar=self.toolbar_stim_editor,
            parent=self,
        )
        self.static_input_controller = StaticInputController(self.stim_editor, self)
        static_input_action = add_action(
            "Design static input",
            self.static_input_controller.open_pattern_designer,
            self.stim_editor.context_menu,
            None,
            self,
            icon="mdi6.pencil-box-outline",
        )
        self.stim_editor.context_menu.insertAction(
            self.stim_editor.action_show_all_stim_arrays, static_input_action
        )
        self.dynamic_input_controller = DynamicInputController(self.stim_editor, self)
        dynamic_input_action = add_action(
            "Design dynamic input",
            self.dynamic_input_controller.open_dynamic_input_designer,
            self.stim_editor.context_menu,
            None,
            self,
            icon="mdi6.pencil-box-multiple-outline",
        )
        self.stim_editor.context_menu.insertAction(
            self.stim_editor.action_show_all_stim_arrays, dynamic_input_action
        )
        self.dynamic_input_controller.simulation_requested.connect(
            self._do_brian2_simulation_threaded
        )
        self.stim_editor.time_range_focus_requested.connect(
            self.trace_controller.focus_time_range
        )
        self.stim_editor.current_stim_changed.connect(self._on_current_stim_changed)
        self.stim_editor.stimulation_host = self.syscon2_panel.stimulation_host
        self.stim_editor.trim_to_mapped_neurons_requested.connect(
            self.btn_trim_to_mapped_neurons_clicked
        )
        self._spatial_footprint_state.data_changed.connect(
            self._on_spatial_footprint_data_changed_for_trim_action
        )
        self.stim_editor.stim_widget.neuron_hovered.connect(
            self._spatial_footprint_state.set_hovered_stim_id
        )
        self._add_stim_array_ops_button_to_data_toolbar()
        split_timeseries.addWidget(self.stim_editor)
        self.trace_widget.set_default_scroll_bars(
            self.stim_editor.x_scrollbar,
            self.stim_editor.y_scrollbar,
        )
        split_timeseries.addWidget(self.trace_widget)
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
        split_timeseries.addWidget(self.spike_widget)
        self.data_widget = DataWidget(self.controller, parent=self)
        self.data_widget.set_default_scroll_bars(
            self.stim_editor.x_scrollbar,
            self.stim_editor.y_scrollbar,
        )
        split_timeseries.addWidget(self.data_widget)
        self.time_slider = TimeSliderWidget(self)
        self.plot_widgets: list[
            TimeseriesWidget | HeatmapWidget | DataWidget | CalciumVideoWidget
        ] = [
            self.stim_editor.stim_widget,
            self.trace_widget,
            self.spike_widget,
            self.data_widget,
            self.heatmap_widget,
            self.calcium_video_widget,
        ]
        for widget in self.plot_widgets:
            self.time_slider.connect_widget(widget)
        self.time_slider.time_changed.connect(self._on_time_changed)
        split_timeseries.addWidget(self.time_slider)
        # Synapses / cell segmentation sit beside the whole timeseries column,
        # not just beside the stim editor, so narrowing them widens every
        # timeseries panel together.
        split_timeseries_row = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split_timeseries_row.addWidget(split_timeseries)
        split_timeseries_row.addWidget(self.split_stim_right)
        split_timeseries_row.setSizes([3000, 1000])
        split_vertical.addWidget(split_timeseries_row)
        self.data_widget.setVisible(False)
        self.trace_widget.setVisible(False)
        self.spike_widget.setVisible(False)
        self.heatmap_widget.setVisible(False)
        self.spatial_footprint_widget.setVisible(True)
        self.custom_analysis_widget.setVisible(False)
        # heatmap / custom analysis widget / console
        split_vertical.addWidget(split_left_right)
        # Reserve 25% of the initial workspace height for the console row.
        split_vertical.setSizes([750, 250])
        main_horizontal_layout.addWidget(split_vertical)
        outer_widget = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        # Zoom controls sit at the top right, beside the project tabs.
        self.zoom_control = ZoomControl()
        tab_row = QtWidgets.QHBoxLayout()
        self._project_tab_row = tab_row
        tab_row.setContentsMargins(0, 0, 4, 0)
        tab_row.addStretch(1)
        tab_row.addWidget(self.zoom_control)
        outer_layout.addLayout(tab_row)
        # Scroll the whole workspace on small screens; nested widgets keep wheel
        # priority until they hit their end (see main_scroll.py).
        self._main_scroll = make_main_scroll_area(central_widget, 1500, 950)
        outer_layout.addWidget(self._main_scroll)
        self._workspace_zoom = WorkspaceZoom(central_widget, 1500, 950)
        self.zoom_control.zoom_changed.connect(self._workspace_zoom.apply)
        self.setCentralWidget(outer_widget)
        self._create_actions()
        for action in self.findChildren(QtGui.QAction):
            if not action.shortcut().isEmpty():
                action.setShortcutContext(
                    QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
                )
                self.addAction(action)
        if not self._embedded:
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
                slice | Sequence[int] | None,
                slice | None,
            ]
            | None
        ) = None
        self._detached_panel_windows: list[DetachedPanelWindow] = []
        # Tracks a panel's detached window only when it was detached via
        # plain "Detach panel" (not "Detach copy of panel") - i.e. exactly
        # the windows _set_widget_visible should close if the docked panel
        # is shown again, so re-showing it doesn't leave both visible at
        # once (which is what "Detach copy" is for).
        self._plain_detached_windows: dict[QtWidgets.QWidget, DetachedPanelWindow] = {}
        self._llm_settings_dialog: LLMSettingsDialog | None = None
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
        # Files dropped from the file explorer are imported (issue #597).
        self.setAcceptDrops(True)

    def _tab_label_for(self, project: Project) -> str:
        if project.filepath is not None:
            return pathlib.Path(project.filepath).name
        if project.project_section and project.project_name:
            return f"{project.project_section} / {project.project_name}"
        return project.b2_model.name or "Untitled"

    def _popup_title(self, title: str) -> str:
        return f"{title} - {self._tab_label_for(self.project)}"

    def set_project_tab_bar(self, tab_bar: QtWidgets.QTabBar) -> None:
        """Place the application's project tabs beside this workspace's zoom."""
        first_item = self._project_tab_row.itemAt(0)
        if first_item is not None and first_item.spacerItem() is not None:
            self._project_tab_row.takeAt(0)
        self._project_tab_row.insertWidget(0, tab_bar, stretch=1)
        tab_bar.show()

    def set_active(self, active: bool) -> None:
        """Pause hidden playback without changing its position or plot state."""
        if not active:
            self.time_slider.timer.stop()
        self.acquisition_controller.set_active(active)

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
        neuron_slice: slice | Sequence[int] | None,
        time_slice: slice | None,
    ):
        if isinstance(analysis_type, AnalysisPlugin) and analysis_type.open_dialog:
            analysis_type.open_dialog(
                self,
                self.project,
                data,
                neuron_slice if isinstance(neuron_slice, slice) else None,
                time_slice,
            )
            return
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
            self.calcium_video_widget.set_cellpose_data(self.project.cellpose_data)
            self.calcium_video_widget.set_stim_arrays(self.project.stim_arrays)
            self.calcium_video_widget.set_video_paths(
                self.project.calcium_video_paths,
                self.project.calcium_video_fps,
                scanimage_index=self.project.calcium_scanimage_index,
                channel=self._recorded_scanimage_channel(
                    self.project.calcium_scanimage_index,
                    self.scanimage_controls.channel_spin.value(),
                ),
                channels=self.project.calcium_video_channels[0],
                shown_channel=self.project.calcium_video_channels[1],
            )

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

    def _prompt_save_before_action(
        self, title: str, message: str, proceed: Callable[[], None]
    ) -> None:
        """Warn that an action may discard unsaved changes, offering to save first.

        Shows a Save / Discard / Cancel prompt. Cancel aborts. Discard runs
        ``proceed`` immediately. Save persists the current project first and
        runs ``proceed`` only after the save completes; if the user backs out
        of the save dialog, ``proceed`` never runs and the current project is
        left untouched.

        Skipped entirely (``proceed`` runs right away) when the active
        project has no unsaved changes - see ``Project.dirty``.
        """
        if not self.project.dirty:
            proceed()
            return
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
        """Return True when the standalone window's close may continue now.

        Skipped (returns True right away) when this workspace's project has
        no unsaved changes - see ``Project.dirty``. Choosing Save returns
        False and closes again once the save has completed.
        """
        if not self.project.dirty:
            self._close_confirmed = True
            return True
        message = (
            "Closing NeuroSeq will discard any unsaved changes to the "
            "current project.\n\nSave the current project first?"
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
        if self._background_job.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another background operation is already in progress.",
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
        *,
        on_failure: Callable[[str], None] | None = None,
    ) -> bool:
        """Run the interactive save flow.

        If the project already has a file path and ``force_dialog`` is False
        (a plain "Save"), it is written straight back to that path with no
        prompts. Otherwise (a "Save As", or a first-ever save with no existing
        path), a file picker is shown. All four project components are always
        included; the options dialog remains available for import workflows.

        Returns True if a save task was dispatched, False if the user cancelled
        at the file dialog (or a background operation is already running).
        ``on_success`` runs after the save completes successfully; it lets a
        caller (e.g. New) chain an action once the current project is safely
        persisted.
        """
        if self._background_job.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another background operation is already in progress.",
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
            self.settings.add_recent_project(str(result))
            self._update_window_title()
            if on_success is not None:
                on_success(result)

        if on_failure is None:
            self._dispatch_io_task(_save_task, f"Saving to {filename.name}…", _on_saved)
        else:
            self._dispatch_io_task(
                _save_task,
                f"Saving to {filename.name}…",
                _on_saved,
                on_failure=on_failure,
            )
        return True

    def btn_edit_clicked(self) -> None:
        old_neuron_count = self.project.b2_model.get_defined_neuron_count()
        mdiag = ModelEditorDialog(
            self.project.b2_model, parent=self, project=self.project
        )
        if mdiag.exec():
            self.project.mark_dirty()
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

    def btn_manage_data_clicked(self) -> None:
        dialog = DataManagerDialog(self.project, parent=self)
        dialog.data_changed.connect(lambda: self.update_project(self.project))
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
            if new_tab and self._embedded:
                self.open_project_requested.emit(loaded)
            else:
                self.update_project(loaded, reset_controller=True)

        if isinstance(project, Project):
            self.logger.info("Project loaded from Project object")
            self.last_cellpose_import = clone_cellpose_result(project.cellpose_data)
            project.cellpose_data = self.last_cellpose_import
            _place(project)
            return

        if self._background_job.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another background operation is already in progress.",
            )
            return
        filename = pathlib.Path(project)

        def _load_task(report: Callable[[int], None]) -> Project:
            return Project(filename, report=report)

        def _on_loaded(result: object) -> None:
            if not isinstance(result, Project):
                return
            self.logger.info(f"Project loaded from {filename}")
            self.settings.add_recent_project(str(filename))
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
        *,
        preserve_dynamic_input: bool = False,
    ) -> None:
        preserve_dynamic_input = preserve_dynamic_input and project is self.project
        if project is not self.project or self._unsubscribe_project_history is None:
            if self._unsubscribe_project_history is not None:
                self._unsubscribe_project_history()
            self._unsubscribe_project_history = project.history.subscribe(
                self._project_history_changed
            )
        self.project = project
        self._saved_vertical_views.clear()
        self._update_window_title()
        # Refresh this workspace's controllers/time-slider range unconditionally.
        # StimEditor.set_project() below
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
        self.syscon2_panel.set_project(self.project)
        self._refresh_video_selector()
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

        self.stim_editor.set_project(
            self.project, keep_undo_history=preserve_dynamic_input
        )
        # A replaced project must not inherit the previous run's Design dynamic
        # input state (plots, log, EPSP history); drop the cached dialog so it
        # is rebuilt fresh on next open.
        if preserve_dynamic_input:
            self.dynamic_input_controller.refresh()
        else:
            self.dynamic_input_controller.discard_dynamic_input_dialogs()
        self.console_widget.set_project(self.project)
        self.chat_widget.set_project(self.project)
        self.syscon2_panel.set_project(self.project)
        self._refresh_video_selector()
        self._update_combo_boxes()
        self._refresh_elise_replay_action()
        if hasattr(self, "spatial_footprint_widget"):
            self.spatial_footprint_widget.set_data(self.project.cellpose_data)
            self.calcium_video_widget.set_cellpose_data(self.project.cellpose_data)
            self.calcium_video_widget.set_stim_arrays(self.project.stim_arrays)
            self.calcium_video_widget.set_video_paths(
                self.project.calcium_video_paths,
                self.project.calcium_video_fps,
                scanimage_index=self.project.calcium_scanimage_index,
                channel=self._recorded_scanimage_channel(
                    self.project.calcium_scanimage_index,
                    self.scanimage_controls.channel_spin.value(),
                ),
                channels=self.project.calcium_video_channels[0],
                shown_channel=self.project.calcium_video_channels[1],
            )
        self.time_slider.set_current_time_ms(self.time_slider.t_ms)

    def _project_history_changed(self, event: str) -> None:
        if self._disposed:
            return
        if event in ("undo", "redo"):
            self._update_data(keep_undo_history=True)
            self._update_window_title()

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
                "onto segmented cells (e.g. for holographic input export).",
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
                    "Resize Input Arrays",
                    "The imported Cellpose segmentation contains "
                    f"{detected_neurons} cells, but the largest existing input "
                    f"array has {current_max_neurons} neurons.\n\n"
                    "Expand all existing input arrays to match the Cellpose "
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
        # not a new user selection. While the project is being replaced the
        # selectors can still name the previous project's data (issue #595),
        # which this project may not have; those are refilled right after.
        trace_name = self._current_data_name(self.cmb_trace, TraceData)
        if trace_name:
            self.project.max_ram_trace_plot_matrix = (
                self.settings.max_ram_trace_plot_matrix
            )
            self.trace_widget.set_data(self.project.get_data(trace_name, TraceData))
            self._keep_cell_selection(self.trace_widget)
        spike_name = self._current_data_name(self.cmb_spike, SpikeData)
        if spike_name:
            self.spike_widget.set_data(self.project.get_data(spike_name, SpikeData))
            self._keep_cell_selection(self.spike_widget)
        synapse_name = self._current_data_name(self.cmb_synapse, SynapseData)
        if synapse_name:
            self.project.max_ram_synapse_plot_matrix = (
                self.settings.max_ram_synapse_plot_matrix
            )
            self.heatmap_widget.set_data(
                self.project.get_data(synapse_name, SynapseData)
            )
        data_name = self.cmb_data.currentText()
        for data_type, suffix in (
            (TraceData, " [trace]"),
            (SpikeData, " [spike]"),
            (SynapseData, " [synapse]"),
        ):
            if data_name.endswith(suffix) and data_name[: -len(suffix)] in (
                self.project.get_data_names(data_type)
            ):
                self._apply_data_selection(data_name)

    def _current_data_name(
        self,
        combo: QtWidgets.QComboBox,
        data_type: type[TraceData | SpikeData | SynapseData],
    ) -> str:
        """The combo's selection, or "" if this project has no such data."""
        name = combo.currentText()
        return name if name in self.project.get_data_names(data_type) else ""

    def _keep_cell_selection(self, widget: TimeseriesWidget | DataWidget) -> None:
        """New data shows every neuron; hide again those that the cells
        selected in the spatial footprints exclude (issue #596)."""
        hidden = self._spatial_footprint_state.greyed_stim_ids
        if hidden:
            widget.set_neurons_greyed(sorted(hidden), [])

    def _apply_trace_selection(self, cmb_name: str) -> None:
        data = self.project.get_data(cmb_name, TraceData)
        self.trace_widget.set_data(data)
        self._keep_cell_selection(self.trace_widget)
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
        # set_data() itself treats a shapeless/zero-length SynapseData (e.g. a
        # model that defines synapses but was never simulated) as empty and
        # leaves self.data unset - check that, not the pre-fetch value, so an
        # empty result doesn't pop the panel open with nothing to show.
        if getattr(self.heatmap_widget, "data", None) is not None and hasattr(
            self, "_auto_show_widget"
        ):
            self._auto_show_widget(self.heatmap_widget)

    def _on_neurons_greyed_out_changed(
        self,
        newly_hidden: list[int],
        newly_shown: list[int],
    ) -> None:
        before = frozenset(self._trace_state.greyed_neuron_ids)
        views = self._vertical_views()
        self._trace_state.set_neurons_greyed(newly_hidden, newly_shown)
        self.stim_editor.stim_widget.set_neurons_greyed(newly_hidden, newly_shown)
        self.spike_widget.set_neurons_greyed(newly_hidden, newly_shown)
        self.data_widget.set_neurons_greyed(newly_hidden, newly_shown)
        after = frozenset(self._trace_state.greyed_neuron_ids)
        if after == before:
            return
        # Hiding rows fits the plots to the remaining neurons. Going back to
        # an earlier selection, e.g. deselecting a cell, restores the vertical
        # zoom and position it had then instead of zooming out (issue #604).
        self._saved_vertical_views[before] = views
        while len(self._saved_vertical_views) > 8:
            del self._saved_vertical_views[next(iter(self._saved_vertical_views))]
        restored = self._saved_vertical_views.pop(after, None)
        if restored is not None:
            for controller, view in zip(
                self._plot_controllers(), restored, strict=True
            ):
                if view is not None:
                    controller.set_vertical_view(view)

    def _focus_traces_on_neuron(self, neuron_id: int) -> None:
        """Centre the traces on a neuron right-clicked in the spatial
        footprints, keeping the zoom and the other traces (issue #603)."""
        if not self.trace_widget.center_on_neuron(neuron_id):
            self.logger.info("Neuron %d has no visible trace to centre on", neuron_id)

    def _plot_controllers(self) -> list[ConstrainedPanZoomController]:
        return [self.controller, self.trace_controller, self.spike_controller]

    def _vertical_views(self) -> list[tuple[float, float] | None]:
        return [controller.vertical_view() for controller in self._plot_controllers()]

    def _current_stim_array(self) -> StimArray | None:
        modifier = self.stim_editor.current_stim_modifier
        return modifier.stim_array if modifier is not None else None

    def _on_current_stim_changed(self, stim_array: StimArray | None) -> None:
        if self.action_overlay_stim_on_traces.isChecked():
            self.trace_widget.set_stim_overlay_data(stim_array)
        self._on_time_changed(self.time_slider.t_ms)

    def _on_time_changed(self, time_ms: int) -> None:
        """Highlight, with a red cross in the Spatial Footprints panel,
        every neuron receiving nonzero stimulation at the current time
        slider position - driven by the currently active stim array, not
        by an interaction with the panel itself."""
        self._spatial_footprint_state.set_active_stim_ids(
            self._stim_ids_active_at(self._current_stim_array(), time_ms)
        )

    @staticmethod
    def _stim_ids_active_at(stim_array: StimArray | None, time_ms: int) -> set[int]:
        if stim_array is None or stim_array.dt <= 0:
            return set()
        time_index = int(time_ms / (stim_array.dt * 1000.0))
        if not (0 <= time_index < stim_array.values.shape[0]):
            return set()
        return {int(i) for i in np.nonzero(stim_array.values[time_index])[0]}

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
        self._keep_cell_selection(self.spike_widget)
        # set_data() itself treats a spike-less SpikeData (e.g. a model that
        # defines a spike monitor but was never simulated) as empty and
        # leaves self.data unset - check that, not the pre-fetch value, so an
        # empty result doesn't pop the panel open with nothing to show.
        if getattr(self.spike_widget, "data", None) is not None and hasattr(
            self, "_auto_show_widget"
        ):
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
        self._keep_cell_selection(self.data_widget)
        if isinstance(data, TraceData):
            self._focus_neuron_range(self.data_widget.trace_widget.n_neurons)
        elif isinstance(data, SpikeData):
            self._focus_neuron_range(self.data_widget.spike_widget.n_neurons)

    @with_wait_cursor
    def set_data(self, cmb_name: str) -> None:
        self._apply_data_selection(cmb_name)

    def btn_exit_clicked(self) -> None:
        if self._embedded:
            self.exit_requested.emit()
        else:
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
            dialog = AddExperimentDataDialog(experiment, self.project, parent=self)
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
            self.update_project(self.project, preserve_dynamic_input=True)
            self._select_experiment_data(experiment.name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Data import error",
                f"Could not import data to Input IDs: {e}",
            )
            self.logger.exception(e)
            return

    def _do_external_simulation(self, simulation_time: float) -> None:
        if self.project.external_simulate_function is None:
            return
        if self._background_job.busy:
            self.logger.warning(
                "External simulation dispatch ignored - another background job is running."
            )
            return
        # External simulators are arbitrary code and may drive Brian2 (e.g.
        # run_brian2_externally.py, the PyNN brian2 backend).
        if not self._acquire_simulation_guard():
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

        cancel_event = threading.Event()
        self._prepare_background_job("Simulating...", cancellable=True)

        worker = ExternalSimWorker(
            simulate_fn=self.project.external_simulate_function,
            stims_dict=stims_dict,
            simulation_time=simulation_time,
            params=params,
            accepts_progress=accepts_progress,
            cancel_event=cancel_event,
        )
        self._background_job.start(
            worker,
            on_progress=self._on_external_sim_progress,
            on_finished=self._on_external_sim_finished,
            on_failed=self._on_external_sim_failed,
            on_cancelled=self._on_external_sim_cancelled,
            cancel=cancel_event.set,
        )

    def _teardown_external_sim(self) -> None:
        self._teardown_background_job()

    @QtCore.Slot(float)
    def _on_external_sim_progress(self, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.progress_dialog.setValue(int(round(clamped * 100)))

    @QtCore.Slot(object)
    def _on_external_sim_finished(self, experiment: Experiment) -> None:
        self._teardown_external_sim()
        self.progress_dialog.setValue(100)
        self.progress_dialog.reset()
        self.logger.info(
            f"External simulation complete: '{experiment.name}' "
            f"{experiment.duration}s, {len(experiment.data)} data objects"
        )
        self._append_experimental_data(experiment)

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
        self.stim_editor.history_enabled = bool(enabled)
        self.stim_editor._stim_values_updated()

    def _iter_menu_actions(self, menu: QtWidgets.QMenu) -> list[QtWidgets.QAction]:
        """Recursively collect every leaf action in ``menu``, including submenus'."""
        actions: list[QtWidgets.QAction] = []
        for action in menu.actions():
            submenu = action.menu()
            if submenu is not None:
                actions.extend(self._iter_menu_actions(submenu))
            else:
                actions.append(action)
        return actions

    def _set_io_actions_enabled(self, enabled: bool) -> None:
        """Gate file-IO entry points while a save/load/simulation worker is running.

        Disables this workspace's File and simulation actions so a simulation
        cannot mutate its project while a file operation is reading it.
        Other workspaces and the application tab bar remain available.

        ``QMenu.setEnabled``/``QToolBar.setEnabled`` only grey out the
        container and block mouse interaction with it -- the ``QAction``s
        inside keep their own ``enabled`` state, so a ``Qt.WindowShortcut``
        such as Ctrl+N/Ctrl+O/Ctrl+E still fires even while the File menu
        looks disabled. Disable every action reachable from the File menu
        (each one doubles as its toolbar button, since ``add_action``
        shares one ``QAction`` between the two) so the shortcuts are
        blocked too.
        """
        for attr in ("menu_file", "toolbar_file"):
            target = getattr(self, attr, None)
            if target is not None:
                target.setEnabled(bool(enabled))
        menu_file = getattr(self, "menu_file", None)
        if menu_file is not None:
            for action in self._iter_menu_actions(menu_file):
                action.setEnabled(bool(enabled))
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
        background job is already in flight.

        If ``cancel_callback`` is given, the progress dialog shows a Cancel
        button wired to it that calls the callback on the GUI thread.
        """
        if self._background_job.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another background operation is in progress. Please wait for it "
                "to finish before starting a new one.",
            )
            return False

        self._prepare_background_job(
            label, cancellable=cancel_callback is not None, block_io=True
        )
        self._io_on_success = on_success
        self._io_failure_title = failure_title
        self._io_failure_prefix = failure_prefix
        self._io_on_failure = on_failure

        self._background_job.start(
            BackgroundIOWorker(task),
            on_progress=self._on_io_progress,
            on_finished=self._on_io_finished,
            on_failed=self._on_io_failed,
            cancel=cancel_callback,
        )
        self.logger.info(f"Dispatching IO task: {label}")
        return True

    def _teardown_io_task(self) -> None:
        self._teardown_background_job()
        self._io_on_success = None
        self._io_on_failure = None
        self._io_failure_title = "File operation failed"
        self._io_failure_prefix = "The file operation could not complete:"

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
        # BackgroundJob reports exceptions raised by completion callbacks.
        if on_success is not None:
            on_success(result)

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
        if self._background_job.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Busy",
                "Another operation is already in progress.",
            )
            # If Design dynamic input requested this run, tell it the request
            # was dropped so its closed loop stops instead of stalling forever
            # waiting for a completion callback that will never arrive. This is
            # a no-op when no routine is awaiting a simulation.
            self.dynamic_input_controller.dynamic_input_simulation_failed(
                "another operation is already in progress"
            )
            return
        if not self._acquire_simulation_guard():
            self.dynamic_input_controller.dynamic_input_simulation_failed(
                "another project is simulating"
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
                0, self.dynamic_input_controller.dynamic_input_simulation_finished
            )

        if not self._dispatch_io_task(
            _brian2_task,
            "Preparing Brian2 simulation…",
            _on_brian2_finished,
            failure_title="Simulation error",
            failure_prefix="Could not run simulation:",
            cancel_callback=_request_brian2_cancel,
            on_failure=self.dynamic_input_controller.dynamic_input_simulation_failed,
        ):
            SimulationGuard.release(self)

    def _acquire_simulation_guard(self) -> bool:
        """Take the application-wide simulation guard, or explain who holds it.

        Released by ``_teardown_background_job`` on every job outcome and by
        ``dispose``.
        """
        busy = SimulationGuard.acquire(self, self._tab_label_for(self.project))
        if busy is None:
            return True
        QtWidgets.QMessageBox.information(
            self,
            "Simulation in progress",
            f"Project '{busy}' is already simulating. Brian2's random seed and "
            "code generation settings are shared by all open projects, so only "
            "one simulation can run at a time. Wait for it to finish or cancel it.",
        )
        return False

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
        self._choose_plot_ram_limit(
            "max_ram_synapse_plot_matrix", "synapse matrix plot"
        )

    def choose_max_ram_trace_plot_matrix(self) -> None:
        self._choose_plot_ram_limit("max_ram_trace_plot_matrix", "trace plot")

    def _choose_plot_ram_limit(self, setting: str, label: str) -> None:
        new_max_ram, ok = QtWidgets.QInputDialog.getText(
            self,
            "RAM limit",
            f"RAM limit in MB for {label} (0 for no limit)",
            QtWidgets.QLineEdit.EchoMode.Normal,
            str(getattr(self.settings, setting)),
        )
        if not ok:
            return
        try:
            limit = max(0, int(new_max_ram))
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse RAM limit '{new_max_ram}' as a number",
            )
            return
        setattr(self.settings, setting, limit)
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

    def btn_import_tif_clicked(
        self, *, paths: list[pathlib.Path] | None = None
    ) -> None:
        """Open dialog to run the Suite2p pipeline on raw 2-photon TIFFs,
        optionally with ``paths`` already listed."""
        from neuroseq.importers.tif_import_dialog import TifImportDialog

        dialog = TifImportDialog(
            self,
            cellpose_import=self.last_cellpose_import,
            stim_arrays=self.project.stim_arrays,
        )
        if paths:
            dialog.add_tif_paths(paths)
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
            if dialog.imported_tif_paths:
                self.project.add_calcium_video(
                    dialog.imported_tif_paths,
                    dialog.imported_fps,
                    name=dialog.experiment_name,
                    channels=dialog.imported_channels,
                    shown_channel=dialog.imported_functional_channel,
                )
                self._refresh_calcium_video_widgets()
                if hasattr(self, "_auto_show_widget"):
                    self._auto_show_widget(self.calcium_video_widget)

    def _on_calcium_video_import_requested(
        self, *, paths: list[pathlib.Path] | None = None
    ) -> None:
        if paths is None:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select raw calcium imaging file(s)",
                "",
                "Imaging files (*.tif *.tiff *.gif *.mp4 *.avi *.mov *.mkv *.webm);;"
                "TIFF files (*.tif *.tiff);;GIF files (*.gif);;"
                "Movie files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
            )
            paths = [pathlib.Path(path) for path in chosen]
        if not paths:
            return
        fps, ok = QtWidgets.QInputDialog.getDouble(
            self, "Frame rate", "Frames per second:", DEFAULT_FS, 0.001, 10000.0, 3
        )
        if not ok:
            return
        tif_paths = [pathlib.Path(path) for path in paths]
        self.project.add_calcium_video(tif_paths, fps)
        self._refresh_calcium_video_widgets()
        self._auto_show_widget(self.calcium_video_widget)

    def _on_calcium_video_remove_requested(self) -> None:
        self.project.remove_calcium_video()
        self._refresh_calcium_video_widgets()

    def _on_calcium_video_channels_changed(
        self, channels: int, shown_channel: int
    ) -> None:
        self.project.set_calcium_video_channels(channels, shown_channel)
        for widget in self.findChildren(CalciumVideoWidget):
            widget.set_channels(channels, shown_channel)

    def _refresh_calcium_video_widgets(self) -> None:
        self._refresh_video_selector()
        for widget in self.findChildren(CalciumVideoWidget):
            widget.set_cellpose_data(self.project.cellpose_data)
            widget.set_stim_arrays(self.project.stim_arrays)
            widget.set_video_paths(
                self.project.calcium_video_paths,
                self.project.calcium_video_fps,
                scanimage_index=self.project.calcium_scanimage_index,
                channel=self._recorded_scanimage_channel(
                    self.project.calcium_scanimage_index,
                    self.scanimage_controls.channel_spin.value(),
                ),
                channels=self.project.calcium_video_channels[0],
                shown_channel=self.project.calcium_video_channels[1],
            )

    def _on_recording_added(self) -> None:
        self._refresh_calcium_video_widgets()
        self._update_combo_boxes()
        self.update_limits()

    def _start_scanimage_recording(self, arguments: dict) -> None:
        return self.acquisition_controller.start_scanimage_recording(arguments)

    def _scanimage_status(self, status: dict) -> None:
        return self.acquisition_controller.scanimage_status(status)

    def _open_image_acquisition_dialog(self) -> None:
        return self.acquisition_controller.open_image_acquisition_dialog()

    def _refresh_video_selector(self) -> None:
        if not hasattr(self, "cmb_video"):
            return
        self.cmb_video.blockSignals(True)
        self.cmb_video.clear()
        for video in self.project.calcium_videos:
            label = video["name"]
            run_id = video.get("run_id")
            record = next(
                (r for r in self.project.imaging_acquisitions if r["run_id"] == run_id),
                None,
            )
            if record is not None and record.get("status") != "completed":
                label = f"{label} · {record.get('status')}"
            self.cmb_video.addItem(label, video["id"])
        if not self.project.calcium_videos:
            self.cmb_video.addItem("No imaging data", None)
        self.cmb_video.setEnabled(bool(self.project.calcium_videos))
        self.cmb_video.setCurrentIndex(
            max(0, self.cmb_video.findData(self.project.selected_calcium_video_id))
        )
        self.cmb_video.blockSignals(False)

    def _on_video_selected(self, index: int) -> None:
        video_id = self.cmb_video.itemData(index)
        if video_id is None or video_id == self.project.selected_calcium_video_id:
            return
        video = next(v for v in self.project.calcium_videos if v["id"] == video_id)
        missing = [path for path in video["paths"] if not pathlib.Path(path).is_file()]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Imaging files missing",
                "These files of the selected imaging data were not found:\n"
                + "\n".join(missing),
            )
            self._refresh_video_selector()
            return
        self.project.select_calcium_video(video_id)
        self.scanimage_controls.live_check.setChecked(False)
        self._refresh_calcium_video_widgets()
        self._auto_show_widget(self.calcium_video_widget)

    def _set_scanimage_live(self, live: bool) -> None:
        return self.acquisition_controller.set_scanimage_live(live)

    def _set_scanimage_channel(self, channel: int) -> None:
        return self.acquisition_controller.set_scanimage_channel(channel)

    def _recorded_scanimage_channel(self, index: dict | None, channel: int) -> int:
        return self.acquisition_controller.recorded_scanimage_channel(index, channel)

    def _scanimage_recording_ready(self, result: dict) -> None:
        return self.acquisition_controller.scanimage_recording_ready(result)

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

    def btn_import_cellpose_data_clicked(
        self, *, segment: pathlib.Path | None = None, model: str | None = None
    ) -> None:
        """Open the cell segmentation dialog; with ``segment``, it segments
        that image with ``model`` right away."""
        dialog = CellposeImportDialog(
            self,
            existing_data=self.last_cellpose_import,
            delete_handler=self._delete_cellpose_data,
        )
        if segment is not None:
            if model is not None:
                dialog.cellpose_model_combo.setCurrentText(model)
            # Once the dialog is shown, so it is there while Cellpose runs.
            QtCore.QTimer.singleShot(0, lambda: dialog.segment_file(str(segment)))
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
                    f"{len(resized_names)} input array(s) to {detected_neurons} neurons."
                )
            QtWidgets.QMessageBox.information(
                self,
                "Cell spatial footprint import",
                message,
            )
            self._warn_if_cellpose_model_mismatch()

    def _complete_stimulus_import(
        self,
        dialog: Syscon2ImportDialog | PatchClampDataInputImportDialog,
        path: pathlib.Path | None = None,
    ) -> None:
        """Load an optional file and apply an accepted stimulus import to the GUI."""
        if path is not None:
            dialog.load_file(str(path))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.project.mark_dirty()
        self.update_project(self.project)
        if dialog.selected_stim_name:
            self.cmb_stim.setCurrentText(dialog.selected_stim_name)

    def btn_import_syscon2_seq_clicked(
        self, *, path: pathlib.Path | None = None
    ) -> None:
        dialog = Syscon2ImportDialog(self.project, parent=self)
        ProjectWorkspace._complete_stimulus_import(self, dialog, path)

    def btn_import_igor2_wave_clicked(
        self, *, csv_only: bool = False, path: pathlib.Path | None = None
    ) -> None:
        dialog = (
            PatchClampDataInputImportDialog(self.project, parent=self, csv_only=True)
            if csv_only
            else PatchClampDataInputImportDialog(self.project, parent=self)
        )
        ProjectWorkspace._complete_stimulus_import(self, dialog, path)

    def btn_import_igor2_trace_clicked(
        self, *, csv_only: bool = False, path: pathlib.Path | None = None
    ) -> None:
        from neuroseq.importers.patch_clamp_data_output_import_dialog import (
            PatchClampDataOutputImportDialog,
        )

        dialog = (
            PatchClampDataOutputImportDialog(self.project, parent=self, csv_only=True)
            if csv_only
            else PatchClampDataOutputImportDialog(self.project, parent=self)
        )
        if path is not None:
            dialog.load_file(str(path))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.project.mark_dirty()
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
            self.calcium_video_widget.set_cellpose_data(None)

    def _on_spatial_footprint_data_changed_for_trim_action(self) -> None:
        """Keep the left toolbar's "Trim to Mapped Neurons…" action enabled
        only once there's an actual mapping to trim to - it lives in
        stim_editor.py's toolbar, not the Spatial Footprints panel itself,
        so it can't gate its own enabled state off the panel's own data."""
        cellpose_data = self._spatial_footprint_state.cellpose_data
        self.stim_editor.action_trim_to_mapped_neurons.setEnabled(
            bool(
                cellpose_data is not None and cellpose_data.stimulus_to_cellpose_mapping
            )
        )

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
                "No Cellpose input mapping is set, so there is nothing to trim to.",
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
            "from every input array and every experiment's trace/spike/synapse "
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

    # Import stimulus wrappers

    def btn_import_stimulus_light_clicked(self) -> None:
        """Pre-dialog: choose software, then import a light stimulus waveform."""
        software, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Import light input",
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
        """Open the patch-clamp input importer directly."""
        self.btn_import_igor2_wave_clicked()

    def btn_import_trace_patchclamp_clicked(self) -> None:
        """Open the patch-clamp output importer directly."""
        self.btn_import_igor2_trace_clicked()

    # Export wrappers

    def btn_export_patch_clamp_stim_clicked(self) -> None:
        show_stimulus_export_dialog(
            self.project,
            "patch-clamp",
            parent=self,
            current_stim_name=self.cmb_stim.currentText(),
        )

    def btn_export_light_stimulus_clicked(self) -> None:
        show_stimulus_export_dialog(
            self.project,
            "light",
            parent=self,
            current_stim_name=self.cmb_stim.currentText(),
            reserved_names=self.syscon2_panel._remote_sequence_names,
            on_generated=self.syscon2_panel.show_generated_preview,
        )

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
        if SimulationGuard.held_by() is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Brian2 backend",
                "A simulation is running; change the backend once it has finished.",
            )
            return
        self.logger.info(f"Setting Brian2 backend to {new_backend}")
        b2.prefs.codegen.target = new_backend
        self.settings.brian2_backend = new_backend

    def choose_stim_padding_secs(self) -> None:
        new_padding_secs, ok = QtWidgets.QInputDialog.getText(
            self,
            "Input padding",
            "Number of seconds of padding for inputs beyond simulation time",
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
        """Apply a light/dark/retro theme to the whole application and all plots."""
        theme = theme_module.normalize_theme(theme)
        self.logger.info(f"Applying {theme} theme")
        theme_module.apply_app_theme(QtWidgets.QApplication.instance(), theme)
        self._apply_right_splitter_style(theme)
        icons.apply_theme(theme)
        for widget in self.plot_widgets:
            widget.apply_theme(theme)
        self.custom_analysis_widget.apply_theme(theme)
        self.spatial_footprint_widget.apply_theme(theme)
        self.dynamic_input_controller.apply_dynamic_input_theme(theme)
        action = getattr(self, "_theme_actions", {}).get(theme)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self.settings.theme = theme
        self.theme_changed.emit(theme)

    def _apply_right_splitter_style(self, theme: str) -> None:
        colors = {
            theme_module.LIGHT: "#b8b8b8",
            theme_module.DARK: "#555555",
            theme_module.RETRO: "#7d8b9d",
        }
        if hasattr(self, "split_stim_right"):
            color = colors.get(theme_module.normalize_theme(theme), "#b8b8b8")
            self.split_stim_right.setStyleSheet(
                f"QSplitter::handle {{ background-color: {color}; }}"
            )

    def open_llm_settings(self) -> None:
        if self._llm_settings_dialog is None:
            self._llm_settings_dialog = LLMSettingsDialog()
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

    def _add_stim_array_ops_button_to_data_toolbar(self) -> None:
        """Add a single dropdown button for the stim array lifecycle actions.

        Create/Edit/Delete/Clear stim array, Trim to Mapped Neurons, and
        Pad/Trim/Loop stim array used to be eight separate buttons on the
        stim editor's left toolbar (issue #535). They now live in one
        dropdown next to the stim array combobox instead, reusing the
        StimEditor QActions directly rather than duplicating them, so state
        like Trim to Mapped Neurons' enabled/disabled status stays in sync
        automatically no matter which menu shows it.
        """
        self.btn_stim_array_ops = QtWidgets.QToolButton(self.toolbar_data)
        self.btn_stim_array_ops.setToolTip("Input array operations")
        icons.set_widget_icon(self.btn_stim_array_ops, "mdi6.text-box-multiple-outline")
        self.btn_stim_array_ops.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.btn_stim_array_ops.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.stim_array_ops_menu = QtWidgets.QMenu(self.btn_stim_array_ops)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_create_stim_array)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_edit_stim_array)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_delete_stim_array)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_clear_stim_array)
        self.stim_array_ops_menu.addSeparator()
        self.stim_array_ops_menu.addAction(
            self.stim_editor.action_trim_to_mapped_neurons
        )
        self.stim_array_ops_menu.addSeparator()
        self.stim_array_ops_menu.addAction(self.stim_editor.action_pad_stim_array)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_trim_stim_array)
        self.stim_array_ops_menu.addAction(self.stim_editor.action_loop_stim_array)
        self.btn_stim_array_ops.setMenu(self.stim_array_ops_menu)
        self.toolbar_data.insertWidget(
            self._stim_array_ops_anchor_action, self.btn_stim_array_ops
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
        self.addToolBarBreak()
        self.addToolBar(self.toolbar_view)

        self.menu_settings = menu.addMenu("Se&ttings")
        assert self.menu_settings is not None
        self.menu_tools = menu.addMenu("&Tools")
        assert self.menu_tools is not None
        self.menu_help = menu.addMenu("&Help")
        assert self.menu_help is not None
        self._create_actions_data(toolbar=self.toolbar_data, menu=None)

    def _create_actions(self) -> None:
        assert self.menu_file is not None
        assert self.menu_simulate is not None
        assert self.menu_playback is not None
        assert self.menu_view is not None
        assert self.menu_settings is not None
        assert self.menu_tools is not None
        assert self.menu_help is not None
        assert self.menu_experiment is not None
        self._create_actions_file(self.menu_file, self.toolbar_file)
        self._create_actions_simulate(self.menu_simulate, self.toolbar_simulate)
        self._create_actions_playback(self.menu_playback, self.toolbar_playback)
        self._create_actions_view(self.menu_view, self.toolbar_view)
        self._create_actions_settings(menu=self.menu_settings, toolbar=None)
        self._create_actions_tools(menu=self.menu_tools)
        self._create_actions_help(menu=self.menu_help, toolbar=None)
        self._create_action_experiment(self.menu_experiment, toolbar=None)
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
        toolbar.addSeparator()
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

        import_stimulus_menu = import_data_menu.addMenu("&Input")
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

        import_trace_menu = import_data_menu.addMenu("&Output")
        assert import_trace_menu is not None
        add_action(
            "&Calcium Imaging (raw .tif)",
            self.btn_import_tif_clicked,
            import_trace_menu,
            None,
            icon="mdi.waves",
        )
        add_action(
            "Calcium Imaging (&Suite2p outputs)",
            self.btn_import_suite2p_data_clicked,
            import_trace_menu,
            None,
            icon="mdi.waves",
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
            "&Patch Clamp Input",
            self.btn_export_patch_clamp_stim_clicked,
            export_data_menu,
            None,
            icon="mdi6.waveform",
        )
        add_action(
            "&Optogenetic Input",
            self.btn_export_light_stimulus_clicked,
            export_data_menu,
            None,
            icon="mdi6.file-code",
        )

        toolbar.addSeparator()
        add_action(
            "&Edit Model",
            self.btn_edit_clicked,
            menu,
            toolbar,
            shortcut="Ctrl+E",
            icon="ph.polygon-light",
        )
        menu.addSeparator()
        toolbar.addSeparator()
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
        add_action(
            "&Manage Data",
            self.btn_manage_data_clicked,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.beaker",
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
        menu.addSeparator()
        toolbar.addSeparator()
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
            ("&Input", self.stim_editor, "mdi6.chart-timeline", True),
            ("&Traces", self.trace_widget, "mdi.waves", False),
            ("&Spikes", self.spike_widget, "ph.barcode-light", False),
            ("&Data", self.data_widget, "mdi6.chart-donut", False),
            ("&Heatmap", self.heatmap_widget, "mdi6.data-matrix", False),
            (
                "&Spatial Footprints",
                self.spatial_footprint_widget,
                "mdi6.movie-open",
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
            ("&Stimulation setup", self.syscon2_panel, "mdi6.laser-pointer", False),
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
        # Keep the child handle addressable for existing integrations and
        # detach requests; both imaging tabs belong to the same panel action.
        self._view_actions[self.calcium_video_widget] = self._view_actions[
            self.spatial_footprint_widget
        ]
        menu.addSeparator()
        toolbar.addSeparator()
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
            "Keep Short Inputs Visible (4 px)",
            self.toggle_minimum_stimulus_size,
            menu,
            toolbar,
            icon="mdi6.square-medium",
            checkable=True,
            is_checked=True,
        )
        menu.addSeparator()
        toolbar.addSeparator()
        # A genuine toggle, so it joins the other view actions directly
        # (reused by reference - see stim_editor.py's comment on the action).
        menu.addAction(self.stim_editor.action_show_all_stim_arrays)
        toolbar.addAction(self.stim_editor.action_show_all_stim_arrays)
        self._add_stim_design_buttons_to_view_toolbar(toolbar)

    def _add_stim_design_buttons_to_view_toolbar(
        self, toolbar: QtWidgets.QToolBar
    ) -> None:
        """Add the Design static/dynamic input buttons here (issue #535).

        Unlike every other button in this toolbar, these carry a text label
        beside their icon: a plain icon glyph for "start designing a
        stimulus" isn't self-explanatory the way e.g. a grid icon is for
        "toggle grid", so they stay rectangular QPushButtons instead of
        collapsing to icon-only QToolButtons like their neighbors.
        """
        self.btn_design_static_input = QtWidgets.QPushButton("Design static input")
        self.btn_design_static_input.setToolTip(
            "Draw a static input pattern across neurons and time"
        )
        icons.set_widget_icon(self.btn_design_static_input, "mdi6.pencil-box-outline")
        self.btn_design_static_input.clicked.connect(
            self.static_input_controller.open_pattern_designer
        )
        toolbar.addWidget(self.btn_design_static_input)
        self.btn_design_dynamic_input = QtWidgets.QPushButton("Design dynamic input")
        self.btn_design_dynamic_input.setToolTip(
            "Design a dynamic input routine (e.g. STDP, rheobase)"
        )
        icons.set_widget_icon(
            self.btn_design_dynamic_input, "mdi6.pencil-box-multiple-outline"
        )
        self.btn_design_dynamic_input.clicked.connect(
            self.dynamic_input_controller.open_dynamic_input_designer
        )
        toolbar.addWidget(self.btn_design_dynamic_input)
        toolbar.addSeparator()
        self.btn_image_acquisition = QtWidgets.QPushButton("Image acquisition")
        self.btn_image_acquisition.setToolTip(
            "Connect to ScanImage to preview and record imaging"
        )
        icons.set_widget_icon(self.btn_image_acquisition, "mdi6.microscope")
        self.btn_image_acquisition.clicked.connect(self._open_image_acquisition_dialog)
        toolbar.addWidget(self.btn_image_acquisition)
        self.cmb_video = QtWidgets.QComboBox()
        self.cmb_video.setMinimumWidth(160)
        self.cmb_video.setToolTip("Imaging data shown in the Imaging tab")
        self.cmb_video.activated.connect(self._on_video_selected)
        toolbar.addWidget(self.cmb_video)
        self._refresh_video_selector()

    def _set_widget_visible(self, widget: QtWidgets.QWidget, toggled: bool) -> None:
        if toggled:
            # Showing a panel that's currently only visible as a plain
            # (non-copy) detached window should bring it back into the
            # dock, not leave both visible - close the floating window
            # rather than leaving it stranded alongside the redocked panel.
            detached_window = self._plain_detached_windows.pop(widget, None)
            if detached_window is not None:
                detached_window.close()
        widget.setVisible(toggled)
        if toggled and widget is self.data_widget:
            self.set_data(self.cmb_data.currentText())

    def _auto_show_widget(self, widget: QtWidgets.QWidget) -> None:
        if widget is self.calcium_video_widget:
            self.spatial_footprint_widget.tabs.setCurrentWidget(widget)
        elif widget is self.spatial_footprint_widget:
            self.spatial_footprint_widget.tabs.setCurrentIndex(0)
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
                self.calcium_video_widget,
                self.calcium_video_widget.widget,
                self.calcium_video_widget.empty_state_widget,
                # Deliberately excludes overlay_view/overlay_view.widget: a
                # right click there is the panel's own lasso-select gesture
                # (see SpatialFootprintWidget), not a request to detach.
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
        if widget in (
            self.calcium_video_widget,
            self.calcium_video_widget.widget,
            self.calcium_video_widget.empty_state_widget,
        ):
            return self.spatial_footprint_widget
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
            if self._plain_detached_windows.get(source) is window:
                del self._plain_detached_windows[source]
            if (
                source is self.spatial_footprint_widget
                and isinstance(widget, SpatialFootprintWidget)
                and widget.calcium_video_widget in self.time_slider.widgets
            ):
                self.time_slider.widgets.remove(widget.calcium_video_widget)

        window = DetachedPanelWindow(
            widget,
            self._popup_title(self._panel_title(source)),
            on_close=_forget_window,
            parent=self,
        )
        if isinstance(widget, SpatialFootprintWidget):
            self.time_slider.connect_widget(widget.calcium_video_widget)
            widget.calcium_video_widget.set_time_ms(self.time_slider.t_ms)
        # A widget's own embedded "close" control (e.g. ConsoleWidget's
        # close_requested) should close the detached window it's actually
        # in, not reach back and toggle the original docked panel.
        close_requested = getattr(widget, "close_requested", None)
        if close_requested is not None:
            close_requested.connect(window.close)
        self._detached_panel_windows.append(window)
        window.show()
        if not keep_original:
            self._plain_detached_windows[source] = window
            action = self._view_actions.get(source)
            if action is not None and action.isChecked():
                action.trigger()

    def _build_fresh_trace_widget(self) -> TraceWidget:
        widget = TraceWidget(
            ConstrainedPanZoomController(),
            parent=None,
            default_controller=self.controller,
            show_axis_link_controls=True,
            state=self._trace_state,
        )
        widget.custom_analysis_requested.connect(self._custom_analysis)
        widget.neuron_hovered.connect(self._spatial_footprint_state.set_hovered_stim_id)
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
        widget.neuron_focus_requested.connect(self._focus_traces_on_neuron)
        widget.calcium_video_widget.import_requested.connect(
            self._on_calcium_video_import_requested
        )
        widget.calcium_video_widget.remove_requested.connect(
            self._on_calcium_video_remove_requested
        )
        widget.calcium_video_widget.channels_changed.connect(
            self._on_calcium_video_channels_changed
        )
        widget.calcium_video_widget.set_cellpose_data(self.project.cellpose_data)
        widget.calcium_video_widget.set_stim_arrays(self.project.stim_arrays)
        widget.calcium_video_widget.set_video_paths(
            self.project.calcium_video_paths,
            self.project.calcium_video_fps,
            scanimage_index=self.project.calcium_scanimage_index,
            channel=self._recorded_scanimage_channel(
                self.project.calcium_scanimage_index,
                self.scanimage_controls.channel_spin.value(),
            ),
            channels=self.project.calcium_video_channels[0],
            shown_channel=self.project.calcium_video_channels[1],
            warn_if_missing_cellpose=False,
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

    def _background_job_busy(self) -> bool:
        job = getattr(self, "_background_job", None)
        return job is not None and job.busy

    def _build_fresh_chat_widget(self) -> ChatWidget:
        widget = ChatWidget(parent=None)
        widget.set_project(self.project)
        widget._history = list(self.chat_widget._history)
        widget._render_history()
        widget.request_settings_dialog.connect(self.open_llm_settings)
        widget.set_busy_check(self._background_job_busy)
        return widget

    def _create_actions_settings(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        add_action("&Brian2 Simulation Backend", self.choose_brian2_backend, menu)
        add_action("&Input Automatic Padding", self.choose_stim_padding_secs, menu)
        add_action("&LLM Assistant", self.open_llm_settings, menu)
        add_action(
            "&Synapse Matrix RAM Limit", self.choose_max_ram_synapse_plot_matrix, menu
        )
        add_action("&Trace Plot RAM Limit", self.choose_max_ram_trace_plot_matrix, menu)
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
            ("&Retro light theme (beta)", theme_module.RETRO),
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

    def _create_actions_tools(self, menu: QtWidgets.QMenu) -> None:
        add_action(
            "&Open Fiji",
            self.btn_open_fiji_clicked,
            menu,
            icon="mdi6.microscope",
        )
        add_action(
            "Open &Cellpose",
            self.btn_open_cellpose_clicked,
            menu,
            icon="mdi6.dots-hexagon",
        )
        menu.addSeparator()
        add_action("&Change Fiji Path…", self.choose_fiji_path, menu)

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
        toolbar.addWidget(QtWidgets.QLabel("  Input data: "))
        toolbar.addWidget(self.cmb_stim)
        # The stim array ops button is inserted here once self.stim_editor
        # exists - see _add_stim_array_ops_button_to_data_toolbar().
        self._stim_array_ops_anchor_action = toolbar.addWidget(
            QtWidgets.QLabel("  Neuron data: ")
        )
        toolbar.addWidget(self.cmb_trace)
        self.action_overlay_stim_on_traces = add_action(
            "Overlay input on traces",
            self._toggle_stim_trace_overlay,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.layers-outline",
            checkable=True,
            is_checked=False,
        )
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("  Spike data: "))
        toolbar.addWidget(self.cmb_spike)
        toolbar.addWidget(QtWidgets.QLabel("  Synapse data: "))
        toolbar.addWidget(self.cmb_synapse)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("  All data: "))
        toolbar.addWidget(self.cmb_data)

    def _create_action_experiment(
        self, menu: QtWidgets.QMenu, toolbar: QtWidgets.QToolBar | None
    ) -> None:
        add_action(
            "&Connect External Input Function",
            self.connect_external_simulation_function,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.code-braces",
        )
        add_action(
            "&Connect Brian2 External Input Function",
            self.connect_external_simulation_function_with_brian2,
            menu,
            toolbar,
            shortcut=None,
            icon="mdi6.function",
        )
        self._do_external_simulation_action = add_action(
            "&Run External Input Function",
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
            "'opsin_inh' inputs."
        )
        self._refresh_elise_replay_action()

    def _refresh_elise_replay_action(self) -> None:
        """Show the ELiSe toolbar actions only for ELiSe projects.

        Neither button is relevant to any other project, so they're hidden
        entirely rather than merely disabled (issue #535) - enabled only
        once their specific inputs are actually ready.
        """
        is_elise_project = self.project.external_simulate_path == "neuroseq:elise"
        replay_action = getattr(self, "elise_replay_action", None)
        if replay_action is not None:
            replay_action.setVisible(is_elise_project)
            replay_action.setEnabled(self._elise_replay_inputs() is not None)
        protocol_action = getattr(self, "elise_protocol_action", None)
        if protocol_action is not None:
            protocol_action.setVisible(is_elise_project)
            protocol_action.setEnabled(
                self._elise_protocol_inputs() is not None
                and not self._background_job.busy
            )

    def _elise_replay_inputs(
        self,
    ) -> tuple[TraceData, np.ndarray, float] | None:
        return elise_workflows.replay_inputs(self.project)

    def _analyse_elise_replay(self) -> None:
        elise_workflows.analyse_replay(self)

    def _elise_protocol_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        return elise_workflows.protocol_inputs(self.project)

    def _run_elise_protocol(self) -> None:
        from neuroseq.elise_support.protocol_dialog import EliseProtocolDialog
        from neuroseq.example_models.elise import run_protocol as elise_run_protocol

        inputs = self._elise_protocol_inputs()
        if inputs is None:
            QtWidgets.QMessageBox.information(
                self,
                "Run ELiSe protocol",
                "This action requires an ELiSe project with 'opsin_exc' and "
                "'opsin_inh' input arrays.",
            )
            return
        if self._background_job.busy:
            QtWidgets.QMessageBox.warning(
                self,
                "Run ELiSe protocol",
                "Another background operation is already in progress.",
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

        cancel_event = threading.Event()
        self._prepare_background_job("Running ELiSe protocol...", cancellable=True)

        self._background_job.start(
            EliseProtocolWorker(
                run_fn=elise_run_protocol,
                kwargs=kwargs,
                cancel_event=cancel_event,
            ),
            on_progress=self._on_elise_protocol_progress,
            on_finished=self._on_elise_protocol_finished,
            on_failed=self._on_elise_protocol_failed,
            on_cancelled=self._on_elise_protocol_cancelled,
            cancel=cancel_event.set,
        )

    def _teardown_elise_protocol(self) -> None:
        self._teardown_background_job()

    @QtCore.Slot(float)
    def _on_elise_protocol_progress(self, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.progress_dialog.setValue(int(round(clamped * 100)))

    def _update_elise_stim_arrays(self, result: dict) -> None:
        elise_workflows.update_stim_arrays(self, result)

    @QtCore.Slot(object)
    def _on_elise_protocol_finished(self, result: dict) -> None:
        self._teardown_elise_protocol()
        self.progress_dialog.setValue(100)
        self.progress_dialog.reset()
        elise_workflows.finish_protocol(self, result)

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

    def _prepare_background_job(
        self, label: str, cancellable: bool, *, block_io: bool = False
    ) -> None:
        self._background_job_label = label
        self.progress_dialog.setLabelText(label)
        self.progress_dialog.setRange(0, 100)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        if cancellable:
            self.progress_dialog.setCancelButtonText("Cancel")
            self.progress_dialog.canceled.connect(self._cancel_background_job)
        else:
            self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.progress_dialog.show()
        self.progress_dialog.raise_()
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        if block_io:
            self._set_io_actions_enabled(False)
        else:
            self._set_simulation_actions_enabled(False)
        if getattr(self, "elise_protocol_action", None) is not None:
            self.elise_protocol_action.setEnabled(False)

    def _cancel_background_job(self) -> None:
        if not self._background_job.busy or not self._background_job.cancellable:
            return
        self._background_job.cancel()
        self.progress_dialog.setLabelText("Cancelling...")
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

    def _teardown_background_job(self) -> None:
        SimulationGuard.release(self)
        self.unsetCursor()
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setMinimumDuration(10)
        with contextlib.suppress(TypeError, RuntimeError):
            self.progress_dialog.canceled.disconnect()
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.reset()
        self._set_io_actions_enabled(True)
        self._refresh_elise_replay_action()

    @staticmethod
    def _dropped_paths(
        event: QtGui.QDropEvent | QtGui.QDragEnterEvent,
    ) -> list[pathlib.Path]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return [
            pathlib.Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()
        ]

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        if event is None:
            return
        if self._dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent | None) -> None:
        if event is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if event is None:
            return
        paths = self._dropped_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        target = self._drop_target(event.position().toPoint())
        # After the drop returns, so the file explorer is not left waiting
        # while an import dialog is open.
        QtCore.QTimer.singleShot(0, lambda: self.import_dropped_files(paths, target))

    def _drop_target(self, pos: QtCore.QPoint) -> str | None:
        """The kind of panel under ``pos``: "segmentation", "imaging",
        "input", "traces", or None elsewhere."""
        widget = self.childAt(pos)
        while widget is not None:
            if widget is self.spatial_footprint_widget:
                return (
                    "imaging"
                    if self.spatial_footprint_widget.tabs.currentIndex() == 1
                    else "segmentation"
                )
            if widget is self.stim_editor:
                return "input"
            if widget in (self.trace_widget, self.spike_widget, self.data_widget):
                return "traces"
            widget = widget.parentWidget()
        return None

    def import_dropped_files(
        self, paths: list[pathlib.Path], target: str | None = None
    ) -> None:
        """Import files dropped onto the window (issue #597), choosing the
        import from each file's type and, where a type fits several, from the
        panel it was dropped on (``target``, see _drop_target). Each import
        opens its usual dialog with the file already loaded, so the data can
        be named and checked before it is added."""
        images = [p for p in paths if p.suffix.lower() in DROP_IMAGE_SUFFIXES]
        unsupported = []
        for path in paths:
            suffix = path.suffix.lower()
            if suffix == ".nseq":
                self.load_project(path)
            elif suffix == ".seq":
                self.btn_import_syscon2_seq_clicked(path=path)
            elif suffix in {".ibw", ".pxp", ".csv", ".txt"}:
                csv_only = suffix in {".csv", ".txt"}
                if target == "input":
                    self.btn_import_igor2_wave_clicked(csv_only=csv_only, path=path)
                else:
                    self.btn_import_igor2_trace_clicked(csv_only=csv_only, path=path)
            elif suffix not in DROP_IMAGE_SUFFIXES:
                unsupported.append(path.name)
        if images:
            self._import_dropped_images(images, target)
        if unsupported:
            QtWidgets.QMessageBox.information(
                self,
                "Cannot import these files",
                "NeuroSeq does not know how to import:\n"
                + "\n".join(unsupported)
                + "\n\nDrop projects (.nseq), SysCon2 sequences (.seq), "
                "patch-clamp waves (.ibw, .pxp, .csv, .txt), or imaging data "
                "(TIFF, PNG, GIF, movies).",
            )

    def _import_dropped_images(
        self, paths: list[pathlib.Path], target: str | None
    ) -> None:
        segmentable = [
            p for p in paths if p.suffix.lower() in {".tif", ".tiff", ".png"}
        ]
        tiffs = [p for p in paths if p.suffix.lower() in {".tif", ".tiff"}]
        movies = [p for p in paths if p.suffix.lower() != ".png"]
        choices = {
            DROP_SEGMENT: bool(segmentable),
            DROP_TRACES: bool(tiffs),
            DROP_VIDEO: bool(movies),
        }
        action = {
            "segmentation": DROP_SEGMENT,
            "traces": DROP_TRACES,
            "imaging": DROP_VIDEO,
        }.get(target or "")
        if action is None or not choices[action]:
            options = [name for name, possible in choices.items() if possible]
            if not options:
                return
            action, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Import imaging data",
                f"What should NeuroSeq do with {paths[0].name}"
                + (f" and {len(paths) - 1} more" if len(paths) > 1 else "")
                + "?",
                options,
                0,
                False,
            )
            if not ok:
                return
        if action == DROP_SEGMENT:
            from neuroseq.cellpose_annotation import CELLPOSE_MODEL_NAMES

            model, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Segment cells",
                f"Cellpose model for {segmentable[0].name}:",
                list(CELLPOSE_MODEL_NAMES),
                0,
                False,
            )
            if ok:
                self.btn_import_cellpose_data_clicked(
                    segment=segmentable[0], model=model
                )
        elif action == DROP_TRACES:
            self.btn_import_tif_clicked(paths=tiffs)
        else:
            self._on_calcium_video_import_requested(paths=movies)

    def close_blockers(self) -> list[str]:
        """Describe work that must finish before this workspace can close.

        Non-destructive: nothing is cancelled or stopped. Cancellable
        background jobs are not blockers; closing cancels them.
        """
        blockers = []
        if self.acquisition_controller.acquisition_in_progress:
            blockers.append(
                "An image acquisition is in progress. Finish or abort it first."
            )
        if self.syscon2_panel.execution_in_progress:
            blockers.append(
                "A HOLO4D sequence upload or run is in progress. Wait for it to "
                "finish or press Stop in the stimulation setup."
            )
        if self._background_job.busy and not self._background_job.cancellable:
            label = self._background_job_label.rstrip(".…") or "A background task"
            blockers.append(
                f"{label} is running and cannot be cancelled. Wait for it to finish."
            )
        return blockers

    def stop_background_job(self) -> bool:
        """Stop remaining work synchronously before this workspace is disposed.

        Callers first pass the close gate (see ``WorkspaceCloseGate``), so a
        background job normally is no longer running here.
        """
        for chat in self.findChildren(ChatWidget):
            if not chat.stop_worker():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Close",
                    "The chat is still stopping; this project stays open. "
                    "Try again in a moment.",
                )
                return False
        if not self._background_job.busy:
            return True
        self._cancel_background_job()
        if not self._background_job.stop(timeout_ms=CLOSE_JOB_TIMEOUT_MS):
            QtWidgets.QMessageBox.warning(
                self,
                "Close",
                "A background job is still stopping; this project stays open. "
                "Try again once it has finished.",
            )
            return False
        self._teardown_background_job()
        return True

    def dispose(self) -> None:
        """Release resources after background jobs have stopped."""
        if self._disposed:
            return
        self._disposed = True
        SimulationGuard.release(self)
        if self._unsubscribe_project_history is not None:
            self._unsubscribe_project_history()
            self._unsubscribe_project_history = None
        self.stim_editor.disconnect_history()
        self.set_active(False)
        self.syscon2_panel.shutdown()
        self.acquisition_controller.dispose()
        if self._owns_devices:
            self.devices.shutdown()
        self.time_slider.close()
        for widget in self.plot_widgets:
            widget.close()
        self.spatial_footprint_widget.close()
        for analysis_window in self.findChildren(CustomAnalysisWindow):
            analysis_window.close()
        for detached_window in self.findChildren(DetachedPanelWindow):
            detached_window.close()
        self.static_input_controller.discard_dialog()
        self.dynamic_input_controller.discard_dynamic_input_dialogs()
        if self._llm_settings_dialog is not None:
            self._llm_settings_dialog.close()

    def closeEvent(self, event):
        if self._disposed:
            super().closeEvent(event)
            return
        if not self._close_gate.ready([self], "Close NeuroSeq", self.close):
            self._close_confirmed = False
            event.ignore()
            return
        if not self._close_confirmed and not self._prompt_save_before_close():
            event.ignore()
            return
        if not self.stop_background_job():
            self._close_confirmed = False
            event.ignore()
            return
        self.dispose()
        super().closeEvent(event)
