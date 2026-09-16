from __future__ import annotations

from math import copysign
from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
from matplotlib.ticker import MultipleLocator
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.input_designer_dialog import EPSPMeasurement
from neuroseq.input_designer_dialog import StimHeatmapWindow
from neuroseq.input_designer_dialog import _set_native_minimized_position
from neuroseq.input_designer_dialog import _WheelToScrollFilter
from neuroseq.resources.analysis_plugins.epsp_amplitudes import _to_mv
from neuroseq.resources.analysis_plugins.epsp_amplitudes import detect_epsps
from neuroseq.stdp_window import classify_window
from neuroseq.stdp_window import delta_t_sweep
from neuroseq.stdp_window import interleave_by_magnitude
from neuroseq.stdp_window import percent_change
from neuroseq.stim_timing import samples_to_milliseconds
from neuroseq.stim_timing import snap_milliseconds_to_dt

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray
    from neuroseq.stim_editor import StimEditor


class _NoEpspDetected(ValueError):
    """No EPSP could be detected in a probe response window.

    Distinct from other measurement ``ValueError``s (missing trace row, trace
    too short) so the sweep can treat an undetectable EPSP as a skipped point
    instead of a fatal error.
    """


class StdpWindowDialog(QtWidgets.QDialog):
    """Sweep spike timing to measure the functional STDP timing window."""

    simulation_requested = QtCore.Signal(float)
    PROBE_PULSES = 5
    PROBE_PERIOD_MS = 100.0
    # Readout modes. EPSP is functional (needs a clean postsynaptic EPSP);
    # weight reads the synapse weight directly and is immune to shunting /
    # feedback inhibition, so it works on recurrent networks.
    EPSP_READOUT = "EPSP amplitude change (%)"
    WEIGHT_READOUT = "Synaptic weight change (%)"
    # Sweep modes. Reset restores a baseline checkpoint before each Δt so every
    # point measures one pairing from the identical state (the clean textbook
    # window). Continuous keeps one running synapse (shows cumulative drift) and
    # visits Δt interleaved by magnitude to stay near baseline.
    RESET_MODE = "Reset to baseline each Δt (independent points)"
    CONTINUOUS_MODE = "Continuous sweep (interleaved, no reset)"

    def __init__(
        self,
        editor: StimEditor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, True)
        self.editor = editor
        self.project = editor.project
        self._running = False
        self._awaiting_simulation = False
        self._phase = "idle"
        self._delta_ts: list[float] = []
        self._index = 0
        self._pairing_repeat = 0
        self._baseline_epsp: float | None = None
        self._last_epsp: float | None = None
        self._pending_delta_t = 0.0
        self._results: list[tuple[float, float]] = []
        self._epsp_trajectory: list[float] = []
        self._probe_pulse_times: list[float] = []
        self._view_block_samples: int | None = None
        self._checkpoint: str | None = None
        self._baseline_sample = 0
        self._point_start = 0
        self._theme = theme_module.LIGHT
        self._expanded_height: int | None = None
        self._compact_height: int | None = None
        self._plot_window_positioned = False
        self._saturation_warned = False
        self._resolution_warning_logged = False

        self.setWindowTitle("STDP timing window (LTP/LTD vs Δt)")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        info_form = QtWidgets.QFormLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_stim_units = QtWidgets.QLabel("-", self)
        self.lbl_stim_dt = QtWidgets.QLabel("-", self)
        self.lbl_stim_neurons = QtWidgets.QLabel("-", self)
        info_form.addRow("Input array:", self.lbl_stim_name)
        info_form.addRow("Units:", self.lbl_stim_units)
        info_form.addRow("dt:", self.lbl_stim_dt)
        info_form.addRow("Neurons:", self.lbl_stim_neurons)
        layout.addLayout(info_form)

        self.btn_toggle_parameters = QtWidgets.QToolButton(self)
        self.btn_toggle_parameters.setText("Hide parameters")
        self.btn_toggle_parameters.setCheckable(True)
        self.btn_toggle_parameters.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_toggle_parameters.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_toggle_parameters.setToolTip(
            "Hide the protocol parameters to keep this window compact."
        )
        self.btn_toggle_parameters.toggled.connect(self._set_parameters_hidden)
        layout.addWidget(self.btn_toggle_parameters)

        self.parameters_panel = QtWidgets.QWidget(self)
        parameters_layout = QtWidgets.QVBoxLayout(self.parameters_panel)
        parameters_layout.setContentsMargins(0, 0, 0, 0)
        parameters_layout.setSpacing(6)
        note = QtWidgets.QLabel(
            "Sweep pre/post spike timing and measure the synaptic change per "
            "pairing. Use a suprathreshold pulse amplitude (the rheobase routine "
            "can estimate one). Reset mode measures each Δt independently from the "
            "baseline weight — the clean textbook window; continuous mode keeps "
            "one running synapse and shows cumulative drift. The EPSP readout "
            "needs a clean postsynaptic EPSP (isolated pair, e.g. the two-neuron "
            "plasticity example); on recurrent networks use the weight readout.",
            self.parameters_panel,
        )
        note.setWordWrap(True)
        parameters_layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.cmb_readout = QtWidgets.QComboBox(self)
        self.cmb_readout.addItems([self.EPSP_READOUT, self.WEIGHT_READOUT])
        self.cmb_sweep_mode = QtWidgets.QComboBox(self)
        self.cmb_sweep_mode.addItems([self.RESET_MODE, self.CONTINUOUS_MODE])
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.cmb_synapse = QtWidgets.QComboBox(self)
        self.spn_presynaptic = QtWidgets.QSpinBox(self)
        self.spn_presynaptic.setMinimum(0)
        self.spn_presynaptic.setValue(0)
        self.spn_postsynaptic = QtWidgets.QSpinBox(self)
        self.spn_postsynaptic.setMinimum(0)
        self.spn_postsynaptic.setValue(1)
        self.spn_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_amplitude.setDecimals(3)
        self.spn_amplitude.setRange(0.001, 1e12)
        self.spn_amplitude.setValue(100.0)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_amplitude)
        amplitude_row.addWidget(self.lbl_units)
        self.spn_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pulse_ms.setDecimals(3)
        self.spn_pulse_ms.setRange(0.001, 1000.0)
        self.spn_pulse_ms.setValue(5.0)
        self.spn_pulse_ms.setSuffix(" ms")
        self.spn_dt_min_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_dt_min_ms.setDecimals(3)
        self.spn_dt_min_ms.setRange(-1000.0, 1000.0)
        self.spn_dt_min_ms.setValue(-100.0)
        self.spn_dt_min_ms.setSuffix(" ms")
        self.spn_dt_max_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_dt_max_ms.setDecimals(3)
        self.spn_dt_max_ms.setRange(-1000.0, 1000.0)
        self.spn_dt_max_ms.setValue(100.0)
        self.spn_dt_max_ms.setSuffix(" ms")
        self.spn_dt_points = QtWidgets.QSpinBox(self)
        self.spn_dt_points.setRange(2, 500)
        self.spn_dt_points.setValue(21)
        self.chk_include_zero = QtWidgets.QCheckBox(self)
        self.chk_include_zero.setChecked(False)
        self.spn_pairings_per_point = QtWidgets.QSpinBox(self)
        self.spn_pairings_per_point.setRange(1, 100)
        self.spn_pairings_per_point.setValue(1)
        self.spn_recovery_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_recovery_ms.setDecimals(3)
        self.spn_recovery_ms.setRange(0.001, 100_000.0)
        self.spn_recovery_ms.setValue(150.0)
        self.spn_recovery_ms.setSuffix(" ms")
        self.spn_response_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_response_ms.setDecimals(3)
        self.spn_response_ms.setRange(0.001, 100_000.0)
        self.spn_response_ms.setValue(50.0)
        self.spn_response_ms.setSuffix(" ms")
        self.chk_follow_input = QtWidgets.QCheckBox(
            "Follow new input on each iteration", self
        )
        self.chk_follow_input.setChecked(True)
        self.spn_history_blocks = QtWidgets.QSpinBox(self)
        self.spn_history_blocks.setRange(0, 100)
        self.spn_history_blocks.setValue(3)
        self.spn_history_blocks.setSuffix(" blocks")

        form.addRow("Readout:", self.cmb_readout)
        form.addRow("Sweep mode:", self.cmb_sweep_mode)
        self.lbl_trace = QtWidgets.QLabel("Trace to analyze:", self)
        form.addRow(self.lbl_trace, self.cmb_trace)
        self.lbl_synapse = QtWidgets.QLabel("Synapse weight:", self)
        form.addRow(self.lbl_synapse, self.cmb_synapse)
        form.addRow("Presynaptic neuron ID:", self.spn_presynaptic)
        form.addRow("Postsynaptic neuron ID:", self.spn_postsynaptic)
        form.addRow("Pulse amplitude:", amplitude_row)
        form.addRow("Pulse duration (ms):", self.spn_pulse_ms)
        form.addRow("Δt minimum (ms):", self.spn_dt_min_ms)
        form.addRow("Δt maximum (ms):", self.spn_dt_max_ms)
        form.addRow("Number of Δt points:", self.spn_dt_points)
        form.addRow("Include Δt = 0:", self.chk_include_zero)
        form.addRow("Pairings per point:", self.spn_pairings_per_point)
        form.addRow("Recovery time (ms):", self.spn_recovery_ms)
        self.lbl_response = QtWidgets.QLabel("Post-pulse window (ms):", self)
        form.addRow(self.lbl_response, self.spn_response_ms)
        form.addRow("Live view:", self.chk_follow_input)
        form.addRow("Visible history:", self.spn_history_blocks)
        parameters_layout.addLayout(form)

        self.parameters_scroll = QtWidgets.QScrollArea(self)
        self.parameters_scroll.setWidget(self.parameters_panel)
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameters_scroll.setMinimumHeight(80)
        layout.addWidget(self.parameters_scroll, 1)
        self._wheel_filter = _WheelToScrollFilter(self.parameters_scroll)
        value_widgets: list[QtWidgets.QWidget] = list(
            self.parameters_panel.findChildren(QtWidgets.QAbstractSpinBox)
        )
        value_widgets += self.parameters_panel.findChildren(QtWidgets.QComboBox)
        for value_widget in value_widgets:
            value_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            value_widget.installEventFilter(self._wheel_filter)

        for widget in (
            self.spn_dt_min_ms,
            self.spn_dt_max_ms,
            self.spn_dt_points,
            self.chk_include_zero,
            self.spn_pairings_per_point,
        ):
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(lambda _value: self._update_estimate())
            else:
                widget.valueChanged.connect(lambda _value: self._update_estimate())
        self.cmb_readout.currentIndexChanged.connect(self._on_readout_changed)
        self.cmb_sweep_mode.currentIndexChanged.connect(
            lambda _index: self._update_estimate()
        )

        self.plot_window = QtWidgets.QDialog(self)
        self.plot_window.setWindowTitle("STDP timing window (LTP/LTD vs Δt) — plots")
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.window_plot = MatplotlibWidget(self.plot_window)
        self.window_plot.setMinimumHeight(500)
        figure = self.window_plot.getFigure()
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.18)
        self.window_axes = figure.add_subplot(grid[0, 0])
        self.trajectory_axes = figure.add_subplot(grid[1, 0])
        plot_layout.addWidget(self.window_plot)
        self.plot_window.resize(720, 560)

        # A third popup: a live, whole-history heatmap of the input pattern
        # (neuron ID x time) - see StimHeatmapWindow's docstring for why.
        self.stim_heatmap_window = StimHeatmapWindow(
            "STDP timing window (LTP/LTD vs Δt) — stimulation pattern", self
        )
        self._stim_heatmap_window_positioned = False

        self.lbl_status = QtWidgets.QLabel("STDP window: waiting for baseline", self)
        layout.addWidget(self.lbl_status)
        self.lbl_estimate = QtWidgets.QLabel("", self)
        layout.addWidget(self.lbl_estimate)
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setMaximumHeight(130)
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.btn_start = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Start", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.btn_stop = cast(
            QtWidgets.QPushButton,
            buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole),
        )
        self.btn_clear = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Clear input array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_reset = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Reset input array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_reset.setToolTip(
            "Trim the input array back to the simulation head so the routine does "
            "not run a long empty tail."
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self.editor.clear_stim_array)
        self.btn_reset.clicked.connect(self.editor.reset_stim_array)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.apply_theme(self._theme)
        self.refresh()
        self._update_readout_controls()
        self._update_estimate()

    def _set_parameters_hidden(self, hidden: bool) -> None:
        if hidden:
            self._expanded_height = self.height()
            panel_height = self.parameters_scroll.height()
            dialog_layout = self.layout()
            spacing = dialog_layout.spacing() if dialog_layout is not None else 0
            self._compact_height = max(1, self.height() - panel_height - spacing)
        self.parameters_scroll.setVisible(not hidden)
        self.btn_toggle_parameters.setText(
            "Show parameters" if hidden else "Hide parameters"
        )
        self.btn_toggle_parameters.setArrowType(
            QtCore.Qt.ArrowType.RightArrow if hidden else QtCore.Qt.ArrowType.DownArrow
        )
        QtCore.QTimer.singleShot(0, self._resize_for_parameter_visibility)

    def _resize_for_parameter_visibility(self) -> None:
        height = (
            self._compact_height
            if self.btn_toggle_parameters.isChecked()
            else self._expanded_height
        )
        if height is not None:
            self.resize(self.width(), max(height, self.minimumSizeHint().height()))

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self.plot_window.show()
        if not self._plot_window_positioned:
            self._plot_window_positioned = True
            self.plot_window.move(self.x() + self.width() + 20, self.y())
        self.plot_window.raise_()
        self.stim_heatmap_window.show()
        if not self._stim_heatmap_window_positioned:
            self._stim_heatmap_window_positioned = True
            self.stim_heatmap_window.position_below(self.plot_window)
        self.stim_heatmap_window.raise_()
        self._update_stim_heatmap()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if self._running:
            self._finish("Routine stopped (dialog closed).")
        self.plot_window.close()
        self.stim_heatmap_window.close()
        super().closeEvent(event)

    def _update_stim_heatmap(self) -> None:
        modifier = self.editor.current_stim_modifier
        stim = modifier.stim_array if modifier is not None else None
        self.stim_heatmap_window.update_from_stim(stim)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.WindowStateChange
            and self.isMinimized()
        ):
            QtCore.QTimer.singleShot(0, self._position_minimized_window)

    def _position_minimized_window(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        margin = 12
        available = screen.availableGeometry()
        minimized_height = 32
        _set_native_minimized_position(
            int(self.winId()),
            available.left() + margin,
            available.bottom() - minimized_height - margin + 1,
        )

    def refresh(self) -> None:
        self.project = self.editor.project
        selected = self.cmb_trace.currentText()
        self.cmb_trace.clear()
        if self.project is not None:
            trace_names = self.project.get_data_names(TraceData)
            self.cmb_trace.addItems(trace_names)
            preferred_trace = self._preferred_trace_name(trace_names, selected)
            if preferred_trace:
                self.cmb_trace.setCurrentText(preferred_trace)
        selected_synapse = self.cmb_synapse.currentText()
        self.cmb_synapse.clear()
        if self.project is not None:
            synapse_names = self.project.get_data_names(SynapseData)
            self.cmb_synapse.addItems(synapse_names)
            if selected_synapse in synapse_names:
                self.cmb_synapse.setCurrentText(selected_synapse)
        modifier = self.editor.current_stim_modifier
        if modifier is not None:
            stim = modifier.stim_array
            n_neurons = stim.values.shape[1]
            maximum = max(0, n_neurons - 1)
            self.spn_presynaptic.setMaximum(maximum)
            self.spn_postsynaptic.setMaximum(maximum)
            self.lbl_units.setText(stim.units)
            self.lbl_stim_name.setText(stim.name)
            self.lbl_stim_units.setText(stim.units or "-")
            self.lbl_stim_dt.setText(f"{stim.dt * 1000.0:g} ms")
            self.lbl_stim_neurons.setText(str(n_neurons))
        else:
            self.lbl_units.setText("")
            self.lbl_stim_name.setText("-")
            self.lbl_stim_units.setText("-")
            self.lbl_stim_dt.setText("-")
            self.lbl_stim_neurons.setText("-")
        self._update_estimate()
        self._update_plot()

    def _preferred_trace_name(
        self, trace_names: list[str], previously_selected: str
    ) -> str:
        if not trace_names:
            return ""
        if self._is_simulation_voltage_trace(previously_selected):
            return previously_selected
        for trace_name in trace_names:
            if self._is_simulation_voltage_trace(trace_name):
                return trace_name
        if previously_selected in trace_names:
            return previously_selected
        return trace_names[0]

    def _is_simulation_voltage_trace(self, trace_name: str) -> bool:
        if self.project is None or not trace_name:
            return False
        if not trace_name.startswith("Simulation :: ") or not trace_name.endswith(
            " / v"
        ):
            return False
        trace = self.project.get_data(trace_name, TraceData)
        if trace is None:
            return False
        return bool(np.any(trace.indices == self.spn_postsynaptic.value()))

    def _readout(self) -> str:
        return self.cmb_readout.currentText()

    def _weight_mode(self) -> bool:
        return self._readout() == self.WEIGHT_READOUT

    def _reset_mode(self) -> bool:
        return self.cmb_sweep_mode.currentText() == self.RESET_MODE

    def _on_readout_changed(self, _index: int) -> None:
        self._update_readout_controls()

    def _update_readout_controls(self) -> None:
        weight = self._weight_mode()
        self.lbl_trace.setVisible(not weight)
        self.cmb_trace.setVisible(not weight)
        self.lbl_response.setVisible(not weight)
        self.spn_response_ms.setVisible(not weight)
        self.lbl_synapse.setVisible(weight)
        self.cmb_synapse.setVisible(weight)
        self._update_estimate()
        self._update_plot()

    def _read_synapse_weight(self) -> float | None:
        """Return the current weight of the pre→post synapse, or None."""
        name = self.cmb_synapse.currentText()
        if not name or self.project is None:
            return None
        data = self.project.get_data(name, SynapseData)
        if data is None or data.values.size == 0:
            return None
        pre = self.spn_presynaptic.value()
        post = self.spn_postsynaptic.value()
        values = data.values
        if pre >= values.shape[0] or post >= values.shape[1]:
            return None
        series = values[pre, post]
        if series.size == 0:
            return None
        return float(series[-1])

    def start(self) -> None:
        self.refresh()
        if self.project is None or self.editor.current_stim_modifier is None:
            self._append_log("No input array is selected.")
            return
        if not self._confirm_start_over_existing_input():
            self._append_log("Routine cancelled; existing input kept.")
            return
        # Start from the simulation head so a long left-over stim array does not
        # make the routine simulate a huge empty tail.
        self.editor.reset_stim_array()
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._finish("Pre- and postsynaptic neuron IDs must differ.")
            return
        if self.spn_dt_max_ms.value() <= self.spn_dt_min_ms.value():
            self._finish("Δt maximum must be greater than Δt minimum.")
            return
        if not self._current_delta_t_values():
            self._finish("No Δt points to sweep.")
            return
        if self._weight_mode():
            needs_init = self._read_synapse_weight() is None
        else:
            needs_init = not self._is_simulation_voltage_trace(
                self.cmb_trace.currentText()
            )
        if needs_init:
            self._running = True
            self._phase = (
                "initialize_weight" if self._weight_mode() else "initialize_trace"
            )
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._append_log(
                "Running an initial 0.001 s simulation to create monitors."
            )
            self._request_simulation(0.001)
            return
        self._begin_sweep(clear_log=True)

    def _begin_sweep(self, *, clear_log: bool) -> None:
        self._delta_ts = self._current_delta_t_values()
        if not self._delta_ts:
            self._finish("No Δt points to sweep.")
            return
        self._running = True
        self._awaiting_simulation = False
        self._phase = "idle"
        self._index = 0
        self._pairing_repeat = 0
        # These hold the generic per-point measurement (EPSP amplitude in EPSP
        # mode, synaptic weight in weight mode).
        self._baseline_epsp = None
        self._last_epsp = None
        self._pending_delta_t = 0.0
        self._results = []
        self._epsp_trajectory = []
        self._probe_pulse_times = []
        self._view_block_samples = None
        self._saturation_warned = False
        self._resolution_warning_logged = False
        self._checkpoint = None
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if clear_log:
            self.log.clear()
        self._append_log(
            f"Sweeping {len(self._delta_ts)} Δt points; estimated simulations "
            f"~{self._estimated_probe_count()}."
        )
        self._maybe_log_delta_t_resolution_warning()
        self._append_log(
            "Pulse amplitude must evoke pre/post spikes during pairings; use "
            "rheobase first if needed."
        )
        self._update_plot()
        if self._reset_mode():
            stim = self._require_stim()
            project = self._require_project()
            self._baseline_sample = int(
                np.ceil(project.b2_model.get_sim_time_secs() / stim.dt)
            )
            self._point_start = self._baseline_sample
            self._checkpoint = f"STDP window baseline {id(self)}"
            project.b2_model.store(self._checkpoint)
        if self._weight_mode():
            weight = self._read_synapse_weight()
            if weight is None:
                self._finish("Could not read the baseline synaptic weight.")
                return
            self._baseline_epsp = weight
            self._last_epsp = weight
            self._epsp_trajectory.append(weight)
            self._append_log(f"Baseline synaptic weight: {weight:.6g}.")
            self._update_plot()
            self._begin_point()
        else:
            self._phase = "baseline_probe"
            self._append_probe()

    def stop(self) -> None:
        self._finish("Routine stopped.")

    def _require_stim(self) -> StimArray:
        modifier = self.editor.current_stim_modifier
        if modifier is None:
            raise RuntimeError("No input array is selected.")
        return modifier.stim_array

    def _require_project(self) -> Project:
        if self.project is None:
            raise RuntimeError("No project is available.")
        return self.project

    def _append_probe(self) -> None:
        stim = self._require_stim()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(self.spn_pulse_ms.value(), stim.dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(
            self.spn_recovery_ms.value(), stim.dt
        )
        period_n, _period_ms = snap_milliseconds_to_dt(
            self.PROBE_PERIOD_MS, stim.dt, minimum_samples=pulse_n + 1
        )
        tail_n, _tail_ms = snap_milliseconds_to_dt(
            self.spn_response_ms.value(), stim.dt
        )
        samples = lead_n + (self.PROBE_PULSES - 1) * period_n + pulse_n + tail_n
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        pulse_offsets = [
            lead_n + index * period_n for index in range(self.PROBE_PULSES)
        ]
        for offset in pulse_offsets:
            block[offset : offset + pulse_n, self.spn_presynaptic.value()] = (
                self.spn_amplitude.value()
            )
        block_start = self._place_block(block)
        self._probe_pulse_times = [
            (block_start + offset) * stim.dt for offset in pulse_offsets
        ]
        self._append_log("Appended five presynaptic probe pulses.")
        self._run_to_sample(block_start + samples)

    def _append_pairing(self, delta_t_ms: float) -> None:
        stim = self._require_stim()
        pulse_n, _pulse_ms = snap_milliseconds_to_dt(self.spn_pulse_ms.value(), stim.dt)
        lead_n, _lead_ms = snap_milliseconds_to_dt(
            self.spn_recovery_ms.value(), stim.dt
        )
        gap_n, gap_ms = snap_milliseconds_to_dt(abs(delta_t_ms), stim.dt)
        self._pending_delta_t = copysign(gap_ms, delta_t_ms)
        delta_t_ms = self._pending_delta_t
        tail_n, _tail_ms = snap_milliseconds_to_dt(
            self.spn_response_ms.value(), stim.dt
        )
        samples = lead_n + gap_n + pulse_n + tail_n
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        pre_before_post = delta_t_ms >= 0.0
        first = (
            self.spn_presynaptic.value()
            if pre_before_post
            else self.spn_postsynaptic.value()
        )
        second = (
            self.spn_postsynaptic.value()
            if pre_before_post
            else self.spn_presynaptic.value()
        )
        block[lead_n : lead_n + pulse_n, first] = self.spn_amplitude.value()
        block[lead_n + gap_n : lead_n + gap_n + pulse_n, second] = (
            self.spn_amplitude.value()
        )
        block_start = self._place_block(block)
        self._run_to_sample(block_start + samples)

    def _restore_baseline(self) -> None:
        """Reset mode: rewind the network to the baseline checkpoint."""
        if self._checkpoint is not None:
            self._require_project().b2_model.restore(self._checkpoint)
        self._point_start = self._baseline_sample

    def _place_block(self, block: np.ndarray) -> int:
        stim = self._require_stim()
        project = self._require_project()
        if self._reset_mode():
            # Overwrite a fixed region: the clock was rewound by _restore_baseline,
            # and blocks within a point are laid down consecutively from there.
            start = self._point_start
            self._point_start = start + block.shape[0]
        else:
            current_sample = int(
                np.ceil(project.b2_model.get_sim_time_secs() / stim.dt)
            )
            start = max(stim.values.shape[0], current_sample)
        self.editor.apply_pattern(
            slice(0, stim.values.shape[1]), slice(start, start + block.shape[0]), block
        )
        if self.chk_follow_input.isChecked():
            if self._view_block_samples is None:
                self._view_block_samples = block.shape[0]
            visible_blocks = max(1, self.spn_history_blocks.value())
            visible_samples = visible_blocks * self._view_block_samples
            block_end = start + block.shape[0]
            history_start = max(0, block_end - visible_samples)
            history_stop = history_start + visible_samples
            self.editor.focus_time_range(
                history_start * stim.dt, history_stop * stim.dt
            )
        self._update_stim_heatmap()
        return start

    def _run_to_sample(self, stop_sample: int) -> None:
        stim = self._require_stim()
        current_time = self._require_project().b2_model.get_sim_time_secs()
        run_seconds = max(stim.dt, stop_sample * stim.dt - current_time)
        self._request_simulation(run_seconds)

    def _request_simulation(self, seconds: float) -> None:
        self._awaiting_simulation = True
        self.simulation_requested.emit(seconds)

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if not self._awaiting_simulation:
            return
        self._awaiting_simulation = False
        if self._phase in ("initialize_trace", "initialize_weight"):
            self.refresh()
            if self._weight_mode():
                if not self.cmb_synapse.currentText():
                    self._finish(
                        "The initial simulation completed, but no synapse weight "
                        "monitor became available. Add a StateMonitor on the "
                        "synapse weight to the model and try again."
                    )
                    return
                self._append_log(
                    "Synapse monitor created; starting the STDP window sweep."
                )
            else:
                if not self._is_simulation_voltage_trace(self.cmb_trace.currentText()):
                    self._finish(
                        "The initial simulation completed, but no simulation "
                        "voltage trace for the postsynaptic neuron became "
                        "available. Add a StateMonitor for v to the model and "
                        "try again."
                    )
                    return
                self._append_log(
                    "Trace monitor created; starting the STDP window sweep."
                )
            self._begin_sweep(clear_log=False)
            return
        if self._phase == "pair":
            if self._pairing_repeat < self.spn_pairings_per_point.value():
                self._apply_next_pairing()
            elif self._weight_mode():
                self._record_weight_point()
            else:
                self._phase = "probe"
                self._append_probe()
            return

        self.refresh()
        trace_name = self.cmb_trace.currentText()
        if not self._is_simulation_voltage_trace(trace_name):
            self._finish(
                "The routine could not find a simulation voltage trace for the "
                "postsynaptic neuron after simulation."
            )
            return
        trace = self.project.get_data(trace_name, TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return
        try:
            measurement: EPSPMeasurement | None = self._measure_epsps(trace)
        except _NoEpspDetected:
            measurement = None
        except ValueError as exc:
            self._finish(f"EPSP measurement failed: {exc}")
            return

        if self._phase == "baseline_probe":
            if measurement is None or measurement.amplitude <= 0.0:
                self._finish(
                    "No baseline EPSP could be measured; the postsynaptic "
                    "response may be shunted by network activity. Try the "
                    "'Synaptic weight' readout or an isolated-synapse project."
                )
                return
            self._baseline_epsp = measurement.amplitude
            self._last_epsp = measurement.amplitude
            self._epsp_trajectory.append(measurement.amplitude)
            self._append_log(
                f"Baseline EPSP: {measurement.amplitude:.6g} {measurement.units}."
            )
            self._update_plot()
            self._index = 0
            self._begin_point()
            return

        if self._phase == "probe":
            self._handle_probe_measurement(measurement)

    def _handle_probe_measurement(self, measurement: EPSPMeasurement | None) -> None:
        if self._baseline_epsp is None or self._last_epsp is None:
            self._finish("Baseline EPSP is unavailable.")
            return
        delta_t = self._pending_delta_t
        label = classify_window(delta_t)
        if measurement is None:
            # An undetectable EPSP is a skipped point, not a fatal error: record
            # NaN and carry on so one shunted response does not abort the sweep.
            self._epsp_trajectory.append(float("nan"))
            self._results.append((delta_t, float("nan")))
            self._append_log(
                f"Δt {delta_t:+.3g} ms → EPSP undetectable; point skipped."
            )
        else:
            self._epsp_trajectory.append(measurement.amplitude)
            percent = self._percent_for(measurement.amplitude)
            self._results.append((delta_t, percent))
            if not self._reset_mode():
                self._last_epsp = measurement.amplitude
            self._append_log(f"Δt {delta_t:+.3g} ms → {percent:+.3f}% EPSP ({label}).")
            self._maybe_warn_saturation(measurement.amplitude)
        self._update_plot()
        self._advance_point()

    def _record_weight_point(self) -> None:
        if self._baseline_epsp is None or self._last_epsp is None:
            self._finish("Baseline synaptic weight is unavailable.")
            return
        weight = self._read_synapse_weight()
        if weight is None:
            self._finish("Could not read the synaptic weight after the pairing.")
            return
        delta_t = self._pending_delta_t
        label = classify_window(delta_t)
        self._epsp_trajectory.append(weight)
        percent = self._percent_for(weight)
        self._results.append((delta_t, percent))
        if not self._reset_mode():
            self._last_epsp = weight
        self._append_log(f"Δt {delta_t:+.3g} ms → {percent:+.3f}% weight ({label}).")
        self._maybe_warn_saturation(weight)
        self._update_plot()
        self._advance_point()

    def _percent_for(self, measure: float) -> float:
        """Percent change: from the fixed baseline (reset) or the running
        value (continuous)."""
        baseline = self._baseline_epsp
        last = self._last_epsp
        assert baseline is not None and last is not None
        reference = baseline if self._reset_mode() else last
        return percent_change(reference, measure, reference)

    def _advance_point(self) -> None:
        self._index += 1
        if self._index >= len(self._delta_ts):
            measured = sum(1 for _dt, percent in self._results if percent == percent)
            self._finish(
                f"STDP window sweep finished; measured {measured}/"
                f"{len(self._results)} points."
            )
            return
        self._begin_point()

    def _begin_point(self) -> None:
        self._pending_delta_t = self._delta_ts[self._index]
        self._pairing_repeat = 0
        if self._reset_mode():
            self._restore_baseline()
        self._apply_next_pairing()

    def _apply_next_pairing(self) -> None:
        self._pairing_repeat += 1
        self._phase = "pair"
        self._append_log(
            f"Δt {self._pending_delta_t:+.3g} ms: pairing "
            f"{self._pairing_repeat}/{self.spn_pairings_per_point.value()}."
        )
        self._append_pairing(self._pending_delta_t)

    def _measure_epsps(self, trace: TraceData) -> EPSPMeasurement:
        neuron_id = self.spn_postsynaptic.value()
        matches = np.flatnonzero(trace.indices == neuron_id)
        if matches.size == 0:
            raise ValueError(
                f"Postsynaptic neuron {neuron_id} is not recorded in the "
                "selected trace."
            )
        trace_mv = _to_mv(trace.traces[int(matches[0])], trace.units)
        dt_ms = 1000.0 * trace.dt
        baseline_ms = 12.0
        response_ms = self.spn_response_ms.value()
        amplitudes: list[float] = []
        for pulse_time in self._probe_pulse_times:
            onset_ms = 1000.0 * pulse_time
            start_ms = max(0.0, onset_ms - baseline_ms)
            stop_ms = onset_ms + response_ms
            start = max(0, int(np.floor(start_ms / dt_ms)))
            stop = min(trace_mv.size, int(np.ceil(stop_ms / dt_ms)) + 1)
            if start >= trace_mv.size:
                trace_duration_ms = trace_mv.size * dt_ms
                raise ValueError(
                    f"The selected trace ends at {trace_duration_ms:.3f} ms, "
                    f"before the probe at {onset_ms:.3f} ms. Select the current "
                    "simulation voltage trace and try again."
                )
            detection = detect_epsps(trace_mv[start:stop], dt_ms)
            relative_onset_ms = onset_ms - start * dt_ms
            sample_tolerance_ms = 0.5 * dt_ms
            candidates = detection.amplitudes_mv[
                (detection.times_ms >= relative_onset_ms - sample_tolerance_ms)
                & (
                    detection.times_ms
                    <= relative_onset_ms + response_ms + sample_tolerance_ms
                )
            ]
            if candidates.size == 0:
                raise _NoEpspDetected(
                    f"No EPSP was detected after the probe at {onset_ms:.3f} ms."
                )
            amplitudes.append(float(np.max(candidates)))
        return EPSPMeasurement(
            amplitude=float(np.mean(amplitudes)),
            pulse_amplitudes=tuple(amplitudes),
        )

    def _maybe_warn_saturation(self, epsp: float) -> None:
        baseline = self._baseline_epsp
        if self._saturation_warned or baseline is None or baseline == 0.0:
            return
        ratio = epsp / baseline
        if ratio < 0.25 or ratio > 4.0:
            self._saturation_warned = True
            self._append_log(
                "EPSP drift is far from baseline; later Δt points may be affected "
                "by synaptic saturation."
            )

    def _current_delta_t_values(self) -> list[float]:
        values = self._snapped_delta_ts(
            delta_t_sweep(
                self.spn_dt_min_ms.value(),
                self.spn_dt_max_ms.value(),
                self.spn_dt_points.value(),
                self.chk_include_zero.isChecked(),
            )
        )
        if not self._reset_mode():
            # Continuous sweep drifts, so visit offsets interleaved by magnitude
            # to keep the running weight near baseline.
            values = interleave_by_magnitude(values)
        return values

    def _raw_delta_t_values(self) -> list[float]:
        return delta_t_sweep(
            self.spn_dt_min_ms.value(),
            self.spn_dt_max_ms.value(),
            self.spn_dt_points.value(),
            self.chk_include_zero.isChecked(),
        )

    def _snapped_delta_ts(self, values: list[float]) -> list[float]:
        if self.editor.current_stim_modifier is None:
            return values
        dt_seconds = self.editor.current_stim_modifier.stim_array.dt
        snapped: list[float] = []
        seen: set[int] = set()
        for value in values:
            samples, effective_ms = snap_milliseconds_to_dt(abs(value), dt_seconds)
            signed_samples = samples if value >= 0.0 else -samples
            if signed_samples in seen:
                continue
            seen.add(signed_samples)
            snapped.append(copysign(effective_ms, value))
        return snapped

    def _maybe_log_delta_t_resolution_warning(self) -> None:
        if self._resolution_warning_logged:
            return
        raw_values = self._raw_delta_t_values()
        if len(raw_values) < 2:
            return
        spacing = min(
            abs(b - a) for a, b in zip(raw_values, raw_values[1:], strict=False)
        )
        one_sample_ms = samples_to_milliseconds(1, self._require_stim().dt)
        if spacing >= one_sample_ms:
            return
        self._resolution_warning_logged = True
        self._append_log(
            f"Input dt is {one_sample_ms:.6g} ms; Δt values are snapped to that "
            "grid, so finer spacing cannot be resolved."
        )

    def _estimated_probe_count(self) -> int:
        n_points = len(self._current_delta_t_values())
        pairings = self.spn_pairings_per_point.value()
        if self._weight_mode():
            # One init run + the pairings; the weight is read directly, no probes.
            return 1 + n_points * pairings
        return 1 + n_points * (pairings + 1)

    def _update_estimate(self) -> None:
        self.lbl_estimate.setText(
            "Estimated simulations for the current settings: "
            f"~{self._estimated_probe_count()}"
        )

    def _update_plot(self) -> None:
        self.window_axes.clear()
        self.trajectory_axes.clear()
        finite_results = [
            (delta_t, percent)
            for delta_t, percent in self._results
            if percent == percent  # drop NaN (undetectable) points
        ]
        if finite_results:
            sorted_results = sorted(finite_results, key=lambda item: item[0])
            x = np.asarray([delta_t for delta_t, _percent in sorted_results])
            y = np.asarray([percent for _delta_t, percent in sorted_results])
            self.window_axes.plot(x, y, color="#2f2f2f", linewidth=1.5, alpha=0.7)
            positive = x > 0.0
            negative = x < 0.0
            zero = x == 0.0
            self.window_axes.scatter(
                x[positive], y[positive], color="#2ca02c", zorder=3
            )
            self.window_axes.scatter(
                x[negative], y[negative], color="#d62728", zorder=3
            )
            if np.any(zero):
                self.window_axes.scatter(x[zero], y[zero], color="#7f7f7f", zorder=3)
        self.window_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.window_axes.axvline(0.0, color="#7f7f7f", linewidth=1)
        self.window_axes.text(
            0.98,
            0.92,
            "LTP",
            color="#2ca02c",
            ha="right",
            transform=self.window_axes.transAxes,
        )
        self.window_axes.text(
            0.02,
            0.92,
            "LTD",
            color="#d62728",
            ha="left",
            transform=self.window_axes.transAxes,
        )
        if self._epsp_trajectory:
            probe_x = np.arange(len(self._epsp_trajectory))
            self.trajectory_axes.plot(
                probe_x,
                self._epsp_trajectory,
                color="#1f77b4",
                linewidth=2,
                marker="o",
            )
            self.trajectory_axes.axhline(
                self._epsp_trajectory[0],
                color="#7f7f7f",
                linewidth=1,
                linestyle="--",
            )
        self._update_status()
        self._style_plot_axes()
        self.window_plot.getFigure().subplots_adjust(
            left=0.13, right=0.98, bottom=0.09, top=0.98, hspace=0.18
        )
        self.window_plot.draw()

    def _update_status(self) -> None:
        if not self._results:
            self.lbl_status.setText("STDP window: waiting for baseline")
            return
        delta_t, percent = self._results[-1]
        if percent != percent:  # NaN → undetectable point
            detail = "undetectable"
        else:
            detail = f"{percent:+.3f}% ({classify_window(delta_t)})"
        self.lbl_status.setText(
            f"{len(self._results)}/{len(self._delta_ts)} points · last Δt "
            f"{delta_t:+.3g} ms → {detail}"
        )

    def _style_plot_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        for axes in (self.window_axes, self.trajectory_axes):
            axes.set_facecolor(face)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        weight = self._weight_mode()
        self.window_axes.set_xlabel("Δt = t_post − t_pre (ms)", color=text)
        self.window_axes.set_ylabel(
            "Weight change (%)" if weight else "EPSP change (%)", color=text
        )
        self.trajectory_axes.set_xlabel("Point" if weight else "Probe", color=text)
        self.trajectory_axes.set_ylabel(
            "Synaptic weight" if weight else "EPSP (mV)", color=text
        )
        self.trajectory_axes.xaxis.set_major_locator(MultipleLocator(1))
        self.window_plot.getFigure().set_facecolor(face)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme_module.normalize_theme(theme)
        self._style_plot_axes()
        self.window_plot.draw()
        self.stim_heatmap_window.apply_theme(theme)

    def simulation_failed(self, message: str) -> None:
        if self._running and self._awaiting_simulation:
            self._finish(f"Simulation failed: {message}")

    def _finish(self, message: str) -> None:
        self._running = False
        self._awaiting_simulation = False
        self._phase = "idle"
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(message)

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _nonempty_stim_array_names(self) -> list[str]:
        project = self.project
        if project is None:
            return []
        return [
            name
            for name, stim in project.stim_arrays.items()
            if bool(np.any(stim.values))
        ]

    def _prompt_existing_input_choice(self, names: list[str]) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Existing input")
        listed = "\n".join(f"  • {name}" for name in names)
        box.setText(
            "The routine appends its pairing and probe pulses after any existing "
            "input, which lengthens every simulation. These input arrays "
            f"already contain input:\n\n{listed}"
        )
        box.setInformativeText(
            "Clear all input arrays first, append after the existing input, or cancel."
        )
        clear_button = box.addButton(
            "Clear all input arrays", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton("Append anyway", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_button or clicked is None:
            return "cancel"
        if clicked is clear_button:
            return "clear"
        return "append"

    def _confirm_start_over_existing_input(self) -> bool:
        names = self._nonempty_stim_array_names()
        if not names:
            return True
        choice = self._prompt_existing_input_choice(names)
        if choice == "cancel":
            return False
        if choice == "clear":
            self.editor.clear_all_stim_arrays()
            self.refresh()
            self._append_log("Cleared all input arrays before starting.")
        return True
