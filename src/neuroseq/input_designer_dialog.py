from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
from matplotlib.ticker import MultipleLocator
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import TraceData
from neuroseq.resources.analysis_plugins.epsp_amplitudes import _to_mv
from neuroseq.resources.analysis_plugins.epsp_amplitudes import detect_epsps

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray
    from neuroseq.stim_editor import StimEditor


@dataclass(frozen=True)
class EPSPMeasurement:
    amplitude: float
    pulse_amplitudes: tuple[float, ...]
    units: str = "mV"


def _set_native_minimized_position(hwnd: int, x: int, y: int) -> bool:
    """Set a Windows iconic-window position without changing restore geometry."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class WindowPlacement(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.UINT),
            ("flags", wintypes.UINT),
            ("show_cmd", wintypes.UINT),
            ("min_position", wintypes.POINT),
            ("max_position", wintypes.POINT),
            ("normal_position", wintypes.RECT),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_placement = user32.GetWindowPlacement
    get_placement.argtypes = [wintypes.HWND, ctypes.POINTER(WindowPlacement)]
    get_placement.restype = wintypes.BOOL
    set_placement = user32.SetWindowPlacement
    set_placement.argtypes = [wintypes.HWND, ctypes.POINTER(WindowPlacement)]
    set_placement.restype = wintypes.BOOL
    placement = WindowPlacement()
    placement.length = ctypes.sizeof(WindowPlacement)
    if not get_placement(hwnd, ctypes.byref(placement)):
        return False
    placement.flags |= 0x0001  # WPF_SETMINPOSITION
    placement.min_position = wintypes.POINT(int(x), int(y))
    return bool(set_placement(hwnd, ctypes.byref(placement)))


class _WheelToScrollFilter(QtCore.QObject):
    """Redirect wheel events from value widgets to a scroll area.

    Spin boxes and combo boxes change their value on a wheel scroll, which makes
    scrolling the parameter panel accidentally edit fields. This filter forwards
    the wheel to the scroll area so the panel scrolls and values stay put.
    """

    def __init__(self, scroll_area: QtWidgets.QScrollArea) -> None:
        super().__init__(scroll_area)
        self._scroll_area = scroll_area

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if event is not None and event.type() == QtCore.QEvent.Type.Wheel:
            QtWidgets.QApplication.sendEvent(self._scroll_area.viewport(), event)
            return True
        return super().eventFilter(obj, event)


class InputDesignerDialog(QtWidgets.QDialog):
    """Closed-loop EPSP targeting with an STDP induction protocol."""

    simulation_requested = QtCore.Signal(float)
    PROBE_PULSES = 5
    PROBE_PERIOD_MS = 100.0
    PULSE_MS = 1.0
    LEAD_MS = 20.0
    RESPONSE_TAIL_MS = 50.0
    FIXED_PAIRING_STRATEGY = "Fixed pre-post pairing time"
    ADAPTIVE_PAIRING_STRATEGY = "Adaptive (single parameter, dt increment)"

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
        self._iteration = 0
        self._view_block_samples: int | None = None
        self._probe_pulse_times: list[float] = []
        self._baseline_epsp: float | None = None
        self._epsp_percentages: list[float] = []
        self._pairing_times_ms: list[float] = []
        self._adaptive_pairing_magnitude_ms = 0.0
        self._theme = theme_module.LIGHT
        self._expanded_height: int | None = None
        self._compact_height: int | None = None
        self.setWindowTitle("Design dynamic input")
        self.resize(640, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        # Current stim-array info at the top, mirroring the Design static input
        # window.
        info_form = QtWidgets.QFormLayout()
        self.lbl_stim_name = QtWidgets.QLabel("-", self)
        self.lbl_stim_units = QtWidgets.QLabel("-", self)
        self.lbl_stim_dt = QtWidgets.QLabel("-", self)
        self.lbl_stim_neurons = QtWidgets.QLabel("-", self)
        info_form.addRow("Stim array:", self.lbl_stim_name)
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
        explanation = QtWidgets.QLabel(
            "Target a postsynaptic EPSP with five-pulse measurements and "
            "single pre/post STDP induction pairings.",
            self.parameters_panel,
        )
        explanation.setWordWrap(True)
        parameters_layout.addWidget(explanation)
        form = QtWidgets.QFormLayout()
        self.cmb_trace = QtWidgets.QComboBox(self)
        self.spn_presynaptic = QtWidgets.QSpinBox(self)
        self.spn_presynaptic.setMinimum(0)
        self.spn_presynaptic.setValue(0)
        self.spn_postsynaptic = QtWidgets.QSpinBox(self)
        self.spn_postsynaptic.setMinimum(0)
        self.spn_postsynaptic.setValue(1)
        self.spn_amplitude = QtWidgets.QDoubleSpinBox(self)
        self.spn_amplitude.setDecimals(3)
        # A positive minimum: a zero-amplitude probe evokes no EPSP and dead-ends
        # the routine at the measurement step.
        self.spn_amplitude.setRange(0.001, 1e12)
        self.spn_amplitude.setValue(1.0)
        self.lbl_units = QtWidgets.QLabel(self)
        amplitude_row = QtWidgets.QHBoxLayout()
        amplitude_row.addWidget(self.spn_amplitude)
        amplitude_row.addWidget(self.lbl_units)
        self.spn_pulse_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pulse_ms.setDecimals(3)
        self.spn_pulse_ms.setRange(0.001, 1000.0)
        self.spn_pulse_ms.setValue(self.PULSE_MS)
        self.spn_pulse_ms.setSuffix(" ms")
        self.cmb_pairing_strategy = QtWidgets.QComboBox(self)
        self.cmb_pairing_strategy.addItems(
            [self.FIXED_PAIRING_STRATEGY, self.ADAPTIVE_PAIRING_STRATEGY]
        )
        self.spn_pairing_interval_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pairing_interval_ms.setRange(0.001, 1000.0)
        self.spn_pairing_interval_ms.setValue(30.0)
        self.spn_pairing_interval_ms.setSuffix(" ms")
        self.spn_initial_pairing_time_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_initial_pairing_time_ms.setRange(0.0, 1000.0)
        self.spn_initial_pairing_time_ms.setValue(30.0)
        self.spn_initial_pairing_time_ms.setSuffix(" ms")
        self.spn_pairing_increment_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_pairing_increment_ms.setRange(0.001, 1000.0)
        self.spn_pairing_increment_ms.setValue(1.0)
        self.spn_pairing_increment_ms.setSuffix(" ms")
        self.spn_target = QtWidgets.QDoubleSpinBox(self)
        self.spn_target.setRange(0.0, 10_000.0)
        self.spn_target.setValue(120.0)
        self.spn_target.setSuffix(" %")
        self.spn_tolerance = QtWidgets.QDoubleSpinBox(self)
        self.spn_tolerance.setRange(0.001, 1000.0)
        self.spn_tolerance.setValue(1.0)
        self.spn_tolerance.setSuffix(" %")
        self.spn_max_iterations = QtWidgets.QSpinBox(self)
        self.spn_max_iterations.setRange(1, 10_000)
        self.spn_max_iterations.setValue(20)
        self.chk_run_all_iterations = QtWidgets.QCheckBox(
            "Run until the last iteration regardless of convergence to the target",
            self,
        )
        self.chk_run_all_iterations.setChecked(False)
        self.chk_run_all_iterations.setToolTip(
            "Continue plasticity induction after the EPSP enters the target "
            "tolerance, stopping only at the maximum iteration count."
        )
        self.chk_follow_input = QtWidgets.QCheckBox(
            "Follow new input on each iteration", self
        )
        self.chk_follow_input.setChecked(True)
        self.spn_history_blocks = QtWidgets.QSpinBox(self)
        self.spn_history_blocks.setRange(0, 100)
        self.spn_history_blocks.setValue(3)
        self.spn_history_blocks.setSuffix(" blocks")
        self.spn_history_blocks.setToolTip(
            "Live-view width, in multiples of a five-pulse probe-block length, "
            "held constant as blocks are appended so the view does not jitter "
            "when shorter pairing blocks are added. Zero shows one probe-block "
            "length (the newest block)."
        )
        form.addRow("Trace to analyze:", self.cmb_trace)
        form.addRow("Presynaptic neuron ID:", self.spn_presynaptic)
        form.addRow("Postsynaptic neuron ID:", self.spn_postsynaptic)
        form.addRow("Pulse amplitude:", amplitude_row)
        form.addRow("Pulse duration (ms):", self.spn_pulse_ms)
        form.addRow("Pairing strategy:", self.cmb_pairing_strategy)
        self.lbl_pairing_interval = QtWidgets.QLabel("Pre-post pairing time:", self)
        form.addRow(self.lbl_pairing_interval, self.spn_pairing_interval_ms)
        self.lbl_initial_pairing_time = QtWidgets.QLabel(
            "Initial guess for pre-post pairing time:", self
        )
        form.addRow(self.lbl_initial_pairing_time, self.spn_initial_pairing_time_ms)
        self.lbl_pairing_increment = QtWidgets.QLabel("dt increment:", self)
        form.addRow(self.lbl_pairing_increment, self.spn_pairing_increment_ms)
        form.addRow("Target relative EPSP amplitude:", self.spn_target)
        form.addRow("Tolerance window (deadband):", self.spn_tolerance)
        form.addRow("Maximum plasticity induction iterations:", self.spn_max_iterations)
        form.addRow("Stopping rule:", self.chk_run_all_iterations)
        form.addRow("Live view:", self.chk_follow_input)
        form.addRow("Visible history:", self.spn_history_blocks)
        parameters_layout.addLayout(form)
        # Wrap the parameters in a scroll area so they scroll instead of being
        # squished into unreadable rows on short screens.
        self.parameters_scroll = QtWidgets.QScrollArea(self)
        self.parameters_scroll.setWidget(self.parameters_panel)
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # A small minimum lets the panel scroll rather than shrink its rows when
        # the dialog is short.
        self.parameters_scroll.setMinimumHeight(80)
        layout.addWidget(self.parameters_scroll, 1)
        # A wheel scroll over a value widget must scroll the panel, not silently
        # change the field, so route those events to the scroll area.
        self._wheel_filter = _WheelToScrollFilter(self.parameters_scroll)
        value_widgets: list[QtWidgets.QWidget] = list(
            self.parameters_panel.findChildren(QtWidgets.QAbstractSpinBox)
        )
        value_widgets += self.parameters_panel.findChildren(QtWidgets.QComboBox)
        for value_widget in value_widgets:
            value_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            value_widget.installEventFilter(self._wheel_filter)
        self.cmb_pairing_strategy.currentIndexChanged.connect(
            self._update_pairing_strategy_controls
        )
        self._update_pairing_strategy_controls()

        # The plots live in their own window so the controls and the plots each
        # get room, and the controls stay compact on small screens.
        self.plot_window = QtWidgets.QDialog(self)
        self.plot_window.setWindowTitle("Design dynamic input — plots")
        self.plot_window.setWindowFlag(
            QtCore.Qt.WindowType.WindowMinimizeButtonHint, True
        )
        self._plot_window_positioned = False
        plot_layout = QtWidgets.QVBoxLayout(self.plot_window)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        self.epsp_plot = MatplotlibWidget(self.plot_window)
        self.epsp_plot.setMinimumHeight(500)
        figure = self.epsp_plot.getFigure()
        grid = figure.add_gridspec(3, 1, height_ratios=(3, 2, 2), hspace=0.14)
        self.epsp_axes = figure.add_subplot(grid[0, 0])
        self.difference_axes = figure.add_subplot(grid[1, 0], sharex=self.epsp_axes)
        self.pairing_time_axes = figure.add_subplot(grid[2, 0], sharex=self.epsp_axes)
        self.spn_target.valueChanged.connect(lambda _value: self._update_plots())
        self.spn_tolerance.valueChanged.connect(lambda _value: self._update_plots())
        # Compatibility alias: both axes now share one tightly packed canvas.
        self.difference_plot = self.epsp_plot
        plot_layout.addWidget(self.epsp_plot)
        self.plot_window.resize(720, 640)
        self.lbl_relative_difference = QtWidgets.QLabel(
            "Relative difference from target: waiting for baseline", self
        )
        layout.addWidget(self.lbl_relative_difference)
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
                "Start routine", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.btn_stop = cast(
            QtWidgets.QPushButton,
            buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole),
        )
        self.btn_clear = cast(
            QtWidgets.QPushButton,
            buttons.addButton(
                "Clear stim array", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            ),
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self.editor.clear_stim_array)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.apply_theme(self._theme)
        self.refresh()

    def _update_pairing_strategy_controls(self) -> None:
        adaptive = (
            self.cmb_pairing_strategy.currentText() == self.ADAPTIVE_PAIRING_STRATEGY
        )
        self.lbl_pairing_interval.setVisible(not adaptive)
        self.spn_pairing_interval_ms.setVisible(not adaptive)
        self.lbl_initial_pairing_time.setVisible(adaptive)
        self.spn_initial_pairing_time_ms.setVisible(adaptive)
        self.lbl_pairing_increment.setVisible(adaptive)
        self.spn_pairing_increment_ms.setVisible(adaptive)

    def _set_parameters_hidden(self, hidden: bool) -> None:
        """Collapse the parameter panel so the plots can use its space."""
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
        self.btn_toggle_parameters.setToolTip(
            "Show the protocol parameters."
            if hidden
            else "Hide the protocol parameters to keep this window compact."
        )
        QtCore.QTimer.singleShot(0, self._resize_for_parameter_visibility)

    def _resize_for_parameter_visibility(self) -> None:
        """Resize after Qt has recalculated the layout's size constraints."""
        height = (
            self._compact_height
            if self.btn_toggle_parameters.isChecked()
            else self._expanded_height
        )
        if height is not None:
            self.resize(self.width(), max(height, self.minimumSizeHint().height()))

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        # The plots live in a separate window; bring it up alongside the dialog.
        self.plot_window.show()
        if not self._plot_window_positioned:
            self._plot_window_positioned = True
            self.plot_window.move(self.x() + self.width() + 20, self.y())
        self.plot_window.raise_()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        # The dialog is cached and reused, so closing only hides it. Stop any
        # running routine first, otherwise it would keep consuming simulation
        # callbacks while hidden and resume mid-run when reopened.
        if self._running:
            self._finish("Routine stopped (dialog closed).")
        self.plot_window.close()
        super().closeEvent(event)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.WindowStateChange
            and self.isMinimized()
        ):
            QtCore.QTimer.singleShot(0, self._position_minimized_window)

    def _position_minimized_window(self) -> None:
        """Place Windows' native minimized dialog at the screen bottom-left."""
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
        if self.project is None:
            return
        self.cmb_trace.addItems(self.project.get_data_names(TraceData))
        self.cmb_trace.setCurrentText(selected)
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
            self.lbl_stim_name.setText("-")
            self.lbl_stim_units.setText("-")
            self.lbl_stim_dt.setText("-")
            self.lbl_stim_neurons.setText("-")

    def start(self) -> None:
        self.refresh()
        if self.project is None or self.editor.current_stim_modifier is None:
            self._append_log("No stimulation array is selected.")
            return
        if not self.cmb_trace.currentText():
            duration = 0.001
            self._running = True
            self._phase = "initialize_trace"
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._append_log(
                f"Running an initial {duration:.3f} s simulation to create "
                "trace monitors."
            )
            self._request_simulation(duration)
            return
        self._begin_protocol(clear_log=True)

    def _begin_protocol(self, *, clear_log: bool) -> None:
        if self.spn_presynaptic.value() == self.spn_postsynaptic.value():
            self._finish("Pre- and postsynaptic neuron IDs must differ.")
            return
        self._running = True
        self._iteration = 0
        self._view_block_samples = None
        self._baseline_epsp = None
        self._epsp_percentages = []
        self._pairing_times_ms = []
        self._adaptive_pairing_magnitude_ms = self.spn_initial_pairing_time_ms.value()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if clear_log:
            self.log.clear()
        self._update_plots()
        self._append_probe()

    def stop(self) -> None:
        self._finish("Routine stopped.")

    def _require_stim(self) -> StimArray:
        modifier = self.editor.current_stim_modifier
        if modifier is None:
            raise RuntimeError("No stimulation array is selected.")
        return modifier.stim_array

    def _require_project(self) -> Project:
        if self.project is None:
            raise RuntimeError("No project is available.")
        return self.project

    def _append_probe(self) -> None:
        stim = self._require_stim()
        pulse_n = max(1, round(self.spn_pulse_ms.value() / 1000.0 / stim.dt))
        lead_n = max(1, round(self.LEAD_MS / 1000.0 / stim.dt))
        period_n = max(pulse_n + 1, round(self.PROBE_PERIOD_MS / 1000.0 / stim.dt))
        tail_n = max(1, round(self.RESPONSE_TAIL_MS / 1000.0 / stim.dt))
        samples = lead_n + (self.PROBE_PULSES - 1) * period_n + pulse_n + tail_n
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        pulse_offsets = [
            lead_n + index * period_n for index in range(self.PROBE_PULSES)
        ]
        for offset in pulse_offsets:
            block[offset : offset + pulse_n, self.spn_presynaptic.value()] = (
                self.spn_amplitude.value()
            )
        block_start = self._append_block(block)
        self._probe_pulse_times = [
            (block_start + offset) * stim.dt for offset in pulse_offsets
        ]
        self._phase = "baseline_probe" if self._baseline_epsp is None else "post_probe"
        self._append_log("Appended five presynaptic probe pulses.")
        self._run_to_sample(block_start + samples)

    def _append_pairing(self, potentiate: bool) -> None:
        stim = self._require_stim()
        pulse_n = max(1, round(self.spn_pulse_ms.value() / 1000.0 / stim.dt))
        lead_n = max(1, round(self.LEAD_MS / 1000.0 / stim.dt))
        adaptive = (
            self.cmb_pairing_strategy.currentText() == self.ADAPTIVE_PAIRING_STRATEGY
        )
        if adaptive:
            if len(self._epsp_percentages) > 1:
                epsp_new = self._epsp_percentages[-1]
                epsp_old = self._epsp_percentages[-2]
                target_error = abs(epsp_new - self.spn_target.value())
                epsp_change = abs(epsp_new - epsp_old)
                increment = self.spn_pairing_increment_ms.value()
                if target_error > epsp_change:
                    self._adaptive_pairing_magnitude_ms -= increment
                else:
                    self._adaptive_pairing_magnitude_ms += increment
            # Keep the magnitude bounded: never below one simulation sample (so
            # pre and post never collapse to the same time) and never above the
            # configured spinbox maximum (so it cannot drift away unboundedly).
            min_magnitude_ms = 1000.0 * stim.dt
            max_magnitude_ms = self.spn_initial_pairing_time_ms.maximum()
            self._adaptive_pairing_magnitude_ms = min(
                max_magnitude_ms,
                max(min_magnitude_ms, self._adaptive_pairing_magnitude_ms),
            )
            pairing_time_ms = (
                self._adaptive_pairing_magnitude_ms
                if potentiate
                else -self._adaptive_pairing_magnitude_ms
            )
        else:
            magnitude_ms = self.spn_pairing_interval_ms.value()
            pairing_time_ms = magnitude_ms if potentiate else -magnitude_ms
        gap_n = max(1, round(abs(pairing_time_ms) / 1000.0 / stim.dt))
        samples = (
            lead_n
            + gap_n
            + pulse_n
            + max(1, round(self.RESPONSE_TAIL_MS / 1000.0 / stim.dt))
        )
        block = np.zeros((samples, stim.values.shape[1]), dtype=np.float32)
        pre_before_post = pairing_time_ms >= 0.0
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
        block_start = self._append_block(block)
        self._phase = "pairing"
        self._pairing_times_ms.append(pairing_time_ms)
        direction = (
            "potentiation (pre → post)"
            if pre_before_post
            else "depression (post → pre)"
        )
        self._append_log(
            f"Appended one {direction} pairing at {pairing_time_ms:+.3f} ms."
        )
        self._update_plots()
        self._run_to_sample(block_start + samples)

    def _append_block(self, block: np.ndarray) -> int:
        stim = self._require_stim()
        project = self._require_project()
        current_sample = int(np.ceil(project.b2_model.get_sim_time_secs() / stim.dt))
        start = max(stim.values.shape[0], current_sample)
        self.editor.apply_pattern(
            slice(0, stim.values.shape[1]), slice(start, start + block.shape[0]), block
        )
        if self.chk_follow_input.isChecked():
            if self._view_block_samples is None:
                # Size the live view in units of the first (probe) block so its
                # width stays constant as blocks are appended; using each block's
                # own length would make the view jitter when the shorter pairing
                # blocks are added.
                self._view_block_samples = block.shape[0]
            visible_blocks = max(1, self.spn_history_blocks.value())
            visible_samples = visible_blocks * self._view_block_samples
            block_end = start + block.shape[0]
            history_start = max(0, block_end - visible_samples)
            history_stop = history_start + visible_samples
            self.editor.focus_time_range(
                history_start * stim.dt, history_stop * stim.dt
            )
        return start

    def _run_to_sample(self, stop_sample: int) -> None:
        stim = self._require_stim()
        current_time = self._require_project().b2_model.get_sim_time_secs()
        run_seconds = max(stim.dt, stop_sample * stim.dt - current_time)
        self._request_simulation(run_seconds)

    def _request_simulation(self, seconds: float) -> None:
        """Ask the host to run a simulation, recording that the designer is now
        waiting for its result so it reacts only to its own run."""
        self._awaiting_simulation = True
        self.simulation_requested.emit(seconds)

    def simulation_finished(self) -> None:
        if not self._running or self.project is None:
            return
        if not self._awaiting_simulation:
            # A simulation the designer did not request (e.g. a manual run)
            # completed; do not advance the closed-loop protocol.
            return
        self._awaiting_simulation = False
        if self._phase == "initialize_trace":
            self.refresh()
            if not self.cmb_trace.currentText():
                self._finish(
                    "The initial simulation completed, but no simulation "
                    "trace monitor became available. Add a StateMonitor to "
                    "the model and try again."
                )
                return
            self._append_log("Trace monitor created; starting the EPSP protocol.")
            self._begin_protocol(clear_log=False)
            return
        if self._phase == "pairing":
            self._append_probe()
            return
        trace = self.project.get_data(self.cmb_trace.currentText(), TraceData)
        if trace is None:
            self._finish("The selected trace is unavailable after simulation.")
            return
        try:
            measurement = self._measure_epsps(trace)
        except ValueError as exc:
            self._finish(f"EPSP measurement failed: {exc}")
            return
        if self._baseline_epsp is None:
            if measurement.amplitude <= 0.0:
                self._finish(
                    "Baseline EPSP must be positive before relative targeting."
                )
                return
            self._baseline_epsp = measurement.amplitude
            percentage = 100.0
            self._append_log(
                f"Baseline EPSP: {measurement.amplitude:.6g} {measurement.units} "
                "(five-pulse average = 100%)."
            )
        else:
            percentage = 100.0 * measurement.amplitude / self._baseline_epsp
            self._append_log(
                f"Current EPSP: {measurement.amplitude:.6g} {measurement.units} "
                f"({percentage:.3f}% of baseline; change from initial "
                f"{percentage - 100.0:+.3f}%)."
            )
        self._epsp_percentages.append(percentage)
        self._update_plots()
        difference = percentage - self.spn_target.value()
        converged = abs(difference) < self.spn_tolerance.value()
        if converged and not self.chk_run_all_iterations.isChecked():
            self._finish("EPSP is within tolerance; target achieved.")
            return
        if self._iteration >= self.spn_max_iterations.value():
            self._finish("Maximum plasticity induction iterations reached.")
            return
        if converged:
            self._append_log(
                "EPSP is within tolerance, but the routine is configured to "
                "run through the last iteration."
            )
        self._iteration += 1
        self._append_pairing(potentiate=difference < 0.0)

    def _measure_epsps(self, trace: TraceData) -> EPSPMeasurement:
        """Associate the shared EPSP plugin's detections with probe pulses."""
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
        response_ms = 40.0
        amplitudes: list[float] = []
        for pulse_time in self._probe_pulse_times:
            onset_ms = 1000.0 * pulse_time
            start_ms = max(0.0, onset_ms - baseline_ms)
            stop_ms = onset_ms + response_ms
            start = max(0, int(np.floor(start_ms / dt_ms)))
            stop = min(trace_mv.size, int(np.ceil(stop_ms / dt_ms)) + 1)
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
                raise ValueError(
                    f"No EPSP was detected after the probe at {onset_ms:.3f} ms."
                )
            # Attribute the dominant depolarisation in the response window to the
            # probe. Taking the largest candidate avoids latching onto a smaller
            # spurious deflection that happens to precede the real EPSP.
            amplitudes.append(float(np.max(candidates)))
        return EPSPMeasurement(
            amplitude=float(np.mean(amplitudes)),
            pulse_amplitudes=tuple(amplitudes),
        )

    def _update_plots(self) -> None:
        x = np.arange(1, len(self._epsp_percentages) + 1)
        values = np.asarray(self._epsp_percentages)
        target = self.spn_target.value()
        tolerance = self.spn_tolerance.value()
        self.epsp_axes.clear()
        self.epsp_axes.axhspan(
            target - tolerance,
            target + tolerance,
            color="#2ca02c",
            alpha=0.18,
            label="Deadband",
        )
        self.epsp_axes.axhline(target, color="#d62728", linewidth=2, linestyle="--")
        self.epsp_axes.plot(x, values, color="#1f77b4", linewidth=2, marker="o")
        self.difference_axes.clear()
        self.difference_axes.axhspan(
            -tolerance,
            tolerance,
            color="#2ca02c",
            alpha=0.18,
        )
        self.difference_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.difference_axes.plot(
            x, values - target, color="#b245a8", linewidth=2, marker="o"
        )
        self.pairing_time_axes.clear()
        pairing_x = np.arange(1, len(self._pairing_times_ms) + 1)
        self.pairing_time_axes.axhline(0.0, color="#7f7f7f", linewidth=1)
        self.pairing_time_axes.plot(
            pairing_x,
            self._pairing_times_ms,
            color="#e28743",
            linewidth=2,
            marker="o",
        )
        if values.size:
            self.difference_axes.set_xlim(0.5, values.size + 0.5)
        if values.size:
            difference = float(values[-1] - target)
            self.lbl_relative_difference.setText(
                f"Relative difference from target: {difference:+.3f}% "
                f"(absolute {abs(difference):.3f}%, tolerance {self.spn_tolerance.value():.3f}%)"
            )
        else:
            self.lbl_relative_difference.setText(
                "Relative difference from target: waiting for baseline"
            )
        self._style_plot_axes()
        self.epsp_plot.getFigure().subplots_adjust(
            left=0.14, right=0.98, bottom=0.09, top=0.98, hspace=0.14
        )
        self.epsp_plot.draw()

    def _style_plot_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        for axes, ylabel in (
            (self.epsp_axes, "EPSP\namplitude (%)"),
            (self.difference_axes, "Δ target\n(%)"),
            (self.pairing_time_axes, "Pairing time\n(ms)"),
        ):
            axes.set_facecolor(face)
            axes.set_ylabel(ylabel, color=text)
            axes.tick_params(colors=text)
            axes.grid(True, alpha=0.25)
            for spine in axes.spines.values():
                spine.set_color(text)
        self.epsp_plot.getFigure().set_facecolor(face)
        self.pairing_time_axes.xaxis.set_major_locator(MultipleLocator(1))
        self.epsp_axes.tick_params(labelbottom=False)
        self.difference_axes.tick_params(labelbottom=False)
        self.epsp_axes.set_xlabel("")
        self.difference_axes.set_xlabel("")
        self.pairing_time_axes.set_xlabel("Measurement / iteration", color=text)

    def apply_theme(self, theme: str) -> None:
        """Recolor both Matplotlib canvases for the application theme."""
        self._theme = theme_module.normalize_theme(theme)
        self._style_plot_axes()
        self.epsp_plot.draw()

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
