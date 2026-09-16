from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.data import TraceData
from neuroseq.input_designer_dialog import EPSPMeasurement
from neuroseq.input_designer_dialog import InputDesignerDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


class _FakeProject:
    def __init__(self) -> None:
        self.trace_available = True
        self.trace = TraceData(
            "Simulation :: P / v",
            np.zeros((2, 2000), dtype=np.float32),
            np.array([0, 1]),
            "mV",
            0.001,
        )
        self.traces = {self.trace.name: self.trace}
        self.b2_model = SimpleNamespace(get_sim_time_secs=lambda: 0.0)

    def get_data_names(self, data_type: type[TraceData]) -> list[str]:
        assert data_type is TraceData
        return list(self.traces) if self.trace_available else []

    def get_data(self, name: str, data_type: type[TraceData]) -> TraceData | None:
        assert data_type is TraceData
        return self.traces.get(name)


class _FakeEditor:
    def __init__(self, project: _FakeProject) -> None:
        self.project = project
        stim = StimArray(
            "Exc_stimulation",
            StimType.Excitation,
            np.zeros((10, 2)),
            "mwatt / um2",
            0.001,
        )
        self.current_stim_modifier = SimpleNamespace(stim_array=stim)
        self.calls: list[np.ndarray] = []
        self.cleared = 0
        self.focused_ranges: list[tuple[float, float]] = []
        self.controller = SimpleNamespace(
            focus_time_range=lambda start, stop: self.focused_ranges.append(
                (start, stop)
            )
        )

    def focus_time_range(self, start: float, stop: float) -> None:
        self.controller.focus_time_range(start, stop)

    def clear_stim_array(self) -> None:
        self.cleared += 1

    def reset_stim_array(self) -> None:
        self.reset_count = getattr(self, "reset_count", 0) + 1

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice, time_slice
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        stim.values = np.vstack((stim.values, values))


def test_input_designer_runs_probe_pairing_probe_and_reaches_target(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    amplitudes = iter([2.0, 2.5])
    measured_neurons: list[int] = []

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        measured_neurons.append(dialog.spn_postsynaptic.value())
        amplitude = next(amplitudes)
        return EPSPMeasurement(amplitude, (amplitude,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(10.0)

    dialog.start()
    assert len(editor.calls) == 1
    assert np.count_nonzero(editor.calls[0][:, 0]) == 5 * 5
    assert np.count_nonzero(editor.calls[0][:, 1]) == 0

    dialog.simulation_finished()
    assert len(editor.calls) == 2
    pairing = editor.calls[1]
    pre_sample = int(np.flatnonzero(pairing[:, 0])[0])
    post_sample = int(np.flatnonzero(pairing[:, 1])[0])
    assert post_sample - pre_sample == 30

    dialog.simulation_finished()
    assert len(editor.calls) == 3
    focused_widths = [stop - start for start, stop in editor.focused_ranges]
    assert focused_widths == pytest.approx([focused_widths[0]] * 3)
    dialog.simulation_finished()

    assert not dialog._running
    assert dialog._epsp_percentages == [100.0, 125.0]
    assert "target achieved" in dialog.log.toPlainText()
    assert "absolute 5.000%" in dialog.lbl_relative_difference.text()
    assert measured_neurons[0] == 1


def test_input_designer_uses_post_pre_pairing_for_depression(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(2.0, (2.0,) * 5),
    )
    dialog.spn_target.setValue(80.0)
    dialog.spn_tolerance.setValue(5.0)

    dialog.start()
    dialog.simulation_finished()

    pairing = editor.calls[1]
    assert np.flatnonzero(pairing[:, 1])[0] < np.flatnonzero(pairing[:, 0])[0]
    assert "depression" in dialog.log.toPlainText()


@pytest.mark.parametrize("visible_blocks", [0, 3])
def test_input_designer_camera_keeps_configured_width(
    qtbot, visible_blocks: int
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.spn_history_blocks.setValue(visible_blocks)

    dialog.start()

    block_samples = editor.calls[0].shape[0]
    expected_blocks = max(1, visible_blocks)
    expected_width = expected_blocks * block_samples
    expected_start = max(0, 10 + block_samples - expected_width)
    assert editor.focused_ranges == [
        (expected_start * 0.001, (expected_start + expected_width) * 0.001)
    ]


def test_input_designer_protocol_defaults(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.spn_pairing_interval_ms.value() == 30.0
    assert dialog.cmb_pairing_strategy.currentText() == dialog.FIXED_PAIRING_STRATEGY
    assert not dialog.spn_pairing_interval_ms.isHidden()
    assert dialog.spn_initial_pairing_time_ms.isHidden()
    assert dialog.spn_pairing_increment_ms.isHidden()
    assert dialog.spn_tolerance.value() == 1.0
    assert dialog.spn_history_blocks.value() == 3
    assert dialog.spn_amplitude.value() == 100.0
    assert dialog.spn_pulse_ms.value() == 5.0
    assert not dialog.chk_run_all_iterations.isChecked()
    assert dialog.windowFlags() & QtCore.Qt.WindowType.WindowMinimizeButtonHint


def test_input_designer_adaptive_strategy_updates_signed_pairing_time(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.cmb_pairing_strategy.setCurrentText(dialog.ADAPTIVE_PAIRING_STRATEGY)
    dialog.spn_initial_pairing_time_ms.setValue(10.0)
    dialog.spn_pairing_increment_ms.setValue(5.0)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(1.0)
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(2.0, (2.0,) * 5),
    )

    assert dialog.spn_pairing_interval_ms.isHidden()
    assert not dialog.spn_initial_pairing_time_ms.isHidden()
    assert not dialog.spn_pairing_increment_ms.isHidden()

    dialog.start()
    dialog.simulation_finished()
    first_pairing = editor.calls[1]
    assert (
        np.flatnonzero(first_pairing[:, 0])[0] < np.flatnonzero(first_pairing[:, 1])[0]
    )
    dialog.simulation_finished()
    dialog.simulation_finished()

    assert dialog._pairing_times_ms == [10.0, 5.0]
    assert list(dialog.pairing_time_axes.lines[-1].get_ydata()) == [10.0, 5.0]


def test_input_designer_adaptive_strategy_increases_time_and_changes_sign(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.cmb_pairing_strategy.setCurrentText(dialog.ADAPTIVE_PAIRING_STRATEGY)
    dialog.spn_initial_pairing_time_ms.setValue(10.0)
    dialog.spn_pairing_increment_ms.setValue(5.0)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(1.0)
    amplitudes = iter([2.0, 2.6])
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(next(amplitudes), (2.0,) * 5),
    )

    dialog.start()
    dialog.simulation_finished()
    dialog.simulation_finished()
    dialog.simulation_finished()

    assert dialog._pairing_times_ms == [10.0, -15.0]
    second_pairing = editor.calls[3]
    assert (
        np.flatnonzero(second_pairing[:, 1])[0]
        < np.flatnonzero(second_pairing[:, 0])[0]
    )


def test_input_designer_can_hide_parameters(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    qtbot.addWidget(dialog.plot_window)
    dialog.show()
    expanded_size = dialog.size()

    assert dialog.parameters_panel.isVisible()
    assert dialog.btn_toggle_parameters.text() == "Hide parameters"

    qtbot.mouseClick(dialog.btn_toggle_parameters, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert not dialog.parameters_panel.isVisible()
    assert dialog.width() == expanded_size.width()
    assert dialog.height() < expanded_size.height()
    assert dialog.btn_toggle_parameters.text() == "Show parameters"
    assert dialog.btn_toggle_parameters.arrowType() == QtCore.Qt.ArrowType.RightArrow

    qtbot.mouseClick(dialog.btn_toggle_parameters, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert dialog.parameters_panel.isVisible()
    assert dialog.size() == expanded_size
    assert dialog.btn_toggle_parameters.text() == "Hide parameters"
    assert dialog.btn_toggle_parameters.arrowType() == QtCore.Qt.ArrowType.DownArrow


def test_input_designer_can_continue_after_convergence(qtbot, monkeypatch) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.spn_target.setValue(100.0)
    dialog.spn_tolerance.setValue(1.0)
    dialog.spn_max_iterations.setValue(1)
    dialog.chk_run_all_iterations.setChecked(True)
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(2.0, (2.0,) * 5),
    )

    dialog.start()
    dialog.simulation_finished()

    assert dialog._running
    assert dialog._phase == "pairing"
    assert dialog._iteration == 1
    assert "configured to run through the last iteration" in dialog.log.toPlainText()


def test_input_designer_positions_native_minimized_window_without_moving_restore(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    host = QtWidgets.QMainWindow()
    host.resize(900, 700)
    qtbot.addWidget(host)
    host.show()
    dialog = InputDesignerDialog(editor, parent=host)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.move(240, 180)
    original_position = dialog.pos()
    placements: list[tuple[int, int, int]] = []

    def record_placement(hwnd: int, x: int, y: int) -> bool:
        placements.append((hwnd, x, y))
        return True

    monkeypatch.setattr(
        "neuroseq.input_designer_dialog._set_native_minimized_position",
        record_placement,
    )

    dialog._position_minimized_window()

    available = dialog.screen().availableGeometry()
    assert placements == [
        (
            int(dialog.winId()),
            available.left() + 12,
            available.bottom() - 32 - 12 + 1,
        )
    ]
    assert dialog.pos() == original_position


def test_input_designer_initializes_missing_trace_then_starts_protocol(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    project.trace_available = False
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        lambda *args: pytest.fail("trace initialization should be silent"),
    )
    requested: list[float] = []
    dialog.simulation_requested.connect(requested.append)

    dialog.start()

    assert requested == [0.001]
    assert dialog._phase == "initialize_trace"
    project.trace_available = True
    dialog.simulation_finished()
    assert len(editor.calls) == 1
    assert requested[1] > 0.001
    assert dialog._phase == "baseline_probe"


def test_input_designer_initializes_when_only_imported_trace_exists(qtbot) -> None:
    project = _FakeProject()
    imported_trace = TraceData(
        "Imported EPSP trace",
        np.zeros((2, 100), dtype=np.float32),
        np.array([0, 1]),
        "mV",
        0.001,
    )
    project.traces = {imported_trace.name: imported_trace}
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    requested: list[float] = []
    dialog.simulation_requested.connect(requested.append)

    dialog.start()

    assert requested == [0.001]
    assert dialog._phase == "initialize_trace"


def test_input_designer_prefers_simulation_voltage_trace_for_measurement(qtbot) -> None:
    project = _FakeProject()
    imported_trace = TraceData(
        "Imported EPSP trace",
        np.zeros((2, 100), dtype=np.float32),
        np.array([0, 1]),
        "mV",
        0.001,
    )
    project.traces = {
        imported_trace.name: imported_trace,
        project.trace.name: project.trace,
    }
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    dialog.cmb_trace.setCurrentText(imported_trace.name)
    dialog.refresh()

    assert dialog.cmb_trace.currentText() == project.trace.name


def test_input_designer_reuses_epsp_plugin_detection(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    pulse_samples = [50, 150, 250, 350, 450]
    x = np.arange(project.trace.traces.shape[1])
    project.trace.traces[1] = -70.0
    for center in pulse_samples:
        project.trace.traces[1] += 5.0 * np.exp(-(((x - center) / 4.0) ** 2))
    dialog._probe_pulse_times = [sample * project.trace.dt for sample in pulse_samples]

    measurement = dialog._measure_epsps(project.trace)

    # The shared plugin's local-baseline window includes the early Gaussian
    # rise, so its measured prominence is slightly below the injected 5 mV.
    assert measurement.amplitude == pytest.approx(4.62, abs=0.05)
    assert len(measurement.pulse_amplitudes) == 5


def test_input_designer_reports_when_selected_trace_ends_before_probe(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog._probe_pulse_times = [2.244]

    with pytest.raises(ValueError, match="selected trace ends at 2000.000 ms"):
        dialog._measure_epsps(project.trace)


def test_input_designer_matplotlib_plots_follow_theme(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.epsp_axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(dialog.difference_axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(dialog.pairing_time_axes.get_facecolor()) == "#1e1e1e"

    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.epsp_axes.get_facecolor()) == "#ffffff"
    assert to_hex(dialog.difference_axes.get_facecolor()) == "#ffffff"
    assert to_hex(dialog.pairing_time_axes.get_facecolor()) == "#ffffff"
    assert dialog.epsp_plot is dialog.difference_plot
    assert dialog.epsp_plot.minimumHeight() == 500
    assert dialog.epsp_axes.get_ylabel() == "EPSP\namplitude (%)"

    dialog._epsp_percentages = [100.0, 101.0, 102.0]
    dialog.spn_target.setValue(101.0)
    dialog.spn_tolerance.setValue(2.0)
    dialog._update_plots()
    visible_ticks = [
        tick for tick in dialog.difference_axes.get_xticks() if 0.5 <= tick <= 3.5
    ]
    assert visible_ticks == [1.0, 2.0, 3.0]
    epsp_patch = dialog.epsp_axes.patches[0]
    epsp_deadband_y = dialog.epsp_axes.transData.inverted().transform(
        epsp_patch.get_transform().transform(epsp_patch.get_path().vertices)
    )[:, 1]
    difference_patch = dialog.difference_axes.patches[0]
    difference_deadband_y = dialog.difference_axes.transData.inverted().transform(
        difference_patch.get_transform().transform(difference_patch.get_path().vertices)
    )[:, 1]
    assert (min(epsp_deadband_y), max(epsp_deadband_y)) == pytest.approx((99.0, 103.0))
    assert (min(difference_deadband_y), max(difference_deadband_y)) == pytest.approx(
        (-2.0, 2.0)
    )
    assert dialog.difference_axes.get_ylabel() == "Δ target\n(%)"


def test_input_designer_stops_routine_when_closed(qtbot, monkeypatch) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog, "_measure_epsps", lambda trace: EPSPMeasurement(2.0, (2.0,) * 5)
    )
    dialog.start()
    assert dialog._running
    assert dialog._awaiting_simulation

    dialog.close()

    assert not dialog._running
    assert not dialog._awaiting_simulation
    assert "dialog closed" in dialog.log.toPlainText()


def test_input_designer_ignores_unrequested_simulation(qtbot, monkeypatch) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    measurements: list[int] = []

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        measurements.append(1)
        return EPSPMeasurement(2.0, (2.0,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)
    dialog.start()
    blocks_after_start = len(editor.calls)

    # A simulation the designer did not request (e.g. a manual Run) completes.
    dialog._awaiting_simulation = False
    dialog.simulation_finished()

    assert measurements == []
    assert len(editor.calls) == blocks_after_start
    assert dialog._running


def test_input_designer_adaptive_magnitude_clamped_to_maximum(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.cmb_pairing_strategy.setCurrentText(dialog.ADAPTIVE_PAIRING_STRATEGY)
    dialog.spn_initial_pairing_time_ms.setMaximum(12.0)
    dialog.spn_initial_pairing_time_ms.setValue(10.0)
    dialog.spn_pairing_increment_ms.setValue(5.0)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(1.0)
    amplitudes = iter([2.0, 2.6])
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(next(amplitudes), (2.0,) * 5),
    )

    dialog.start()
    dialog.simulation_finished()  # baseline -> pairing +10
    dialog.simulation_finished()  # probe 130% -> increase would be 15, clamp to 12
    dialog.simulation_finished()

    # Without the clamp the second magnitude would be 15; the spinbox max is 12.
    assert dialog._pairing_times_ms == [10.0, -12.0]


def test_input_designer_adaptive_magnitude_floored_above_zero(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.cmb_pairing_strategy.setCurrentText(dialog.ADAPTIVE_PAIRING_STRATEGY)
    dialog.spn_initial_pairing_time_ms.setValue(10.0)
    dialog.spn_pairing_increment_ms.setValue(5.0)
    dialog.spn_target.setValue(200.0)  # never reached, so magnitude keeps shrinking
    dialog.spn_tolerance.setValue(1.0)
    dialog.spn_max_iterations.setValue(6)
    monkeypatch.setattr(
        dialog, "_measure_epsps", lambda trace: EPSPMeasurement(2.0, (2.0,) * 5)
    )

    dialog.start()
    for _ in range(14):
        if not dialog._running:
            break
        dialog.simulation_finished()

    # dt = 0.001 s -> one-sample floor of 1 ms; magnitude shrinks 10 -> 5 -> 1
    # and never collapses to 0.
    assert dialog._pairing_times_ms[:3] == [10.0, 5.0, 1.0]
    assert 0.0 not in dialog._pairing_times_ms
    assert min(abs(t) for t in dialog._pairing_times_ms) >= 1.0 - 1e-9


def test_input_designer_measures_dominant_epsp_not_earlier_blip(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    x = np.arange(project.trace.traces.shape[1])
    project.trace.traces[1] = -70.0
    onset = 100
    # A small spurious bump just after the probe, then the real, larger EPSP.
    project.trace.traces[1] += 2.0 * np.exp(-(((x - (onset + 5)) / 2.0) ** 2))
    project.trace.traces[1] += 6.0 * np.exp(-(((x - (onset + 18)) / 3.0) ** 2))
    dialog._probe_pulse_times = [onset * project.trace.dt]

    measurement = dialog._measure_epsps(project.trace)

    # The larger EPSP (~6 mV) is selected, not the ~2 mV early blip.
    assert measurement.amplitude > 4.0


def test_input_designer_pulse_amplitude_minimum_is_positive(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.spn_amplitude.minimum() > 0.0
    # The spin box must clamp an attempt to set zero up to its positive minimum.
    dialog.spn_amplitude.setValue(0.0)
    assert dialog.spn_amplitude.value() > 0.0


def test_input_designer_parameters_are_scrollable(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.parameters_scroll.widget() is dialog.parameters_panel
    assert dialog.parameters_scroll.widgetResizable()
    # The panel can shrink below its natural height, forcing a scrollbar.
    assert (
        dialog.parameters_scroll.minimumHeight()
        < dialog.parameters_panel.sizeHint().height()
    )


def test_input_designer_wheel_over_field_does_not_change_value(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    spin = dialog.spn_max_iterations
    before = spin.value()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )

    # The filter intercepts the wheel (routes it to the scroll area) ...
    assert dialog._wheel_filter.eventFilter(spin, wheel) is True
    # ... so delivering a wheel scroll to the field leaves its value unchanged.
    QtWidgets.QApplication.sendEvent(spin, wheel)
    assert spin.value() == before


def test_input_designer_plots_open_in_separate_window(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    qtbot.addWidget(dialog.plot_window)

    # The plot canvas lives in its own top-level window, not the dialog itself.
    assert dialog.plot_window is not dialog
    assert dialog.plot_window.isWindow()
    assert dialog.epsp_plot.window() is dialog.plot_window

    # Showing the dialog brings up the plot window; closing it hides the plots.
    dialog.show()
    qtbot.waitUntil(lambda: dialog.plot_window.isVisible())
    dialog.close()
    qtbot.waitUntil(lambda: not dialog.plot_window.isVisible())


def test_input_designer_shows_stim_array_info_and_clear_button(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    # refresh() (run in __init__) fills the top info from the current array.
    assert dialog.lbl_stim_name.text() == "Exc_stimulation"
    assert dialog.lbl_stim_units.text() == "mwatt / um2"
    assert dialog.lbl_stim_dt.text() == "1 ms"  # dt = 0.001 s
    assert dialog.lbl_stim_neurons.text() == "2"

    # The Clear input array button delegates to the editor.
    assert dialog.btn_clear.text() == "Clear input array"
    dialog.btn_clear.click()
    assert editor.cleared == 1


def test_input_designer_pulse_duration_controls_probe_width(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.spn_pulse_ms.value() == 5.0

    # dt = 1 ms, so a 3 ms pulse spans 3 samples; five probe pulses -> 15 samples.
    dialog.spn_pulse_ms.setValue(3.0)
    dialog.start()
    probe = editor.calls[0]
    assert np.count_nonzero(probe[:, 0]) == 5 * 3


def test_input_designer_pairing_time_recorded_as_effective_grid_value(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.cmb_pairing_strategy.setCurrentText(dialog.ADAPTIVE_PAIRING_STRATEGY)
    dialog.spn_initial_pairing_time_ms.setValue(10.0)
    dialog.spn_pairing_increment_ms.setValue(0.25)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(1.0)
    amplitudes = iter([2.0, 2.6])
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(next(amplitudes), (2.0,) * 5),
    )

    dialog.start()
    dialog.simulation_finished()
    dialog.simulation_finished()
    dialog.simulation_finished()

    assert dialog._pairing_times_ms == [10.0, -10.0]
    assert "pairing times are snapped" in dialog.log.toPlainText()


def test_input_designer_stim_heatmap_opens_in_its_own_window(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    qtbot.addWidget(dialog.stim_heatmap_window)

    assert dialog.stim_heatmap_window not in (dialog, dialog.plot_window)
    assert dialog.stim_heatmap_window.isWindow()
    assert dialog.stim_heatmap_plot.window() is dialog.stim_heatmap_window

    dialog.show()
    qtbot.waitUntil(lambda: dialog.stim_heatmap_window.isVisible())
    dialog.close()
    qtbot.waitUntil(lambda: not dialog.stim_heatmap_window.isVisible())


def test_input_designer_stim_heatmap_shows_whole_history_without_panning(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog, "_measure_epsps", lambda trace: EPSPMeasurement(2.0, (2.0,) * 5)
    )

    dialog.start()
    image = dialog._stim_heatmap_image
    assert image is not None
    stim = editor.current_stim_modifier.stim_array
    first_n_samples = stim.values.shape[0]
    # the heatmap always starts at t=0 and ends at the array's current length -
    # it never pans a fixed-width window the way the main follow-view does.
    assert image.get_extent()[0] == 0.0
    assert image.get_extent()[1] == pytest.approx(first_n_samples * stim.dt * 1000.0)
    assert dialog.stim_heatmap_axes.get_ylabel() == "Neuron ID"
    assert dialog.stim_heatmap_axes.get_xlabel() == "Time (ms)"

    dialog.simulation_finished()  # appends a pairing block, growing the array
    # the same image artist is reused (updated in place), not rebuilt/replaced
    assert dialog._stim_heatmap_image is image
    grown_n_samples = stim.values.shape[0]
    assert grown_n_samples > first_n_samples
    assert image.get_extent()[1] == pytest.approx(grown_n_samples * stim.dt * 1000.0)
    # every previously appended sample is still part of the shown data - the
    # start of the array is never dropped/panned away
    assert image.get_array().shape[1] == grown_n_samples


def test_input_designer_stim_heatmap_follows_theme(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.start()

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.stim_heatmap_axes.get_facecolor()) == "#1e1e1e"
    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.stim_heatmap_axes.get_facecolor()) == "#ffffff"
