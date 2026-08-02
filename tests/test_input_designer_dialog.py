from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtCore
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
        self.b2_model = SimpleNamespace(get_sim_time_secs=lambda: 0.0)

    def get_data_names(self, data_type: type[TraceData]) -> list[str]:
        assert data_type is TraceData
        return [self.trace.name] if self.trace_available else []

    def get_data(self, name: str, data_type: type[TraceData]) -> TraceData:
        assert name == self.trace.name
        assert data_type is TraceData
        return self.trace


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
        self.focused_ranges: list[tuple[float, float]] = []
        self.controller = SimpleNamespace(
            focus_time_range=lambda start, stop: self.focused_ranges.append(
                (start, stop)
            )
        )

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
    assert np.count_nonzero(editor.calls[0][:, 0]) == 5
    assert np.count_nonzero(editor.calls[0][:, 1]) == 0

    dialog.simulation_finished()
    assert len(editor.calls) == 2
    pairing = editor.calls[1]
    pre_sample = int(np.flatnonzero(pairing[:, 0])[0])
    post_sample = int(np.flatnonzero(pairing[:, 1])[0])
    assert post_sample - pre_sample == 30

    dialog.simulation_finished()
    assert len(editor.calls) == 3
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


def test_input_designer_camera_includes_three_prior_blocks(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.spn_history_blocks.setValue(0)

    dialog.start()

    block_samples = editor.calls[0].shape[0]
    expected_start = 10 * 0.001
    assert editor.focused_ranges == [(expected_start, (10 + block_samples) * 0.001)]


def test_input_designer_protocol_defaults(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    assert dialog.spn_pairing_interval_ms.value() == 30.0
    assert dialog.spn_tolerance.value() == 1.0
    assert dialog.spn_history_blocks.value() == 3
    assert not dialog.chk_run_all_iterations.isChecked()
    assert dialog.windowFlags() & QtCore.Qt.WindowType.WindowMinimizeButtonHint


def test_input_designer_can_hide_parameters_to_expand_plots(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    dialog.show()
    expanded_size = dialog.size()

    assert dialog.parameters_panel.isVisible()
    assert dialog.btn_toggle_parameters.text() == "Hide parameters"

    qtbot.mouseClick(dialog.btn_toggle_parameters, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert not dialog.parameters_panel.isVisible()
    assert dialog.epsp_plot.isVisible()
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
    input_arguments: list[tuple] = []

    def get_duration(*args):
        input_arguments.append(args)
        return 0.00149, True

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        get_duration,
    )
    requested: list[float] = []
    dialog.simulation_requested.connect(requested.append)

    dialog.start()

    assert requested == [0.001]
    assert input_arguments[0][4] == 0.001
    assert input_arguments[0][6] == 3
    assert dialog._phase == "initialize_trace"
    project.trace_available = True
    dialog.simulation_finished()
    assert len(editor.calls) == 1
    assert requested[1] > 0.001
    assert dialog._phase == "baseline_probe"


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


def test_input_designer_matplotlib_plots_follow_theme(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.epsp_axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(dialog.difference_axes.get_facecolor()) == "#1e1e1e"

    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.epsp_axes.get_facecolor()) == "#ffffff"
    assert to_hex(dialog.difference_axes.get_facecolor()) == "#ffffff"
    assert dialog.epsp_plot is dialog.difference_plot
    assert dialog.epsp_plot.minimumHeight() == 430
    assert dialog.epsp_axes.get_ylabel() == "EPSP\namplitude (%)"

    dialog._epsp_percentages = [100.0, 101.0, 102.0]
    dialog._update_plots()
    visible_ticks = [
        tick for tick in dialog.difference_axes.get_xticks() if 0.5 <= tick <= 3.5
    ]
    assert visible_ticks == [1.0, 2.0, 3.0]
    assert dialog.difference_axes.get_ylabel() == "Δ target\n(%)"
