from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.data import TraceData
from neuroseq.rheobase_dialog import RheobaseDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


class _FakeProject:
    def __init__(self) -> None:
        self.trace_available = True
        self.trace = TraceData(
            "Simulation :: P / v",
            np.full((2, 2000), -70.0, dtype=np.float32),
            np.array([0, 1]),
            "mV",
            0.001,
        )
        self.stored_checkpoints: list[str] = []
        self.restored_checkpoints: list[str] = []
        self.network = SimpleNamespace(
            store=self.stored_checkpoints.append,
            restore=self.restored_checkpoints.append,
        )
        self.b2_model = SimpleNamespace(
            get_sim_time_secs=lambda: 0.0,
            network=self.network,
        )

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
        self.time_slices: list[slice] = []
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

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice
        self.time_slices.append(time_slice)
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        stim.values = np.vstack((stim.values, values))


def _make_dialog(qtbot) -> tuple[_FakeProject, _FakeEditor, RheobaseDialog]:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = RheobaseDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    return project, editor, dialog


def _finish_search(dialog: RheobaseDialog, limit: int = 100) -> None:
    for _ in range(limit):
        if not dialog._running:
            return
        dialog.simulation_finished()
    raise AssertionError("Rheobase search did not finish.")


def _use_fixed_duration_strategy(dialog: RheobaseDialog) -> None:
    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
    )


def test_rheobase_brackets_then_bisects_to_threshold(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    tested: list[float] = []

    def measure(trace: TraceData) -> int:
        del trace
        amplitude = dialog._pending_amplitude
        tested.append(amplitude)
        return int(amplitude >= 2.6)

    monkeypatch.setattr(dialog, "_measure_spikes", measure)
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    assert dialog._rheobase is not None
    assert 2.6 <= dialog._rheobase <= 2.6 + dialog.spn_tolerance.value()
    assert tested[:2] == [100.0, 1.0]
    assert len(editor.calls) == len(tested)
    for block in editor.calls:
        assert np.count_nonzero(block[:, 1]) == 0


def test_rheobase_restores_probe_baseline_each_iteration(qtbot, monkeypatch) -> None:
    project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(
        dialog, "_measure_spikes", lambda trace: int(dialog._pending_amplitude >= 2.6)
    )
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    assert len(project.stored_checkpoints) == 1
    assert project.stored_checkpoints[0].startswith("Rheobase baseline ")
    assert project.restored_checkpoints == project.stored_checkpoints * len(
        editor.calls
    )
    assert {time_slice.start for time_slice in editor.time_slices} == {0}


def test_rheobase_sweeping_strategy_controls(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)

    assert dialog.cmb_sweeping_strategy.currentText() == (
        dialog.AMPLITUDE_AND_DURATION_STRATEGY
    )
    assert dialog.spn_pulse_ms.value() == 10.0
    assert dialog.spn_max_pulse_ms.value() == 50.0
    assert dialog.spn_duration_tolerance_ms.value() == 1.0
    assert dialog.spn_min_amplitude.value() == 1.0
    assert dialog.spn_max_amplitude.value() == 100.0
    assert dialog.lbl_amplitude_range.text() == "Pulse amplitude range/tolerance:"
    assert not dialog.spn_min_amplitude.isHidden()
    assert not dialog.spn_max_amplitude.isHidden()
    assert not dialog.spn_tolerance.isHidden()
    assert dialog.spn_fixed_amplitude.isHidden()

    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
    )
    assert dialog.spn_min_amplitude.isHidden()
    assert dialog.spn_max_amplitude.isHidden()
    assert dialog.spn_tolerance.isHidden()
    assert not dialog.spn_fixed_amplitude.isHidden()
    assert not dialog.spn_max_pulse_ms.isHidden()
    assert not dialog.spn_duration_tolerance_ms.isHidden()

    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
    )
    assert not dialog.spn_min_amplitude.isHidden()
    assert not dialog.spn_max_amplitude.isHidden()
    assert not dialog.spn_tolerance.isHidden()
    assert dialog.spn_fixed_amplitude.isHidden()
    assert dialog.spn_max_pulse_ms.isHidden()
    assert dialog.spn_duration_tolerance_ms.isHidden()


def test_rheobase_reports_when_no_spike_at_max(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 0)

    dialog.start()
    dialog.simulation_finished()

    assert not dialog._running
    assert len(editor.calls) == 1
    assert "increase the range" in dialog.log.toPlainText()


def test_rheobase_reports_when_spikes_at_min(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()
    dialog.simulation_finished()
    dialog.simulation_finished()

    assert not dialog._running
    assert len(editor.calls) == 2
    assert dialog._rheobase == pytest.approx(dialog.spn_min_amplitude.value())
    assert "threshold <= minimum" in dialog.log.toPlainText()


def test_rheobase_stops_when_dialog_closed(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)
    dialog.start()
    assert dialog._running
    assert dialog._awaiting_simulation

    dialog.close()

    assert not dialog._running
    assert not dialog._awaiting_simulation
    assert "dialog closed" in dialog.log.toPlainText()


def test_rheobase_ignores_unrequested_simulation(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    measurements: list[int] = []

    def measure(trace: TraceData) -> int:
        del trace
        measurements.append(1)
        return 1

    monkeypatch.setattr(dialog, "_measure_spikes", measure)
    dialog.start()
    blocks_after_start = len(editor.calls)
    dialog._awaiting_simulation = False

    dialog.simulation_finished()

    assert measurements == []
    assert len(editor.calls) == blocks_after_start
    assert dialog._running


def test_rheobase_plot_marks_converged_rheobase(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(
        dialog, "_measure_spikes", lambda trace: int(dialog._pending_amplitude >= 2.6)
    )
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    assert dialog._rheobase is not None
    guide_lines = [
        line for line in dialog.rheobase_axes.lines if line.get_linestyle() == "--"
    ]
    assert any(np.allclose(line.get_ydata(), dialog._rheobase) for line in guide_lines)
    assert any(
        np.allclose(line.get_ydata(), 2.0 * dialog._rheobase) for line in guide_lines
    )
    assert dialog.rheobase_axes.lines[0].get_color() == "#000000"
    assert dialog.rheobase_axes.get_xlabel() == "Stimulus duration (ms)"
    assert dialog.rheobase_axes.get_ylabel() == "Stimulus strength\n(mwatt / um2)"
    assert dialog.rheobase_axes.get_title() == ""
    assert dialog.search_axes.get_title() == ""
    assert "Rheobase ~=" in dialog.lbl_status.text()


def test_rheobase_sweeps_duration_range_and_marks_chronaxie(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_pulse_ms.setValue(10.0)
    dialog.spn_max_pulse_ms.setValue(30.0)
    dialog.spn_duration_tolerance_ms.setValue(0.5)
    dialog.spn_tolerance.setValue(0.1)

    def measure(trace: TraceData) -> int:
        del trace
        threshold = 30.0 / dialog._current_duration_ms
        return int(dialog._pending_amplitude >= threshold)

    monkeypatch.setattr(dialog, "_measure_spikes", measure)

    dialog.start()
    _finish_search(dialog)

    threshold_pairs = sorted(dialog._thresholds, key=lambda item: item[0])
    durations = [duration for duration, _threshold in threshold_pairs]
    thresholds = [threshold for _duration, threshold in threshold_pairs]
    assert min(durations) == pytest.approx(10.0)
    assert max(durations) == pytest.approx(30.0)
    assert any(14.0 < duration < 16.0 for duration in durations)
    assert thresholds == sorted(thresholds, reverse=True)
    assert dialog._rheobase == pytest.approx(min(thresholds))
    assert dialog._chronaxie_ms is not None
    assert 10.0 < dialog._chronaxie_ms < 30.0
    guide_lines = [
        line for line in dialog.rheobase_axes.lines if line.get_linestyle() == "--"
    ]
    assert any(
        np.allclose(line.get_xdata(), dialog._chronaxie_ms) for line in guide_lines
    )
    assert "chronaxie ~=" in dialog.lbl_status.text()


def test_rheobase_fixed_amplitude_varies_duration(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_AMPLITUDE_VARYING_DURATION_STRATEGY
    )
    dialog.spn_fixed_amplitude.setValue(2.0)
    dialog.spn_pulse_ms.setValue(10.0)
    dialog.spn_max_pulse_ms.setValue(30.0)
    dialog.spn_duration_tolerance_ms.setValue(1.0)

    monkeypatch.setattr(
        dialog,
        "_measure_spikes",
        lambda trace: int(dialog._current_duration_ms >= 20.0),
    )

    dialog.start()
    _finish_search(dialog)

    assert len(editor.calls) > 3
    assert dialog._thresholds == []
    assert dialog._rheobase is None
    assert dialog._fixed_spiking_duration_ms == pytest.approx(20.0)
    sampled_durations = [sample[0] for sample in dialog._samples]
    assert sampled_durations[:3] == pytest.approx([30.0, 10.0, 20.0])
    assert [sample[1] for sample in dialog._samples] == pytest.approx(
        [2.0] * len(dialog._samples)
    )
    assert "first spiking duration" in dialog.log.toPlainText()
    guide_lines = [
        line for line in dialog.rheobase_axes.lines if line.get_linestyle() == "--"
    ]
    assert any(
        np.allclose(line.get_xdata(), dialog._fixed_spiking_duration_ms)
        for line in guide_lines
    )


def test_rheobase_fixed_duration_varies_amplitude(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
    )
    dialog.spn_pulse_ms.setValue(12.0)
    dialog.spn_max_pulse_ms.setValue(30.0)
    dialog.spn_duration_tolerance_ms.setValue(1.0)
    monkeypatch.setattr(
        dialog, "_measure_spikes", lambda trace: int(dialog._pending_amplitude >= 2.6)
    )
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    assert [duration for duration, _threshold in dialog._thresholds] == [12.0]
    assert dialog._rheobase is not None
    assert 2.6 <= dialog._rheobase <= 2.6 + dialog.spn_tolerance.value()


def test_rheobase_pulse_only_on_target_column(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)
    dialog.spn_neuron.setValue(1)
    dialog.spn_max_amplitude.setValue(4.0)

    dialog.start()

    block = editor.calls[0]
    recovery_n = round(dialog.spn_recovery_ms.value() / 1000.0 / 0.001)
    pulse_n = round(dialog.spn_pulse_ms.value() / 1000.0 / 0.001)
    assert np.count_nonzero(block[:, 0]) == 0
    assert np.all(block[:recovery_n, 1] == 0.0)
    assert np.all(block[recovery_n : recovery_n + pulse_n, 1] == 4.0)
    assert np.all(block[recovery_n + pulse_n :, 1] == 0.0)


def test_rheobase_plots_open_in_separate_window(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    qtbot.addWidget(dialog.plot_window)

    assert dialog.plot_window is not dialog
    assert dialog.plot_window.isWindow()
    assert dialog.rheobase_plot.parent() is dialog.plot_window


def test_rheobase_initializes_missing_trace_then_starts_search(
    qtbot, monkeypatch
) -> None:
    project, editor, dialog = _make_dialog(qtbot)
    project.trace_available = False
    dialog.refresh()
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
    assert dialog._phase == "probe_max"


def test_rheobase_measures_voltage_threshold_crossings(qtbot) -> None:
    project, _editor, dialog = _make_dialog(qtbot)
    project.trace.traces[1] = -70.0
    project.trace.traces[1, 120:130] = 25.0
    project.trace.traces[1, 170:180] = 20.0
    dialog.spn_neuron.setValue(1)
    dialog._pulse_onset_seconds = 0.1
    dialog.spn_response_ms.setValue(100.0)

    assert dialog._measure_spikes(project.trace) == 2


def test_rheobase_matplotlib_plots_follow_theme(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.rheobase_axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(dialog.search_axes.get_facecolor()) == "#1e1e1e"

    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.rheobase_axes.get_facecolor()) == "#ffffff"
    assert to_hex(dialog.search_axes.get_facecolor()) == "#ffffff"
