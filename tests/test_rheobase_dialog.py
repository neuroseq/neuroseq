from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from tests.fake_stimulation_host import FakeStimulationHost


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
        self.stim_arrays: dict[str, StimArray] = {}
        self.spikes: dict[str, SpikeData] = {}
        #: Imported calcium traces, by name.
        self.traces: dict[str, TraceData] = {}
        self.b2_model = SimpleNamespace(get_sim_time_secs=lambda: 0.0)

    def get_data_names(self, data_type: type) -> list[str]:
        if data_type is SpikeData:
            return list(self.spikes)
        assert data_type is TraceData
        return ([self.trace.name] if self.trace_available else []) + list(self.traces)

    def get_data(self, name: str, data_type: type):
        if data_type is SpikeData:
            return self.spikes[name]
        assert data_type is TraceData
        if name in self.traces:
            return self.traces[name]
        assert name == self.trace.name
        return self.trace


class _FakeEditor:
    def __init__(self, project: _FakeProject) -> None:
        self.project = project
        self.stimulation_host = None
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
        self.pad_calls: list[float] = []
        self.focused_ranges: list[tuple[float, float]] = []
        self.controller = SimpleNamespace(
            focus_time_range=lambda start, stop: self.focused_ranges.append(
                (start, stop)
            )
        )

    def prepare_syscon_timestep(self, parent=None):
        return (
            abs(self.current_stim_modifier.stim_array.dt - _SYSCON2_TTL_PULSE_QUANTUM_S)
            < 1e-12
        )

    def focus_time_range(self, start: float, stop: float) -> None:
        self.controller.focus_time_range(start, stop)

    def clear_stim_array(self) -> None:
        self.cleared += 1

    def zero_current_stim_array(self) -> None:
        self.cleared += 1
        stim = self.current_stim_modifier.stim_array
        stim.values = np.zeros_like(stim.values)

    def reset_stim_array(self) -> None:
        self.reset_count = getattr(self, "reset_count", 0) + 1

    def current_stim_array_has_unsimulated_tail(self) -> bool:
        stim = self.current_stim_modifier.stim_array
        target_secs = max(self.project.b2_model.get_sim_time_secs(), float(stim.dt))
        n_keep = seconds_to_samples_nearest(target_secs, float(stim.dt))
        return bool(np.any(stim.values[n_keep:, :]))

    def current_stim_array_has_any_data(self) -> bool:
        return bool(np.any(self.current_stim_modifier.stim_array.values))

    def current_stim_array_last_stimulus_sample(self) -> int:
        values = self.current_stim_modifier.stim_array.values
        nonzero_rows = np.flatnonzero(np.any(values != 0, axis=1))
        if nonzero_rows.size == 0:
            return 0
        return int(nonzero_rows[-1]) + 1

    def window(self) -> SimpleNamespace:
        return SimpleNamespace(raise_=lambda: None, activateWindow=lambda: None)

    def _ensure_len(self, n_samples: int) -> None:
        stim = self.current_stim_modifier.stim_array
        if stim.values.shape[0] < n_samples:
            pad = n_samples - stim.values.shape[0]
            stim.values = np.pad(stim.values, ((0, pad), (0, 0)))

    def pad_stim_array_to(self, total_length_secs: float) -> None:
        self.pad_calls.append(total_length_secs)
        stim = self.current_stim_modifier.stim_array
        self._ensure_len(round(total_length_secs / stim.dt))

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice
        self.time_slices.append(time_slice)
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        self._ensure_len(time_slice.stop)
        stim.values[time_slice.start : time_slice.stop, :] = values


def _make_dialog(qtbot) -> tuple[_FakeProject, _FakeEditor, RheobaseDialog]:
    project = _FakeProject()
    editor = _FakeEditor(project)
    stim = editor.current_stim_modifier.stim_array
    project.stim_arrays = {stim.name: stim}
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


def test_rheobase_appends_each_probe_forward(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(
        dialog, "_measure_spikes", lambda trace: int(dialog._pending_amplitude >= 2.6)
    )
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    starts = [time_slice.start for time_slice in editor.time_slices]
    # Each probe is appended after the previous one (STDP-style), never
    # overwriting a fixed region, so block starts strictly increase.
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # The fixture's stim array is 10 samples but entirely zero, so there is
    # no "last stimulus" to continue after - probes start at sample 0.
    assert starts[0] == 0


def test_rheobase_pads_ahead_and_focuses_camera_once_on_first_probe(
    qtbot, monkeypatch
) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()

    # One pad call (ahead of the first block) and one matching camera focus -
    # the array-cursor plumbing is exercised, not just a bare append.
    assert len(editor.pad_calls) == 1
    assert len(editor.focused_ranges) == 1
    stim = editor.current_stim_modifier.stim_array
    assert stim.values.shape[0] > editor.time_slices[0].stop


def test_rheobase_trims_leftover_padding_when_search_converges_on_its_own(
    qtbot, monkeypatch
) -> None:
    """The backend can call finish() on itself (search converged) without
    going through the dialog's own stop()/_finish() path; _on_backend_finished
    - wired as RoutineBackend.on_finished - must still run and trim any
    padding placed ahead of the last block."""
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(
        dialog, "_measure_spikes", lambda trace: int(dialog._pending_amplitude >= 2.6)
    )
    dialog.spn_tolerance.setValue(0.1)

    dialog.start()
    _finish_search(dialog)

    assert not dialog._running
    assert dialog._array_cursor is None
    # reset_stim_array() (which trims any padding still ahead of the
    # simulation head) is what StimEditor uses for the actual trim; here we
    # only need to know _on_backend_finished actually reached it.
    assert editor.reset_count >= 1


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
    assert dialog.rheobase_axes.get_xlabel() == "Input duration (ms)"
    assert dialog.rheobase_axes.get_ylabel() == "Input strength\n(mwatt / um2)"
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
    assert dialog.rheobase_plot.window() is dialog.plot_window


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


def test_rheobase_no_warning_when_stim_array_is_empty(qtbot, monkeypatch) -> None:
    """An all-zero array needs no start-routine prompt at all -
    confirm_start_routine() itself short-circuits to "continue" without
    showing any dialog, so a routine starts immediately."""
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()

    assert len(editor.calls) >= 1


def test_rheobase_continue_preserves_existing_stimulus_and_catches_up_simulation(
    qtbot, monkeypatch
) -> None:
    """ "Continue" keeps existing stimulus content - even content that has
    never been simulated (e.g. an example project's pre-authored pattern) -
    rather than discarding it as an "unsimulated tail". When the simulation
    has not run over that content yet, it is caught up first so the network
    actually experiences it before probes are appended after it."""
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    editor.current_stim_modifier.stim_array.values[0, 0] = 1.0
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.rheobase_dialog.confirm_start_routine",
        lambda *args, **kwargs: "continue",
    )
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()

    # The fake simulation never advances its own clock, so the routine must
    # catch it up before appending any probes - not jump straight to the
    # search as it would if the simulation were already at the last sample.
    assert editor.calls == []
    assert dialog._running
    assert dialog._phase == "initialize_trace"
    # "Continue" itself never trims - unlike "clear" it does not touch the
    # array at all before appending.
    assert getattr(editor, "reset_count", 0) == 0

    _finish_search(dialog)

    assert editor.calls
    assert editor.current_stim_modifier.stim_array.values[0, 0] == 1.0
    assert editor.time_slices[0].start == 1  # right after the last stimulus


def test_rheobase_cancel_keeps_existing_input(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    editor.current_stim_modifier.stim_array.values[5, 0] = 1.0
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.rheobase_dialog.confirm_start_routine",
        lambda *args, **kwargs: "cancel",
    )
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()

    assert not dialog._running
    assert editor.calls == []
    assert "cancelled" in dialog.log.toPlainText()


def test_rheobase_clear_zeroes_existing_input_before_start(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    editor.current_stim_modifier.stim_array.values[5, 0] = 1.0
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.rheobase_dialog.confirm_start_routine",
        lambda *args, **kwargs: "clear",
    )
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()

    assert len(editor.calls) >= 1
    assert editor.cleared == 1
    # Trimmed back to the simulation head afterwards, same as reset_stim_array()
    # always did - there is nothing real left to lose once the array is zeroed.
    assert getattr(editor, "reset_count", 0) == 1
    # Starts from (almost) the very beginning - reset_stim_array()'s own
    # convention keeps at least one baseline sample rather than an empty
    # array, so this is 1 sample in, not literally 0.
    assert editor.time_slices[0].start == 1


def test_rheobase_estimate_updates_with_settings(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    dialog.spn_min_amplitude.setValue(0.0)
    dialog.spn_max_amplitude.setValue(8.0)
    dialog.spn_tolerance.setValue(1.0)

    # span 8, tol 1 -> 3 bisections + 2 bracket probes.
    assert dialog._estimated_probe_count() == 5
    assert "~5" in dialog.lbl_estimate.text()

    dialog.spn_max_amplitude.setValue(16.0)  # span 16 -> 4 bisections.
    assert dialog._estimated_probe_count() == 6
    assert "~6" in dialog.lbl_estimate.text()


def test_rheobase_estimate_scales_with_duration_sweep(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.cmb_sweeping_strategy.setCurrentText(dialog.AMPLITUDE_AND_DURATION_STRATEGY)
    dialog.spn_min_amplitude.setValue(0.0)
    dialog.spn_max_amplitude.setValue(8.0)
    dialog.spn_tolerance.setValue(1.0)
    dialog.spn_pulse_ms.setValue(0.0)
    dialog.spn_max_pulse_ms.setValue(8.0)
    dialog.spn_duration_tolerance_ms.setValue(1.0)

    # A full amplitude search (5) per duration stage (2 + 3 bisections).
    assert dialog._estimated_probe_count() == (2 + 3) * 5


def test_rheobase_response_window_scales_with_pulse_duration(qtbot) -> None:
    project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_neuron.setValue(1)
    dialog.spn_response_ms.setValue(20.0)
    dialog._current_duration_ms = 10.0
    dialog._pulse_onset_seconds = 0.1  # 100 ms; window is [100, 130] ms.
    project.trace.traces[1] = -70.0
    project.trace.traces[1, 105:107] = 25.0  # inside the window.
    project.trace.traces[1, 160:162] = 25.0  # after the window, must be ignored.

    assert dialog._measure_spikes(project.trace) == 1


def test_strength_duration_points_snap_to_dt_grid(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    _use_fixed_duration_strategy(dialog)
    dialog.spn_pulse_ms.setValue(15.4)
    monkeypatch.setattr(dialog, "_measure_spikes", lambda trace: 1)

    dialog.start()
    _finish_search(dialog)

    assert {duration for duration, _amp, _count in dialog._samples} == {15.0}
    assert dialog._thresholds == [(15.0, dialog.spn_min_amplitude.value())]


def test_duration_bisection_stops_at_one_sample(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_pulse_ms.setValue(10.0)
    dialog.spn_max_pulse_ms.setValue(12.0)
    dialog.spn_duration_tolerance_ms.setValue(0.01)
    dialog.spn_tolerance.setValue(0.5)
    dialog.spn_min_amplitude.setValue(0.0)
    dialog.spn_max_amplitude.setValue(8.0)

    def measure(trace: TraceData) -> int:
        del trace
        threshold = 20.0 / dialog._current_duration_ms
        return int(dialog._pending_amplitude >= threshold)

    monkeypatch.setattr(dialog, "_measure_spikes", measure)

    dialog.start()
    _finish_search(dialog)

    threshold_durations = {duration for duration, _threshold in dialog._thresholds}
    assert threshold_durations <= {10.0, 11.0, 12.0}
    assert "durations are snapped" in dialog.log.toPlainText()


def test_rheobase_pause_defers_next_simulation_and_resets_on_finish(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    requested: list[float] = []
    dialog.simulation_requested.connect(requested.append)
    dialog._running = True
    dialog.btn_pause.setEnabled(True)

    dialog.toggle_pause()
    dialog._request_simulation(0.5)
    assert dialog._paused
    assert dialog._pending_simulation_seconds == 0.5
    assert not dialog._awaiting_simulation
    assert requested == []

    dialog.toggle_pause()
    assert requested == [0.5]
    assert dialog._awaiting_simulation
    dialog._finish("done")
    assert not dialog._paused
    assert dialog._pending_simulation_seconds is None
    assert not dialog.btn_pause.isEnabled()
    assert dialog.btn_pause.text() == "Pause"


def test_rheobase_plot_window_is_scrollable(qtbot) -> None:
    """So a short screen scrolls the plots instead of squeezing the figure
    until axis labels overlap or get clipped off."""
    _, _, dialog = _make_dialog(qtbot)

    scroll_areas = dialog.plot_window.findChildren(QtWidgets.QScrollArea)
    assert len(scroll_areas) == 1
    assert scroll_areas[0].widget() is dialog.rheobase_plot
    assert scroll_areas[0].widgetResizable()


def _syscon2_dialog(qtbot):
    project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.dt = _SYSCON2_TTL_PULSE_QUANTUM_S
    host = FakeStimulationHost()
    editor.stimulation_host = host  # type: ignore[assignment]
    dialog.refresh()
    dialog.cmb_target.setCurrentText(dialog.SYSCON2_TARGET)
    # Manual delivery: the user runs, records and imports each block.
    dialog.hardware.chk_automatic.setChecked(False)
    dialog.cmb_calcium_readout.setCurrentText(dialog.SPIKES_READOUT)
    _use_fixed_duration_strategy(dialog)
    dialog.spn_min_amplitude.setValue(0.1)
    dialog.spn_max_amplitude.setValue(1.0)
    dialog.spn_tolerance.setValue(0.3)
    dialog.spn_recovery_ms.setValue(100.0)
    return project, editor, dialog, host


def _import_spikes(project, dialog, name: str, *, spiking: bool) -> None:
    onset = (
        dialog._pulse_onset_seconds
        - dialog._block_range[0] * _SYSCON2_TTL_PULSE_QUANTUM_S
    )
    times = [onset + 0.005] if spiking else [onset + 5.0]
    project.spikes[name] = SpikeData(name, np.array([0]), np.array(times))
    dialog.refresh()
    dialog.cmb_spikes.setCurrentText(name)


def test_syscon2_target_unavailable_without_host(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    item = dialog.cmb_target.model().item(1)
    assert not item.isEnabled()


def test_syscon2_probe_is_sent_and_waits_for_import(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)

    dialog.start()

    assert len(host.sent) == 1
    start, stop, preserve = host.sent[0]
    assert (start, stop) == dialog._block_range
    assert preserve
    assert dialog.btn_data_imported.isEnabled()
    assert dialog._awaiting_simulation


def test_syscon2_search_converges_from_imported_calcium_spikes(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()

    for i in range(20):
        if not dialog._running:
            break
        spiking = dialog._pending_amplitude >= 0.5
        _import_spikes(project, dialog, f"rec{i}", spiking=spiking)
        dialog.btn_data_imported.click()

    assert not dialog._running
    assert dialog._rheobase is not None
    assert 0.5 <= dialog._rheobase <= 0.5 + dialog.spn_tolerance.value()
    assert all(preserve for _, _, preserve in host.sent)
    assert not dialog.btn_data_imported.isEnabled()


def test_syscon2_reused_dataset_does_not_advance_search(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    _import_spikes(project, dialog, "rec0", spiking=True)
    dialog.btn_data_imported.click()
    sent_before = len(host.sent)

    # The same dataset again for the next probe: must be rejected, not re-measured.
    dialog.btn_data_imported.click()

    assert len(host.sent) == sent_before
    assert dialog._running
    assert dialog.btn_data_imported.isEnabled()


def test_syscon2_failure_finishes_routine(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    host.block_failed.emit(dialog, "pipe closed")
    assert not dialog._running
    assert not dialog.btn_data_imported.isEnabled()


def test_syscon2_rejects_amplitudes_above_one(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.spn_max_amplitude.setValue(5.0)
    dialog.start()
    assert not host.sent
    assert not dialog._running


def test_syscon2_requires_ttl_quantum_dt(qtbot) -> None:
    _project, editor, dialog, host = _syscon2_dialog(qtbot)
    editor.current_stim_modifier.stim_array.dt = 0.001
    dialog.start()
    assert not host.sent
    assert not dialog._running


def test_syscon2_picks_up_newly_imported_dataset_without_manual_selection(
    qtbot,
) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    _import_spikes(project, dialog, "rec0", spiking=True)
    dialog.btn_data_imported.click()
    sent = len(host.sent)

    # Next probe: a new dataset appears in the project but is never selected by hand.
    onset = dialog._pulse_onset_seconds - dialog._block_range[0] * (
        _SYSCON2_TTL_PULSE_QUANTUM_S
    )
    project.spikes["rec1"] = SpikeData("rec1", np.array([0]), np.array([onset + 0.005]))
    dialog.btn_data_imported.click()

    assert len(host.sent) > sent or not dialog._running
    assert dialog._last_measured_dataset == "rec1"


def test_syscon2_first_probe_rejects_existing_recording_and_picks_new_import(qtbot):
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    _import_spikes(project, dialog, "old", spiking=False)
    dialog.start()
    dialog.btn_data_imported.click()
    assert dialog._running
    assert len(host.sent) == 1
    assert dialog.btn_data_imported.isEnabled()
    onset = dialog._pulse_onset_seconds
    project.spikes["new"] = SpikeData("new", [0], [onset + 0.005])
    dialog.btn_data_imported.click()
    assert dialog._last_measured_dataset == "new"
    assert len(host.sent) == 2


def test_syscon2_accepts_replacement_recording_with_same_name(qtbot):
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    _import_spikes(project, dialog, "recording", spiking=False)
    dialog.start()
    _import_spikes(project, dialog, "recording", spiking=True)
    dialog.btn_data_imported.click()
    assert len(host.sent) == 2
    dialog.btn_data_imported.click()
    assert len(host.sent) == 2
    _import_spikes(project, dialog, "recording", spiking=True)
    dialog.btn_data_imported.click()
    assert len(host.sent) > 2 or not dialog._running


def test_syscon2_does_not_accept_simulation_data_as_an_import(qtbot):
    _project, _editor, dialog, _host = _syscon2_dialog(qtbot)
    assert not dialog._is_fresh_dataset("Simulation :: P / v")


def test_dynamic_syscon_dt_button_resamples_before_start(qtbot, mocker):
    _project, editor, dialog, _host = _syscon2_dialog(qtbot)
    stim = editor.current_stim_modifier.stim_array
    stim.dt = 0.001

    def fix(parent):
        stim.dt = _SYSCON2_TTL_PULSE_QUANTUM_S
        return True

    prepare = mocker.patch.object(editor, "prepare_syscon_timestep", side_effect=fix)
    dialog.btn_syscon_dt.click()
    prepare.assert_called_once_with(dialog)
    assert "0.5208125" in dialog.lbl_stim_dt.text()


def _dff_trace(dialog, name: str, *, responding: bool) -> TraceData:
    """ΔF/F of a recording starting with the probe block, as automatic blocks yield."""
    dt = 1 / 30
    onset = (
        dialog._pulse_onset_seconds
        - dialog._block_range[0] * _SYSCON2_TTL_PULSE_QUANTUM_S
    )
    values = np.zeros((1, 120), dtype=np.float32)
    if responding:
        values[0, int(onset / dt) + 1 :] = 0.5
    return TraceData("dF/F", values, np.array([0]), "dF/F", dt)


def _automatic_dialog(qtbot):
    project, editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.hardware.chk_automatic.setChecked(True)
    dialog.cmb_calcium_readout.setCurrentText(dialog.DFF_READOUT)
    dialog.spn_response_ms.setValue(300.0)
    return project, editor, dialog, host


def test_syscon2_automatic_probe_is_recorded_and_measured_as_dff(qtbot) -> None:
    project, _editor, dialog, host = _automatic_dialog(qtbot)
    assert not dialog.btn_data_imported.isVisibleTo(dialog)
    dialog.hardware.spn_tail_s.setValue(2.0)
    dialog.hardware.spn_channel.setValue(2)

    dialog.start()

    assert host.automatic == [True]
    recording = host.recordings[0]
    start, stop = dialog._block_range
    assert recording.stim_block.values.shape[0] == stop - start
    assert (recording.tail_s, recording.functional_channel) == (2.0, 2)
    assert host.owner is dialog
    for i in range(20):
        if not dialog._running:
            break
        name = f"Block {i} :: dF/F"
        project.traces[name] = _dff_trace(
            dialog, name, responding=dialog._pending_amplitude >= 0.5
        )
        host.complete({"trace": name})

    assert not dialog._running
    assert 0.5 <= dialog._rheobase <= 0.5 + dialog.spn_tolerance.value()
    assert all(host.automatic)


def test_syscon2_automatic_probe_can_be_measured_from_deconvolved_spikes(
    qtbot,
) -> None:
    project, _editor, dialog, host = _automatic_dialog(qtbot)
    dialog.cmb_calcium_readout.setCurrentText(dialog.SPIKES_READOUT)
    assert dialog.cmb_calcium_readout.isEnabled()
    assert not dialog.cmb_spikes.isEnabled()  # the block names its spikes
    assert dialog.hardware.spn_spike_threshold.isVisibleTo(dialog)
    assert not dialog.hardware.spn_threshold.isVisibleTo(dialog)
    dialog.hardware.spn_spike_threshold.setValue(0.25)
    dialog.start()
    assert host.recordings[0].spike_threshold == 0.25
    frame = 1 / 30
    for i in range(20):
        if not dialog._running:
            break
        onset = (
            dialog._pulse_onset_seconds
            - dialog._block_range[0] * _SYSCON2_TTL_PULSE_QUANTUM_S
        )
        trace_name, spikes_name = f"Block {i} :: dF/F", f"Block {i} :: spikes"
        project.traces[trace_name] = _dff_trace(dialog, trace_name, responding=False)
        # Timed at the start of the frame holding the onset: counts with tolerance.
        times = [frame * int(onset / frame)] if dialog._pending_amplitude >= 0.5 else []
        project.spikes[spikes_name] = SpikeData(
            spikes_name, np.zeros(len(times)), np.array(times)
        )
        host.complete({"trace": trace_name, "spikes": spikes_name})

    assert not dialog._running
    assert 0.5 <= dialog._rheobase <= 0.5 + dialog.spn_tolerance.value()
    assert "Deconvolved spikes after the probe: 1." in dialog.log.toPlainText()


def test_syscon2_automatic_needs_a_recorder(qtbot) -> None:
    _project, _editor, dialog, host = _automatic_dialog(qtbot)
    host.unavailable = "Connect to ScanImage first."
    dialog.start()
    assert not host.sent
    assert not dialog._running
    assert "Connect to ScanImage first." in dialog.log.toPlainText()


def test_syscon2_automatic_block_without_trace_finishes(qtbot) -> None:
    _project, _editor, dialog, host = _automatic_dialog(qtbot)
    dialog.start()
    host.complete({})
    assert not dialog._running
    assert "no calcium trace" in dialog.log.toPlainText()


def test_syscon2_ignores_blocks_of_other_routines(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    other = object()
    host.block_failed.emit(other, "someone else's block failed")
    host.block_completed.emit(other, None)
    assert dialog._running
    assert len(host.sent) == 1


def test_syscon2_stop_cancels_only_its_own_block(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    host.owner = object()  # another routine's block is in flight now
    dialog.stop()
    assert host.cancelled == 0
    assert not dialog._running


def test_syscon2_dispose_stops_the_routine_and_releases_the_host(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.start()
    dialog.dispose()
    assert not dialog._running
    assert host.cancelled == 1
    host.block_completed.emit(dialog, None)  # no longer connected
    assert not dialog._running


def test_syscon2_manual_dff_readout_uses_the_imported_trace(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.cmb_calcium_readout.setCurrentText(dialog.DFF_READOUT)
    dialog.spn_response_ms.setValue(300.0)
    assert dialog.cmb_trace.isEnabled()
    assert not dialog.cmb_spikes.isEnabled()
    dialog.start()
    project.traces["rec0"] = _dff_trace(dialog, "rec0", responding=True)
    dialog.btn_data_imported.click()
    assert dialog._last_measured_dataset == "rec0"
    assert len(host.sent) == 2


def test_hardware_settings_edit_the_hosts_export_settings(qtbot) -> None:
    _project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.hardware.spn_power_mw.setValue(23.0)
    dialog.hardware.cmb_laser.setCurrentIndex(dialog.hardware.cmb_laser.findData(1100))
    dialog.hardware.chk_spiral.setChecked(True)
    assert host.settings.target_power_mw == 23.0
    assert host.settings.laser_wavelength_nm == 1100
    assert host.settings.spiralling
    host.settings = host.settings.__class__(config_text="", target_power_mw=5.0)
    dialog.refresh()
    assert dialog.hardware.spn_power_mw.value() == 5.0
