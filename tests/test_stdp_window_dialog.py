from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.stimulation.dynamic_input.stdp_synaptic_tuning_dialog import (
    EPSPMeasurement,
)
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowDialog
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import _NoEpspDetected
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from tests.fake_stimulation_host import FakeStimulationHost


class _FakeProject:
    def __init__(self) -> None:
        self.trace_available = True
        self.trace = TraceData(
            "Simulation :: P / v",
            np.zeros((3, 4000), dtype=np.float32),
            np.array([0, 1, 2]),
            "mV",
            0.001,
        )
        self.traces = {self.trace.name: self.trace}
        self.synapses: dict[str, SynapseData] = {}
        self.stim_arrays: dict[str, StimArray] = {}
        self.stored_checkpoints: list[str] = []
        self.restored_checkpoints: list[str] = []
        self.b2_model = SimpleNamespace(
            get_sim_time_secs=lambda: 0.0,
            store=self.stored_checkpoints.append,
            restore=self.restored_checkpoints.append,
        )

    def get_data_names(self, data_type: type) -> list[str]:
        if data_type is TraceData:
            return list(self.traces) if self.trace_available else []
        if data_type is SynapseData:
            return list(self.synapses)
        return []

    def get_data(self, name: str, data_type: type):
        if data_type is TraceData:
            return self.traces.get(name)
        if data_type is SynapseData:
            return self.synapses.get(name)
        return None


class _FakeEditor:
    def __init__(self, project: _FakeProject) -> None:
        self.project = project
        self.stimulation_host = None
        self.bss2_host = None
        stim = StimArray(
            "Exc_stimulation",
            StimType.Excitation,
            np.zeros((10, 3)),
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
            self.project.stim_arrays[stim.name] = stim

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
        self._ensure_len(int(time_slice.stop))
        stim.values[time_slice, :] = values


def _make_dialog(qtbot) -> tuple[_FakeProject, _FakeEditor, StdpWindowDialog]:
    project = _FakeProject()
    editor = _FakeEditor(project)
    stim = editor.current_stim_modifier.stim_array
    project.stim_arrays = {stim.name: stim}
    dialog = StdpWindowDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    return project, editor, dialog


def _real_blocks(editor: "_FakeEditor") -> list[np.ndarray]:
    """The actual probe/pairing blocks, excluding reset-mode's per-point
    zero-out write (see _restore_baseline() in stdp_window_dialog.py) - that
    write goes through the same apply_pattern() path as every other block,
    but is entirely zero, so filtering it out is unambiguous."""
    return [block for block in editor.calls if np.any(block)]


def _finish_sweep(dialog: StdpWindowDialog, limit: int = 200) -> None:
    for _ in range(limit):
        if not dialog._running:
            return
        dialog.simulation_finished()
    raise AssertionError("STDP window sweep did not finish.")


def _patch_directional_measurement(
    monkeypatch, dialog: StdpWindowDialog, baseline: float = 2.0
) -> list[str]:
    # Reset-mode friendly: the measured EPSP depends only on the current Δt
    # (each point is independent), so LTP/LTD signs hold regardless of order.
    phases: list[str] = []

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        phases.append(dialog._phase)
        if dialog._phase == "baseline_probe":
            value = baseline
        elif dialog._pending_delta_t > 0.0:
            value = baseline + 0.2
        elif dialog._pending_delta_t < 0.0:
            value = baseline - 0.2
        else:
            value = baseline
        return EPSPMeasurement(value, (value,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)
    return phases


def test_sweeps_all_delta_t_points_and_builds_window(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    _finish_sweep(dialog)

    assert [delta_t for delta_t, _percent in dialog._results] == [
        -100.0,
        -50.0,
        50.0,
        100.0,
    ]
    for delta_t, percent in dialog._results:
        assert percent > 0.0 if delta_t > 0.0 else percent < 0.0


def test_baseline_probe_measured_before_first_pairing(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    phases = _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    assert np.count_nonzero(editor.calls[0][:, 1]) == 0
    dialog.simulation_finished()

    assert phases == ["baseline_probe"]
    assert dialog._phase == "pair"
    assert dialog._baseline_epsp == pytest.approx(2.0)


def test_pairings_per_point_applies_n_pairings_then_one_probe(
    qtbot, monkeypatch
) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)
    dialog.spn_pairings_per_point.setValue(3)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    _finish_sweep(dialog)

    assert len(dialog._results) == 4
    assert len(_real_blocks(editor)) == 1 + 4 * (3 + 1)
    assert "pairing 3/3" in dialog.log.toPlainText()


def test_continuous_mode_appends_each_block_forward(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    dialog.cmb_sweep_mode.setCurrentText(dialog.CONTINUOUS_MODE)
    dialog.spn_dt_points.setValue(3)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    _finish_sweep(dialog)

    starts = [time_slice.start for time_slice in editor.time_slices]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    # No checkpoints in continuous mode.
    assert _project.stored_checkpoints == []
    # Continuous mode grows forward through a RoutineArrayCursor, so the
    # array gets padded ahead (and the camera focused) at least once.
    assert editor.pad_calls
    assert editor.focused_ranges
    assert dialog._array_cursor is None  # cleared once the sweep finished
    assert editor.reset_count >= 1  # leftover padding trimmed on finish


def test_reset_mode_restores_baseline_and_overwrites_region(qtbot, monkeypatch) -> None:
    project, editor, dialog = _make_dialog(qtbot)  # reset is the default
    dialog.spn_dt_points.setValue(3)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    _finish_sweep(dialog)

    # A baseline checkpoint is stored once and restored before each Δt point.
    assert len(project.stored_checkpoints) == 1
    assert len(project.restored_checkpoints) == len(dialog._results)
    # Each point overwrites the same fixed region, so the pairing blocks reuse a
    # small set of start samples rather than marching forward indefinitely.
    starts = [time_slice.start for time_slice in editor.time_slices]
    assert min(starts) == 0
    assert max(starts) < sum(block.shape[0] for block in editor.calls)
    # Unlike continuous mode, reset mode pads the region it will reuse
    # exactly once, up front - every point overwrites the same space
    # rather than growing forward, so there is nothing to re-pad or jump
    # the camera for.
    assert len(editor.pad_calls) == 1
    assert len(editor.focused_ranges) == 1
    assert dialog._array_cursor is None


def test_pairing_only_on_pre_and_post_columns(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    dialog.spn_presynaptic.setValue(0)
    dialog.spn_postsynaptic.setValue(1)
    dialog.spn_dt_min_ms.setValue(20.0)
    dialog.spn_dt_max_ms.setValue(30.0)
    dialog.spn_dt_points.setValue(2)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    dialog.simulation_finished()
    pairing = _real_blocks(editor)[1]  # [0] is the baseline probe

    assert np.count_nonzero(pairing[:, 0]) > 0
    assert np.count_nonzero(pairing[:, 1]) > 0
    assert np.count_nonzero(pairing[:, 2]) == 0


def test_probe_only_on_pre_column(qtbot) -> None:
    _project, editor, dialog = _make_dialog(qtbot)

    dialog.start()
    probe = editor.calls[0]

    assert np.count_nonzero(probe[:, 0]) > 0
    assert np.count_nonzero(probe[:, 1]) == 0
    assert np.count_nonzero(probe[:, 2]) == 0


def test_reports_saturation_warning_when_epsp_drifts_far(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    values = iter([2.0, 10.0])

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        value = next(values)
        return EPSPMeasurement(value, (value,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)
    dialog.spn_dt_min_ms.setValue(20.0)
    dialog.spn_dt_max_ms.setValue(30.0)
    dialog.spn_dt_points.setValue(2)

    dialog.start()
    dialog.simulation_finished()
    dialog.simulation_finished()
    dialog.simulation_finished()

    assert "synaptic saturation" in dialog.log.toPlainText()


def test_estimate_matches_run_count(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)
    dialog.spn_pairings_per_point.setValue(2)

    assert dialog._estimated_probe_count() == 1 + 4 * (2 + 1)
    assert "~13" in dialog.lbl_estimate.text()


def test_continue_preserves_existing_stimulus_in_continuous_mode(
    qtbot, monkeypatch
) -> None:
    """ "Continue" keeps existing stimulus content - even content that has
    never been simulated - rather than discarding it. Continuous mode is
    selected explicitly: reset mode always anchors to the actual simulation
    head (see test_reset_mode_restores_baseline_and_overwrites_region), so
    this is where "continue" vs. "clear" actually changes the cursor."""
    _project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.values[5, 0] = 1.0
    dialog.cmb_sweep_mode.setCurrentText(dialog.CONTINUOUS_MODE)
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.stdp_window_dialog.confirm_start_routine",
        lambda *args, **kwargs: "continue",
    )
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    # The fake simulation never advances its own clock, so the routine must
    # catch it up to the pre-existing stimulus before appending any probes.
    assert editor.calls == []
    assert dialog._running
    assert dialog._phase == "initialize_trace"
    # "Continue" itself never trims - unlike "clear" it does not touch the
    # array at all before appending.
    assert getattr(editor, "reset_count", 0) == 0

    dialog.simulation_finished()

    assert len(editor.calls) == 1
    assert editor.current_stim_modifier.stim_array.values[5, 0] == 1.0
    assert editor.time_slices[0].start == 6  # right after the last stimulus


def test_clear_zeroes_existing_input_before_start(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.values[5, 0] = 1.0
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.stdp_window_dialog.confirm_start_routine",
        lambda *args, **kwargs: "clear",
    )
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    assert len(editor.calls) == 1
    assert editor.cleared == 1
    # Trimmed back to the simulation head afterwards, same as reset_stim_array()
    # always did - there is nothing real left to lose once the array is zeroed.
    assert getattr(editor, "reset_count", 0) == 1
    # Default sweep mode is reset-mode, which always anchors to the actual
    # simulation head (here 0, the fake's constant sim time) regardless of
    # clear/continue - "clear" only matters for continuous mode's cursor.
    assert editor.time_slices[0].start == 0


def test_cancel_keeps_existing_input(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.values[5, 0] = 1.0
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.stdp_window_dialog.confirm_start_routine",
        lambda *args, **kwargs: "cancel",
    )

    dialog.start()

    assert not dialog._running
    assert editor.calls == []


def test_no_warning_when_stim_array_is_empty(qtbot, monkeypatch) -> None:
    """An all-zero array needs no start-routine prompt at all -
    confirm_start_routine() itself short-circuits to "continue" without
    showing any dialog, so a sweep starts immediately."""
    _project, editor, dialog = _make_dialog(qtbot)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    assert len(editor.calls) == 1


def test_stops_when_dialog_closed(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    assert dialog._running
    dialog.close()

    assert not dialog._running
    assert "dialog closed" in dialog.log.toPlainText()


def test_ignores_unrequested_simulation(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    phases = _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    blocks_after_start = len(editor.calls)
    dialog._awaiting_simulation = False
    dialog.simulation_finished()

    assert phases == []
    assert len(editor.calls) == blocks_after_start
    assert dialog._running


def test_initializes_missing_trace_then_starts_sweep(qtbot, monkeypatch) -> None:
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
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    assert requested == [0.001]
    assert dialog._phase == "initialize_trace"
    project.trace_available = True
    dialog.simulation_finished()
    assert len(editor.calls) == 1
    assert dialog._phase == "baseline_probe"


def test_plots_open_in_separate_window(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    qtbot.addWidget(dialog.plot_window)

    assert dialog.plot_window is not dialog
    assert dialog.plot_window.isWindow()
    assert dialog.window_plot.window() is dialog.plot_window


def test_stdp_window_plot_window_is_scrollable(qtbot) -> None:
    """So a short screen scrolls the plots instead of squeezing the figure
    until axis labels overlap or get clipped off."""
    _project, _editor, dialog = _make_dialog(qtbot)
    qtbot.addWidget(dialog.plot_window)

    scroll_areas = dialog.plot_window.findChildren(QtWidgets.QScrollArea)
    assert len(scroll_areas) == 1
    assert scroll_areas[0].widget() is dialog.window_plot
    assert scroll_areas[0].widgetResizable()


def test_matplotlib_plots_follow_theme(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.window_axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(dialog.trajectory_axes.get_facecolor()) == "#1e1e1e"

    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.window_axes.get_facecolor()) == "#ffffff"
    assert to_hex(dialog.trajectory_axes.get_facecolor()) == "#ffffff"


def _use_weight_readout(project, dialog) -> None:
    # A synapse weight monitor must exist for the weight readout.
    project.synapses = {
        "Simulation :: Ce / we": SynapseData(
            "Simulation :: Ce / we",
            np.full((3, 3, 2), 6.0, dtype=np.float32),
            np.array([0.0, 0.001], dtype=np.float32),
        )
    }
    dialog.cmb_readout.setCurrentText(dialog.WEIGHT_READOUT)
    dialog.refresh()


def test_weight_readout_hides_epsp_only_controls(qtbot) -> None:
    project, _editor, dialog = _make_dialog(qtbot)
    _use_weight_readout(project, dialog)

    assert dialog.cmb_synapse.isVisibleTo(dialog.parameters_panel)
    assert not dialog.cmb_trace.isVisibleTo(dialog.parameters_panel)
    assert not dialog.spn_response_ms.isVisibleTo(dialog.parameters_panel)
    # No probe runs in weight mode: estimate is init + pairings only.
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)
    dialog.spn_pairings_per_point.setValue(1)
    assert dialog._estimated_probe_count() == 1 + 4 * 1


def test_weight_readout_sweeps_without_probes(qtbot, monkeypatch) -> None:
    project, editor, dialog = _make_dialog(qtbot)
    _use_weight_readout(project, dialog)
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)

    def read_weight() -> float:
        # Reset mode: weight after one pairing depends only on the current Δt.
        if dialog._pending_delta_t > 0.0:
            return 6.5
        if dialog._pending_delta_t < 0.0:
            return 5.5
        return 6.0  # baseline read (pending Δt still 0)

    monkeypatch.setattr(dialog, "_read_synapse_weight", read_weight)

    dialog.start()
    _finish_sweep(dialog)

    assert [delta_t for delta_t, _p in dialog._results] == [-100.0, -50.0, 50.0, 100.0]
    for delta_t, percent in dialog._results:
        assert percent > 0.0 if delta_t > 0.0 else percent < 0.0
    # Every appended block is a two-column pairing (no pre-only probe blocks).
    for block in _real_blocks(editor):
        assert np.count_nonzero(block[:, 0]) > 0
        assert np.count_nonzero(block[:, 1]) > 0
    assert "% weight" in dialog.log.toPlainText()


def test_undetectable_epsp_point_is_recorded_nan_and_continues(
    qtbot, monkeypatch
) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(-100.0)
    dialog.spn_dt_max_ms.setValue(100.0)
    dialog.spn_dt_points.setValue(5)
    call = {"n": 0}

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        call["n"] += 1
        if call["n"] == 2:  # first post-pairing probe: undetectable
            raise _NoEpspDetected("no EPSP")
        return EPSPMeasurement(2.0, (2.0,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)

    dialog.start()
    _finish_sweep(dialog)

    # All four points recorded; the first is NaN, the rest finite, sweep finished.
    assert len(dialog._results) == 4
    assert dialog._results[0][1] != dialog._results[0][1]  # NaN
    assert all(p == p for _dt, p in dialog._results[1:])
    assert "undetectable" in dialog.log.toPlainText()
    assert "finished" in dialog.log.toPlainText()


def test_no_baseline_epsp_aborts_with_weight_hint(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        raise _NoEpspDetected("no EPSP")

    monkeypatch.setattr(dialog, "_measure_epsps", measure)

    dialog.start()
    dialog.simulation_finished()

    assert not dialog._running
    assert "Synaptic weight" in dialog.log.toPlainText()


def test_delta_t_recorded_as_effective_grid_value(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(20.0)
    dialog.spn_dt_max_ms.setValue(20.4)
    dialog.spn_dt_points.setValue(2)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    _finish_sweep(dialog)

    assert dialog._results == [(20.0, pytest.approx(10.0))]
    pairing = _real_blocks(editor)[1]  # [0] is the baseline probe
    pre_sample = int(np.flatnonzero(pairing[:, 0])[0])
    post_sample = int(np.flatnonzero(pairing[:, 1])[0])
    assert post_sample - pre_sample == 20


def test_warns_when_delta_t_step_finer_than_dt(qtbot, monkeypatch) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    dialog.spn_dt_min_ms.setValue(20.0)
    dialog.spn_dt_max_ms.setValue(20.4)
    dialog.spn_dt_points.setValue(2)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    assert "Δt values are snapped" in dialog.log.toPlainText()


def test_wheel_over_field_does_not_change_value(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    spin = dialog.spn_dt_points
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

    assert dialog._wheel_filter.eventFilter(spin, wheel) is True
    QtWidgets.QApplication.sendEvent(spin, wheel)
    assert spin.value() == before


def test_stdp_pause_defers_next_simulation_and_resets_on_finish(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    requested: list[float] = []
    dialog.simulation_requested.connect(requested.append)
    dialog._running = True
    dialog.btn_pause.setEnabled(True)

    dialog.toggle_pause()
    dialog._request_simulation(0.75)
    assert dialog._paused
    assert dialog._pending_simulation_seconds == 0.75
    assert not dialog._awaiting_simulation
    assert requested == []

    dialog.toggle_pause()
    assert requested == [0.75]
    assert dialog._awaiting_simulation
    dialog._finish("done")
    assert not dialog._paused
    assert dialog._pending_simulation_seconds is None
    assert not dialog.btn_pause.isEnabled()
    assert dialog.btn_pause.text() == "Pause"


def _syscon2_dialog(qtbot):
    project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.dt = _SYSCON2_TTL_PULSE_QUANTUM_S
    host = FakeStimulationHost()
    editor.stimulation_host = host  # type: ignore[assignment]
    dialog.refresh()
    dialog.cmb_target.setCurrentText(dialog.SYSCON2_TARGET)
    # Manual delivery: the user runs, records and imports each block.
    dialog.hardware.chk_automatic.setChecked(False)
    dialog.spn_amplitude.setValue(0.8)
    dialog.spn_dt_points.setValue(2)
    dialog.chk_include_zero.setChecked(True)
    dialog.spn_dt_min_ms.setValue(-10.0)
    dialog.spn_dt_max_ms.setValue(10.0)
    dialog.spn_response_ms.setValue(60.0)
    return project, editor, dialog, host


def _import_calcium(project, dialog, name: str, amplitude: float) -> None:
    block_start = dialog._block_range[0] * _SYSCON2_TTL_PULSE_QUANTUM_S
    dt = 1 / 30
    values = np.zeros((3, 600), dtype=np.float32)
    for onset in dialog._probe_pulse_times:
        idx = int((onset - block_start) / dt) + 1
        values[1, idx : idx + 2] = amplitude
    project.traces[name] = TraceData(name, values, np.array([0, 1, 2]), "dF/F", dt)
    dialog.refresh()
    dialog.cmb_trace.setCurrentText(name)


def test_syscon2_target_forces_continuous_calcium_mode(qtbot) -> None:
    _project, _editor, dialog, _host = _syscon2_dialog(qtbot)
    assert not dialog._reset_mode()
    assert not dialog._weight_mode()
    assert not dialog.cmb_sweep_mode.isEnabled()
    assert not dialog.cmb_readout.isEnabled()


def test_syscon2_target_unavailable_without_host(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    assert not dialog.cmb_target.model().item(1).isEnabled()


def test_syscon2_sweep_runs_from_imported_calcium_traces(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace  # something selectable at start
    dialog.cmb_trace.setCurrentText("seed")
    dialog.start()
    assert dialog._running

    responses = iter([1.0, 1.5, 0.5])  # baseline, then one probe per dt point
    for i in range(30):
        if not dialog._running:
            break
        assert dialog.btn_data_imported.isEnabled()
        if dialog._phase != "pair":
            _import_calcium(project, dialog, f"rec{i}", next(responses))
        dialog.btn_data_imported.click()

    assert not dialog._running
    assert [round(p) for _dt, p in dialog._backend.results] == [50, -67]
    assert all(preserve for _a, _b, preserve in host.sent)
    # baseline probe + (pair + probe) per dt point
    assert len(host.sent) == 5
    assert not dialog.btn_data_imported.isEnabled()


def test_syscon2_pairing_block_needs_no_new_dataset(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace
    dialog.cmb_trace.setCurrentText("seed")
    dialog.start()
    _import_calcium(project, dialog, "base", 1.0)
    dialog.btn_data_imported.click()
    assert dialog._phase == "pair"
    sent = len(host.sent)
    # A second click on the pairing block advances without any dataset change.
    dialog.btn_data_imported.click()
    assert len(host.sent) == sent + 1
    assert dialog._phase == "baseline_probe" or dialog._phase == "probe"


def test_syscon2_reused_trace_does_not_advance(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace
    dialog.cmb_trace.setCurrentText("seed")
    dialog.start()
    _import_calcium(project, dialog, "base", 1.0)
    dialog.btn_data_imported.click()  # baseline measured, pairing sent
    dialog.btn_data_imported.click()  # pairing done, probe sent
    sent = len(host.sent)
    dialog.cmb_trace.setCurrentText("base")  # stale: same as baseline probe
    dialog.btn_data_imported.click()
    assert len(host.sent) == sent
    assert dialog._running
    assert dialog.btn_data_imported.isEnabled()


def test_syscon2_failure_finishes_sweep(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace
    dialog.cmb_trace.setCurrentText("seed")
    dialog.start()
    host.block_failed.emit(dialog, "pipe closed")
    assert not dialog._running


def test_syscon2_rejects_amplitude_above_one_and_wrong_dt(qtbot) -> None:
    project, editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace
    dialog.cmb_trace.setCurrentText("seed")
    dialog.spn_amplitude.setValue(5.0)
    dialog.start()
    assert not host.sent and not dialog._running
    dialog.spn_amplitude.setValue(0.5)
    editor.current_stim_modifier.stim_array.dt = 0.001
    dialog.start()
    assert not host.sent and not dialog._running


def test_syscon2_picks_up_newly_imported_trace_without_manual_selection(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["seed"] = project.trace
    dialog.cmb_trace.setCurrentText("seed")
    dialog.start()
    _import_calcium(project, dialog, "base", 1.0)
    dialog.btn_data_imported.click()  # baseline measured
    assert dialog._last_measured_dataset == "base"
    dialog.btn_data_imported.click()  # pairing done, probe sent

    # New recording is added to the project but never selected by hand.
    _import_calcium(project, dialog, "after", 1.5)
    dialog.cmb_trace.setCurrentText("base")  # selection still the previous probe's
    dialog.btn_data_imported.click()

    assert dialog._last_measured_dataset == "after"


def test_syscon2_first_probe_rejects_existing_trace_and_picks_new_import(qtbot):
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["old"] = project.trace
    dialog.refresh()
    dialog.cmb_trace.setCurrentText("old")
    dialog.start()
    dialog.btn_data_imported.click()
    assert dialog._running
    assert len(host.sent) == 1
    assert dialog.btn_data_imported.isEnabled()
    _import_calcium(project, dialog, "new", 1.0)
    dialog.cmb_trace.setCurrentText("old")
    dialog.btn_data_imported.click()
    assert dialog._last_measured_dataset == "new"
    assert len(host.sent) == 2


def test_syscon2_accepts_replacement_trace_with_same_name(qtbot):
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces["recording"] = project.trace
    dialog.refresh()
    dialog.cmb_trace.setCurrentText("recording")
    dialog.start()
    _import_calcium(project, dialog, "recording", 1.0)
    dialog.btn_data_imported.click()
    assert len(host.sent) == 2
    dialog.btn_data_imported.click()
    assert len(host.sent) == 3
    dialog.btn_data_imported.click()
    assert len(host.sent) == 3
    _import_calcium(project, dialog, "recording", 1.5)
    dialog.btn_data_imported.click()
    assert len(host.sent) > 3


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


def test_syscon2_automatic_records_only_baseline_and_probe_blocks(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    dialog.hardware.chk_automatic.setChecked(True)
    assert not dialog.btn_data_imported.isVisibleTo(dialog)
    dialog.start()
    assert dialog._running

    responses = iter([1.0, 1.5, 0.5])  # baseline, then one probe per dt point
    for i in range(30):
        if not dialog._running:
            break
        if dialog._phase == "pair":
            assert host.recordings[-1] is None
            host.complete({})
            continue
        assert host.recordings[-1] is not None
        name = f"Block {i} :: dF/F"
        _import_calcium(project, dialog, name, next(responses))
        dialog.cmb_trace.setCurrentText("")  # the payload names the trace
        host.complete({"trace": name})

    assert not dialog._running
    assert [round(p) for _dt, p in dialog._backend.results] == [50, -67]
    assert host.automatic == [True] * 5
    assert [r is not None for r in host.recordings] == [True, False, True, False, True]


def test_syscon2_start_no_longer_needs_a_selected_trace(qtbot) -> None:
    project, _editor, dialog, host = _syscon2_dialog(qtbot)
    project.traces.clear()
    dialog.refresh()
    dialog.start()
    assert dialog._running
    assert len(host.sent) == 1


def test_target_is_locked_while_the_sweep_runs(qtbot) -> None:
    _project, _editor, dialog, _host = _syscon2_dialog(qtbot)
    dialog.start()
    assert dialog._running and not dialog.cmb_target.isEnabled()
    dialog.stop()
    assert dialog.cmb_target.isEnabled()
