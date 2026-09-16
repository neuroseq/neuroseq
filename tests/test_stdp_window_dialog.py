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
from neuroseq.input_designer_dialog import EPSPMeasurement
from neuroseq.stdp_window_dialog import StdpWindowDialog
from neuroseq.stdp_window_dialog import _NoEpspDetected
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


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
        self.cleared_all = 0
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

    def clear_all_stim_arrays(self) -> None:
        self.cleared_all += 1
        for stim in self.project.stim_arrays.values():
            stim.values[:] = 0.0

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice
        self.time_slices.append(time_slice)
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        stop = int(time_slice.stop)
        if stop > stim.values.shape[0]:
            pad = np.zeros((stop - stim.values.shape[0], stim.values.shape[1]))
            stim.values = np.vstack((stim.values, pad))
            self.project.stim_arrays[stim.name] = stim
        stim.values[time_slice, :] = values


def _make_dialog(qtbot) -> tuple[_FakeProject, _FakeEditor, StdpWindowDialog]:
    project = _FakeProject()
    editor = _FakeEditor(project)
    stim = editor.current_stim_modifier.stim_array
    project.stim_arrays = {stim.name: stim}
    dialog = StdpWindowDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    return project, editor, dialog


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
    assert len(editor.calls) == 1 + 4 * (3 + 1)
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
    pairing = editor.calls[1]

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


def test_warns_and_can_clear_existing_input(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.values[0, 0] = 1.0
    monkeypatch.setattr(dialog, "_prompt_existing_input_choice", lambda names: "clear")
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()

    assert editor.cleared_all == 1
    assert editor.current_stim_modifier.stim_array.values[0, 0] == 0.0
    assert len(editor.calls) == 1


def test_cancel_existing_input_keeps_input(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    editor.current_stim_modifier.stim_array.values[0, 0] = 1.0
    monkeypatch.setattr(dialog, "_prompt_existing_input_choice", lambda names: "cancel")

    dialog.start()

    assert not dialog._running
    assert editor.calls == []
    assert editor.cleared_all == 0


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
    assert dialog.window_plot.parent() is dialog.plot_window


def test_stim_heatmap_opens_in_its_own_window(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)
    qtbot.addWidget(dialog.stim_heatmap_window)

    assert dialog.stim_heatmap_window not in (dialog, dialog.plot_window)
    assert dialog.stim_heatmap_window.isWindow()

    dialog.show()
    qtbot.waitUntil(lambda: dialog.stim_heatmap_window.isVisible())
    dialog.close()
    qtbot.waitUntil(lambda: not dialog.stim_heatmap_window.isVisible())


def test_updates_stim_heatmap_as_blocks_are_appended(qtbot, monkeypatch) -> None:
    _project, editor, dialog = _make_dialog(qtbot)
    _patch_directional_measurement(monkeypatch, dialog)

    dialog.start()
    image = dialog.stim_heatmap_window._image
    assert image is not None
    stim = editor.current_stim_modifier.stim_array
    assert image.get_array().shape[1] == stim.values.shape[0]


def test_forwards_theme_to_stim_heatmap(qtbot) -> None:
    _project, _editor, dialog = _make_dialog(qtbot)

    dialog.apply_theme(theme.DARK)
    assert to_hex(dialog.stim_heatmap_window.axes.get_facecolor()) == "#1e1e1e"
    dialog.apply_theme(theme.LIGHT)
    assert to_hex(dialog.stim_heatmap_window.axes.get_facecolor()) == "#ffffff"


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
    for block in editor.calls:
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
    pairing = editor.calls[1]
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
