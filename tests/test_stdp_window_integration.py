"""End-to-end STDP window sweep on a real Brian2 model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from neuroseq.example_projects import two_hh_neurons_with_plasticity
from neuroseq.stdp_window_dialog import StdpWindowDialog


class _BrianEditor:
    def __init__(self, project) -> None:
        self.project = project
        self.current_stim_modifier = SimpleNamespace(
            stim_array=project.stim_arrays["Exc_stimulation"]
        )
        self.calls: list[np.ndarray] = []
        self.focused_ranges: list[tuple[float, float]] = []
        self.cleared = 0
        self.cleared_all = 0

    def clear_stim_array(self) -> None:
        self.cleared += 1
        self.current_stim_modifier.stim_array.values[:] = 0.0

    def clear_all_stim_arrays(self) -> None:
        self.cleared_all += 1
        for stim in self.project.stim_arrays.values():
            stim.values[:] = 0.0

    def focus_time_range(self, start: float, stop: float) -> None:
        self.focused_ranges.append((start, stop))

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        stop = int(time_slice.stop)
        if stop > stim.values.shape[0]:
            pad = np.zeros((stop - stim.values.shape[0], stim.values.shape[1]))
            stim.values = np.vstack((stim.values, pad))
            self.project.stim_arrays[stim.name] = stim
            self.current_stim_modifier.stim_array = stim
        stim.values[time_slice, :] = values


def test_stdp_window_closed_loop_measures_ltd_and_ltp_on_real_model(
    qtbot, monkeypatch
) -> None:
    project = two_hh_neurons_with_plasticity()
    b2_model = project.b2_model
    b2_model.define_init("P", "v", "El")
    b2_model.define_init("P", "ge", "0*nS")
    b2_model.define_init("P", "gi", "0*nS")
    b2_model.define_init("Ce", "we", "6*nS")

    editor = _BrianEditor(project)
    dialog = StdpWindowDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_prompt_existing_input_choice", lambda names: "clear")

    def run(seconds: float) -> None:
        b2_model.run_simulation(seconds, project.stim_arrays)
        dialog.simulation_finished()

    dialog.simulation_requested.connect(run)

    # Continuous mode: append-and-run-forward, no checkpoints.
    dialog.cmb_sweep_mode.setCurrentText(dialog.CONTINUOUS_MODE)
    dialog.spn_presynaptic.setValue(0)
    dialog.spn_postsynaptic.setValue(1)
    dialog.spn_dt_min_ms.setValue(-30.0)
    dialog.spn_dt_max_ms.setValue(30.0)
    dialog.spn_dt_points.setValue(3)
    dialog.chk_include_zero.setChecked(False)
    dialog.spn_pairings_per_point.setValue(1)
    dialog.spn_recovery_ms.setValue(150.0)
    dialog.spn_response_ms.setValue(50.0)
    dialog.spn_pulse_ms.setValue(5.0)
    dialog.spn_amplitude.setValue(100.0)
    dialog.chk_follow_input.setChecked(False)

    dialog.start()

    assert not dialog._running
    result_by_delta = dict(dialog._results)
    assert result_by_delta[-30.0] < 0.0
    assert result_by_delta[30.0] > 0.0
    assert dialog.cmb_trace.currentText().startswith("Simulation :: ")
    assert project.stim_arrays["Exc_stimulation"].values.shape[0] > 2000
    assert editor.cleared_all == 1
    assert b2_model.checkpoints == ["Initial state"]


def test_stdp_window_weight_readout_measures_ltd_and_ltp(qtbot, monkeypatch) -> None:
    """The direct-weight readout traces the window without needing an EPSP."""
    project = two_hh_neurons_with_plasticity()
    b2_model = project.b2_model
    b2_model.define_init("P", "v", "El")
    b2_model.define_init("P", "ge", "0*nS")
    b2_model.define_init("P", "gi", "0*nS")
    b2_model.define_init("Ce", "we", "6*nS")

    editor = _BrianEditor(project)
    dialog = StdpWindowDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "_prompt_existing_input_choice", lambda names: "clear")

    def run(seconds: float) -> None:
        b2_model.run_simulation(seconds, project.stim_arrays)
        dialog.simulation_finished()

    dialog.simulation_requested.connect(run)

    # Reset mode (the default): each Δt is measured from the baseline weight via
    # a store/restore checkpoint, giving a clean window on the real synapse.
    dialog.cmb_readout.setCurrentText(dialog.WEIGHT_READOUT)
    dialog.refresh()
    dialog.spn_presynaptic.setValue(0)
    dialog.spn_postsynaptic.setValue(1)
    dialog.spn_dt_min_ms.setValue(-30.0)
    dialog.spn_dt_max_ms.setValue(30.0)
    dialog.spn_dt_points.setValue(3)
    dialog.spn_pairings_per_point.setValue(1)
    dialog.spn_recovery_ms.setValue(150.0)
    dialog.spn_pulse_ms.setValue(2.0)
    dialog.spn_amplitude.setValue(100.0)
    dialog.chk_follow_input.setChecked(False)

    dialog.start()

    assert not dialog._running
    result_by_delta = dict(dialog._results)
    assert result_by_delta[-30.0] < 0.0  # post-before-pre depresses (LTD)
    assert result_by_delta[30.0] > 0.0  # pre-before-post potentiates (LTP)
    # Weight mode reads the monitor, so no pre-only probe blocks are appended.
    for block in editor.calls:
        assert np.count_nonzero(block[:, 0]) > 0
        assert np.count_nonzero(block[:, 1]) > 0
    # Reset mode stored a baseline checkpoint and restored it before each point.
    assert any("STDP window baseline" in name for name in b2_model.checkpoints)
