"""End-to-end STDP window sweep on a real Brian2 model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq.example_projects import two_hh_neurons_with_plasticity
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowDialog

# Real Hodgkin-Huxley sweeps run several Brian2 simulation segments using the
# NumPy backend. Allow slower CI runners more than the default 60 seconds.
pytestmark = pytest.mark.timeout(300)


class _BrianEditor:
    def __init__(self, project) -> None:
        self.project = project
        self.stimulation_host = None
        self.bss2_host = None
        self.current_stim_modifier = SimpleNamespace(
            stim_array=project.stim_arrays["Exc_stimulation"]
        )
        self.calls: list[np.ndarray] = []
        self.focused_ranges: list[tuple[float, float]] = []
        self.pad_calls: list[float] = []
        self.cleared = 0

    def clear_stim_array(self) -> None:
        self.cleared += 1
        self.current_stim_modifier.stim_array.values[:] = 0.0

    def zero_current_stim_array(self) -> None:
        self.cleared += 1
        self.current_stim_modifier.stim_array.values[:] = 0.0

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

    def pad_stim_array_to(self, total_length_secs: float) -> None:
        self.pad_calls.append(total_length_secs)
        stim = self.current_stim_modifier.stim_array
        target_samples = int(round(total_length_secs / stim.dt))
        if stim.values.shape[0] < target_samples:
            pad = np.zeros(
                (target_samples - stim.values.shape[0], stim.values.shape[1])
            )
            stim.values = np.vstack((stim.values, pad))
            self.project.stim_arrays[stim.name] = stim
            self.current_stim_modifier.stim_array = stim

    def _target_samples(self) -> int:
        stim = self.current_stim_modifier.stim_array
        return max(1, int(np.ceil(self.project.b2_model.get_sim_time_secs() / stim.dt)))

    def reset_stim_array(self) -> None:
        # Mirrors StimEditor.reset_stim_array(): trim the array back to the
        # current simulation head (never growing it). dynamic_input_blocks
        # .append_block() relies on this actually trimming - the example
        # project's stim arrays ship pre-authored at several seconds long,
        # so without a real trim here every appended probe would land far
        # past the end of the recorded trace.
        self.reset_count = getattr(self, "reset_count", 0) + 1
        stim = self.current_stim_modifier.stim_array
        target_samples = self._target_samples()
        if stim.values.shape[0] > target_samples:
            stim.values = stim.values[:target_samples]

    def current_stim_array_has_unsimulated_tail(self) -> bool:
        stim = self.current_stim_modifier.stim_array
        return bool(np.any(stim.values[self._target_samples() :, :]))

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
    # The fixture ships a stim array with pre-authored content past the
    # (unsimulated) start; confirm clearing it so the sweep runs from a
    # clean slate at the simulation head, matching the carefully controlled
    # initial state set up above - this also exercises the
    # confirm-start-routine path. ("Continue" would instead preserve that
    # pre-authored content and simulate through it first, which is exactly
    # what test_continue_preserves_existing_stimulus_in_continuous_mode
    # covers on a fast fake model.)
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.stdp_window_dialog.confirm_start_routine",
        lambda *args, **kwargs: "clear",
    )

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

    dialog.start()

    assert not dialog._running
    result_by_delta = dict(dialog._results)
    assert result_by_delta[-30.0] < 0.0
    assert result_by_delta[30.0] > 0.0
    assert dialog.cmb_trace.currentText().startswith("Simulation :: ")
    assert project.stim_arrays["Exc_stimulation"].values.shape[0] > 2000
    assert editor.reset_count >= 1
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
    monkeypatch.setattr(
        "neuroseq.stimulation.dynamic_input.stdp_window_dialog.confirm_start_routine",
        lambda *args, **kwargs: "continue",
    )

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

    dialog.start()

    assert not dialog._running
    result_by_delta = dict(dialog._results)
    assert result_by_delta[-30.0] < 0.0  # post-before-pre depresses (LTD)
    assert result_by_delta[30.0] > 0.0  # pre-before-post potentiates (LTP)
    # Weight mode reads the monitor, so no pre-only probe blocks are appended
    # (reset mode's own per-point zero-out write - see _restore_baseline() in
    # stdp_window_dialog.py - is excluded here, since it is neither).
    for block in editor.calls:
        if not np.any(block):
            continue
        assert np.count_nonzero(block[:, 0]) > 0
        assert np.count_nonzero(block[:, 1]) > 0
    # Reset mode stored a baseline checkpoint and restored it before each point.
    assert any("STDP window baseline" in name for name in b2_model.checkpoints)
