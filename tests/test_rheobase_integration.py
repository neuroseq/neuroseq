"""End-to-end closed-loop test of the rheobase routine on a real Brian2 model.

The dialog's unit tests fake ``_measure_spikes`` and the editor, so the risky
new behaviour — appending probe blocks, having Brian2 rebuild its ``TimedArray``
from the grown stim array on every run, and detecting spikes in the real
recorded trace — is never exercised there. This drives the whole chain against
the ``test_project`` HH model with deterministic initial conditions.
"""

from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.rheobase_dialog import RheobaseDialog
from neuroseq.stim_editor import StimEditor


def test_rheobase_closed_loop_finds_threshold_on_real_model(
    qtbot, test_project, monkeypatch
) -> None:
    project = test_project
    b2_model = project.b2_model
    # Deterministic rest state: sit at El with no synaptic drive, so the only
    # depolarisation is the probe pulse. Removes the fixture's RNG initial
    # conditions (and their spontaneous spikes), making the threshold stable.
    b2_model.define_init("P", "v", "El")
    b2_model.define_init("P", "ge", "0*nS")
    b2_model.define_init("P", "gi", "0*nS")
    # The fixture's stim feeds an excitatory conductance; relabel it to
    # nanosiemens so the (steady-state ~20 nS) rheobase is representable in the
    # amplitude spin box's three decimals.
    project.stim_arrays["stimulus"].units = "nS"

    controller = ConstrainedPanZoomController()
    editor = StimEditor(controller, QtWidgets.QComboBox(), QtWidgets.QToolBar())
    qtbot.addWidget(editor)
    editor.set_project(project)

    dialog = RheobaseDialog(editor)
    qtbot.addWidget(dialog)

    # The fixture ships a stim array with pre-authored content past the
    # (unsimulated) start; confirm discarding it so the probes run from the
    # simulation head (this also exercises the confirm-and-discard path).
    monkeypatch.setattr(
        "neuroseq.rheobase_dialog.confirm_discard_unsimulated_input",
        lambda *args, **kwargs: True,
    )

    def run(seconds: float) -> None:
        b2_model.run_simulation(seconds, project.stim_arrays)
        dialog.simulation_finished()

    dialog.simulation_requested.connect(run)

    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
    )
    dialog.spn_neuron.setValue(0)
    dialog.spn_recovery_ms.setValue(40.0)
    dialog.spn_pulse_ms.setValue(40.0)
    dialog.spn_response_ms.setValue(15.0)
    dialog.spn_threshold_mv.setValue(-20.0)
    dialog.spn_min_amplitude.setValue(0.0)
    dialog.spn_max_amplitude.setValue(100.0)  # nS
    dialog.spn_tolerance.setValue(0.5)  # nS bracket
    dialog.spn_max_iterations.setValue(30)
    dialog.chk_follow_input.setChecked(False)

    # start() drives the whole search synchronously through the connected slot.
    dialog.start()

    assert not dialog._running
    assert dialog._rheobase is not None
    # A genuine interior threshold, not pinned to either amplitude bound.
    assert 0.0 < dialog._rheobase < 100.0
    # The trace really came from the model, not a fixture double.
    assert dialog.cmb_trace.currentText().startswith("Simulation :: ")
    # Spike detection was monotone in amplitude: everything at/above the
    # reported rheobase spiked, and something below it did not.
    spiking = [amp for _dur, amp, count in dialog._samples if count > 0]
    silent = [amp for _dur, amp, count in dialog._samples if count == 0]
    assert spiking and silent
    assert min(spiking) >= max(silent) - dialog.spn_tolerance.value()
    # Probes were appended past the fixture's original 200-sample stim array.
    assert project.stim_arrays["stimulus"].values.shape[0] > 200
    # The clock advanced through every appended probe.
    assert b2_model.get_sim_time_secs() > 0.0
