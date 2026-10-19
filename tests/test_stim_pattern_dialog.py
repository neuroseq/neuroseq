from __future__ import annotations

import numpy as np
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.project import Project
from neuroseq.stimulation.static_input.static_input_controller import (
    StaticInputController,
)
from neuroseq.stimulation.static_input.stim_pattern_dialog import StimPatternDialog
from neuroseq.stimulation.static_input.stim_patterns import generate_fuer_elise
from neuroseq.stimulation.static_input.stim_patterns import generate_per_neuron
from neuroseq.stimulation.static_input.stim_patterns import generate_ramp
from neuroseq.stimulation.static_input.stim_patterns import generate_sequential
from neuroseq.stimulation.static_input.stim_patterns import generate_simultaneous
from neuroseq.stimulation.stim import DrawStimType
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_editor import StimEditor
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)


def _make_editor(
    qtbot: QtBot,
    *,
    length_samples: int = 500,
    n_neurons: int = 10,
    include_second_stim: bool = False,
) -> tuple[StimEditor, Project, StaticInputController]:
    project = Project()
    project.add_stim_array(
        StimArray(
            "a",
            StimType.Excitation,
            np.zeros((length_samples, n_neurons), dtype=np.float32),
            "siemens",
            0.001,
        )
    )
    if include_second_stim:
        project.add_stim_array(
            StimArray(
                "b",
                StimType.Excitation,
                np.zeros((length_samples, max(1, n_neurons - 4)), dtype=np.float32),
                "pA",
                0.001,
            )
        )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    editor = StimEditor(controller, cmb_stim, toolbar)
    static_input = StaticInputController(editor, editor)
    qtbot.addWidget(editor)
    editor.show()
    qtbot.waitUntil(lambda: editor.isVisible())
    editor.set_project(project, stim_to_select="a")
    return editor, project, static_input


def _open_dialog(controller: StaticInputController) -> StimPatternDialog:
    controller.open_pattern_designer()
    assert controller._pattern_dialog is not None
    return controller._pattern_dialog


def test_dialog_has_no_qt_parent(qtbot: QtBot) -> None:
    """No Qt parent - see _SelfRaisingDialog's docstring for why a shared
    WM_TRANSIENT_FOR chain with a sibling popup breaks click-to-front."""
    editor, _project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)
    assert dialog.parent() is None


def test_dialog_apply_sequential_writes_pattern_and_advances_cursor(
    qtbot: QtBot,
) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_sequential.setChecked(True)
    dialog.btn_apply.click()

    expected = generate_sequential(10, 100, 18, 1.0, 0.001)
    np.testing.assert_allclose(project.stim_arrays["a"].values[:1180, :], expected)
    assert dialog.spin_start_time_ms.value() == 1180.0


def test_dialog_apply_twice_stacks_two_patterns(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=2000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_sequential.setChecked(True)
    dialog.btn_apply.click()
    dialog.radio_simultaneous.setChecked(True)
    dialog.spin_repeats.setValue(1)
    dialog.btn_apply.click()

    sequential = generate_sequential(10, 100, 18, 1.0, 0.001)
    simultaneous = generate_simultaneous(10, 100, 18, 1.0, 1, 0.001)
    values = project.stim_arrays["a"].values
    np.testing.assert_allclose(values[: sequential.shape[0], :], sequential)
    np.testing.assert_allclose(
        values[
            sequential.shape[0] : sequential.shape[0] + simultaneous.shape[0],
            :,
        ],
        simultaneous,
    )

    editor.undo()
    editor.undo()
    assert np.allclose(project.stim_arrays["a"].values, 0.0)


def test_dialog_apply_undo_redo(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.btn_apply.click()
    expected = generate_sequential(10, 100, 18, 1.0, 0.001)
    editor.undo()
    assert np.allclose(project.stim_arrays["a"].values, 0.0)
    editor.redo()
    np.testing.assert_allclose(project.stim_arrays["a"].values[:1180, :], expected)


def test_dialog_apply_with_padding_is_one_undo(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=100)
    dialog = _open_dialog(static_input)

    original_length = project.stim_arrays["a"].values.shape[0]
    dialog.btn_apply.click()
    assert project.max_time_secs > original_length * 0.001
    assert project.stim_arrays["a"].values.shape[0] == 1180

    editor.undo()
    assert project.stim_arrays["a"].values.shape[0] == original_length
    assert np.allclose(project.stim_arrays["a"].values, 0.0)


def test_dialog_clear_button_clears_via_undoable_path(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=2000)
    dialog = _open_dialog(static_input)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    dialog.btn_apply.click()
    dialog.btn_clear.click()

    assert np.allclose(project.stim_arrays["a"].values, 0.0)
    assert dialog.spin_start_time_ms.value() == 0.0
    editor.undo()
    assert not np.allclose(project.stim_arrays["a"].values, 0.0)
    editor.undo()
    assert np.allclose(project.stim_arrays["a"].values, 0.0)


def test_dialog_specific_neuron_constant_writes_only_that_column(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("2")
    dialog.radio_simultaneous.setChecked(True)
    dialog.spin_magnitude.setValue(0.7)
    dialog.spin_repeats.setValue(3)
    dialog.btn_apply.click()

    expected = generate_per_neuron(
        neuron_index=2,
        n_neurons=10,
        pulse_ms=100,
        ipi_ms=18,
        profile=DrawStimType.Constant,
        value1=0.7,
        value2=0.0,
        repeats=3,
        dt_seconds=0.001,
    )
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )


def test_dialog_specific_multiple_neurons_and_range(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("1, 4, 6-8")
    dialog.radio_simultaneous.setChecked(True)
    dialog.spin_magnitude.setValue(0.5)
    dialog.spin_repeats.setValue(2)
    dialog.btn_apply.click()

    expected = generate_per_neuron(
        neuron_index=[1, 4, 6, 7, 8],
        n_neurons=10,
        pulse_ms=100,
        ipi_ms=18,
        profile=DrawStimType.Constant,
        value1=0.5,
        value2=0.0,
        repeats=2,
        dt_seconds=0.001,
    )
    values = project.stim_arrays["a"].values
    np.testing.assert_allclose(values[: expected.shape[0], :], expected)
    for col in [1, 4, 6, 7, 8]:
        assert np.any(values[:, col] != 0.0)
    for col in [0, 2, 3, 5, 9]:
        assert np.all(values[:, col] == 0.0)


def test_dialog_specific_neuron_does_not_erase_other_neurons_same_window(
    qtbot: QtBot,
) -> None:
    """Regression test for issue #517.

    Applying a specific-neuron pattern must not clear a pre-existing
    stimulus on a different neuron in the same time window.
    """
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("1")
    dialog.radio_simultaneous.setChecked(True)
    dialog.spin_magnitude.setValue(0.7)
    dialog.spin_repeats.setValue(1)
    dialog.spin_start_time_ms.setValue(0)
    dialog.btn_apply.click()
    neuron_1_values = project.stim_arrays["a"].values[:, 1].copy()
    assert np.any(neuron_1_values != 0.0)

    dialog.edit_neuron_indices.setText("2")
    dialog.spin_start_time_ms.setValue(0)
    dialog.btn_apply.click()

    values = project.stim_arrays["a"].values
    np.testing.assert_allclose(values[:, 1], neuron_1_values)
    assert np.any(values[:, 2] != 0.0)


def test_dialog_specific_invalid_index_shows_warning_and_no_change(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("2, 99")
    dialog.radio_simultaneous.setChecked(True)
    dialog.btn_apply.click()

    assert qmessagebox_calls[-1]["name"] == "warning"
    assert "out of range" in qmessagebox_calls[-1]["text"]
    assert np.allclose(project.stim_arrays["a"].values, 0.0)


def test_dialog_apply_ramp_multiple_neurons(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("0, 3-4")
    dialog.radio_ramp.setChecked(True)
    dialog.spin_value1.setValue(0.0)
    dialog.spin_value2.setValue(1.0)
    dialog.spin_repeats.setValue(1)
    dialog.btn_apply.click()

    expected = generate_ramp(10, 100, 18, 0.0, 1.0, 1, 0.001, neuron_index=[0, 3, 4])
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    for col in [1, 2, 5, 6, 7, 8, 9]:
        assert np.all(project.stim_arrays["a"].values[:, col] == 0.0)


def test_dialog_fuer_elise_clipped_to_six_neurons(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(
        qtbot, length_samples=4000, n_neurons=6
    )
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_fuer_elise.setChecked(True)
    dialog.btn_apply.click()

    expected = generate_fuer_elise(6, 100, 18, 1.0, 1, 0.001)
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )


def test_dialog_fuer_elise_has_independent_pulse_and_sequence_repeats(
    qtbot: QtBot,
) -> None:
    editor, project, static_input = _make_editor(
        qtbot, length_samples=20_000, n_neurons=13
    )
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_fuer_elise.setChecked(True)
    assert dialog.spin_pulse_repetitions.isVisible()
    assert dialog.spin_repeats.isVisible()
    dialog.spin_pulse_ms.setValue(10)
    dialog.spin_ipi_ms.setValue(5)
    dialog.spin_pulse_repetitions.setValue(3)
    dialog.spin_repeats.setValue(2)
    dialog.btn_apply.click()

    expected = generate_fuer_elise(
        13,
        10,
        5,
        1.0,
        repeats=2,
        dt_seconds=0.001,
        pulse_repetitions=3,
    )
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    assert dialog.spin_start_time_ms.value() == expected.shape[0]


def test_dialog_context_updates_when_user_switches_stim_array(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, include_second_stim=True)
    dialog = _open_dialog(static_input)

    assert dialog.lbl_stim_name.text() == "a"
    assert dialog.lbl_units.text() == "siemens"
    assert "0–9" in dialog.lbl_neuron_hint.text()

    editor.cmb_stim.setCurrentText("b")

    assert dialog.lbl_stim_name.text() == "b"
    assert dialog.lbl_units.text() == "pA"
    assert "0–5" in dialog.lbl_neuron_hint.text()


def test_dialog_invalid_pulse_duration_shows_warning_and_no_change(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.spin_pulse_ms.setValue(0.0)
    dialog.btn_apply.click()

    assert qmessagebox_calls[-1]["name"] == "warning"
    assert "Pulse duration" in qmessagebox_calls[-1]["text"]
    assert np.allclose(project.stim_arrays["a"].values, 0.0)


def test_dialog_apply_ramp_all_neurons(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_ramp.setChecked(True)
    dialog.spin_value1.setValue(0.0)
    dialog.spin_value2.setValue(1.0)
    dialog.spin_repeats.setValue(2)
    dialog.btn_apply.click()

    expected = generate_ramp(10, 100, 18, 0.0, 1.0, 2, 0.001)
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    assert dialog.spin_start_time_ms.value() == expected.shape[0] * 1.0


def test_dialog_apply_ramp_specific_neuron(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("2")
    dialog.radio_ramp.setChecked(True)
    dialog.spin_value1.setValue(0.0)
    dialog.spin_value2.setValue(1.0)
    dialog.spin_repeats.setValue(1)
    dialog.btn_apply.click()

    expected = generate_ramp(10, 100, 18, 0.0, 1.0, 1, 0.001, neuron_index=2)
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    for col in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
        assert np.all(project.stim_arrays["a"].values[:, col] == 0.0)


def test_dialog_ramp_visible_in_both_scopes(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    assert dialog.radio_ramp.isVisible()
    dialog.radio_scope_specific.setChecked(True)
    assert dialog.radio_ramp.isVisible()


def test_dialog_sequential_visible_in_both_scopes(qtbot: QtBot) -> None:
    """Feature (#582): Sequential used to only be selectable for "All
    neurons" scope."""
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    assert dialog.radio_sequential.isVisible()
    dialog.radio_scope_specific.setChecked(True)
    assert dialog.radio_sequential.isVisible()


def test_dialog_apply_sequential_specific_neurons_writes_only_those_columns(
    qtbot: QtBot,
) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=1000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_specific.setChecked(True)
    # parse_neuron_indices sorts/dedupes the typed text (same as every other
    # pattern), so "3, 1" fires neuron 1 first, then 3.
    dialog.edit_neuron_indices.setText("3, 1")
    dialog.radio_sequential.setChecked(True)
    dialog.spin_magnitude.setValue(0.8)
    dialog.btn_apply.click()

    expected = generate_sequential(10, 100, 18, 0.8, 0.001, neuron_index=[1, 3])
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    for col in [0, 2, 4, 5, 6, 7, 8, 9]:
        assert np.all(project.stim_arrays["a"].values[:, col] == 0.0)


def test_dialog_sequential_shows_repeats_in_both_scopes(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_sequential.setChecked(True)
    assert dialog.lbl_repeats.isVisible()

    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("0")
    dialog.radio_sequential.setChecked(True)
    assert dialog.lbl_repeats.isVisible()


def test_dialog_sequential_repeats_replay_the_whole_sequence(qtbot: QtBot) -> None:
    """#586: Repeats on a sequential pattern plays the sequence N times."""
    editor, project, static_input = _make_editor(qtbot, length_samples=4000)
    dialog = _open_dialog(static_input)

    dialog.radio_scope_all.setChecked(True)
    dialog.radio_sequential.setChecked(True)
    dialog.spin_repeats.setValue(3)
    dialog.btn_apply.click()

    expected = generate_sequential(10, 100, 18, 1.0, 0.001, repeats=3)
    assert expected.shape[0] == 3 * 1180
    np.testing.assert_allclose(
        project.stim_arrays["a"].values[: expected.shape[0], :], expected
    )
    assert dialog.spin_start_time_ms.value() == 3540.0


def test_separate_editors_keep_their_own_designer_values(qtbot: QtBot) -> None:
    _, _, first = _make_editor(qtbot)
    _, _, second = _make_editor(qtbot)
    first_dialog = _open_dialog(first)
    first_dialog.spin_pulse_ms.setValue(250.0)
    second_dialog = _open_dialog(second)
    assert second_dialog is not first_dialog
    assert second_dialog.spin_pulse_ms.value() == 100.0
    second_dialog.spin_pulse_ms.setValue(5.0)
    assert _open_dialog(first).spin_pulse_ms.value() == 250.0
    assert _open_dialog(second).spin_pulse_ms.value() == 5.0


def test_dialog_values_survive_refreshing_the_same_project(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)
    dialog.spin_pulse_ms.setValue(250.0)

    editor.set_project(project)

    assert dialog.spin_pulse_ms.value() == 250.0


def test_designer_without_a_stimulation_array_stays_closed(qtbot: QtBot) -> None:
    editor, _, static_input = _make_editor(qtbot)
    editor.set_project(Project())

    static_input.open_pattern_designer()

    assert static_input._pattern_dialog is None


def test_designer_is_disabled_for_an_empty_project(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    editor.set_project(Project())
    assert not dialog.isEnabled()

    editor.set_project(project)
    assert dialog.isEnabled()
    assert dialog.lbl_stim_name.text() == "a"


def test_discarding_designer_releases_its_dialog(qtbot: QtBot) -> None:
    _, _, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)
    static_input.discard_dialog()
    static_input.discard_dialog()
    assert not dialog.isVisible()
    assert static_input._pattern_dialog is None


def test_dialog_uses_the_same_pattern_names_in_both_scopes(qtbot: QtBot) -> None:
    """Issue #613: the same patterns have the same names whichever scope is
    chosen; only Für Elise, on fixed neurons, is for all neurons alone."""
    editor, _, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)

    def shown() -> list[str]:
        return [b.text() for b in dialog._pattern_buttons if b.isVisibleTo(dialog)]

    dialog.radio_scope_all.setChecked(True)
    assert shown() == [
        "Sequential",
        "Simultaneous",
        "Linear gradient",
        "Ramp",
        "Für Elise",
    ]
    dialog.radio_fuer_elise.setChecked(True)
    dialog.radio_scope_specific.setChecked(True)
    assert shown() == ["Sequential", "Simultaneous", "Linear gradient", "Ramp"]
    assert dialog.radio_simultaneous.isChecked()  # Für Elise is not offered


def test_dialog_labels_say_what_each_value_means(qtbot: QtBot) -> None:
    editor, _, static_input = _make_editor(qtbot)
    dialog = _open_dialog(static_input)
    dialog.radio_scope_all.setChecked(True)

    dialog.radio_sequential.setChecked(True)
    assert dialog.lbl_repeats.text() == "Sequence repetitions:"
    assert dialog.lbl_pulse_repetitions.text() == "Pulses per neuron:"
    assert dialog.lbl_pulse_repetitions.isVisibleTo(dialog)

    dialog.radio_simultaneous.setChecked(True)
    assert dialog.lbl_repeats.text() == "Number of pulses:"
    assert not dialog.lbl_pulse_repetitions.isVisibleTo(dialog)

    dialog.radio_linear.setChecked(True)
    assert dialog.lbl_value1.text() == "First pulse magnitude:"
    assert dialog.lbl_value2.text() == "Last pulse magnitude:"

    dialog.radio_ramp.setChecked(True)
    assert dialog.lbl_pulse_ms.text() == "Ramp duration (ms):"
    assert dialog.lbl_value1.text() == "Start magnitude:"
    assert dialog.lbl_repeats.text() == "Number of ramps:"

    dialog.radio_fuer_elise.setChecked(True)
    assert dialog.lbl_pulse_repetitions.text() == "Pulses per note:"


def test_dialog_sequential_repeats_each_neurons_pulse(qtbot: QtBot) -> None:
    """Issue #613 (comment): pulses per neuron, as for Für Elise's notes."""
    editor, project, static_input = _make_editor(qtbot, length_samples=3000)
    dialog = _open_dialog(static_input)
    dialog.radio_scope_specific.setChecked(True)
    dialog.edit_neuron_indices.setText("2, 5")
    dialog.radio_sequential.setChecked(True)
    dialog.spin_pulse_ms.setValue(10.0)
    dialog.spin_ipi_ms.setValue(5.0)
    dialog.spin_repeats.setValue(1)
    dialog.spin_pulse_repetitions.setValue(3)
    dialog.btn_apply.click()

    values = project.stim_arrays["a"].values
    onsets = {
        neuron: np.flatnonzero(np.diff(np.r_[0.0, values[:, neuron]]) > 0).tolist()
        for neuron in (2, 5)
    }
    assert onsets == {2: [0, 15, 30], 5: [45, 60, 75]}


def test_dialog_linear_gradient_for_all_neurons(qtbot: QtBot) -> None:
    editor, project, static_input = _make_editor(qtbot, length_samples=2000)
    dialog = _open_dialog(static_input)
    dialog.radio_scope_all.setChecked(True)
    dialog.radio_linear.setChecked(True)
    dialog.spin_pulse_ms.setValue(10.0)
    dialog.spin_ipi_ms.setValue(10.0)
    dialog.spin_value1.setValue(1.0)
    dialog.spin_value2.setValue(3.0)
    dialog.spin_repeats.setValue(3)
    dialog.btn_apply.click()

    values = project.stim_arrays["a"].values
    np.testing.assert_allclose(
        values[[0, 20, 40], :], [[1.0] * 10, [2.0] * 10, [3.0] * 10]
    )
