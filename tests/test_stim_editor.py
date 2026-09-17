from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import brian2.numpy_ as np
import pytest
from brian2.tests.utils import assert_allclose
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.project import Project
from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim import StimArray
from neuroseq.stim_editor import DrawMode
from neuroseq.stim_editor import SelectMode
from neuroseq.stim_editor import StimEditor
from neuroseq.utils import InteractMode


class MockStimIntensityDialog:
    draw_stim = DrawStim(DrawStimType.Constant, 1.0, 1.0)
    draw_stim_length = 1
    # Set by __init__ each time the dialog is constructed, so a test can
    # assert what choose_stim_intensity() passed in without keeping a
    # reference to the (never-stored) dialog instance itself.
    received_draw_stim_length: int | None = None
    received_dt_seconds: float | None = None

    def __init__(
        self,
        draw_stim,
        stim_units,
        stim_type,
        draw_stim_length,
        dt_seconds,
        parent=None,
    ):
        MockStimIntensityDialog.received_draw_stim_length = draw_stim_length
        MockStimIntensityDialog.received_dt_seconds = dt_seconds

    def exec(self) -> bool:
        return True


def test_stim_editor_no_data(qtbot: QtBot) -> None:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.close()


def test_switching_mode_shows_only_the_relevant_action_group(qtbot: QtBot) -> None:
    """Draw-mode and select-mode actions take toolbar/menu space only in their

    own mode, rather than merely greying out (issue #535 space optimization).
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    # default mode is DRAW
    assert all(action.isVisible() for action in stim_editor.actions_draw)
    assert all(not action.isVisible() for action in stim_editor.actions_edit)

    stim_editor.set_mode(InteractMode.SELECT)
    assert all(not action.isVisible() for action in stim_editor.actions_draw)
    assert all(action.isVisible() for action in stim_editor.actions_edit)

    stim_editor.set_mode(InteractMode.DRAW)
    assert all(action.isVisible() for action in stim_editor.actions_draw)
    assert all(not action.isVisible() for action in stim_editor.actions_edit)
    stim_editor.close()


def test_drawing_actions_sit_directly_under_draw_mode_button(qtbot: QtBot) -> None:
    """Draw input / Erase input / Set input value/length… are specific to Draw

    mode, so they sit right after the Draw mode toggle - between it and
    Select mode - rather than far below near Undo/Redo (issue #535).
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    texts = [a.text() for a in toolbar.actions()]
    draw_mode_index = texts.index("Draw mode (1)")
    select_mode_index = texts.index("Select mode (2)")
    assert texts[draw_mode_index + 1 : select_mode_index] == [
        "Draw input",
        "Erase input",
        "Set input value/length…",
    ]
    stim_editor.close()


def test_selection_actions_sit_directly_under_select_mode_button(
    qtbot: QtBot,
) -> None:
    """Select all / Copy / Cut / Paste / etc. are specific to Select mode, so

    they sit right after the Select mode toggle - the mirror image of the
    drawing tools under Draw mode - rather than far below near Undo/Redo
    (issue #535).
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    actions = toolbar.actions()
    texts = [a.text() for a in actions]
    select_mode_index = texts.index("Select mode (2)")
    # Create/Edit/Delete/Clear stim array, Trim to Mapped Neurons, Pad/Trim/
    # Loop stim array, Design static/dynamic input, and Show all stim arrays
    # all moved out of this toolbar (issue #535), so the next thing here
    # after the selection actions is Undo.
    undo_index = texts.index("Undo (Ctrl+Z)")
    assert actions[undo_index - 1].isSeparator()
    assert texts[select_mode_index + 1 : undo_index - 1] == [
        "Select all (Ctrl+A)",
        "Copy (Ctrl+C)",
        "Cut (Ctrl+X)",
        "Paste (Ctrl+V)",
        "Paste inverted (Ctrl+Shift+V)",
        "Save to file",
        "Load from file",
        "Delete selected input region",
    ]
    stim_editor.close()


def test_stim_array_lifecycle_actions_are_not_in_the_left_toolbar(
    qtbot: QtBot,
) -> None:
    """Create/Edit/Delete/Clear input array and Trim to Mapped Neurons moved

    to a dropdown in the top Data toolbar next to the stim data combobox
    (issue #535) - Gui._add_stim_array_ops_button_to_data_toolbar() adds
    them there. They stay registered on the "Actions" context menu, in the
    same order, but no longer duplicate onto this left toolbar.
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    array_op_names = [
        "Create input array",
        "Edit input array",
        "Delete input array",
        "Clear input array",
        "Trim to Mapped Neurons…",
    ]
    toolbar_texts = {a.text() for a in toolbar.actions()}
    assert toolbar_texts.isdisjoint(array_op_names)

    context_menu_texts = [a.text() for a in stim_editor.context_menu.actions()]
    assert [t for t in context_menu_texts if t in array_op_names] == array_op_names
    assert not stim_editor.action_trim_to_mapped_neurons.isEnabled()
    stim_editor.close()


def test_pad_trim_loop_actions_are_not_in_the_left_toolbar(qtbot: QtBot) -> None:
    """Pad/Trim/Loop stim array moved to the stim array ops dropdown in the

    top Data toolbar too (issue #535), alongside the array lifecycle
    actions. They stay registered on the "Actions" context menu, in the
    same order, but no longer duplicate onto this left toolbar.
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    time_op_names = [
        "Pad input array (Ctrl+Shift+Plus)",
        "Trim input array (Ctrl+Shift+Minus)",
        "Loop input array",
    ]
    toolbar_texts = {a.text() for a in toolbar.actions()}
    assert toolbar_texts.isdisjoint(time_op_names)

    context_menu_texts = [a.text() for a in stim_editor.context_menu.actions()]
    assert [t for t in context_menu_texts if t in time_op_names] == time_op_names
    stim_editor.close()


def test_design_and_show_all_actions_are_not_in_the_left_toolbar(
    qtbot: QtBot,
) -> None:
    """Design static input / Design dynamic input / Show all input arrays

    moved to the top View toolbar (issue #535) -
    Gui._add_stim_design_buttons_to_view_toolbar and the Show all stim
    arrays action reused directly in Gui._create_actions_view. They stay
    registered on the "Actions" context menu, in the same order, but no
    longer duplicate onto this left toolbar.
    """
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    design_op_names = [
        "Design static input",
        "Design dynamic input",
        "Show all input arrays",
    ]
    toolbar_texts = {a.text() for a in toolbar.actions()}
    assert toolbar_texts.isdisjoint(design_op_names)

    context_menu_texts = [a.text() for a in stim_editor.context_menu.actions()]
    assert [t for t in context_menu_texts if t in design_op_names] == design_op_names
    assert stim_editor.action_show_all_stim_arrays.isChecked()
    stim_editor.close()


def test_trim_to_mapped_neurons_action_emits_its_own_signal(qtbot: QtBot) -> None:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.action_trim_to_mapped_neurons.setEnabled(True)

    with qtbot.waitSignal(stim_editor.trim_to_mapped_neurons_requested):
        stim_editor.action_trim_to_mapped_neurons.trigger()
    stim_editor.close()
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


def _make_ramp_editor(qtbot: QtBot) -> tuple[StimEditor, Project]:
    project = Project()
    project.stim_arrays = {
        "a": StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.zeros((200, 4)),
            "mV",
            0.001,
        )
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    stim_editor.draw_stim = DrawStim(DrawStimType.Linear, 0.0, 1.0)
    stim_editor.draw_stim_length = 4
    return stim_editor, project


@pytest.mark.parametrize("amplitude", [2.5, -3.25, 0.0])
def test_set_selected_stim_amplitude(qtbot, monkeypatch, amplitude):
    editor, project = _make_ramp_editor(qtbot)
    original = np.ones((200, 4))
    original[2:5, 1:3] = [[0, 2], [-1, 0], [3, -4]]
    editor.replace_stim_values_with_undo("a", original)
    action = editor.action_set_amplitude
    assert action in editor.context_menu.actions()
    assert action in editor.toolbar.actions()
    assert not action.isEnabled()
    editor.set_mode(InteractMode.SELECT)
    assert action.isEnabled()
    editor.stim_widget.select_neurons(slice(1, 3), slice(2, 5))

    def get_amplitude(*args):
        assert "mV" in args[2]
        return str(amplitude), True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", get_amplitude)
    action.trigger()
    expected = original.copy()
    region = expected[2:5, 1:3]
    region[region != 0] = amplitude
    assert_allclose(project.stim_arrays["a"].values, expected)
    editor.undo()
    assert_allclose(project.stim_arrays["a"].values, original)
    editor.redo()
    assert_allclose(project.stim_arrays["a"].values, expected)


@pytest.mark.parametrize(
    "response", [("7", False), ("invalid", True), ("nan", True), ("inf", True)]
)
def test_set_selected_stim_amplitude_rejects_invalid_or_cancelled_input(
    qtbot, monkeypatch, response
):
    editor, project = _make_ramp_editor(qtbot)
    original = np.ones((200, 4))
    editor.replace_stim_values_with_undo("a", original)
    editor.stim_widget.select_neurons(slice(1, 3), slice(2, 5))
    modifier = editor.current_stim_modifier
    index = modifier.current_index
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: response)
    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args)
    )
    editor.set_selected_stim_amplitude()
    assert_allclose(project.stim_arrays["a"].values, original)
    assert modifier.current_index == index
    assert bool(errors) == response[1]


def test_set_selected_stim_amplitude_without_selection_or_nonzero_values(
    qtbot, monkeypatch
):
    editor, project = _make_ramp_editor(qtbot)

    def unexpected_dialog(*args):
        pytest.fail("No amplitude dialog should open for an empty selection")

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", unexpected_dialog)
    editor.set_selected_stim_amplitude()
    editor.stim_widget.select_neurons(slice(1, 3), slice(2, 5))
    editor.set_selected_stim_amplitude()
    assert not editor.action_undo.isEnabled()
    assert not np.any(project.stim_arrays["a"].values)
    editor.current_stim_modifier = None
    editor.set_selected_stim_amplitude()


def test_ramp_drag_paints_single_monotonic_ramp_anchored_at_click(
    qtbot: QtBot,
) -> None:
    stim_editor, project = _make_ramp_editor(qtbot)

    # Replay the sequence the StimWidget actually emits during a drag:
    # the first emit comes from pointer_down with initial == current ==
    # the click position; subsequent emits carry initial = *previous*
    # sampled position, not the original click. Without the editor's
    # drag-origin tracking this leaves many short ramps stamped between
    # consecutive sampled positions.
    stim_editor.stim_widget_mouse_dragged(1, 1, 20, 20, 1)  # pointer_down
    stim_editor.stim_widget_mouse_dragged(1, 1, 20, 30, 1)
    stim_editor.stim_widget_mouse_dragged(1, 1, 30, 50, 1)
    stim_editor.stim_widget_mouse_dragged(1, 1, 50, 70, 1)
    # Release at sample 70 — must NOT stamp a fresh draw_stim_length-wide
    # ramp at the right end of the long one.
    stim_editor.stim_widget_mouse_clicked(1, 70, 1)

    values = project.stim_arrays["a"].values
    drawn = values[20:74, 1]
    assert drawn.shape == (54,)
    assert drawn[0] == approx(0.0)
    assert drawn[-1] == approx(1.0)
    # No discontinuity: a single ramp must be strictly increasing.
    assert np.all(np.diff(drawn) > 0)
    # Drag-origin tracker is cleared on release.
    assert stim_editor._draw_drag_origin_time_index is None
    # Other columns untouched.
    assert np.all(values[:, [0, 2, 3]] == 0.0)


def test_ramp_backward_drag_keeps_value1_at_click_position(qtbot: QtBot) -> None:
    stim_editor, project = _make_ramp_editor(qtbot)

    # Click at sample 70, then drag leftward to sample 20.
    stim_editor.stim_widget_mouse_dragged(1, 1, 70, 70, 1)
    stim_editor.stim_widget_mouse_dragged(1, 1, 70, 50, 1)
    stim_editor.stim_widget_mouse_dragged(1, 1, 50, 20, 1)
    stim_editor.stim_widget_mouse_clicked(1, 20, 1)

    values = project.stim_arrays["a"].values
    drawn = values[20:74, 1]
    # Slice spans min(20, 70)=20 to max(20, 70)+4=74. value1 belongs at the
    # click position (=70 ≈ right end of slice), value2 at the release
    # position (=20 = left end of slice). So the ramp DESCENDS left-to-right.
    assert drawn[0] == approx(1.0)
    assert drawn[-1] == approx(0.0)
    assert np.all(np.diff(drawn) < 0)


def test_stim_editor_zoom_pan(qtbot: QtBot) -> None:
    project = Project()
    project.stim_arrays = {
        "a": StimArray(
            "a", neuroseq.stim.StimType.Excitation, np.zeros((127, 17)), "mV", 0.001
        )
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    # zoom/pan using the sliders and zoom buttons
    delay = 50
    for _ in range(4):
        stim_editor.x_btn_zoom_in.click()
        qtbot.wait(delay)
        stim_editor.y_btn_zoom_in.click()
        qtbot.wait(delay)
    stim_editor.x_scrollbar.setValue(512)
    qtbot.wait(delay)
    stim_editor.x_scrollbar.setValue(0)
    qtbot.wait(delay)
    stim_editor.y_scrollbar.setValue(317)
    qtbot.wait(delay)
    stim_editor.y_scrollbar.setValue(0)
    qtbot.wait(delay)
    for _ in range(7):
        stim_editor.x_btn_zoom_out.click()
        qtbot.wait(delay)
        stim_editor.y_btn_zoom_out.click()
        qtbot.wait(delay)
    stim_editor.close()


def test_stim_editor_pad_and_trim(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
):
    project = Project()
    project.stim_arrays = {
        "a": StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.random.random_sample((500, 10)),
            "mV",
            0.001,
        )
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    # pad with invalid number of seconds
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("notanumber", True)
    )
    assert len(qmessagebox_calls) == 0
    stim_editor.pad_stim_array()
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "parse" in qmessagebox_calls[0]["text"].lower()
    # pad with negative number of seconds
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("-2", True))
    assert len(qmessagebox_calls) == 1
    stim_editor.pad_stim_array()
    assert len(qmessagebox_calls) == 2
    assert qmessagebox_calls[1]["name"] == "critical"
    assert "positive" in qmessagebox_calls[1]["text"].lower()
    # pad with 1 second
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1", True))
    assert len(qmessagebox_calls) == 2
    stim_editor.pad_stim_array()
    assert len(qmessagebox_calls) == 2
    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.values.shape[0] == 1500
    # trim with invalid number of seconds
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("notanumber", True)
    )
    assert len(qmessagebox_calls) == 2
    stim_editor.trim_stim_array()
    assert len(qmessagebox_calls) == 3
    assert qmessagebox_calls[2]["name"] == "critical"
    assert "parse" in qmessagebox_calls[2]["text"].lower()
    # trim with negative number of seconds
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("-2", True))
    assert len(qmessagebox_calls) == 3
    stim_editor.trim_stim_array()
    assert len(qmessagebox_calls) == 4
    assert qmessagebox_calls[3]["name"] == "critical"
    assert "positive" in qmessagebox_calls[3]["text"].lower()
    # trim with too large number of seconds
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("5", True))
    assert len(qmessagebox_calls) == 4
    stim_editor.trim_stim_array()
    assert len(qmessagebox_calls) == 5
    assert qmessagebox_calls[4]["name"] == "critical"
    assert "greater than" in qmessagebox_calls[4]["text"].lower()
    # trim by 0.5 seconds (no confirmation dialog since all stimuli being removed are zero)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.5", True))
    assert len(qmessagebox_calls) == 5
    stim_editor.trim_stim_array()
    assert len(qmessagebox_calls) == 5
    assert stim_editor.current_stim_modifier.stim_array.values.shape[0] == 1000
    # trim with 0.7 seconds (confirmation dialog since some non-zero stimuli will be removed)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.7", True))
    # say no when asked to confirm
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.No,
    )
    stim_editor.trim_stim_array()
    assert stim_editor.current_stim_modifier.stim_array.values.shape[0] == 1000
    # do it again and this time say yes when asked to confirm
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    stim_editor.trim_stim_array()
    assert stim_editor.current_stim_modifier.stim_array.values.shape[0] == 300
    stim_editor.close()


def test_stim_editor_trim_is_undoable(monkeypatch: MonkeyPatch, qtbot: QtBot):
    project = Project()
    values = np.zeros((1000, 4))
    values[900:950, 0] = 1.0  # non-zero tail that trimming would remove
    project.stim_arrays = {
        "a": StimArray("a", neuroseq.stim.StimType.Excitation, values, "mV", 0.001)
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)

    # Trim 0.4 s off; confirm the "would delete data" prompt with Yes.
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.4", True))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    stim_editor.trim_stim_array()
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None
    assert modifier.stim_array.values.shape[0] == 600
    # The trim is now on the undo stack instead of clearing it.
    assert modifier.can_undo() is True

    stim_editor.undo()
    assert modifier.stim_array.values.shape[0] == 1000
    assert np.allclose(modifier.stim_array.values[900:950, 0], 1.0)

    stim_editor.redo()
    assert modifier.stim_array.values.shape[0] == 600
    stim_editor.close()


def test_trim_warns_when_cutting_into_simulated_trace(
    monkeypatch: MonkeyPatch, qtbot: QtBot
):
    from neuroseq.example_projects import default_project

    project = default_project()
    project.run_simulation(0.05)
    sim_time = project.b2_model.get_sim_time_secs()
    assert sim_time > 0

    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.set_project(project)
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None

    stim = modifier.stim_array
    current_secs = stim.dt * stim.values.shape[0]
    new_length = sim_time / 2.0  # cut into the recorded simulation
    trim = current_secs - new_length
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a: (str(trim), True))

    captured: dict[str, str] = {}

    def fake_question(parent, title, text, *a, **k):
        captured["text"] = text
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", fake_question)

    before = stim.values.shape[0]
    stim_editor.trim_stim_array()
    # The simulated-trace caveat is surfaced, and declining leaves the stim as is.
    assert "resumes from the full, untrimmed end" in captured["text"]
    assert modifier.stim_array.values.shape[0] == before

    # Accepting the warning proceeds with the trim.
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    stim_editor.trim_stim_array()
    assert modifier.stim_array.values.shape[0] < before
    stim_editor.close()


def test_routine_dialogs_have_reset_stim_array_button(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    for routine, attr in (
        (StimEditor.TARGETED_STDP_ROUTINE, "_input_designer_dialog"),
        (StimEditor.RHEOBASE_ROUTINE, "_rheobase_dialog"),
        (StimEditor.STDP_WINDOW_ROUTINE, "_stdp_window_dialog"),
    ):
        stim_editor.open_input_designer()
        selector = stim_editor._routine_selector_dialog
        assert selector is not None
        selector.cmb_routine.setCurrentText(routine)
        selector._on_continue()
        qtbot.waitUntil(lambda s=selector: not s.isVisible())
        dialog = getattr(stim_editor, attr)
        assert dialog is not None
        assert dialog.btn_reset.text() == "Reset input array"
        dialog.close()
        qtbot.waitUntil(lambda d=dialog: not d.isVisible())
    stim_editor.close()
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


def test_routine_reset_button_trims_stim_to_simulation_head(qtbot: QtBot) -> None:
    stim_editor, project = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None
    selector.cmb_routine.setCurrentText(StimEditor.TARGETED_STDP_ROUTINE)
    selector._on_continue()
    qtbot.waitUntil(lambda: not selector.isVisible())
    dialog = stim_editor._input_designer_dialog
    assert dialog is not None

    # No simulation yet, so the reset button trims the array to a single
    # baseline sample (never empty).
    assert project.b2_model.get_sim_time_secs() == 0
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None
    assert modifier.stim_array.values.shape[0] == 200
    dialog.btn_reset.click()
    assert modifier.stim_array.values.shape[0] == 1
    dialog.close()
    qtbot.waitUntil(lambda: not dialog.isVisible())
    stim_editor.close()
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


def _make_editor_for_project(qtbot: QtBot, project: Project) -> StimEditor:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.set_project(project)
    return stim_editor


def test_reset_stim_array_trims_to_simulation_head(qtbot: QtBot):
    from neuroseq.example_projects import default_project
    from neuroseq.stim_timing import seconds_to_samples_nearest

    project = default_project()
    project.run_simulation(0.05)
    sim_time = project.b2_model.get_sim_time_secs()
    # Leave a long empty tail past the simulated region.
    project.pad_stim_arrays(2.0)

    stim_editor = _make_editor_for_project(qtbot, project)
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None
    dt = modifier.stim_array.dt
    before = modifier.stim_array.values.shape[0]
    assert before == seconds_to_samples_nearest(2.0, dt)

    stim_editor.reset_stim_array()
    after = modifier.stim_array.values.shape[0]
    assert after == seconds_to_samples_nearest(sim_time, dt)
    assert after < before

    # The reset is undoable (restores the trimmed tail).
    stim_editor.undo()
    assert modifier.stim_array.values.shape[0] == before
    stim_editor.close()


def test_reset_stim_array_no_op_when_already_at_simulation_head(qtbot: QtBot):
    from neuroseq.example_projects import default_project

    project = default_project()
    project.run_simulation(0.05)
    stim_editor = _make_editor_for_project(qtbot, project)
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None

    stim_editor.reset_stim_array()  # trims the tail down to the sim head
    length_at_head = modifier.stim_array.values.shape[0]
    stack_len = len(modifier.action_stack)

    # A second reset has nothing to trim: no change and no new undo entry.
    stim_editor.reset_stim_array()
    assert modifier.stim_array.values.shape[0] == length_at_head
    assert len(modifier.action_stack) == stack_len
    stim_editor.close()


def test_stim_editor_pad_is_undoable(monkeypatch: MonkeyPatch, qtbot: QtBot):
    project = Project()
    values = np.zeros((1000, 4))
    values[900:950, 0] = 1.0  # non-zero content that pad + undo must preserve
    project.stim_arrays = {
        "a": StimArray("a", neuroseq.stim.StimType.Excitation, values, "mV", 0.001)
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)

    # Pad 0.5 s (500 samples) onto the 1 s array.
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.5", True))
    stim_editor.pad_stim_array()
    modifier = stim_editor.current_stim_modifier
    assert modifier is not None
    assert modifier.stim_array.values.shape[0] == 1500
    # The pad is on the undo stack instead of clearing it.
    assert modifier.can_undo() is True

    stim_editor.undo()
    assert modifier.stim_array.values.shape[0] == 1000
    assert np.allclose(modifier.stim_array.values[900:950, 0], 1.0)

    stim_editor.redo()
    assert modifier.stim_array.values.shape[0] == 1500
    assert np.allclose(modifier.stim_array.values[1000:], 0.0)
    stim_editor.close()


def test_edit_stim_array_dt_change_requires_confirmation(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    class MockEditStimDialog:
        def __init__(self, stim_array, existing_names, parent=None):
            self.name = stim_array.name
            self.stim_type = stim_array.stim_type
            self.n_neurons = stim_array.values.shape[1]
            self.units = stim_array.units
            self.dt = 0.002

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    project = Project()
    project.add_stim_array(
        StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.array([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            "mV",
            0.001,
        )
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    stim_editor.draw_stim_length = 4
    monkeypatch.setattr(neuroseq.stim_editor, "EditStimDialog", MockEditStimDialog)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.No,
    )

    stim_editor.edit_stim_array()

    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.dt == approx(0.001)
    assert stim_editor.current_stim_modifier.stim_array.values.shape == (5, 1)
    assert stim_editor.draw_stim_length == 4

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    stim_editor.edit_stim_array()

    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.dt == approx(0.002)
    assert stim_editor.current_stim_modifier.stim_array.values.shape == (3, 1)
    assert stim_editor.current_stim_modifier.stim_array.values[0, 0] == approx(1.0)
    assert stim_editor.current_stim_modifier.stim_array.values[1, 0] == approx(0.0)
    assert stim_editor.current_stim_modifier.stim_array.values[2, 0] == approx(0.0)
    assert stim_editor.draw_stim_length == 2
    stim_editor.close()


def test_choose_stim_intensity_passes_and_reads_back_draw_stim_length(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    """ "Draw designer stimuli" and "Modify stimulus length" were merged into

    one "Set input value/length…" dialog/action (issue #535); choose_stim_intensity
    now carries the stim length through StimIntensityDialog too.
    """
    monkeypatch.setattr(
        neuroseq.stim_editor,
        "StimIntensityDialog",
        MockStimIntensityDialog,
    )
    project = Project()
    project.add_stim_array(
        StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "mV",
            0.0005208125,
        )
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    stim_editor.draw_stim_length = 5

    MockStimIntensityDialog.draw_stim_length = 9
    stim_editor.choose_stim_intensity()

    # the dialog is constructed with the editor's current length/dt...
    assert MockStimIntensityDialog.received_draw_stim_length == 5
    assert MockStimIntensityDialog.received_dt_seconds == approx(0.0005208125)
    # ...and the editor adopts whatever the dialog reports back on accept.
    assert stim_editor.draw_stim_length == 9
    stim_editor.close()


def test_stim_editor_left_toolbar_has_one_design_stimulus_action(
    qtbot: QtBot,
) -> None:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)

    texts = [a.text() for a in toolbar.actions()]
    assert texts.count("Set input value/length…") == 1
    assert "Draw designer stimuli" not in texts
    assert "Modify stimulus length" not in texts
    stim_editor.close()


def test_selecting_stim_preserves_physical_draw_duration(qtbot: QtBot) -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    project.add_stim_array(
        StimArray(
            "b",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "mV",
            0.002,
        )
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project, stim_to_select="a")
    stim_editor.draw_stim_length = 4
    stim_editor.select_stim("b")

    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.name == "b"
    assert stim_editor.draw_stim_length == 2
    assert stim_editor._draw_stim_length_milliseconds() == approx(4.0)
    stim_editor.close()


def test_selecting_smaller_stim_refocuses_y_view() -> None:
    wide_modifier = SimpleNamespace(
        stim_array=StimArray(
            "wide",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 50), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    single_modifier = SimpleNamespace(
        stim_array=StimArray(
            "single",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 1), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    focused_ranges: list[tuple[int, int]] = []
    fake_stim_widget = SimpleNamespace(
        set_data=lambda stim_array: None,
        set_background_data=lambda stim_array: None,
    )
    fake_editor = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        stims_dict={"wide": wide_modifier, "single": single_modifier},
        current_stim_modifier=wide_modifier,
        draw_stim_length=4,
        draw_stim=DrawStim(DrawStimType.Constant, 1.0, 1.0),
        stim_type=neuroseq.stim.StimType.Excitation,
        stim_widget=fake_stim_widget,
        controller=SimpleNamespace(
            focus_neuron_range=lambda start, stop: focused_ranges.append((start, stop))
        ),
        set_mode=lambda mode: None,
        set_draw_mode=lambda mode: None,
        _stim_values_updated=lambda: None,
        _current_stim_dt_seconds=lambda: 0.001,
        current_stim_changed=SimpleNamespace(emit=lambda stim_array: None),
    )

    StimEditor.select_stim(fake_editor, "single")  # type: ignore[arg-type]

    assert fake_editor.current_stim_modifier is single_modifier
    assert focused_ranges == [(0, 1)]


def test_create_stim_array_defaults_to_selected_stim_dt(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    captured = {}

    class MockEditStimDialog:
        def __init__(self, stim_array, existing_names, parent=None):
            captured["dt"] = stim_array.dt
            self.name = stim_array.name
            self.stim_type = stim_array.stim_type
            self.n_neurons = stim_array.values.shape[1]
            self.units = stim_array.units
            self.dt = stim_array.dt

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(neuroseq.stim_editor, "EditStimDialog", MockEditStimDialog)
    project = Project()
    project.add_stim_array(
        StimArray(
            "a",
            neuroseq.stim.StimType.Excitation,
            np.zeros((10, 2), dtype=np.float32),
            "mV",
            0.002,
        )
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project, stim_to_select="a")

    stim_editor.create_stim_array()

    assert captured["dt"] == approx(0.002)
    stim_editor.close()


def test_stim_editor_mouse_click_draws_and_erases() -> None:
    stim_array = StimArray(
        "igor_wave",
        neuroseq.stim.StimType.External,
        np.zeros((6, 1), dtype=np.float32),
        "pA",
        0.001,
    )
    draw_calls: list[tuple[slice, slice, bool]] = []

    def _draw_stim_region(
        neuron_slice: slice,
        time_slice: slice,
        erase: bool = False,
    ) -> None:
        draw_calls.append((neuron_slice, time_slice, erase))
        stim_array.values[time_slice, neuron_slice] = 0.0 if erase else 2.5

    fake_editor: Any = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        current_stim_modifier=SimpleNamespace(stim_array=stim_array),
        mode=InteractMode.DRAW,
        draw_mode=DrawMode.DRAW,
        draw_stim_length=2,
        draw_stim=DrawStim(DrawStimType.Constant, 2.5, 2.5),
        _is_valid_slice=lambda neuron_slice, time_slice: True,
        draw_stim_region=_draw_stim_region,
        _draw_drag_origin_time_index=None,
    )

    StimEditor.stim_widget_mouse_clicked(fake_editor, 0, 2, 1)

    assert draw_calls == [(slice(0, 1), slice(2, 4), False)]
    assert stim_array.values[2, 0] == approx(2.5)
    assert stim_array.values[3, 0] == approx(2.5)

    StimEditor.stim_widget_mouse_clicked(fake_editor, 0, 2, 2)

    assert draw_calls[-1] == (slice(0, 1), slice(2, 4), True)
    assert stim_array.values[2, 0] == approx(0.0)
    assert stim_array.values[3, 0] == approx(0.0)


def test_stim_editor_draw_region_extends_short_stim_for_follow_up_input() -> None:
    stim_array = StimArray(
        "igor_wave",
        neuroseq.stim.StimType.External,
        np.zeros((1, 1), dtype=np.float32),
        "pA",
        0.001,
    )
    project = Project()
    project.add_stim_array(stim_array)
    draw_calls: list[tuple[slice, slice, float]] = []
    stim_widget = SimpleNamespace(n_neurons=1, n_timestamps=1)

    class _FakeModifier:
        def __init__(self, stim_array: StimArray) -> None:
            self.stim_array = stim_array

        def do(
            self,
            neuron_slice: slice,
            time_slice: slice,
            values: float,
        ) -> None:
            draw_calls.append((neuron_slice, time_slice, values))
            self.stim_array.values[time_slice, neuron_slice] = values

    modifier = _FakeModifier(stim_array)

    def _set_project(
        project: Project,
        keep_undo_history: bool = False,
        stim_to_select: str = "",
    ) -> None:
        stim_widget.n_timestamps = project.stim_arrays["igor_wave"].values.shape[0]
        modifier.stim_array = project.stim_arrays["igor_wave"]

    fake_editor: Any = SimpleNamespace(
        current_stim_modifier=modifier,
        project=project,
        draw_stim=DrawStim(DrawStimType.Constant, 2.5, 2.5),
        draw_stim_length=1,
        stim_widget=stim_widget,
        set_project=_set_project,
        _is_valid_slice=lambda neuron_slice, time_slice: (
            neuron_slice.start >= 0
            and neuron_slice.stop <= stim_widget.n_neurons
            and time_slice.start >= 0
            and time_slice.stop <= stim_widget.n_timestamps
        ),
        _current_draw_stim_values=lambda erase=False, length=None, reverse=False: (
            0.0 if erase else 2.5
        ),
        _stim_values_updated=lambda: None,
    )
    fake_editor._ensure_time_capacity_for_slice = lambda time_slice: (
        StimEditor._ensure_time_capacity_for_slice(
            fake_editor,
            time_slice,
        )
    )
    fake_editor._stim_data_modified = lambda: StimEditor._stim_data_modified(
        fake_editor
    )

    project.dirty = False  # reset after the add_stim_array() setup above
    StimEditor.draw_stim_region(fake_editor, slice(0, 1), slice(1, 2))

    assert project.stim_arrays["igor_wave"].values.shape == (2, 1)
    assert project.stim_arrays["igor_wave"].values[1, 0] == approx(2.5)
    assert draw_calls == [(slice(0, 1), slice(1, 2), 2.5)]
    assert project.dirty is True


def test_create_stim_array_defaults_to_one_millisecond_without_selection(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    captured = {}

    class MockEditStimDialog:
        def __init__(self, stim_array, existing_names, parent=None):
            captured["dt"] = stim_array.dt
            self.name = stim_array.name
            self.stim_type = stim_array.stim_type
            self.n_neurons = stim_array.values.shape[1]
            self.units = stim_array.units
            self.dt = stim_array.dt

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(neuroseq.stim_editor, "EditStimDialog", MockEditStimDialog)
    project = Project()
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)

    stim_editor.create_stim_array()

    assert captured["dt"] == approx(0.001)
    stim_editor.close()


def test_stim_editor_actions(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    n_neurons = 10
    n_timestamps = 20
    project = Project()
    arr = np.ones((n_timestamps, n_neurons))
    selection = np.random.randint(3, size=(5, 2))
    selection_inverted = 2 - selection
    arr[2:7, 1:3] = selection
    project.stim_arrays = {
        "a": StimArray("a", neuroseq.stim.StimType.Excitation, arr, "mV", 0.001)
    }
    monkeypatch.setattr(
        neuroseq.stim_editor,
        "StimIntensityDialog",
        MockStimIntensityDialog,
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    assert stim_editor.action_draw_mode.isChecked()
    assert not stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    assert len(stim_editor.stims_dict) == 0
    stim_editor.set_project(project)
    stim_editor.set_mode(InteractMode.SELECT)
    assert "a" in stim_editor.stims_dict
    assert_allclose(stim_editor.stims_dict["a"].stim_array.values, arr)
    assert stim_editor.current_stim_modifier == stim_editor.stims_dict["a"]
    # delete a slice
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    stim_editor.delete_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[10:20, 1:3],
        np.zeros((10, 2)),
    )
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    # undo delete
    stim_editor.undo()
    assert_allclose(stim_editor.current_stim_modifier.stim_array.values, arr)
    assert not stim_editor.action_undo.isEnabled()
    assert stim_editor.action_redo.isEnabled()
    # redo delete
    stim_editor.redo()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[10:20, 1:3],
        np.zeros((10, 2)),
    )
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    # copy a slice
    stim_editor.copy_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[2:7, 1:3], selection
    )
    # cut a slice
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    stim_editor.cut_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[2:7, 1:3],
        np.zeros_like(selection),
    )
    # paste an inverted slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    stim_editor.select_mode = SelectMode.PASTE_INVERTED
    stim_editor.paste_selected_stim(1, 10)
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[10:15, 1:3],
        selection_inverted,
    )
    # paste a slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[10:15, 1:3],
        np.zeros_like(selection),
    )
    stim_editor.select_mode = SelectMode.PASTE
    stim_editor.paste_selected_stim(1, 10)
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[10:15, 1:3], selection
    )
    # save a slice to a file
    filename = tmp_path / "test.npy"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    assert stim_editor.stim_widget.selected_time_slice == slice(10, 15)
    assert stim_editor.stim_widget.selected_neuron_slice == slice(1, 3)
    stim_editor.save_selected_stim()
    assert filename.exists()
    assert np.allclose(np.load(filename), selection)
    # undo all actions
    while stim_editor.action_undo.isEnabled():
        stim_editor.undo()
    assert not stim_editor.action_undo.isEnabled()
    assert stim_editor.action_redo.isEnabled()
    assert_allclose(stim_editor.current_stim_modifier.stim_array.values, arr)
    # redo all actions
    while stim_editor.action_redo.isEnabled():
        stim_editor.redo()
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    assert np.allclose(np.load(filename), selection)
    # delete all stimuli
    stim_editor.stim_widget.select_neurons(slice(0, None), slice(0, None))
    stim_editor.delete_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values,
        np.zeros((n_timestamps, n_neurons)),
    )
    # load a slice from a file: neurons start at 0, timeslice starts at current timeline
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (f"{filename}", True)
    )
    stim_editor.load_selected_stim()
    stim_editor.paste_selected_stim(0, 5)
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[5:10, 0:2], selection
    )
    # invalid slices are not loaded and an error message is shown
    error_messages: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: error_messages.append(args[2])
    )
    np.save(filename, np.zeros((4, 6, 7)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 1
    assert "wrong shape" in error_messages[-1]
    np.save(filename, np.zeros((25, 2)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 2
    assert "too many time points" in error_messages[-1]
    np.save(filename, np.zeros((2, 25)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.stim_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 3
    assert "too many neurons" in error_messages[-1]
    # default stim drawing is constant intensity with the maximum value of the existing stimuli
    assert stim_editor.draw_stim.draw_stim_type == DrawStimType.Constant
    assert stim_editor.draw_stim.value1 == pytest.approx(
        stim_editor.current_stim_modifier.stim_array.values.max()
    )
    # set a valid value for stim drawing intensity
    new_stim_value = 1.85
    MockStimIntensityDialog.draw_stim.value1 = new_stim_value
    stim_editor.choose_stim_intensity()
    assert stim_editor.draw_stim.value1 == pytest.approx(new_stim_value)
    # 2ms stim is drawn at mouse click location if draw mode is active with mode DRAW
    stim_editor.draw_stim_length = 2
    stim_editor.set_mode(InteractMode.DRAW)
    stim_editor.set_draw_mode(DrawMode.DRAW)
    # 1ms is two timesteps in the current stim
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_dragged(8, 8, 18, 18, 1)
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_dragged(7, 7, 16, 16, 1)
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(
        new_stim_value
    )
    # 1ms stim with value 0 is drawn at mouse click location if draw mode is active with mode ERASE
    stim_editor.action_draw_mode.setChecked(True)
    stim_editor.set_draw_mode(DrawMode.ERASE)
    # 1ms is two timesteps in the current stim
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(
        new_stim_value
    )
    stim_editor.stim_widget_mouse_dragged(8, 8, 18, 18, 1)
    # drawn stimuli have a magnitude of 1.0
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(
        new_stim_value
    )
    stim_editor.stim_widget_mouse_dragged(7, 7, 16, 16, 1)
    assert stim_editor.current_stim_modifier.stim_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.stim_array.values[17, 7] == approx(0)
    stim_editor.close()


def _routine_combo_items(selector: Any) -> list[str]:
    return [
        selector.cmb_routine.itemText(i) for i in range(selector.cmb_routine.count())
    ]


def test_open_input_designer_shows_selector_with_preview(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()

    selector = stim_editor._routine_selector_dialog
    assert selector is not None
    assert selector.isVisible()
    assert selector.windowTitle() == "Design dynamic input"
    assert _routine_combo_items(selector) == [
        "Targeted STDP synaptic strength",
        "Minimal input for spiking (rheobase)",
        "STDP timing window (LTP/LTD vs Δt)",
    ]
    # The preview is always on: the initial selection already shows goal text
    # and a sample plot in the same window as the dropdown.
    assert selector.lbl_goal.text()
    assert selector.lbl_params.text()
    assert selector._axes.lines

    # Selecting a routine and clicking Continue opens the matching dialog.
    selector.cmb_routine.setCurrentText(StimEditor.TARGETED_STDP_ROUTINE)
    assert stim_editor._input_designer_dialog is None
    selector._on_continue()
    assert stim_editor._input_designer_dialog is not None
    assert stim_editor._input_designer_dialog.isVisible()
    stim_editor.close()


def test_routine_selector_preview_updates_on_selection_change(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None

    selector.cmb_routine.setCurrentText(StimEditor.RHEOBASE_ROUTINE)
    assert "rheobase" in selector.lbl_goal.text().lower()
    selector.cmb_routine.setCurrentText(StimEditor.STDP_WINDOW_ROUTINE)
    assert "stdp" in selector.lbl_goal.text().lower()
    stim_editor.close()


def test_open_input_designer_opens_rheobase_when_selected(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None

    selector.cmb_routine.setCurrentText(StimEditor.RHEOBASE_ROUTINE)
    assert stim_editor._rheobase_dialog is None  # only opens on Continue
    selector._on_continue()

    assert stim_editor._rheobase_dialog is not None
    assert stim_editor._rheobase_dialog.isVisible()
    stim_editor.close()


def test_routine_dialogs_have_no_qt_parent(qtbot: QtBot) -> None:
    """A parent sets WM_TRANSIENT_FOR on X11; on KWin the routine dialog and
    its own two popups (also parentless) would then all share one owner and
    get treated as a single "raise group" that can't be reordered by
    clicking any individual member - not even this dialog itself."""
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None

    selector.cmb_routine.setCurrentText(StimEditor.RHEOBASE_ROUTINE)
    selector._on_continue()

    assert stim_editor._rheobase_dialog is not None
    assert stim_editor._rheobase_dialog.parent() is None
    stim_editor.close()


def test_open_input_designer_opens_stdp_window_when_selected(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None

    selector.cmb_routine.setCurrentText(StimEditor.STDP_WINDOW_ROUTINE)
    selector._on_continue()

    assert stim_editor._stdp_window_dialog is not None
    assert stim_editor._stdp_window_dialog.isVisible()
    stim_editor.discard_input_designer()
    assert stim_editor._stdp_window_dialog is None
    stim_editor.close()


def test_routine_selector_cancel_does_not_open_routine_dialog(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    stim_editor.open_input_designer()
    selector = stim_editor._routine_selector_dialog
    assert selector is not None and selector.isVisible()

    selector.reject()  # user backs out of the selector
    assert stim_editor._input_designer_dialog is None
    assert stim_editor._rheobase_dialog is None
    assert stim_editor._stdp_window_dialog is None
    stim_editor.close()


def test_discard_input_designer_clears_cached_dialog(qtbot: QtBot) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)

    def _open(routine: str) -> None:
        stim_editor.open_input_designer()
        selector = stim_editor._routine_selector_dialog
        assert selector is not None
        selector.cmb_routine.setCurrentText(routine)
        selector._on_continue()

    _open(StimEditor.TARGETED_STDP_ROUTINE)
    assert stim_editor._input_designer_dialog is not None
    _open(StimEditor.RHEOBASE_ROUTINE)
    assert stim_editor._rheobase_dialog is not None
    _open(StimEditor.STDP_WINDOW_ROUTINE)
    assert stim_editor._stdp_window_dialog is not None

    # Replacing the project must not carry the previous designer state over.
    stim_editor.discard_input_designer()
    assert stim_editor._input_designer_dialog is None
    assert stim_editor._rheobase_dialog is None
    assert stim_editor._stdp_window_dialog is None
    assert stim_editor._routine_selector_dialog is None

    # Discarding again is a no-op and must not raise.
    stim_editor.discard_input_designer()
    assert stim_editor._input_designer_dialog is None
    assert stim_editor._rheobase_dialog is None
    assert stim_editor._stdp_window_dialog is None
    stim_editor.close()


def test_clear_all_stim_arrays_zeroes_every_array(qtbot: QtBot) -> None:
    project = Project()
    project.stim_arrays = {
        "a": StimArray(
            "a", neuroseq.stim.StimType.Excitation, np.ones((5, 3)), "mV", 0.001
        ),
        "b": StimArray(
            "b", neuroseq.stim.StimType.Inhibition, np.full((5, 3), 2.0), "mV", 0.001
        ),
    }
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.set_project(project)

    stim_editor.clear_all_stim_arrays()

    for stim in project.stim_arrays.values():
        assert not np.any(stim.values)
    stim_editor.close()
