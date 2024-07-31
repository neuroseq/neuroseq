from __future__ import annotations

from pathlib import Path

import brian2 as b2
import brian2.numpy_ as np
from brian2.tests.utils import assert_allclose
from formozov_project.b2_model import B2Model
from formozov_project.stim_editor import DrawMode
from formozov_project.stim_editor import StimEditor
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets


def test_stim_editor_no_data(qtbot: QtBot) -> None:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    stim_editor = StimEditor(cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.close()
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


def test_stim_editor_actions(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    n_neurons = 10
    n_timestamps = 20
    dt = 0.5 * b2.ms
    timed_array = b2.TimedArray(np.ones((n_timestamps, n_neurons)), dt=dt)
    selection = np.random.randint(3, size=(5, 2))
    selection_inverted = 2 - selection
    timed_array.values[2:7, 1:3] = selection
    model = B2Model()
    model.parameters = {"a": timed_array, "irrelevant": 4}
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    stim_editor = StimEditor(cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    assert not stim_editor.action_toggle_draw.isChecked()
    assert not stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    assert len(stim_editor.stims_dict) == 0
    stim_editor.set_model(model)
    assert "a" in stim_editor.stims_dict
    assert_allclose(stim_editor.stims_dict["a"].timed_array.values, timed_array.values)
    assert stim_editor.current_stim_modifier == stim_editor.stims_dict["a"]
    # delete a slice
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    stim_editor.delete_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:20, 1:3],
        np.zeros((10, 2)),
    )
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    # undo delete
    stim_editor.undo()
    assert_allclose(
        stim_editor.current_stim_modifier.timed_array.values, timed_array.values
    )
    assert not stim_editor.action_undo.isEnabled()
    assert stim_editor.action_redo.isEnabled()
    # redo delete
    stim_editor.redo()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:20, 1:3],
        np.zeros((10, 2)),
    )
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    # move a slice up
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    stim_editor.move_selected_stim_up()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 2:4], selection
    )
    # undo move up (this requires undo twice as a move is implemented as a cut and then a paste)
    stim_editor.undo()
    stim_editor.undo()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    assert stim_editor.action_undo.isEnabled()
    assert stim_editor.action_redo.isEnabled()
    # redo move up
    stim_editor.redo()
    stim_editor.redo()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 2:4], selection
    )
    assert stim_editor.action_undo.isEnabled()
    assert not stim_editor.action_redo.isEnabled()
    # move a slice down
    stim_editor.stim_widget.select_neurons(slice(2, 4), slice(2, 7))
    stim_editor.move_selected_stim_down()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    # move a slice left
    stim_editor.move_selected_stim_left()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[1:6, 1:3], selection
    )
    # move a slice right
    stim_editor.move_selected_stim_right()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    # invalid moves are ignored, and all neurons are deselected
    stim_editor.move_selected_stim(99, 27)
    assert stim_editor.stim_widget.selected_neuron_slice is None
    assert stim_editor.stim_widget.selected_time_slice is None
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    # copy a slice
    stim_editor.copy_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    # cut a slice
    stim_editor.cut_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3],
        np.zeros_like(selection),
    )
    # paste an inverted slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    stim_editor.stim_widget.timeline.setX(0.5 * n_timestamps * dt / b2.second)
    stim_editor.paste_selected_stim(invert=True)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3],
        selection_inverted,
    )
    # paste a slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    stim_editor.stim_widget.timeline.setX(0.5 * n_timestamps * dt / b2.second)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3],
        np.zeros_like(selection),
    )
    stim_editor.paste_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3], selection
    )
    # save a slice to a file
    filename = tmp_path / "test.npy"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
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
    assert_allclose(
        stim_editor.current_stim_modifier.timed_array.values, timed_array.values
    )
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
        stim_editor.current_stim_modifier.timed_array.values,
        np.zeros((n_timestamps, n_neurons)),
    )
    # set timeline to the 5th timestamp
    stim_editor.stim_widget.timeline.setX(0.25 * n_timestamps * dt / b2.second)
    # load a slice from a file: neurons start at 0, timeslice starts at current timeline
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (f"{filename}", True)
    )
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[5:10, 0:2], selection
    )
    # loaded slice is automatically selected
    assert stim_editor.stim_widget.selected_time_slice == slice(5, 10)
    assert stim_editor.stim_widget.selected_neuron_slice == slice(0, 2)
    # invalid slices are not loaded and an error message is shown
    error_messages: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: error_messages.append(args[2])
    )
    np.save(filename, np.zeros((4, 6, 7)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 1
    assert "wrong shape" in error_messages[-1]
    np.save(filename, np.zeros((25, 2)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 2
    assert "too many time points" in error_messages[-1]
    np.save(filename, np.zeros((2, 25)))
    stim_editor.load_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[5:10, 0:2], selection
    )
    assert len(error_messages) == 3
    assert "too many neurons" in error_messages[-1]
    # selection doesn't draw stim if draw mode is not active
    stim_editor.action_toggle_draw.setChecked(False)
    stim_editor.stim_widget_selection_made(slice(0, 5), slice(0, 5))
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[0:5, 0:5], np.zeros((5, 5))
    )
    # selection is drawn if draw mode is active with mode REGION
    stim_editor.action_toggle_draw.setChecked(True)
    stim_editor.draw_mode_updated(DrawMode.REGION)
    stim_editor.stim_widget_selection_made(slice(0, 5), slice(0, 5))
    # drawn stimuli have magnitude of the maximum value of the existing stimuli
    new_stim_value = np.max(selection)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[0:5, 0:5],
        new_stim_value * np.ones((5, 5)),
    )
    # 1ms stim is drawn at mouse click location if draw mode is active with mode SINGLE_STIM
    stim_editor.action_toggle_draw.setChecked(True)
    stim_editor.draw_mode_updated(DrawMode.SINGLE_STIM)
    # 1ms is two timesteps in the current stim
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_clicked(8, 18)
    # drawn stimuli have magnitude of the maximum value of the existing stimuli
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_clicked(7, 16)
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(
        new_stim_value
    )
    # 1ms stim with value 0 is drawn at mouse click location if draw mode is active with mode SINGLE_STIM_ERASE
    stim_editor.action_toggle_draw.setChecked(True)
    stim_editor.draw_mode_updated(DrawMode.SINGLE_STIM_ERASE)
    # 1ms is two timesteps in the current stim
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(
        new_stim_value
    )
    stim_editor.stim_widget_mouse_clicked(8, 18)
    # drawn stimuli have magnitude of the maximum value of the existing stimuli
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(
        new_stim_value
    )
    stim_editor.stim_widget_mouse_clicked(7, 16)
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
