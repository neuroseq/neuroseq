from __future__ import annotations

from pathlib import Path

import brian2 as b2
import brian2.numpy_ as np
from brian2.tests.utils import assert_allclose
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.b2_model import B2Model
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.stim_editor import DrawMode
from neuroseq.stim_editor import SelectMode
from neuroseq.stim_editor import StimEditor
from neuroseq.utils import InteractMode


def test_stim_editor_no_data(qtbot: QtBot) -> None:
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
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
    dt = 1.0 * b2.ms
    timed_array = b2.TimedArray(np.ones((n_timestamps, n_neurons)), dt=dt)
    selection = np.random.randint(3, size=(5, 2))
    selection_inverted = 2 - selection
    timed_array.values[2:7, 1:3] = selection
    model = B2Model()
    model.parameters = {"a": timed_array, "irrelevant": 4}
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
    stim_editor.set_model(model)
    stim_editor.set_mode(InteractMode.SELECT)
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
    # copy a slice
    stim_editor.copy_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3], selection
    )
    # cut a slice
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    stim_editor.cut_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[2:7, 1:3],
        np.zeros_like(selection),
    )
    # paste an inverted slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    stim_editor.select_mode = SelectMode.PASTE_INVERTED
    stim_editor.paste_selected_stim(1, 10)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3],
        selection_inverted,
    )
    # paste a slice into empty region starting at timeslice 10
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 15))
    stim_editor.delete_selected_stim()
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3],
        np.zeros_like(selection),
    )
    stim_editor.select_mode = SelectMode.PASTE
    stim_editor.paste_selected_stim(1, 10)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[10:15, 1:3], selection
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
    # load a slice from a file: neurons start at 0, timeslice starts at current timeline
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (f"{filename}", True)
    )
    stim_editor.load_selected_stim()
    stim_editor.paste_selected_stim(0, 5)
    assert np.allclose(
        stim_editor.current_stim_modifier.timed_array.values[5:10, 0:2], selection
    )
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
    # 2ms stim is drawn at mouse click location if draw mode is active with mode DRAW
    stim_editor.draw_stim_length = 2
    stim_editor.set_mode(InteractMode.DRAW)
    stim_editor.set_draw_mode(DrawMode.DRAW)
    # 1ms is two timesteps in the current stim
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_dragged(8, 8, 18, 18, 1)
    new_stim_value = 1.0
    # drawn stimuli have magnitude of the maximum value of the existing stimuli
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
    stim_editor.stim_widget_mouse_dragged(7, 7, 16, 16, 1)
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
    # 1ms stim with value 0 is drawn at mouse click location if draw mode is active with mode ERASE
    stim_editor.action_draw_mode.setChecked(True)
    stim_editor.set_draw_mode(DrawMode.ERASE)
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
    stim_editor.stim_widget_mouse_dragged(8, 8, 18, 18, 1)
    # drawn stimuli have a magnitude of 1.0
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(
        new_stim_value
    )
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(
        new_stim_value
    )
    stim_editor.stim_widget_mouse_dragged(7, 7, 16, 16, 1)
    assert stim_editor.current_stim_modifier.timed_array.values[18, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[19, 8] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[16, 7] == approx(0)
    assert stim_editor.current_stim_modifier.timed_array.values[17, 7] == approx(0)
