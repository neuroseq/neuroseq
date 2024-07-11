from __future__ import annotations

import brian2 as b2
import brian2.numpy_ as np
from brian2.tests.utils import assert_allclose
from formozov_project.b2_model import B2Model
from formozov_project.stim_editor import StimEditor
from pytestqt.qtbot import QtBot


def test_stim_editor_no_data(qtbot: QtBot) -> None:
    stim_editor = StimEditor()
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.close()
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


def test_stim_editor_edit_mode_actions(qtbot: QtBot) -> None:
    n_neurons = 10
    n_timestamps = 20
    dt = 10 * b2.ms
    timed_array = b2.TimedArray(np.ones((n_timestamps, n_neurons)), dt=dt)
    selection = np.random.randint(3, size=(5, 2))
    timed_array.values[2:7, 1:3] = selection
    model = B2Model()
    model.parameters = {"a": timed_array, "irrelevant": 4}
    stim_editor = StimEditor()
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    assert len(stim_editor.stims_dict) == 0
    stim_editor.set_model(model)
    assert "a" in stim_editor.stims_dict
    assert_allclose(stim_editor.stims_dict["a"].values, timed_array.values)
    assert stim_editor.current_stim == stim_editor.stims_dict["a"]
    # delete a slice
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    stim_editor.delete_selected_stim()
    assert np.allclose(stim_editor.current_stim.values[10:20, 1:3], np.zeros((10, 2)))
    # move a slice up
    stim_editor.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    stim_editor.move_selected_stim_up()
    assert np.allclose(stim_editor.current_stim.values[2:7, 2:4], selection)
    # move a slice down
    stim_editor.move_selected_stim_down()
    assert np.allclose(stim_editor.current_stim.values[2:7, 1:3], selection)
    # move a slice left
    stim_editor.move_selected_stim_left()
    assert np.allclose(stim_editor.current_stim.values[1:6, 1:3], selection)
    # move a slice right
    stim_editor.move_selected_stim_right()
    assert np.allclose(stim_editor.current_stim.values[2:7, 1:3], selection)
    # copy a slice
    stim_editor.copy_selected_stim()
    assert np.allclose(stim_editor.current_stim.values[2:7, 1:3], selection)
    # cut a slice
    stim_editor.cut_selected_stim()
    assert np.allclose(
        stim_editor.current_stim.values[2:7, 1:3], np.zeros_like(selection)
    )
    # paste a slice into empty region starting at timeslice 10
    stim_editor.stim_widget.timeline.setX(0.5 * n_timestamps * dt / b2.second)
    assert np.allclose(
        stim_editor.current_stim.values[10:15, 1:3], np.zeros_like(selection)
    )
    stim_editor.paste_selected_stim()
    assert np.allclose(stim_editor.current_stim.values[10:15, 1:3], selection)
