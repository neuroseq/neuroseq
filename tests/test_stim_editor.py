from __future__ import annotations

from pathlib import Path

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

    def __init__(self, *args, **kwargs):
        pass

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
    qtbot.waitUntil(lambda: not stim_editor.isVisible())


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


def test_choose_stim_length_snaps_to_selected_stim_dt(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
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
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1.0", True))

    stim_editor.choose_stim_length()

    assert stim_editor.draw_stim_length == 2
    assert stim_editor._draw_stim_length_milliseconds() == approx(1.041625)
    stim_editor.close()


def test_choose_stim_length_rejects_non_finite_input(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list[dict[str, str]]
) -> None:
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
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(lambda: stim_editor.isVisible())
    stim_editor.set_project(project)
    stim_editor.draw_stim_length = 3
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("inf", True))

    stim_editor.choose_stim_length()

    assert stim_editor.draw_stim_length == 3
    assert qmessagebox_calls[-1]["name"] == "critical"
    assert qmessagebox_calls[-1]["title"] == "Invalid input"
    assert "positive number of milliseconds" in qmessagebox_calls[-1]["text"]
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
