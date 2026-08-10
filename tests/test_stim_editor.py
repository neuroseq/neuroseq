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

    StimEditor.draw_stim_region(fake_editor, slice(0, 1), slice(1, 2))

    assert project.stim_arrays["igor_wave"].values.shape == (2, 1)
    assert project.stim_arrays["igor_wave"].values[1, 0] == approx(2.5)
    assert draw_calls == [(slice(0, 1), slice(1, 2), 2.5)]


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


def test_open_input_designer_offers_routine_selection(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    captured: dict[str, Any] = {}

    def fake_get_item(parent, title, label, items, current, editable):
        del parent, current, editable
        captured["title"] = title
        captured["items"] = list(items)
        return (StimEditor.TARGETED_STDP_ROUTINE, True)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", fake_get_item)
    stim_editor.open_input_designer()

    assert captured["title"] == "Design dynamic input"
    assert captured["items"] == [
        "Targeted STDP synaptic strength",
        "Minimal stimulus for spiking (rheobase)",
        "STDP timing window (LTP/LTD vs Δt)",
    ]
    assert stim_editor._input_designer_dialog is not None
    assert stim_editor._input_designer_dialog.isVisible()
    stim_editor.close()


def test_open_input_designer_opens_rheobase_when_selected(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args: (StimEditor.RHEOBASE_ROUTINE, True),
    )

    stim_editor.open_input_designer()

    assert stim_editor._rheobase_dialog is not None
    assert stim_editor._rheobase_dialog.isVisible()
    stim_editor.close()


def test_open_input_designer_opens_stdp_window_when_selected(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args: (StimEditor.STDP_WINDOW_ROUTINE, True),
    )

    stim_editor.open_input_designer()

    assert stim_editor._stdp_window_dialog is not None
    assert stim_editor._stdp_window_dialog.isVisible()
    stim_editor.discard_input_designer()
    assert stim_editor._stdp_window_dialog is None
    stim_editor.close()


def test_open_input_designer_cancel_does_not_open_dialog(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args: ("", False))
    stim_editor.open_input_designer()

    assert stim_editor._input_designer_dialog is None
    assert stim_editor._rheobase_dialog is None
    assert stim_editor._stdp_window_dialog is None
    stim_editor.close()


def test_discard_input_designer_clears_cached_dialog(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    stim_editor, _ = _make_ramp_editor(qtbot)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args: (StimEditor.TARGETED_STDP_ROUTINE, True),
    )
    stim_editor.open_input_designer()
    assert stim_editor._input_designer_dialog is not None
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args: (StimEditor.RHEOBASE_ROUTINE, True),
    )
    stim_editor.open_input_designer()
    assert stim_editor._rheobase_dialog is not None
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args: (StimEditor.STDP_WINDOW_ROUTINE, True),
    )
    stim_editor.open_input_designer()
    assert stim_editor._stdp_window_dialog is not None

    # Replacing the project must not carry the previous designer state over.
    stim_editor.discard_input_designer()
    assert stim_editor._input_designer_dialog is None
    assert stim_editor._rheobase_dialog is None
    assert stim_editor._stdp_window_dialog is None

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
