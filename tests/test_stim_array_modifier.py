from __future__ import annotations

import numpy as np
from pytest import approx
from pytestqt.qtbot import QtBot

from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_array_modifier import PadReversibleAction
from neuroseq.stimulation.stim_array_modifier import StimArrayModifier
from neuroseq.stimulation.stim_array_modifier import TrimReversibleAction
from neuroseq.visualization.stim_widget import StimWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)


def test_timed_array_modifier(qtbot: QtBot) -> None:
    stim_array = StimArray(
        "stim", StimType.Excitation, np.zeros((100, 10)), "mV", 0.001
    )
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    tma = StimArrayModifier(stim_array, stim_widget)
    assert tma.can_undo() is False
    assert tma.can_redo() is False
    assert len(tma) == 0
    # do an action with an array of new values
    tma.do(slice(2, 4), slice(10, 30), np.ones((20, 2)))
    assert np.allclose(stim_array.values[10:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 1
    # do an action with a float as new values
    tma.do(slice(2, 3), slice(17, 19), 2.6)
    assert np.allclose(stim_array.values[17:19, 2:3], 2.6)
    assert np.allclose(stim_array.values[10:30, 3:4], 1)
    assert np.allclose(stim_array.values[19:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 2
    # undo the last action
    tma.undo()
    assert np.allclose(stim_array.values[10:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo again
    tma.undo()
    assert np.allclose(stim_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
    # redo
    tma.redo()
    assert np.allclose(stim_array.values[10:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # do a new action - new action added at current point in stack, redos are removed
    tma.do(slice(6, 7), slice(15, 16), 0.5)
    assert stim_array.values[15, 6] == approx(0.5)
    assert np.allclose(stim_array.values[10:15, 2:4], 1)
    assert np.allclose(stim_array.values[16:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 2
    # undo the last action
    tma.undo()
    assert np.allclose(stim_array.values[10:30, 2:4], 1)
    assert np.allclose(stim_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo again
    tma.undo()
    assert np.allclose(stim_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo when there are no undos left is a no-op
    tma.undo()
    assert np.allclose(stim_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
    stim_widget.close()


def test_trim_reversible_action_restores_all_arrays(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)

    project = Project()
    long_vals = np.zeros((1000, 3))
    long_vals[900:950, 0] = 1.0  # non-zero data in the tail that gets trimmed
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, long_vals, "mV", 0.001)
    )
    project.add_stim_array(
        StimArray("stim2", StimType.Inhibition, np.ones((1000, 3)), "mV", 0.001)
    )

    action = TrimReversibleAction(project, 0.4, stim_widget, "stim")
    action(project.stim_arrays["stim"], stim_widget)
    # Every array is trimmed to the same 0.4 s timeline.
    assert project.stim_arrays["stim"].values.shape[0] == 400
    assert project.stim_arrays["stim2"].values.shape[0] == 400
    assert project.simulation_timeline_seconds() == approx(0.4)

    action.undo(project.stim_arrays["stim"], stim_widget)
    # Undo restores the original lengths and the trimmed tail contents.
    assert project.stim_arrays["stim"].values.shape[0] == 1000
    assert project.stim_arrays["stim2"].values.shape[0] == 1000
    assert np.allclose(project.stim_arrays["stim"].values[900:950, 0], 1.0)
    assert np.allclose(project.stim_arrays["stim2"].values, 1.0)
    assert project.simulation_timeline_seconds() == approx(1.0)
    stim_widget.close()


def test_trim_reversible_action_only_one_array(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)

    project = Project()
    project.add_stim_array(
        StimArray("a", StimType.Excitation, np.zeros((1000, 2)), "mV", 0.001)
    )
    project.add_stim_array(
        StimArray("b", StimType.Inhibition, np.ones((1000, 2)), "mV", 0.001)
    )

    action = TrimReversibleAction(project, 0.4, stim_widget, "a", only_stim_name="a")
    action(project.stim_arrays["a"], stim_widget)
    # Only "a" is trimmed; "b" is left untouched.
    assert project.stim_arrays["a"].values.shape[0] == 400
    assert project.stim_arrays["b"].values.shape[0] == 1000

    action.undo(project.stim_arrays["a"], stim_widget)
    assert project.stim_arrays["a"].values.shape[0] == 1000
    assert project.stim_arrays["b"].values.shape[0] == 1000
    stim_widget.close()


def test_pad_reversible_action_restores_all_arrays(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)

    project = Project()
    vals = np.zeros((500, 3))
    vals[400:450, 0] = 1.0  # non-zero content that must survive pad + undo
    project.add_stim_array(StimArray("stim", StimType.Excitation, vals, "mV", 0.001))
    project.add_stim_array(
        StimArray("stim2", StimType.Inhibition, np.ones((500, 3)), "mV", 0.001)
    )

    action = PadReversibleAction(project, 0.9, stim_widget, "stim")
    action(project.stim_arrays["stim"], stim_widget)
    # Every array grows to the same 0.9 s timeline; new rows are zeros.
    assert project.stim_arrays["stim"].values.shape[0] == 900
    assert project.stim_arrays["stim2"].values.shape[0] == 900
    assert np.allclose(project.stim_arrays["stim"].values[500:], 0.0)
    assert np.allclose(project.stim_arrays["stim2"].values[500:], 0.0)
    assert project.simulation_timeline_seconds() == approx(0.9)

    action.undo(project.stim_arrays["stim"], stim_widget)
    # Undo strips the padding and leaves the original content untouched.
    assert project.stim_arrays["stim"].values.shape[0] == 500
    assert project.stim_arrays["stim2"].values.shape[0] == 500
    assert np.allclose(project.stim_arrays["stim"].values[400:450, 0], 1.0)
    assert np.allclose(project.stim_arrays["stim2"].values, 1.0)
    assert project.simulation_timeline_seconds() == approx(0.5)
    stim_widget.close()
