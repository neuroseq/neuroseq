from __future__ import annotations

import brian2 as b2
from brian2 import numpy_ as np
from pytest import approx
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.stim_widget import StimWidget
from neuroseq.timed_array_modifer import TimedArrayModifier


def test_timed_array_modifier(qtbot: QtBot) -> None:
    timed_array = b2.TimedArray(np.zeros((100, 10)), dt=1 * b2.ms)
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    tma = TimedArrayModifier(timed_array, stim_widget)
    assert tma.can_undo() is False
    assert tma.can_redo() is False
    assert len(tma) == 0
    # do an action with an array of new values
    tma.do(slice(2, 4), slice(10, 30), np.ones((20, 2)))
    assert np.allclose(timed_array.values[10:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 1
    # do an action with a float as new values
    tma.do(slice(2, 3), slice(17, 19), 2.6)
    assert np.allclose(timed_array.values[17:19, 2:3], 2.6)
    assert np.allclose(timed_array.values[10:30, 3:4], 1)
    assert np.allclose(timed_array.values[19:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 2
    # undo the last action
    tma.undo()
    assert np.allclose(timed_array.values[10:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo again
    tma.undo()
    assert np.allclose(timed_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
    # redo
    tma.redo()
    assert np.allclose(timed_array.values[10:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # do a new action - new action added at current point in stack, redos are removed
    tma.do(slice(6, 7), slice(15, 16), 0.5)
    assert timed_array.values[15, 6] == approx(0.5)
    assert np.allclose(timed_array.values[10:15, 2:4], 1)
    assert np.allclose(timed_array.values[16:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is False
    assert len(tma) == 2
    # undo the last action
    tma.undo()
    assert np.allclose(timed_array.values[10:30, 2:4], 1)
    assert np.allclose(timed_array.values[30:, 4:], 0)
    assert tma.can_undo() is True
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo again
    tma.undo()
    assert np.allclose(timed_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
    # undo when there are no undos left is a no-op
    tma.undo()
    assert np.allclose(timed_array.values, 0)
    assert tma.can_undo() is False
    assert tma.can_redo() is True
    assert len(tma) == 2
