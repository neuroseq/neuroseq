from __future__ import annotations

import brian2 as b2
import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.stim_widget import StimWidget
from neuroseq.utils import InteractMode


def mouse_click(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    pos: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    delay_ms=10,
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseClick(
        widget.viewport(),
        button=button,
        pos=pos,
    )


def mouse_click_and_drag(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    start: QtCore.QPoint,
    end: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    delay_ms=10,
    intermediate_points=4,
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseMove(widget.windowHandle(), pos=start)
    qtbot.wait(delay_ms)
    qtbot.mousePress(
        widget.windowHandle(),
        button,
        pos=start,
    )
    n_points = intermediate_points + 2
    for x, y in zip(
        np.linspace(start.x(), end.x(), n_points),
        np.linspace(start.y(), end.y(), n_points),
        strict=True,
    ):
        qtbot.wait(delay_ms)
        qtbot.mouseMove(
            widget.windowHandle(),
            pos=QtCore.QPoint(int(x), int(y)),
        )
    qtbot.wait(delay_ms)
    qtbot.mouseRelease(
        widget,
        button,
        pos=end,
    )
    qtbot.wait(delay_ms)


@pytest.fixture
def stims() -> b2.TimedArray:
    n_neurons = 20
    n_stim_per_neuron = 5
    n_timestamps = 100
    dt = 0.683
    stim = np.zeros((n_neurons, n_timestamps))
    for row in stim:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    return b2.TimedArray(stim.T, dt=dt * b2.second)


def test_stim_widget_select_mode(qtbot: QtBot, stims: b2.TimedArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_widget.set_mode(InteractMode.SELECT)
    assert stim_widget.n_neurons == 0
    n_timestamps = stims.values.shape[0]
    n_neurons = stims.values.shape[1]
    stim_widget.set_data(stims)
    assert stim_widget.n_neurons == n_neurons
    assert stim_widget.n_timestamps == n_timestamps
    # initially no neurons are selected
    assert stim_widget.selected_neuron_slice is None
    assert stim_widget.selected_time_slice is None
    stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.waitUntil(lambda: stim_widget.selected_neuron_slice == slice(1, 3))
    assert stim_widget.selected_time_slice == slice(10, 20)
    stim_widget.unselect_neurons()
    qtbot.waitUntil(lambda: stim_widget.selected_neuron_slice is None)
    assert stim_widget.selected_time_slice is None
    # a missing slice argument should default to the full range
    stim_widget.select_neurons(time_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(0, n_neurons)
    assert stim_widget.selected_time_slice == slice(1, 3)
    stim_widget.select_neurons(neuron_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(1, 3)
    assert stim_widget.selected_time_slice == slice(0, n_timestamps)
    stim_widget.select_neurons()
    assert stim_widget.selected_neuron_slice == slice(0, n_neurons)
    assert stim_widget.selected_time_slice == slice(0, n_timestamps)
    stim_widget.unselect_neurons()
    ## Select about half the neurons and half the time points
    x0 = stim_widget.size().width() // 4
    y0 = stim_widget.size().height() // 4
    mouse_click_and_drag(
        qtbot, stim_widget, QtCore.QPoint(x0, y0), QtCore.QPoint(x0 * 3, y0 * 3)
    )
    assert stim_widget.selected_neuron_slice is not None
    assert stim_widget.selected_time_slice is not None
    assert 0.15 * n_neurons < stim_widget.selected_neuron_slice.start < 0.35 * n_neurons
    assert 0.65 * n_neurons < stim_widget.selected_neuron_slice.stop < 0.85 * n_neurons
    assert (
        0.15 * n_timestamps
        < stim_widget.selected_time_slice.start
        < 0.35 * n_timestamps
    )
    assert (
        0.65 * n_timestamps < stim_widget.selected_time_slice.stop < 0.85 * n_timestamps
    )


def test_stim_widget_draw_mode(qtbot: QtBot, stims: b2.TimedArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_widget.set_mode(InteractMode.DRAW)
    assert stim_widget.n_neurons == 0
    n_timestamps = stims.values.shape[0]
    n_neurons = stims.values.shape[1]
    stim_widget.set_data(stims)
    assert stim_widget.n_neurons == n_neurons
    x0 = stim_widget.size().width() // 4
    y0 = stim_widget.size().height() // 4
    x1 = x0 * 2
    y1 = y0 * 2
    x2 = x0 * 3
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # mouse clicks
    stim_widget.setFocus()
    qtbot.wait(20)
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(qtbot, stim_widget, QtCore.QPoint(x0, y0))
        neuron_index_offset, time_index, mouse_button = signal.args
        assert 0 < time_index / n_timestamps < 0.4
        assert 0 < neuron_index_offset / n_neurons < 1.0
        assert mouse_button == 1
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(
            qtbot,
            stim_widget.figure.canvas,
            QtCore.QPoint(x1, y0),
            QtCore.Qt.MouseButton.RightButton,
        )
        neuron_index_offset, time_index, mouse_button = signal.args
        assert 0.3 < time_index / n_timestamps < 0.7
        assert 0 < neuron_index_offset / n_neurons < 1.0
        assert mouse_button == 2
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(qtbot, stim_widget.figure.canvas, QtCore.QPoint(x2, y1))
        neuron_index_offset, time_index, mouse_button = signal.args
        assert 0.6 < time_index / n_timestamps < 1.0
        assert 0 < neuron_index_offset / n_neurons < 1.0
        assert mouse_button == 1
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # todo: mouse drag in draw mode should draw a line
