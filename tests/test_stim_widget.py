from __future__ import annotations

from typing import cast

import brian2 as b2
import brian2.numpy_ as np
import pytest
import qtpy
from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.stim_widget import StimWidget
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy.QtTest import QTest


def mouse_ctrl_click(
    qtbot: QtBot, widget: QtGui.QWidget, pos: QtCore.QPoint, delay_ms=10
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseClick(
        widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        pos=pos,
    )


def mouse_ctrl_click_and_drag(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    start: QtCore.QPoint,
    end: QtCore.QPoint,
    delay_ms=10,
    intermediate_points=4,
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseMove(widget.viewport(), pos=start)
    qtbot.wait(delay_ms)
    qtbot.mousePress(
        widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        pos=start,
    )
    n_points = intermediate_points + 2
    for x, y in zip(
        np.linspace(start.x(), end.x(), n_points),
        np.linspace(start.y(), end.y(), n_points),
        strict=True,
    ):
        qtbot.wait(delay_ms)
        QTest.mouseEvent(
            QTest.MouseAction.MouseMove,
            widget.viewport(),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.QPoint(int(x), int(y)),
            0,
        )
    qtbot.wait(delay_ms)
    qtbot.mouseRelease(
        widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
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


def test_stim_widget_no_ctrl_key(qtbot: QtBot, stims: b2.TimedArray) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0
    n_timestamps = stims.values.shape[0]
    n_neurons = stims.values.shape[1]
    timestamps = np.arange(n_timestamps) * stims.dt
    stim_widget.set_data(stims)
    assert len(stim_widget.neuron_plots) == n_neurons
    neuron_plot = cast(SelectableBarPlotItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    # bar plot x-values are the center of the bars, so shifted forwards by 0.5dt compared to timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps + 0.5 * stims.dt)
    # set dt instead of providing timestamps array, same x-values should be constructed
    stim_widget.set_data(stims)
    assert len(stim_widget.neuron_plots) == n_neurons
    neuron_plot = cast(SelectableBarPlotItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps + 0.5 * stims.dt)
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


@pytest.mark.skipif(
    not qtpy.PYSIDE6,
    reason="click&drag with ctrl key uses mouseEvent to be able to add the ctrl key modifier, which is apparently only provided by qtest in pyside6",
)
def test_stim_widget_mouse_with_ctrl_key(qtbot: QtBot, stims: b2.TimedArray) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0
    n_timestamps = stims.values.shape[0]
    n_neurons = stims.values.shape[1]
    timestamps = np.arange(n_timestamps) * stims.dt
    stim_widget.set_data(stims)
    assert len(stim_widget.neuron_plots) == n_neurons
    x0 = stim_widget.size().width() // 4
    y0 = stim_widget.size().height() // 4
    x1 = x0 * 2
    y1 = y0 * 2
    x2 = x0 * 3
    y2 = y0 * 3
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # ctrl+mouse click signal
    stim_widget.setFocus()
    qtbot.wait(20)
    assert stim_widget.timeline.x() == 0
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_ctrl_click(qtbot, stim_widget, QtCore.QPoint(x0, y0))
        neuron_index_offset, time_index = signal.args
        assert 0 < time_index / n_timestamps < 0.4
        assert 0 < neuron_index_offset / n_neurons < 1.0
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_ctrl_click(qtbot, stim_widget, QtCore.QPoint(x1, y0))
        neuron_index_offset, time_index = signal.args
        assert 0.3 < time_index / n_timestamps < 0.7
        assert 0 < neuron_index_offset / n_neurons < 1.0
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_ctrl_click(qtbot, stim_widget, QtCore.QPoint(x2, y1))
        neuron_index_offset, time_index = signal.args
        assert 0.6 < time_index / n_timestamps < 1.0
        assert 0 < neuron_index_offset / n_neurons < 1.0
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # mouse drag selects neurons
    with qtbot.waitSignal(stim_widget.selection_made) as signal:
        mouse_ctrl_click_and_drag(
            qtbot, stim_widget, QtCore.QPoint(x0, y0), QtCore.QPoint(x2, y2)
        )
        neuron_slice, time_slice = signal.args
        assert neuron_slice == stim_widget.selected_neuron_slice
        assert time_slice == stim_widget.selected_time_slice

    assert stim_widget.selected_time_slice is not None
    assert stim_widget.selected_time_slice.start > 0
    assert (
        stim_widget.selected_time_slice.start
        < stim_widget.selected_time_slice.stop
        < len(timestamps)
    )
    assert stim_widget.selected_neuron_slice is not None
    assert stim_widget.selected_neuron_slice.start > 0
    assert (
        stim_widget.selected_neuron_slice.start
        < stim_widget.selected_neuron_slice.stop
        < n_neurons
    )

    # mouse drag starting from inside selected region causes a selection_moved signal
    # drag up and to the left
    with qtbot.waitSignal(stim_widget.selection_moved) as signal:
        mouse_ctrl_click_and_drag(
            qtbot, stim_widget, QtCore.QPoint(x1, y1), QtCore.QPoint(x0, y0)
        )
        neuron_index_offset, time_index_offset = signal.args
        assert neuron_index_offset > 0
        assert time_index_offset < 0
