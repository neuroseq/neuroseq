from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
import pytest
from formozov_project.stim_widget import StimWidget
from pytestqt.qtbot import QtBot
from qtpy import QtCore


@pytest.fixture
def stim_timestamps_dt() -> tuple[np.ndarray, np.ndarray, float]:
    n_neurons = 20
    n_stim_per_neuron = 5
    n_timestamps = 100
    dt = 0.683
    timestamps = dt * np.arange(n_timestamps)
    stim = np.zeros((n_neurons, n_timestamps))
    for row in stim:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    return stim, timestamps, dt


def test_stim_widget_no_mouse_capture(
    qtbot: QtBot, stim_timestamps_dt: tuple[np.ndarray, np.ndarray, float]
) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0
    stim, timestamps, dt = stim_timestamps_dt
    n_timestamps = len(timestamps)
    stim_widget.set_data(stim, timestamps)
    assert len(stim_widget.neuron_plots) == len(stim)
    neuron_plot = cast(pg.BarGraphItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    # bar plot x-values are the center of the bars, so shifted forwards by 0.5dt compared to timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps + 0.5 * dt)
    # set dt instead of providing timestamps array, same x-values should be constructed
    stim_widget.set_data(stim, dt=dt)
    assert len(stim_widget.neuron_plots) == len(stim)
    neuron_plot = cast(pg.BarGraphItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps + 0.5 * dt)
    # initially no neurons are selected
    assert stim_widget.selected_neuron_slice is None
    assert stim_widget.selected_time_slice is None
    stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.wait(10)
    assert stim_widget.selected_neuron_slice == slice(1, 3)
    assert stim_widget.selected_time_slice == slice(10, 20)
    stim_widget.unselect_neurons()
    qtbot.wait(10)
    assert stim_widget.selected_neuron_slice is None
    assert stim_widget.selected_time_slice is None
    # a missing slice argument should default to the full range
    stim_widget.select_neurons(time_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(0, -1)
    assert stim_widget.selected_time_slice == slice(1, 3)
    stim_widget.select_neurons(neuron_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(1, 3)
    assert stim_widget.selected_time_slice == slice(0, -1)
    stim_widget.select_neurons()
    assert stim_widget.selected_neuron_slice == slice(0, -1)
    assert stim_widget.selected_time_slice == slice(0, -1)
    stim_widget.repaint()


def test_stim_widget_mouse_capture(
    qtbot: QtBot, stim_timestamps_dt: tuple[np.ndarray, np.ndarray, float]
) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.capture_mouse_events = True
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0
    stim, timestamps, dt = stim_timestamps_dt
    dt = timestamps[1] - timestamps[0]
    stim_widget.set_data(stim, dt=dt)
    assert len(stim_widget.neuron_plots) == len(stim)
    print(stim_widget.size())
    x0 = stim_widget.size().width() // 3
    y0 = stim_widget.size().height() // 3
    x1 = x0 * 2
    y1 = y0 * 2
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # mouse click sets timeline position
    stim_widget.setFocus()
    qtbot.wait(10)
    assert stim_widget.timeline.x() == 0
    qtbot.mouseClick(
        stim_widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(x0, y0),
    )
    assert 0 < stim_widget.timeline.x() < timestamps[-1] / 2
    qtbot.mouseClick(
        stim_widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(x1, y0),
    )
    assert timestamps[-1] / 2 < stim_widget.timeline.x() < timestamps[-1]
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # mouse drag selects neurons
    qtbot.mouseMove(stim_widget.viewport(), pos=QtCore.QPoint(x0, y0))
    qtbot.mousePress(
        stim_widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(x0, y0),
    )
    qtbot.mouseMove(stim_widget.viewport(), pos=QtCore.QPoint(x1, y1))
    qtbot.mouseRelease(
        stim_widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(x1, y1),
    )
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
        < stim.shape[0]
    )
