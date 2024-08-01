from __future__ import annotations

from typing import cast

import brian2.numpy_ as np
import pyqtgraph as pg
from neuroseq.trace_widget import TraceWidget
from pytestqt.qtbot import QtBot


def test_trace_widget(qtbot: QtBot) -> None:
    trace_widget = TraceWidget()
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    assert len(trace_widget.neuron_plots) == 0
    n_neurons = 20
    n_timestamps = 100
    dt = 0.0991
    traces = np.random.rand(n_neurons, n_timestamps)
    timestamps = dt * np.arange(n_timestamps)
    trace_widget.set_data(traces, timestamps)
    assert len(trace_widget.neuron_plots) == traces.shape[0]
    neuron_plot = cast(pg.PlotDataItem, trace_widget.neuron_plots[0])
    assert len(neuron_plot.xData) == n_timestamps
    assert np.allclose(neuron_plot.xData, timestamps)
    # set dt instead of providing timestamps array, same x-values should be constructed
    trace_widget.set_data(traces, dt=dt)
    assert len(trace_widget.neuron_plots) == traces.shape[0]
    neuron_plot = cast(pg.PlotDataItem, trace_widget.neuron_plots[0])
    assert len(neuron_plot.xData) == n_timestamps
    assert np.allclose(neuron_plot.xData, timestamps)
    # initially no neurons are selected
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
    trace_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice == slice(1, 3)
    assert trace_widget.selected_time_slice == slice(10, 20)
    # de-selecting neurons reverts to displaying all of them as selected
    trace_widget.unselect_neurons()
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
