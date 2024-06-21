from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
from formozov_project.trace_widget import TraceWidget
from pytestqt.qtbot import QtBot


def test_trace_widget(qtbot: QtBot) -> None:
    trace_widget = TraceWidget()
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    assert len(trace_widget.neuron_plots) == 0
    assert trace_widget.selected_neuron is None
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
    assert trace_widget.selected_neuron is None
    trace_widget.select_neuron(14)
    assert trace_widget.selected_neuron == 14
    trace_widget.select_neuron(14)
    assert trace_widget.selected_neuron is None
