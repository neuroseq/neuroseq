from __future__ import annotations

import numpy as np
from formozov_project.trace_widget import TraceWidget
from pytestqt.qtbot import QtBot


def test_trace_widget_no_data(qtbot: QtBot) -> None:
    trace_widget = TraceWidget()
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    assert len(trace_widget.neuron_plots) == 0
    assert trace_widget.selected_neuron is None


def test_trace_widget_data(qtbot: QtBot) -> None:
    traces = []
    n_neurons = 20
    n_timestamps = 100
    timestamps = np.arange(n_timestamps)
    for _ in range(n_neurons):
        traces.append(np.random.rand(n_timestamps))
    trace_widget = TraceWidget(traces=traces, timestamps=timestamps)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    assert len(trace_widget.neuron_plots) == len(traces)
    assert trace_widget.selected_neuron is None
    trace_widget.select_neuron(14)
    assert trace_widget.selected_neuron == 14
    trace_widget.select_neuron(14)
    assert trace_widget.selected_neuron is None
