from __future__ import annotations

import numpy as np
import pytest
from formozov_project.stim_trace_widget import StimTraceWidget
from pytestqt.qtbot import QtBot


def test_stim_widget(qtbot: QtBot) -> None:
    stim_trace_widget = StimTraceWidget()
    qtbot.addWidget(stim_trace_widget)
    stim_trace_widget.show()
    qtbot.waitUntil(lambda: stim_trace_widget.isVisible())
    assert len(stim_trace_widget.neuron_plots) == 0
    n_neurons = 20
    n_stim_per_neuron = 20
    n_timestamps = 200
    dt = 0.0683
    timestamps = dt * np.arange(n_timestamps)
    stims = np.zeros((n_neurons, n_timestamps))
    for row in stims:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    trace = np.random.rand(n_neurons, n_timestamps)
    stim_trace_widget.set_data(stims, trace, timestamps)
    assert len(stim_trace_widget.neuron_plots) == len(stims)
    assert len(stim_trace_widget.neuron_plots[0][0].getData()[0]) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0][0].getData()[0], timestamps)
    assert len(stim_trace_widget.neuron_plots[0][1].xData) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0][1].xData, timestamps)
    # set dt instead of providing timestamps array, same x-values should be constructed
    stim_trace_widget.set_data(stims, trace, dt=dt)
    assert len(stim_trace_widget.neuron_plots) == len(stims)
    assert len(stim_trace_widget.neuron_plots[0][0].getData()[0]) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0][0].getData()[0], timestamps)
    assert len(stim_trace_widget.neuron_plots[0][1].xData) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0][1].xData, timestamps)
    assert stim_trace_widget.selected_neuron is None
    stim_trace_widget.select_neuron(8)
    assert stim_trace_widget.selected_neuron == 8
    stim_trace_widget.select_neuron(8)
    assert stim_trace_widget.selected_neuron is None
    stim_trace_widget.select_neuron(13)
    assert stim_trace_widget.selected_neuron == 13
    stim_trace_widget.select_neuron(None)
    assert stim_trace_widget.selected_neuron is None
    # set empty data to clear plot
    stim_trace_widget.set_data()
    assert len(stim_trace_widget.neuron_plots) == 0
    # setting only stim or only traces also results in an empty plot
    stim_trace_widget.set_data(stims=stims)
    assert len(stim_trace_widget.neuron_plots) == 0
    stim_trace_widget.set_data(traces=trace)
    assert len(stim_trace_widget.neuron_plots) == 0
    with pytest.raises(ValueError):
        # stim and trace arrays must have the same shape
        stim_trace_widget.set_data(stims, np.random.rand(12, 12))
