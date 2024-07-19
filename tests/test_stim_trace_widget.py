from __future__ import annotations

import brian2.numpy_ as np
import pytest
from formozov_project.stim_trace_widget import StimTraceWidget
from pytestqt.qtbot import QtBot

delay_for_widgets_to_update = 50


def test_stim_trace_widget(qtbot: QtBot) -> None:
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
    assert len(stim_trace_widget.neuron_plots[0].bar.getData()[0]) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0].bar.getData()[0], timestamps)
    assert len(stim_trace_widget.neuron_plots[0].line.xData) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0].line.xData, timestamps)
    # set dt instead of providing timestamps array, same x-values should be constructed
    stim_trace_widget.set_data(stims, trace, dt=dt)
    assert len(stim_trace_widget.neuron_plots) == len(stims)
    assert len(stim_trace_widget.neuron_plots[0].bar.getData()[0]) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0].bar.getData()[0], timestamps)
    assert len(stim_trace_widget.neuron_plots[0].line.xData) == n_timestamps
    assert np.allclose(stim_trace_widget.neuron_plots[0].line.xData, timestamps)
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
    stim_trace_widget.set_data(stims, trace, timestamps)
    # initially no neurons are selected
    assert stim_trace_widget.selected_neuron_slice is None
    assert stim_trace_widget.selected_time_slice is None
    stim_trace_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.wait(10)
    assert stim_trace_widget.selected_neuron_slice == slice(1, 3)
    assert stim_trace_widget.selected_time_slice == slice(10, 20)
    stim_trace_widget.unselect_neurons()
    qtbot.wait(10)
    assert stim_trace_widget.selected_neuron_slice is None
    assert stim_trace_widget.selected_time_slice is None
