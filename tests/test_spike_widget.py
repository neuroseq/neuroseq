from __future__ import annotations

import numpy as np
from formozov_project.spike_widget import SpikeWidget
from pytestqt.qtbot import QtBot


def test_spike_widget_no_data(qtbot: QtBot) -> None:
    spike_widget = SpikeWidget()
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert len(spike_widget.neuron_plots) == 0


def test_spike_widget_data(qtbot: QtBot) -> None:
    spikes = []
    n_neurons = 20
    n_spikes = 5
    n_timestamps = 100
    timestamps = np.arange(n_timestamps)
    for _ in range(n_neurons):
        spike = np.zeros(n_timestamps)
        for _ in range(n_spikes):
            spike[np.random.randint(0, n_timestamps)] = 5.4
        spikes.append(spike)
    spike_widget = SpikeWidget(spikes=spikes, timestamps=timestamps)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert len(spike_widget.neuron_plots) == len(spikes)
    assert spike_widget.selected_neuron is None
    spike_widget.select_neuron(8)
    assert spike_widget.selected_neuron == 8
    spike_widget.select_neuron(8)
    assert spike_widget.selected_neuron is None
    spike_widget.select_neuron(13)
    assert spike_widget.selected_neuron == 13
    spike_widget.select_neuron(None)
    assert spike_widget.selected_neuron is None
