from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
from formozov_project.spike_widget import SpikeWidget
from pytestqt.qtbot import QtBot


def test_spike_widget(qtbot: QtBot) -> None:
    spike_widget = SpikeWidget()
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert len(spike_widget.neuron_plots) == 0
    n_neurons = 20
    n_spikes = 100
    n_timesteps = 1000
    dt = 0.33
    # random indices, with no spikes with index zero
    spike_indices = np.random.randint(1, n_neurons, n_spikes)
    spike_times = np.arange(n_spikes) * dt * (
        n_timesteps - 1
    ) / n_spikes + 0.2 * np.random.rand(n_spikes)
    spike_widget.set_data(spike_indices, spike_times, dt, n_neurons, n_timesteps)
    n_spikes_neuron_11 = np.count_nonzero(spike_indices == 11)
    assert len(spike_widget.neuron_plots) == n_neurons
    neuron_0_plot = cast(pg.BarGraphItem, spike_widget.neuron_plots[0])
    assert len(neuron_0_plot.getData()[0]) == 0
    neuron_11_plot = cast(pg.BarGraphItem, spike_widget.neuron_plots[11])
    assert len(neuron_11_plot.getData()[0]) == n_spikes_neuron_11
    assert spike_widget.selected_neuron is None
    spike_widget.select_neuron(8)
    assert spike_widget.selected_neuron == 8
    spike_widget.select_neuron(8)
    assert spike_widget.selected_neuron is None
    spike_widget.select_neuron(13)
    assert spike_widget.selected_neuron == 13
    spike_widget.select_neuron(None)
    assert spike_widget.selected_neuron is None
    spike_widget.clear_data()
    assert len(spike_widget.neuron_plots) == 0
