from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
from formozov_project.stim_widget import StimWidget
from pytestqt.qtbot import QtBot


def test_stim_widget(qtbot: QtBot) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0
    n_neurons = 20
    n_stim_per_neuron = 5
    n_timestamps = 100
    dt = 0.683
    timestamps = dt * np.arange(n_timestamps)
    stim = np.zeros((n_neurons, n_timestamps))
    for row in stim:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    stim_widget.set_data(stim, timestamps)
    assert len(stim_widget.neuron_plots) == len(stim)
    neuron_plot = cast(pg.BarGraphItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps)
    # set dt instead of providing timestamps array, same x-values should be constructed
    stim_widget.set_data(stim, dt=dt)
    assert len(stim_widget.neuron_plots) == len(stim)
    neuron_plot = cast(pg.BarGraphItem, stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], timestamps)
    assert stim_widget.selected_neuron is None
    stim_widget.select_neuron(8)
    assert stim_widget.selected_neuron == 8
    stim_widget.select_neuron(8)
    assert stim_widget.selected_neuron is None
    stim_widget.select_neuron(13)
    assert stim_widget.selected_neuron == 13
    stim_widget.select_neuron(None)
    assert stim_widget.selected_neuron is None
    stim_widget.clear_data()
    assert len(stim_widget.neuron_plots) == 0
