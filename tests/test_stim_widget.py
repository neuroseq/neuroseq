from __future__ import annotations

import numpy as np
from formozov_project.stim_widget import StimWidget
from pytestqt.qtbot import QtBot


def test_stim_widget_no_data(qtbot: QtBot) -> None:
    stim_widget = StimWidget()
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == 0


def test_stim_widget_data(qtbot: QtBot) -> None:
    n_neurons = 20
    n_stim_per_neuron = 5
    n_timestamps = 100
    timestamps = np.arange(n_timestamps)
    stim = np.zeros((n_neurons, n_timestamps))
    for row in stim:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    stim_widget = StimWidget(stim=stim, timestamps=timestamps)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    assert len(stim_widget.neuron_plots) == len(stim)
    assert stim_widget.selected_neuron is None
    stim_widget.select_neuron(8)
    assert stim_widget.selected_neuron == 8
    stim_widget.select_neuron(8)
    assert stim_widget.selected_neuron is None
    stim_widget.select_neuron(13)
    assert stim_widget.selected_neuron == 13
    stim_widget.select_neuron(None)
    assert stim_widget.selected_neuron is None
