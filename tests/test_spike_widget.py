from __future__ import annotations

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import SpikeData
from neuroseq.spike_widget import SpikeWidget


def test_spike_widget(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert spike_widget.n_neurons == 0
    n_neurons = 20
    n_spikes = 100
    n_timesteps = 1000
    dt = 0.33
    # random indices, with no spikes with index zero
    spike_indices = np.random.randint(1, n_neurons, n_spikes)
    spike_times = np.arange(n_spikes) * dt * (
        n_timesteps - 1
    ) / n_spikes + 0.2 * np.random.rand(n_spikes)
    data = SpikeData("spikes", spike_indices, spike_times)
    spike_widget.set_data(data)
    assert spike_widget.n_neurons == max(spike_indices) + 1
    spike_widget.close()
