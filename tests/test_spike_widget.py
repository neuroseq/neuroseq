from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.spike_widget import SpikeWidget


def test_spike_widget(qtbot: QtBot, make_spike_data) -> None:
    controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert spike_widget.n_neurons == 0
    spike_data = make_spike_data(n_neurons=10, n_spikes=50, max_time_secs=1.0)
    spike_widget.set_data(spike_data)
    assert spike_widget.n_neurons == max(spike_data.spike_indices) + 1
    spike_widget.close()
