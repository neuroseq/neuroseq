from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data_widget import DataWidget


def test_data_widget(
    qtbot: QtBot, make_trace_data, make_spike_data, make_synapse_data
) -> None:
    controller = ConstrainedPanZoomController()
    data_widget = DataWidget(controller=controller)
    qtbot.addWidget(data_widget)
    data_widget.show()
    qtbot.waitUntil(lambda: data_widget.isVisible())
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_trace_data(n_neurons=3))
    assert data_widget.trace_widget.n_neurons == 3
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_synapse_data(n_i=6, n_j=4))
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_spike_data(n_neurons=5))
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons <= 5
    data_widget.clear_data()
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.close()
