from __future__ import annotations

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.data_widget import DataWidget


def test_data_widget(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    data_widget = DataWidget(controller=controller)
    qtbot.addWidget(data_widget)
    data_widget.show()
    qtbot.waitUntil(lambda: data_widget.isVisible())
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(
        TraceData("trace", np.random.random_sample((3, 6)), "mV", 0.001)
    )
    assert data_widget.trace_widget.n_neurons == 3
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(
        SynapseData(
            "synapse",
            np.random.random_sample((3, 3, 5)),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        )
    )
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(
        SpikeData(
            "spike", np.array([2, 6, 4, 3, 1, 7, 6, 4, 1, 4]), np.linspace(0, 0.1, 10)
        )
    )
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 8
    data_widget.clear_data()
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.close()
