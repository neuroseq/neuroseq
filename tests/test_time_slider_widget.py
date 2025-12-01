from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_widget import StimWidget
from neuroseq.time_slider_widget import TimeSliderWidget
from neuroseq.trace_widget import TraceWidget


def test_time_slider_widget(qtbot: QtBot) -> None:
    slider = TimeSliderWidget()
    qtbot.addWidget(slider)
    slider.show()
    qtbot.waitUntil(lambda: slider.isVisible())

    # create some widgets that should respond to slider time changes
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    trace_widget.set_data(TraceData("tr", np.zeros((1, 1800)), "mV", 0.001))
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    spike_widget = SpikeWidget(controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    spike_widget.set_data(SpikeData("sp", np.zeros(10), np.linspace(0, 2, 10)))
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    stim_widget.set_data(
        StimArray("stim", StimType.Excitation, np.zeros((2200, 1)), "mV", 0.001)
    )
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    heatmap_widget.set_data(
        SynapseData("syn", np.zeros((5, 5, 20)), np.linspace(0, 2, 20))
    )
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())

    # connect them to the slider
    slider.connect_widget(trace_widget)
    slider.connect_widget(spike_widget)
    slider.connect_widget(stim_widget)
    slider.connect_widget(heatmap_widget)

    # initial state is t=0 for all
    assert slider.t_max_ms == 0
    assert slider.t_ms == 0
    assert trace_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert heatmap_widget.get_timestamp_index() == 0

    # set a max time for the slider
    slider.set_max_time_secs(2)
    assert slider.t_max_ms == 2000

    # select max value of underlying slider
    slider.slider.setValue(999999)
    assert slider.t_ms == 2000
    assert trace_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert heatmap_widget.get_timestamp_index() == heatmap_widget.timestamps.size - 1

    # select min value of underlying slider
    slider.slider.setValue(-100)
    assert slider.t_ms == 0
    assert trace_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert heatmap_widget.get_timestamp_index() == 0

    # select mid value of underlying slider
    slider.slider.setValue(1000)
    assert slider.t_ms == 1000
    assert trace_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert heatmap_widget.get_timestamp_index() == pytest.approx(10, abs=1)
