from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.visualization.heatmap_widget import HeatmapWidget
from neuroseq.visualization.spike_widget import SpikeWidget
from neuroseq.visualization.stim_widget import StimWidget
from neuroseq.visualization.trace_widget import TraceWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from neuroseq.visualization_support.time_slider_widget import TimeSliderWidget


def test_time_slider_widget(
    qtbot: QtBot, make_spike_data, make_synapse_data, make_trace_data
) -> None:
    slider = TimeSliderWidget()
    qtbot.addWidget(slider)
    slider.show()
    qtbot.waitUntil(lambda: slider.isVisible())

    # create some widgets that should respond to slider time changes
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    trace_widget.set_data(make_trace_data(n_neurons=2, n_timestamps=1800))
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    spike_widget = SpikeWidget(controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    spike_widget.set_data(make_spike_data(n_neurons=2, max_time_secs=2.0))
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
        make_synapse_data(n_i=3, n_j=2, n_timestamps=20, max_time_secs=2.0)
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
    assert heatmap_widget.idx == 0

    # set a max time for the slider
    slider.set_max_time_secs(2)
    assert slider.t_max_ms == 2000

    # select max value of underlying slider
    slider.slider.setValue(999999)
    assert slider.t_ms == 2000
    assert trace_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(2000.0, abs=0.1)
    assert heatmap_widget.timestamps is not None
    assert heatmap_widget.idx == heatmap_widget.timestamps.size - 1

    # select min value of underlying slider
    slider.slider.setValue(-100)
    assert slider.t_ms == 0
    assert trace_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(0.0, abs=0.1)
    assert heatmap_widget.idx == 0

    # select mid value of underlying slider
    slider.slider.setValue(1000)
    assert slider.t_ms == 1000
    assert trace_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert spike_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert stim_widget.timeline.t == pytest.approx(1000.0, abs=0.1)
    assert heatmap_widget.idx == pytest.approx(10, abs=1)

    trace_widget.close()
    spike_widget.close()
    stim_widget.close()
    heatmap_widget.close()
    slider.close()


def test_time_slider_emits_time_changed_on_drag_and_explicit_set(qtbot: QtBot) -> None:
    slider = TimeSliderWidget()
    qtbot.addWidget(slider)
    slider.set_max_time_secs(2)
    emitted: list[int] = []
    slider.time_changed.connect(emitted.append)

    slider.slider.setValue(500)
    assert emitted == [500]

    # set_current_time_ms propagates unconditionally (see its own docstring)
    # in addition to the slider's own valueChanged - so a value that actually
    # changes fires time_changed via both paths, same as it already does for
    # each connected widget's set_time_ms()
    slider.set_current_time_ms(1500)
    assert emitted == [500, 1500, 1500]
    slider.close()


def test_slider_tooltip_shows_seconds_and_movie_frame(qtbot: QtBot, tmp_path) -> None:
    import tifffile

    from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget

    slider = TimeSliderWidget()
    qtbot.addWidget(slider)
    assert slider.slider.describe_value(12345) == "12.345 s (12345 ms)"

    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, np.zeros((30, 8, 8), dtype=np.uint16))
    video = CalciumVideoWidget()
    qtbot.addWidget(video)
    video.set_video_paths([path], 10.0, warn_if_missing_cellpose=False)
    slider.connect_widget(video)

    # Frames are counted from 1, as in Fiji; past the end shows the last one.
    assert slider.slider.describe_value(0) == "0.000 s (0 ms)\nframe 1 / 30"
    assert slider.slider.describe_value(1250) == "1.250 s (1250 ms)\nframe 13 / 30"
    assert slider.slider.describe_value(9000).endswith("frame 30 / 30")
