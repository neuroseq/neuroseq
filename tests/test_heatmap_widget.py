from __future__ import annotations

from time import perf_counter

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.data import SynapseData
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.time_slider_widget import TimeSliderWidget


def test_heatmap_widget_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.close()


@pytest.mark.parametrize("relative_playback_speed", [0.5, 1.0, 2.0])
def test_heatmap_widget_3d_data_with_timeline(
    qtbot: QtBot, relative_playback_speed: float
) -> None:
    heatmap_widget = HeatmapWidget()
    time_slider = TimeSliderWidget()
    qtbot.addWidget(heatmap_widget)
    qtbot.addWidget(time_slider)
    heatmap_widget.show()
    time_slider.show()
    time_slider.connect_widget(heatmap_widget)
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(23, 18, 10)
    interval_s = 0.05
    timepoints_s = interval_s * np.arange(values.shape[2])
    total_playback_time_secs = timepoints_s[-1] / relative_playback_speed
    time_slider.set_max_time_secs(timepoints_s[-1])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    time_slider.set_relative_playback_speed(relative_playback_speed)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape == values.T.shape
    assert heatmap_widget.timestamps is not None
    assert heatmap_widget.timestamps.shape == timepoints_s.shape
    time_slider.toggle_pause()
    start_time_secs = perf_counter()
    qtbot.waitUntil(
        lambda: heatmap_widget.get_timestamp_index() == timepoints_s.size - 1
    )
    elapsed_time_secs = perf_counter() - start_time_secs
    # timing should be correct within 25%
    assert elapsed_time_secs == pytest.approx(total_playback_time_secs, rel=0.25)
    heatmap_widget.close()
    qtbot.waitUntil(lambda: not heatmap_widget.isVisible())


def test_heatmap_widget_clear_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(10, 10, 5)
    interval_s = 0.05
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape[0] == values.shape[2]
    heatmap_widget.clear_data()
    assert heatmap_widget.data is None
    heatmap_widget.close()


@pytest.mark.parametrize("data_shape", [(15, 12, 8), (20, 5, 10), (8, 8, 4)])
def test_heatmap_widget_set_data(qtbot: QtBot, data_shape: tuple[int]) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(*data_shape)
    interval_s = 0.1
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape == values.T.shape
    assert heatmap_widget.timestamps is not None
    assert heatmap_widget.timestamps.shape == timepoints_s.shape
    heatmap_widget.close()


def test_heatmap_widget_set_data_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.set_data(None)
    assert heatmap_widget.data is None
    heatmap_widget.close()


def test_heatmap_widget_set_time_ms(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(10, 10, 5)
    interval_s = 0.1
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    for idx in range(values.shape[2]):
        time_ms = 1000 * timepoints_s[idx]
        heatmap_widget.set_time_ms(time_ms)
        assert heatmap_widget.get_timestamp_index() == idx
    heatmap_widget.close()
