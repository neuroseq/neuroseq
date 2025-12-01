from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.data import SynapseData
from neuroseq.heatmap_widget import HeatmapWidget


def test_heatmap_widget_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())


@pytest.mark.parametrize("relative_playback_speed", [0.5, 1.0, 2.0])
def test_heatmap_widget_3d_data_with_timeline(
    qtbot: QtBot, relative_playback_speed: float
) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(23, 18, 10)
    interval_s = 0.05
    timepoints_s = interval_s * np.arange(values.shape[2])
    # currently setting the relative playback speed does not affect the playback rate
    # reconfigure this test when playback speed is implemented
    # total_playback_time_secs = timepoints_s[-1] / relative_playback_speed
    data = SynapseData("synapse", values, timepoints_s)
    assert not heatmap_widget.slider.isEnabled()
    heatmap_widget.set_data(data)
    heatmap_widget.set_relative_playback_speed(relative_playback_speed)
    assert (
        heatmap_widget.image_widget.data[0].shape
        == np.transpose(values, (2, 0, 1)).shape
    )
    assert heatmap_widget.slider.maximum() == values.shape[2] - 1
    assert heatmap_widget.slider.isEnabled()
    assert heatmap_widget.nframes() == values.shape[2]
    assert heatmap_widget.timestamps.shape == timepoints_s.shape
    # heatmap_widget.toggle_pause()
    # start_time_secs = perf_counter()
    # qtbot.waitUntil(lambda: heatmap_widget.get_heatmap_index() == timepoints_s.size - 1)
    # elapsed_time_secs = perf_counter() - start_time_secs
    # timing should be correct within 25%
    # assert elapsed_time_secs == approx(total_playback_time_secs, rel=0.25)


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
    assert heatmap_widget.nframes() == values.shape[2]
    heatmap_widget.clear_data()
    assert heatmap_widget.nframes() == 1


@pytest.mark.parametrize("data_shape", [(15, 12, 8), (20, 5, 10), (8, 8, 4)])
def test_heatmap_widget_set_data(qtbot: QtBot, data_shape) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(*data_shape)
    interval_s = 0.1
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    assert not heatmap_widget.slider.isEnabled()
    heatmap_widget.set_data(data)
    assert (
        heatmap_widget.image_widget.data[0].shape
        == np.transpose(values, (2, 0, 1)).shape
    )
    assert heatmap_widget.timestamps.shape == timepoints_s.shape
    assert heatmap_widget.slider.maximum() == data_shape[2] - 1
    assert heatmap_widget.slider.isEnabled()


def test_heatmap_widget_set_data_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.set_data(None)
    assert heatmap_widget.nframes() == 1
    heatmap_widget.set_data(SynapseData("synapse", np.empty((0, 0, 0)), np.empty((0,))))
    assert heatmap_widget.nframes() == 1


def test_heatmap_widget_configure_subplot(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.configure_subplot()
    assert not heatmap_widget.subplot.toolbar
    assert not heatmap_widget.subplot.axes.y.ticks
    assert not heatmap_widget.subplot.axes.y.points.visible
    assert not heatmap_widget.subplot.axes.y.line.visible
    assert heatmap_widget.subplot.axes.y.tick_format(0) == ""
    assert not heatmap_widget.subplot.axes.x.ticks
    assert not heatmap_widget.subplot.axes.x.points.visible
    assert not heatmap_widget.subplot.axes.x.line.visible
    assert heatmap_widget.subplot.axes.x.tick_format(0) == ""
    # check if assertions hold after setting data without calling configure_subplot again
    values = np.random.rand(10, 10, 5)
    interval_s = 0.1
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    assert not heatmap_widget.subplot.toolbar
    assert not heatmap_widget.subplot.axes.y.ticks
    assert not heatmap_widget.subplot.axes.y.points.visible
    assert not heatmap_widget.subplot.axes.y.line.visible
    assert heatmap_widget.subplot.axes.y.tick_format(0) == ""
    assert not heatmap_widget.subplot.axes.x.ticks
    assert not heatmap_widget.subplot.axes.x.points.visible
    assert not heatmap_widget.subplot.axes.x.line.visible
    assert heatmap_widget.subplot.axes.x.tick_format(0) == ""


def test_heatmap_widget_set_current_index(qtbot: QtBot) -> None:
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
        heatmap_widget.set_current_index(idx)
        assert heatmap_widget.get_heatmap_index() == idx
        assert heatmap_widget.slider.value() == idx


def test_heatmap_widget_set_heatmap_index(qtbot: QtBot) -> None:
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
        heatmap_widget.set_heatmap_index(idx)
        assert heatmap_widget.get_heatmap_index() == idx


def test_heatmap_widget_update_heatmap(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    values = np.random.rand(10, 10, 5)
    interval_s = 0.1
    timepoints_s = interval_s * np.arange(values.shape[2])
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    heatmap_widget.set_current_index(3)
    # calling update_heatmap should not change the index if not playing
    heatmap_widget.update_heatmap()
    assert heatmap_widget.get_heatmap_index() == 3
    heatmap_widget.set_current_index(0)
    heatmap_widget.toggle_pause()  # start playing
    for expected_idx in range(1, values.shape[2]):
        heatmap_widget.update_heatmap()
        assert heatmap_widget.get_heatmap_index() == expected_idx
    # updating once more should reset the index to the first frame
    heatmap_widget.update_heatmap()
    assert heatmap_widget.get_heatmap_index() == 0
