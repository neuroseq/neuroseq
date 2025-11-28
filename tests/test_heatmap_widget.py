from __future__ import annotations

from time import perf_counter

import brian2.numpy_ as np
import pytest
from pytest import approx
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
    total_playback_time_secs = timepoints_s[-1] / relative_playback_speed
    data = SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    heatmap_widget.set_relative_playback_speed(relative_playback_speed)
    assert heatmap_widget.getProcessedImage().shape == values.T.shape
    assert heatmap_widget.timeLine.isVisible()
    heatmap_widget.togglePause()
    start_time_secs = perf_counter()
    qtbot.waitUntil(lambda: heatmap_widget.currentIndex == timepoints_s.size - 1)
    elapsed_time_secs = perf_counter() - start_time_secs
    # timing should be correct within 25%
    assert elapsed_time_secs == approx(total_playback_time_secs, rel=0.25)
