from __future__ import annotations

from time import perf_counter

import brian2.numpy_ as np
from neuroseq.heatmap_widget import HeatmapWidget
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore


def test_heatmap_widget_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())


def test_heatmap_widget_2d_data_no_timeline(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    assert heatmap_widget.getImageItem().image is None
    assert not heatmap_widget.timeLine.isVisible()
    random_2d_array = np.random.rand(100, 100)
    heatmap_widget.set_data(random_2d_array)
    assert heatmap_widget.getProcessedImage().shape == random_2d_array.shape
    assert not heatmap_widget.timeLine.isVisible()


def test_heatmap_widget_3d_data_with_timeline(qtbot: QtBot) -> None:
    data = np.random.rand(47, 51)
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.set_data(data)
    assert not heatmap_widget.timeLine.isVisible()
    assert heatmap_widget.getProcessedImage().shape == data.T.shape
    data = np.random.rand(23, 18, 10)
    interval_secs = 0.05
    timepoints_secs = interval_secs * np.arange(data.shape[2])
    total_time_secs = timepoints_secs[-1]
    heatmap_widget.set_data(data, timepoints_secs)
    assert heatmap_widget.getProcessedImage().shape == data.T.shape
    assert heatmap_widget.timeLine.isVisible()
    # press space-bar to start real-time playback
    qtbot.keyPress(heatmap_widget, QtCore.Qt.Key.Key_Space)
    start_time_secs = perf_counter()
    qtbot.waitUntil(lambda: heatmap_widget.currentIndex == timepoints_secs.size - 1)
    elapsed_time_secs = perf_counter() - start_time_secs
    # timing should be correct within 25%
    assert elapsed_time_secs == approx(total_time_secs, rel=0.25)
