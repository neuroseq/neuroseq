from __future__ import annotations

import numpy as np
from formozov_project.heatmap_widget import HeatmapWidget
from pytestqt.qtbot import QtBot


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
    heatmap_widget.setImage(random_2d_array)
    assert heatmap_widget.getProcessedImage().shape == random_2d_array.shape
    assert not heatmap_widget.timeLine.isVisible()


def test_heatmap_widget_3d_data_with_timeline(qtbot: QtBot) -> None:
    data = np.random.rand(47, 51)
    heatmap_widget = HeatmapWidget(data)
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    assert not heatmap_widget.timeLine.isVisible()
    assert heatmap_widget.getProcessedImage().shape == data.shape
    data = np.random.rand(100, 100, 100)
    heatmap_widget.setImage(data)
    assert heatmap_widget.getProcessedImage().shape == data.shape
    assert heatmap_widget.timeLine.isVisible()
