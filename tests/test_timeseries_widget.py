from __future__ import annotations

import numpy as np
import pytest
from formozov_project.timeseries_widget import TimeseriesWidget
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets


class InvalidWidget(TimeseriesWidget):
    def __init__(
        self,
        data: list[np.ndarray] | None = None,
        timestamps: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(data, timestamps, parent)


def test_timeseries_widget_plot_data_not_implemented(qtbot: QtBot) -> None:
    # child classes of TimeseriesWidget must implement _plot_data
    invalid_widget = InvalidWidget()
    qtbot.addWidget(invalid_widget)
    with pytest.raises(NotImplementedError):
        invalid_widget.set_data([np.random.rand(5)], np.random.rand(5))
