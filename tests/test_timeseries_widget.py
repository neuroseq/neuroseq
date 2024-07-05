from __future__ import annotations

import brian2.numpy_ as np
import pytest
from formozov_project.timeseries_widget import TimeseriesWidget
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets


class InvalidTimeseriesWidget(TimeseriesWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)


def test_timeseries_widget_plot_data_not_implemented(qtbot: QtBot) -> None:
    # child classes of TimeseriesWidget must implement _plot_neuron_data
    invalid_timeseries_widget = InvalidTimeseriesWidget()
    qtbot.addWidget(invalid_timeseries_widget)
    with pytest.raises(NotImplementedError):
        invalid_timeseries_widget._plot_data(np.zeros((5, 5)))
