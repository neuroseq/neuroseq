from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.timeseries_widget import TimeseriesWidget


class InvalidTimeseriesWidget(TimeseriesWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)


def test_timeseries_widget_plot_data_not_implemented(qtbot: QtBot) -> None:
    # child classes of TimeseriesWidget must implement _plot_neuron_data
    with pytest.raises((TypeError, NotImplementedError), match="_plot_neuron_data"):
        # this line should raise a TypeError due to _plot_neuron_data not being implemented:
        invalid_timeseries_widget = InvalidTimeseriesWidget()
        qtbot.addWidget(invalid_timeseries_widget)
        # pyside6 doesn't support ABC correctly, so we only get a fallback NotImplementedError here:
        invalid_timeseries_widget._plot_data(np.zeros((5, 5)))
