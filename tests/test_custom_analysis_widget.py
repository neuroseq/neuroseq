from __future__ import annotations

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot

from neuroseq.custom_analysis_widget import CustomAnalysisDataType
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget

gui_delay_in_ms = 50


def test_custom_analysis_widget_no_data(qtbot: QtBot) -> None:
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.wait(gui_delay_in_ms)


def test_custom_analysis_widget_trace_histogram(qtbot: QtBot) -> None:
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.wait(gui_delay_in_ms)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = {"dt": 0.1, "data": np.random.rand(10, 1000)}
    # no data, no plot
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram, None, None, {}
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram, None, None, data
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram,
        slice(5, 6),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram,
        slice(5, 7),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s)
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(0, 105),
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0." in custom_analysis_widget.subplot.get_title().lower()
    assert "10." in custom_analysis_widget.subplot.get_title().lower()


def test_custom_analysis_widget_spike_histogram(qtbot: QtBot) -> None:
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.wait(gui_delay_in_ms)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = {
        "dt": 0.01,
        "spike_times": np.array([0.0, 0.3, 0.8, 1.0]),
        "spike_indices": np.array([7, 11, 11, 21]),
    }
    # no data, no plot
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram, None, None, {}
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike, CustomAnalysisType.Histogram, None, None, data
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram,
        slice(7, 8),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram,
        slice(8, 18),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s)
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(15, 105),
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0.1" in custom_analysis_widget.subplot.get_title().lower()
    assert "1.0" in custom_analysis_widget.subplot.get_title().lower()


def test_custom_analysis_widget_trace_histogram2d(qtbot: QtBot) -> None:
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.wait(gui_delay_in_ms)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = {"dt": 0.1, "data": np.random.rand(10, 1000)}
    # no data, no plot
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram2d, None, None, {}
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram2d, None, None, data
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram2d,
        slice(5, 6),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram2d,
        slice(5, 7),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s)
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(0, 105),
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0." in custom_analysis_widget.subplot.get_title().lower()
    assert "10." in custom_analysis_widget.subplot.get_title().lower()


def test_custom_analysis_widget_spike_histogram2d(qtbot: QtBot) -> None:
    custom_analysis_widget = CustomAnalysisWidget()
    qtbot.addWidget(custom_analysis_widget)
    custom_analysis_widget.show()
    qtbot.wait(gui_delay_in_ms)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = {
        "dt": 0.01,
        "spike_times": np.array([0.0, 0.3, 0.8, 1.0]),
        "spike_indices": np.array([7, 11, 11, 21]),
    }
    # no data, no plot
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Trace, CustomAnalysisType.Histogram2d, None, None, {}
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike, CustomAnalysisType.Histogram2d, None, None, data
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram2d,
        slice(7, 8),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram2d,
        slice(8, 18),
        None,
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s)
    custom_analysis_widget.set_data(
        CustomAnalysisDataType.Spike,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(15, 105),
        data,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0.1" in custom_analysis_widget.subplot.get_title().lower()
    assert "1.0" in custom_analysis_widget.subplot.get_title().lower()
