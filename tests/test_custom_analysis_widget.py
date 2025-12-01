from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.data import SpikeData
from neuroseq.data import TraceData

gui_delay_in_ms = 50


def make_custom_analysis_widget(
    qtbot: QtBot, new_window: bool
) -> tuple[CustomAnalysisWidget, CustomAnalysisWindow | None]:
    if new_window:
        win = CustomAnalysisWindow()
        qtbot.addWidget(win)
        qtbot.addWidget(win.analysis_widget)
        win.show()
        qtbot.wait(gui_delay_in_ms)
        return win.analysis_widget, win
    else:
        custom_analysis_widget = CustomAnalysisWidget()
        qtbot.addWidget(custom_analysis_widget)
        custom_analysis_widget.show()
        qtbot.wait(gui_delay_in_ms)
        return custom_analysis_widget, None


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_no_data(qtbot: QtBot, new_window: bool) -> None:
    make_custom_analysis_widget(qtbot, new_window)


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_histogram(qtbot: QtBot, new_window: bool) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = TraceData("trace", np.random.rand(10, 1000), "mV", 0.1)
    assert custom_analysis_widget.subplot.get_title() == ""
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 6),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 7),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(0, 105),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_histogram(qtbot: QtBot, new_window: bool) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = SpikeData(
        "spikes",
        np.array([7, 11, 11, 21]),
        np.array([0.0, 0.3, 0.8, 1.0]),
    )
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(7, 8),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(8, 18),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s: spike widget has fixed 1ms resolution)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(150, 1050),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_histogram2d(
    qtbot: QtBot, new_window: bool
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = TraceData("trace", np.random.rand(10, 1000), "mV", 0.1)
    # no data, no plot
    custom_analysis_widget.set_data(None, CustomAnalysisType.Histogram2d, None, None)
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 6),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 7),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s: spike widget has fixed 1ms resolution)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(0, 105),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_histogram2d(
    qtbot: QtBot, new_window: bool
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = SpikeData(
        "spikes", np.array([7, 11, 11, 29]), np.array([0.0, 0.3, 0.8, 1.0])
    )
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(7, 8),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(8, 18),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(150, 1050),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()
