from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow

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
def test_custom_analysis_widget_trace_histogram(
    qtbot: QtBot, new_window: bool, make_trace_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_trace_data(n_neurons=10, dt=0.1)
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
def test_custom_analysis_widget_spike_histogram(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_spike_data(n_neurons=24)
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
    qtbot: QtBot, new_window: bool, make_trace_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_trace_data(n_neurons=10, n_timestamps=1000, dt=0.1)
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
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_spike_data(n_neurons=24)
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


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test the SpikeCountPerNeuron analysis with all data and no slices."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert (
        "spike count per neuron" in custom_analysis_widget.subplot.get_title().lower()
    )
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # Verify histogram was created with correct axes
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron Index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Spike Count"


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron_single_neuron(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test SpikeCountPerNeuron with single neuron slice."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(7, 8),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron Index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Spike Count"


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron_neuron_range(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test SpikeCountPerNeuron with neuron range slice."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(8, 18),
        None,
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron Index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Spike Count"


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron_time_slice(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test SpikeCountPerNeuron with time slice filtering."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # set time slice (0.15-1.05s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        slice(150, 1050),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron Index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Spike Count"


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron_both_slices(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test SpikeCountPerNeuron with both neuron and time slices."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # set neuronslice and timeslice
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(5, 10),
        slice(150, 1050),
    )
    qtbot.wait(gui_delay_in_ms)
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron Index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Spike Count"


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_no_spikes_in_selection(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test SpikeCountPerNeuron when no spikes exist in the selected region."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible())
    data = make_spike_data(n_neurons=24)
    # set a time slice far beyond existing data (should result in no spikes)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        slice(50000, 100000),
    )
    qtbot.wait(gui_delay_in_ms)
    # Should show "No spikes in selected region" message
    text_children = [
        child
        for child in custom_analysis_widget.subplot.get_children()
        if hasattr(child, "get_text") and "no spikes" in child.get_text().lower()
    ]
    assert len(text_children) > 0
