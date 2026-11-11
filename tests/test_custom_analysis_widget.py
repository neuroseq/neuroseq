from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisType
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow
from tests.wait_timeout import WAIT_TIMEOUT_MS


def make_custom_analysis_widget(
    qtbot: QtBot, new_window: bool
) -> tuple[CustomAnalysisWidget, CustomAnalysisWindow | None]:
    if new_window:
        win = CustomAnalysisWindow()
        qtbot.addWidget(win)
        qtbot.addWidget(win.analysis_widget)
        with qtbot.waitExposed(win, timeout=WAIT_TIMEOUT_MS):
            win.show()
        return win.analysis_widget, win
    else:
        custom_analysis_widget = CustomAnalysisWidget()
        qtbot.addWidget(custom_analysis_widget)
        with qtbot.waitExposed(custom_analysis_widget, timeout=WAIT_TIMEOUT_MS):
            custom_analysis_widget.show()
        return custom_analysis_widget, None


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_no_data(qtbot: QtBot, new_window: bool) -> None:
    make_custom_analysis_widget(qtbot, new_window)


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_histogram(
    qtbot: QtBot, new_window: bool, make_trace_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_trace_data(n_neurons=10, dt=0.1)
    assert custom_analysis_widget.subplot.get_title() == ""
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 6),
        None,
    )
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 7),
        None,
    )
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(0, 105),
    )
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_histogram(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_spike_data(n_neurons=24)
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        None,
        None,
    )
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(7, 8),
        None,
    )
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(8, 18),
        None,
    )
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s: spike widget has fixed 1ms resolution)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram,
        slice(5, 10),
        slice(150, 1050),
    )
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_histogram2d(
    qtbot: QtBot, new_window: bool, make_trace_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_trace_data(n_neurons=10, n_timestamps=1000, dt=0.1)
    # no data, no plot
    custom_analysis_widget.set_data(None, CustomAnalysisType.Histogram2d, None, None)
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 6),
        None,
    )
    assert "neuron 5" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 7),
        None,
    )
    assert "neurons 5-6" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0-10.5s: spike widget has fixed 1ms resolution)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(0, 105),
    )
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 0" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_histogram2d(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    assert custom_analysis_widget.subplot.get_title() == ""
    data = make_spike_data(n_neurons=24)
    # no data, no plot
    custom_analysis_widget.set_data(
        None,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    assert custom_analysis_widget.subplot.get_title() == ""
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        None,
        None,
    )
    assert "all neurons" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(7, 8),
        None,
    )
    assert "neuron 7" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(8, 18),
        None,
    )
    assert "neurons 8-17" in custom_analysis_widget.subplot.get_title().lower()
    assert "all time" in custom_analysis_widget.subplot.get_title().lower()
    # set neuronslice and timeslice (0.15-1.05s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.Histogram2d,
        slice(5, 10),
        slice(150, 1050),
    )
    assert "neurons 5-9" in custom_analysis_widget.subplot.get_title().lower()
    assert "time 150" in custom_analysis_widget.subplot.get_title().lower()
    assert "1050" in custom_analysis_widget.subplot.get_title().lower()


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_spike_count_per_neuron(
    qtbot: QtBot, new_window: bool, make_spike_data
) -> None:
    """Test the SpikeCountPerNeuron analysis with all data and no slices."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # with no slices provided, all data is used
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        None,
    )
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # set single neuron
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(7, 8),
        None,
    )
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # set slice of neurons
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(8, 18),
        None,
    )
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # set time slice (0.15-1.05s)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        slice(150, 1050),
    )
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # set neuronslice and timeslice
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        slice(5, 10),
        slice(150, 1050),
    )
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
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = make_spike_data(n_neurons=24)
    # set a time slice far beyond existing data (should result in no spikes)
    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.SpikeCountPerNeuron,
        None,
        slice(50000, 100000),
    )
    # Should show "No spikes in selected region" message
    text_children = [
        child
        for child in custom_analysis_widget.subplot.get_children()
        if hasattr(child, "get_text") and "no spikes" in child.get_text().lower()
    ]
    assert len(text_children) > 0


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_pearson_correlation(
    qtbot: QtBot, new_window: bool
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    traces = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    data = TraceData(
        name="corr_traces",
        traces=traces,
        indices=np.asarray([0, 1, 2], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )

    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.PearsonCorrelation,
        None,
        None,
    )

    assert "pearson correlation" in custom_analysis_widget.subplot.get_title().lower()
    assert custom_analysis_widget.subplot.get_xlabel() == "Neuron index"
    assert custom_analysis_widget.subplot.get_ylabel() == "Neuron index"
    assert len(custom_analysis_widget.subplot.images) == 1
    corr = custom_analysis_widget.subplot.images[0].get_array()
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.isclose(corr[0, 1], 1.0)
    assert np.isclose(corr[0, 2], -1.0)


def test_custom_analysis_widget_trace_pearson_correlation_title_wraps_after_label(
    qtbot: QtBot,
) -> None:
    """Regression (#546): a long neuron/time description used to run the
    title past the window edge on one line - it should now wrap onto a
    second line right after "Pearson correlation:"."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, False)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = TraceData(
        name="corr_traces",
        traces=np.random.random_sample((20, 50)).astype(np.float32),
        indices=np.arange(20, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )

    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.PearsonCorrelation,
        slice(5, 17),
        slice(0, 50),
    )

    title = custom_analysis_widget.subplot.get_title()
    assert "pearson correlation" in title.lower()
    first_line, _, rest = title.partition("\n")
    assert first_line.endswith(":")
    assert "neurons 5-16" in rest.lower()


@pytest.mark.parametrize(
    "neuron_slice,expected_ids",
    [
        (None, [0, 1, 2, 3, 4]),
        (slice(2, 5), [2, 3, 4]),
        ([1, 4], [1, 4]),
    ],
)
def test_custom_analysis_widget_trace_pearson_correlation_ticks_use_real_neuron_ids(
    qtbot: QtBot, neuron_slice, expected_ids
) -> None:
    """Regression (#546): the correlation matrix's ticks used to always read
    0, 1, 2, ... regardless of which neurons were actually selected."""
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, False)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = TraceData(
        name="corr_traces",
        traces=np.random.random_sample((5, 10)).astype(np.float32),
        indices=np.arange(5, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )

    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.PearsonCorrelation,
        neuron_slice,
        None,
    )

    assert custom_analysis_widget.neuron_ids == expected_ids
    formatter = custom_analysis_widget.subplot.xaxis.get_major_formatter()
    labels = [formatter(i, None) for i in range(len(expected_ids))]
    assert labels == [str(n) for n in expected_ids]


def test_custom_analysis_widget_trace_pearson_correlation_ticks_are_integers(
    qtbot: QtBot,
) -> None:
    """Regression (#560): with enough neurons, matplotlib's default tick
    locator can pick fractional positions (e.g. 42.5) that don't correspond
    to any real neuron id - the axes should only ever tick at whole numbers.
    """
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, False)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    n_neurons = 18  # with this axes size, matplotlib's default locator
    # picks a fractional step (multiples of 2.5) without MaxNLocator(integer=True).
    data = TraceData(
        name="corr_traces",
        traces=np.random.random_sample((n_neurons, 20)).astype(np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )

    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.PearsonCorrelation,
        None,
        None,
    )

    for axis in (
        custom_analysis_widget.subplot.xaxis,
        custom_analysis_widget.subplot.yaxis,
    ):
        ticks = axis.get_major_locator().tick_values(0, n_neurons - 1)
        assert all(float(t).is_integer() for t in ticks)


@pytest.mark.parametrize("new_window", [False, True])
def test_custom_analysis_widget_trace_pearson_correlation_short_selection(
    qtbot: QtBot, new_window: bool
) -> None:
    custom_analysis_widget, _ = make_custom_analysis_widget(qtbot, new_window)
    qtbot.waitUntil(lambda: custom_analysis_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    data = TraceData(
        name="short_corr_traces",
        traces=np.asarray([[1.0], [2.0]], dtype=np.float32),
        indices=np.asarray([0, 1], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )

    custom_analysis_widget.set_data(
        data,
        CustomAnalysisType.PearsonCorrelation,
        None,
        None,
    )

    text_children = [
        child
        for child in custom_analysis_widget.subplot.get_children()
        if hasattr(child, "get_text")
        and "need at least one neuron and two time samples" in child.get_text().lower()
    ]
    assert len(text_children) > 0


@pytest.mark.parametrize(
    ("neurons", "expected_ids", "expected_values"),
    [
        (slice(0, 1), [0], [[2.0, 2.0]]),
        (slice(5, 6), [5], [[3.0, 3.0]]),
        (None, [0, 1, 5], [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]]),
        ([5, 1], [5, 1], [[3.0, 3.0], [1.0, 1.0]]),
    ],
)
def test_custom_analysis_widget_selects_trace_rows_by_neuron_id(
    qtbot: QtBot, neurons, expected_ids, expected_values
) -> None:
    """TraceData rows follow ``indices``, not the neuron ids themselves."""
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    data = TraceData(
        "t",
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        np.array([1, 0, 5]),
        "a.u.",
        0.001,
    )

    widget.set_data(data, CustomAnalysisType.Histogram, neurons, None)

    assert widget.neuron_ids == expected_ids
    np.testing.assert_allclose(widget.traces, expected_values)


def _plugin(plugin_id: str):
    from neuroseq.analysis_plugins import discover_analysis_plugins

    return next(p for p in discover_analysis_plugins() if p.id == plugin_id)


def _layout_widgets(widget: CustomAnalysisWidget) -> list:
    layout = widget.layout()
    return [layout.itemAt(i).widget() for i in range(layout.count())]


def test_custom_analysis_widget_drops_colorbar_and_twin_axes_on_reuse(
    qtbot: QtBot,
) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    rng = np.random.default_rng(0)
    data = TraceData(
        "t", rng.random((4, 50)).astype(np.float32), np.arange(4), "mV", 0.001
    )
    for _ in range(3):
        widget.set_data(data, CustomAnalysisType.PearsonCorrelation, None, None)
        # The plot axes plus its single colorbar.
        assert len(widget.getFigure().axes) == 2
    widget.set_data(data, CustomAnalysisType.Histogram, None, None)
    assert widget.getFigure().axes == [widget.subplot]

    widget.subplot.twinx()  # e.g. left behind by a plugin's message path
    widget.clear_data()
    assert widget.getFigure().axes == [widget.subplot]


def test_custom_analysis_widget_removes_plugin_controls_on_reuse(
    qtbot: QtBot,
) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    traces = TraceData(
        "t",
        (np.random.default_rng(0).random((3, 500)) * 0.01 - 0.07).astype(np.float32),
        np.arange(3),
        "V",
        0.0001,
    )
    spikes = SpikeData(
        name="s",
        spike_indices=np.array([0, 1, 2]),
        spike_times=np.array([0.01, 0.02, 0.03]),
    )
    widget.set_data(traces, _plugin("epsp_amplitudes"), None, None)
    epsp_bar = widget._epsp_amplitude_controls
    assert epsp_bar in _layout_widgets(widget)

    widget.set_data(spikes, _plugin("advanced_mean_activity"), None, None)
    bars = [
        w
        for w in _layout_widgets(widget)
        if w.objectName() == CustomAnalysisWidget.PLUGIN_CONTROLS_OBJECT_NAME
    ]
    assert len(bars) == 1
    assert epsp_bar not in _layout_widgets(widget)

    widget.set_data(spikes, CustomAnalysisType.Histogram, None, None)
    assert not any(
        w.objectName() == CustomAnalysisWidget.PLUGIN_CONTROLS_OBJECT_NAME
        for w in _layout_widgets(widget)
    )

    widget.set_data(traces, _plugin("replay_mse"), None, None)
    widget.clear_data()
    assert not any(
        w.objectName() == CustomAnalysisWidget.PLUGIN_CONTROLS_OBJECT_NAME
        for w in _layout_widgets(widget)
    )
