import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.data import TraceData
from neuroseq.stdp_synaptic_tuning_dialog import StimHeatmapWindow
from neuroseq.stdp_synaptic_tuning_dialog import _SelfRaisingDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.utils import get_cmap


def _make_stim(
    n_time: int,
    n_neurons: int,
    dt: float = 0.001,
    stim_type: StimType = StimType.Excitation,
) -> StimArray:
    return StimArray(
        "stim",
        stim_type,
        np.zeros((n_time, n_neurons), dtype=np.float32),
        "mV",
        dt,
    )


def _make_trace(
    n_neurons: int, n_time: int, indices: np.ndarray | None = None, dt: float = 0.001
) -> TraceData:
    return TraceData(
        "Simulation :: P / v",
        np.zeros((n_neurons, n_time), dtype=np.float32),
        indices if indices is not None else np.arange(n_neurons, dtype=np.uint32),
        "mV",
        dt,
    )


def test_stim_heatmap_window_shows_whole_history_without_panning(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    stim = _make_stim(10, 3)
    window.update_view(stim, [0, 1, 2])
    image = window._image
    assert image is not None
    # always starts at t=0 and ends at the array's current length - never a
    # fixed-width window that pans as more samples are appended.
    assert image.get_extent()[0] == 0.0
    assert image.get_extent()[1] == pytest.approx(10 * stim.dt * 1000.0)
    assert window.trace_axes.get_xlabel() == "Time (ms)"
    assert window.axes.get_ylabel() == "Neuron ID"

    # growing the array in place (as a routine appending blocks would) keeps
    # the same image artist, updated rather than rebuilt, and shows every
    # sample - old ones are never dropped/panned away.
    stim.values = np.vstack([stim.values, np.zeros((5, 3), dtype=np.float32)])
    window.update_view(stim, [0, 1, 2])
    assert window._image is image
    assert image.get_extent()[1] == pytest.approx(15 * stim.dt * 1000.0)
    array = image.get_array()
    assert array is not None
    assert array.shape[1] == 15


def test_stim_heatmap_window_plot_is_scrollable(qtbot) -> None:
    """So a short screen scrolls the plot instead of squeezing the figure
    until axis labels overlap or get clipped off."""
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    scroll_areas = window.findChildren(QtWidgets.QScrollArea)
    assert len(scroll_areas) == 1
    assert scroll_areas[0].widget() is window.plot
    assert scroll_areas[0].widgetResizable()


def test_stim_heatmap_window_ignores_absent_or_empty_stim(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    window.update_view(None, [0])
    assert window._image is None

    window.update_view(_make_stim(0, 3), [0, 1, 2])
    assert window._image is None

    window.update_view(_make_stim(5, 3), [])
    assert window._image is None


def test_stim_heatmap_window_shows_only_targeted_neurons(qtbot) -> None:
    """Only the routine's own target neurons should be shown - and their
    color scale must come from just those columns. A much bigger pulse on
    some unrelated neuron elsewhere in the array must not wash out the
    targeted neurons' own (smaller) pulses (the reported "too faint,
    becomes invisible" bug)."""
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    stim = _make_stim(4, 5)
    stim.values[0, 1] = 1.0  # targeted neuron 1's own pulse
    stim.values[1, 4] = 1000.0  # unrelated neuron 4, much larger amplitude

    window.update_view(stim, [1, 3])

    image = window._image
    assert image is not None
    array = image.get_array()
    assert array is not None
    # only neurons 1 and 3 are shown, in ascending order
    assert array.shape[0] == 2
    assert window.axes.get_yticklabels()[0].get_text() == "1"
    assert window.axes.get_yticklabels()[1].get_text() == "3"
    # the color scale is normalized to the targeted neurons' own peak (1.0),
    # not drowned out by the unrelated neuron's much larger pulse - so
    # neuron 1's pulse renders at full saturation, not a faint fraction of it
    expected_color = get_cmap(StimType.Excitation).to_mpl()(1.0)
    assert tuple(array[0, 0]) == pytest.approx(expected_color, abs=1e-6)


def test_stim_heatmap_window_recolors_when_stim_type_changes(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    excitatory = _make_stim(5, 2, stim_type=StimType.Excitation)
    excitatory.values[0, 0] = 1.0
    window.update_view(excitatory, [0, 1])
    image = window._image
    assert image is not None
    array = image.get_array()
    assert array is not None
    excitatory_pixel = tuple(array[0, 0])

    inhibitory = _make_stim(5, 2, stim_type=StimType.Inhibition)
    inhibitory.values[0, 0] = 1.0
    window.update_view(inhibitory, [0, 1])
    image = window._image
    assert image is not None
    array = image.get_array()
    assert array is not None
    inhibitory_pixel = tuple(array[0, 0])

    assert excitatory_pixel != inhibitory_pixel


def test_stim_heatmap_window_keeps_small_pulses_fully_opaque(qtbot) -> None:
    """A pulse much smaller than the neuron's own peak (e.g. an early,
    low-amplitude rheobase probe next to a much larger later one) must
    still render fully opaque - only an exact-zero sample is transparent.
    get_cmap()'s own alpha ramps with magnitude, which faded smaller (but
    real) pulses toward invisible here as the peak grows (issue: "as the
    amplitude decreases they become invisible almost")."""
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    stim = _make_stim(3, 1)
    stim.values[0, 0] = 0.01  # a tiny pulse...
    stim.values[1, 0] = 100.0  # ...next to a much bigger one
    # index 2 stays exactly zero (no stimulus)

    window.update_view(stim, [0])

    image = window._image
    assert image is not None
    array = image.get_array()
    assert array is not None
    assert array[0, 0, 3] == pytest.approx(1.0)
    assert array[0, 1, 3] == pytest.approx(1.0)
    assert array[0, 2, 3] == pytest.approx(0.0)


def test_stim_heatmap_window_plots_only_targeted_neuron_traces(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    trace = _make_trace(4, 20, indices=np.array([0, 1, 2, 3], dtype=np.uint32))
    trace.traces[1, :] = 5.0
    trace.traces[3, :] = 7.0

    window.update_view(_make_stim(20, 4), [1, 3], trace, [1, 3])

    assert len(window.trace_axes.lines) == 2
    plotted_y = {tuple(line.get_ydata()) for line in window.trace_axes.lines}
    assert plotted_y == {(5.0,) * 20, (7.0,) * 20}
    assert window.trace_axes.get_ylabel() == "mV"
    assert window.trace_axes.legend_ is not None


def test_stim_heatmap_window_aligns_trace_window_to_stim_duration(qtbot) -> None:
    """The trace panel must show the same *relative* time window as the
    heatmap - both ending at "now" - rather than the trace's own full
    absolute history, which can be far longer (e.g. a StateMonitor that has
    been recording since long before this routine's stim array was last
    reset to a fresh, much shorter start). Otherwise the routine's own
    pulses get squeezed into a sliver of a shared x-axis dominated by
    unrelated earlier history."""
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    # stim covers only 5 ms (5 samples at dt=1ms); the trace covers 20 ms -
    # only its last 5 ms should end up shown, rebased to start at t=0.
    stim = _make_stim(5, 1)
    stim.values[2, 0] = 1.0
    trace = _make_trace(1, 20, indices=np.array([0], dtype=np.uint32))
    trace.traces[0, 15:] = np.arange(5, dtype=np.float32)

    window.update_view(stim, [0], trace, [0])

    line = window.trace_axes.lines[0]
    assert list(line.get_xdata()) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert list(line.get_ydata()) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    # the shared x-view spans just the stim's own duration, not the trace's
    # much longer absolute history
    assert window.axes.get_xlim() == pytest.approx((0.0, 5.0))


def test_stim_heatmap_window_skips_traces_absent_from_the_recording(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    # only neuron 1 was actually recorded (e.g. presynaptic voltage isn't
    # monitored, only postsynaptic) - the other target id must be skipped,
    # not raise.
    trace = _make_trace(1, 10, indices=np.array([1], dtype=np.uint32))

    window.update_view(_make_stim(10, 4), [0, 1], trace, [0, 1])

    assert len(window.trace_axes.lines) == 1


def test_stim_heatmap_window_clears_traces_when_none_available(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    trace = _make_trace(2, 10)
    window.update_view(_make_stim(10, 2), [0, 1], trace, [0, 1])
    assert len(window.trace_axes.lines) == 2

    window.update_view(_make_stim(10, 2), [0, 1], None)
    assert len(window.trace_axes.lines) == 0


def test_stim_heatmap_window_follows_theme(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    window.apply_theme(theme.DARK)
    assert to_hex(window.axes.get_facecolor()) == "#1e1e1e"
    assert to_hex(window.trace_axes.get_facecolor()) == "#1e1e1e"
    window.apply_theme(theme.LIGHT)
    assert to_hex(window.axes.get_facecolor()) == "#ffffff"
    assert to_hex(window.trace_axes.get_facecolor()) == "#ffffff"


def test_stim_heatmap_window_position_below_clamped_to_screen(qtbot) -> None:
    reference = QtWidgets.QDialog()
    qtbot.addWidget(reference)
    reference.resize(720, 640)
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    screen = reference.screen() or QtWidgets.QApplication.primaryScreen()
    available = screen.availableGeometry()
    # Force the reference window near the bottom of the screen: placing the
    # heatmap window naively beneath it (its y + its own height) would then
    # land past the bottom edge, off-screen.
    reference.move(available.left(), available.bottom() - 100)

    window.position_below(reference)

    pos = window.pos()
    size = window.size()
    assert pos.x() >= available.left()
    assert pos.y() >= available.top()
    assert pos.x() + size.width() <= available.right() + 1
    assert pos.y() + size.height() <= available.bottom() + 1


def test_self_raising_dialog_raises_itself_when_activated(qtbot) -> None:
    """Some window managers raise a whole transient-dialog chain as one unit
    on click rather than reordering siblings within it, so a sibling popup
    activated by the user doesn't otherwise come to the front of the others.
    _SelfRaisingDialog (the base every popup here uses) forces that itself
    via ActivationChange rather than relying on the window manager."""
    dialog = _SelfRaisingDialog()
    qtbot.addWidget(dialog)
    raised = []
    dialog.raise_ = lambda: raised.append(True)  # type: ignore[method-assign]

    event = QtCore.QEvent(QtCore.QEvent.Type.ActivationChange)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(dialog), "isActiveWindow", lambda self: True)
        dialog.changeEvent(event)
    assert raised == [True]

    raised.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(dialog), "isActiveWindow", lambda self: False)
        dialog.changeEvent(QtCore.QEvent(QtCore.QEvent.Type.ActivationChange))
    assert raised == []
