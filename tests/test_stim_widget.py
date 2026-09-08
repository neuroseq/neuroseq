from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.stim_widget import StimArray
from neuroseq.stim_widget import StimType
from neuroseq.stim_widget import StimWidget
from neuroseq.utils import InteractMode


def get_event_target(widget):
    # return a windowHandle() for the widget if it exists, otherwise return the widget itself
    # note that rendercanvas Qt backend may wrap the real interactive widget inside `canvas._subwidget`
    # so we first check for that and use the subwidget instead if present
    widget = getattr(widget, "_subwidget", widget)
    handle = widget.windowHandle()
    return handle if handle is not None else widget


def mouse_click(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    pos: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    delay_ms=10,
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseClick(
        get_event_target(widget),
        button,
        pos=pos,
    )


def wait_for_canvas(qtbot: QtBot, stim_widget, *points: QtCore.QPoint) -> None:
    # ensure screen->world mapping succeeds for the given screen points
    qtbot.waitUntil(
        lambda: all(
            stim_widget.subplot.map_screen_to_world((p.x(), p.y())) is not None
            for p in points
        )
    )


def mouse_click_and_drag(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    start: QtCore.QPoint,
    end: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    delay_ms=10,
    intermediate_points=4,
) -> None:
    qtbot.wait(delay_ms)
    qtbot.mouseMove(get_event_target(widget), pos=start)
    qtbot.wait(delay_ms)
    qtbot.mousePress(
        get_event_target(widget),
        button,
        pos=start,
    )
    n_points = intermediate_points + 2
    for x, y in zip(
        np.linspace(start.x(), end.x(), n_points),
        np.linspace(start.y(), end.y(), n_points),
        strict=True,
    ):
        qtbot.wait(delay_ms)
        qtbot.mouseMove(
            get_event_target(widget),
            pos=QtCore.QPoint(int(x), int(y)),
        )
    qtbot.wait(delay_ms)
    qtbot.mouseRelease(
        widget,
        button,
        pos=end,
    )
    qtbot.wait(delay_ms)


@pytest.fixture
def positive_stims() -> StimArray:
    n_neurons = 20
    n_stim_per_neuron = 5
    n_timestamps = 100
    dt = 0.001  # 1 ms in seconds
    stim = np.zeros((n_neurons, n_timestamps))
    for row in stim:
        for _ in range(n_stim_per_neuron):
            row[np.random.randint(0, n_timestamps)] = 1.4
    return StimArray("", StimType.Excitation, stim.T, "mV", dt=dt)


@pytest.fixture
def negative_and_positive_stims() -> StimArray:
    # stim array with values from -2.0 to 3.0
    n_neurons = 3
    n_timestamps = 3
    dt = 0.001  # 1 ms in seconds
    stim = np.zeros((n_neurons, n_timestamps))
    stim[0, 0] = -1.18
    stim[0, 1] = 0.0
    stim[0, 2] = 1.0
    stim[1, 0] = -2.0
    stim[1, 1] = 0.0
    stim[1, 2] = 1.5
    stim[2, 0] = -0.11
    stim[2, 1] = 0.0
    stim[2, 2] = 3.0
    return StimArray("", StimType.External, stim.T, "mV", dt=dt)


def test_stim_widget_background_stims(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    s1 = StimArray(
        "s1", StimType.Excitation, np.random.random((10, 100)), "mV", dt=0.001
    )
    s2 = StimArray(
        "s2", StimType.Inhibition, np.random.random((10, 100)), "mV", dt=0.001
    )
    stim_widget.set_data(s1)
    stim_widget.set_background_data(s2)
    assert len(stim_widget.background_graphics) == 1
    # toggle background visibility off and on
    stim_widget.set_background_data_visible(False)
    for graphic in stim_widget.background_graphics:
        assert graphic.visible is False
    stim_widget.set_background_data_visible(True)
    for graphic in stim_widget.background_graphics:
        assert graphic.visible is True
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_background_stims_use_their_own_dt(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    selected_stim = StimArray(
        "selected", StimType.Excitation, np.random.random((10, 8)), "mV", dt=0.001
    )
    background_stim = StimArray(
        "background", StimType.Inhibition, np.random.random((10, 8)), "mV", dt=0.0005
    )

    stim_widget.set_data(selected_stim)
    stim_widget.set_background_data(background_stim)

    assert stim_widget.graphic is not None
    assert len(stim_widget.background_graphics) == 1
    background_graphic = stim_widget.background_graphics[0]
    assert stim_widget.graphic.world_object.world.scale[0] == pytest.approx(1.0)
    assert background_graphic.world_object.world.scale[0] == pytest.approx(0.5)
    assert background_graphic.offset[0] == pytest.approx(0.25)
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_hover_uses_background_values_in_show_all_mode(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    foreground = StimArray(
        "foreground",
        StimType.Excitation,
        np.zeros((5, 3), dtype=float),
        "pA",
        dt=0.001,
    )
    background_values = np.zeros((5, 3), dtype=float)
    background_values[2, 1] = 4.5  # inhibitory arrays store positive magnitudes
    background = StimArray(
        "background",
        StimType.Inhibition,
        background_values,
        "pA",
        dt=0.001,
    )

    stim_widget.set_data(foreground)
    stim_widget.set_background_data(background)
    captured: dict[str, str] = {}

    def capture_text(pos: np.ndarray, additional_text: str) -> None:
        captured["additional_text"] = additional_text

    monkeypatch.setattr(stim_widget.text, "set_pos", capture_text)
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([2.0, 1.0, 0.0]),
    )

    stim_widget.pointer_move(None)

    assert len(stim_widget._background_stim_arrays) == 1
    assert captured["additional_text"] == "-4.5 pA"
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_clear_data_resets_background_stim_arrays(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    foreground = StimArray(
        "foreground",
        StimType.Excitation,
        np.zeros((5, 3), dtype=float),
        "pA",
        dt=0.001,
    )
    background = StimArray(
        "background",
        StimType.Inhibition,
        np.ones((5, 3), dtype=float),
        "pA",
        dt=0.001,
    )

    stim_widget.set_data(foreground)
    stim_widget.set_background_data(background)
    stim_widget.clear_data()

    assert stim_widget._background_stim_arrays == []
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_set_neuron_greyed_adds_and_removes_overlay(
    qtbot: QtBot,
    positive_stims: StimArray,
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(positive_stims)

    assert not stim_widget.is_neuron_greyed(2)
    stim_widget.set_neuron_greyed(2, True)
    assert stim_widget.is_neuron_greyed(2)
    stim_widget.set_neuron_greyed(2, True)
    assert stim_widget._greyed_neuron_ids == {2}
    stim_widget.set_neuron_greyed(2, False)
    assert not stim_widget.is_neuron_greyed(2)
    stim_widget.close()


def test_stim_widget_clear_data_clears_greyed_neurons(
    qtbot: QtBot,
    positive_stims: StimArray,
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(positive_stims)
    stim_widget.set_neuron_greyed(1, True)

    stim_widget.set_data(positive_stims)

    assert not stim_widget.is_neuron_greyed(1)
    stim_widget.close()


def test_stim_widget_greying_hides_and_compacts_rows(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((5, 3))  # (n_timestamps, n_neurons)
    values[:, 0] = 0.1
    values[:, 1] = 0.5
    values[:, 2] = 0.9
    stim_array = StimArray("", StimType.Excitation, values, "mV", dt=0.001)
    original_values = stim_array.values.copy()
    stim_widget.set_data(stim_array)
    assert stim_widget.graphic is not None
    assert stim_widget.data_values is not None

    stim_widget.set_neuron_greyed(1, True)

    rendered = np.asarray(stim_widget.graphic.data.value)
    assert np.allclose(rendered[0, :], 0.1)
    assert np.allclose(rendered[1, :], 0.9)  # neuron 2 shifted up to fill the gap
    assert np.allclose(rendered[2, :], 0.0)  # trailing row zeroed (renders transparent)
    assert stim_widget.n_neurons == 2
    assert stim_widget.n_neurons_total == 3
    assert np.allclose(stim_widget.data_values, original_values)  # untouched

    stim_widget.set_neuron_greyed(1, False)
    rendered = np.asarray(stim_widget.graphic.data.value)
    assert np.allclose(rendered[0, :], 0.1)
    assert np.allclose(rendered[1, :], 0.5)
    assert np.allclose(rendered[2, :], 0.9)
    assert stim_widget.n_neurons == 3
    stim_widget.close()


def test_stim_widget_draw_mode_translates_display_row_to_real_neuron_when_greyed(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((5, 3))
    stim_array = StimArray("", StimType.Excitation, values, "mV", dt=0.001)
    stim_widget.set_data(stim_array)
    stim_widget.set_neuron_greyed(1, True)
    stim_widget.set_mode(InteractMode.DRAW)

    # display row 1 now corresponds to real neuron 2, since real neuron 1 is hidden
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 1.0, 0.0]),
    )
    with qtbot.waitSignal(stim_widget.mouse_dragged) as blocker:
        stim_widget.pointer_down(SimpleNamespace(button=1))

    assert blocker.args[0] == 2
    assert blocker.args[1] == 2
    stim_widget.close()


def test_stim_widget_gridlines_follow_stim_dt(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_array = StimArray(
        "stim",
        StimType.Excitation,
        np.random.random((10, 8)),
        "mV",
        dt=0.0005208125,
    )

    stim_widget.set_data(stim_array)

    assert stim_widget.grid_lines is not None
    assert stim_widget.grid_lines.dt_ms == pytest.approx(0.5208125)
    positions = stim_widget.grid_lines.gridline_1sample.geometry.positions.data[0::2, 0]
    assert positions[1] - positions[0] == pytest.approx(0.5208125)
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_short_stimuli_minimum_size_toggle(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((100_000, 2))
    values[50_000, 1] = 1.0
    stim_widget.set_data(
        StimArray("short", StimType.Excitation, values, "mV", dt=0.001)
    )

    graphic = stim_widget.minimum_size_graphic
    assert graphic is not None
    assert graphic.sizes == pytest.approx(4.0)
    assert graphic.size_space == "screen"
    assert stim_widget.graphic is not None
    assert stim_widget.graphic.offset[2] > 199.0
    assert np.all(graphic.data.value[:, 2] > 199.0)
    assert graphic.visible

    stim_widget.set_keep_short_stimuli_visible(False)
    assert not graphic.visible
    assert values[50_000, 1] == 1.0
    stim_widget.set_keep_short_stimuli_visible(True)
    assert graphic.visible


def test_stim_widget_hover_preview_snaps_to_sample_on_pointer_move(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_array = StimArray(
        "stim",
        StimType.Excitation,
        np.random.random((10, 8)),
        "mV",
        dt=0.0005,
    )
    stim_widget.set_data(stim_array)
    stim_widget.mouse_pos = np.array([0.0, 2.2, 0.0])
    stim_widget.set_hover_slice(np.ones((2, 1)))
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([1.3, 2.2, 0.0]),
    )

    # The event argument is ignored because map_screen_to_world is monkeypatched above.
    stim_widget.pointer_move(None)

    assert stim_widget.hover_slice_graphic is not None
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)
    assert stim_widget.hover_slice_graphic.offset[1] == pytest.approx(2.0)
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_select_neurons_translates_display_rows_when_greyed(
    qtbot: QtBot, positive_stims: StimArray
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(positive_stims)
    stim_widget.set_neuron_greyed(1, True)
    assert stim_widget.n_neurons == 19

    # display rows 1-2 now correspond to real neurons 2-3, since real neuron 1
    # is hidden and neuron 2 shifted up to display row 1.
    stim_widget.select_neurons(slice(1, 3), slice(10, 20))

    assert stim_widget.selected_neuron_slice == slice(2, 4)
    stim_widget.close()


def test_stim_widget_select_mode(qtbot: QtBot, positive_stims: StimArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_widget.set_mode(InteractMode.SELECT)
    assert stim_widget.n_neurons == 0
    n_timestamps = positive_stims.values.shape[0]
    n_neurons = positive_stims.values.shape[1]
    controller.set_limits(n_timestamps * positive_stims.dt, n_neurons)
    stim_widget.set_data(positive_stims)
    # toggle selection text on and off again
    stim_widget.toggle_selection_text()
    stim_widget.toggle_selection_text()
    assert stim_widget.n_neurons == n_neurons
    assert stim_widget.n_timestamps == n_timestamps
    # initially no neurons are selected
    assert stim_widget.selected_neuron_slice is None
    assert stim_widget.selected_time_slice is None
    stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.waitUntil(lambda: stim_widget.selected_neuron_slice == slice(1, 3))
    assert stim_widget.selected_time_slice == slice(10, 20)
    stim_widget.unselect_neurons()
    qtbot.waitUntil(lambda: stim_widget.selected_neuron_slice is None)
    assert stim_widget.selected_time_slice is None
    # a missing slice argument should default to the full range
    stim_widget.select_neurons(time_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(0, n_neurons)
    assert stim_widget.selected_time_slice == slice(1, 3)
    stim_widget.select_neurons(neuron_index_slice=slice(1, 3))
    assert stim_widget.selected_neuron_slice == slice(1, 3)
    assert stim_widget.selected_time_slice == slice(0, n_timestamps)
    stim_widget.select_neurons()
    assert stim_widget.selected_neuron_slice == slice(0, n_neurons)
    assert stim_widget.selected_time_slice == slice(0, n_timestamps)
    stim_widget.unselect_neurons()
    ## Select about half the neurons and half the time points
    x0 = stim_widget.size().width() // 4
    y0 = stim_widget.size().height() // 4
    start = QtCore.QPoint(x0, y0)
    end = QtCore.QPoint(x0 * 3, y0 * 3)
    wait_for_canvas(qtbot, stim_widget, start, end)
    mouse_click_and_drag(qtbot, stim_widget, start, end)
    assert stim_widget.selected_neuron_slice is not None
    assert stim_widget.selected_time_slice is not None
    assert 0.15 * n_neurons < stim_widget.selected_neuron_slice.start < 0.35 * n_neurons
    assert 0.65 * n_neurons < stim_widget.selected_neuron_slice.stop < 0.90 * n_neurons
    assert (
        0.15 * n_timestamps
        < stim_widget.selected_time_slice.start
        < 0.35 * n_timestamps
    )
    assert (
        0.65 * n_timestamps < stim_widget.selected_time_slice.stop < 0.90 * n_timestamps
    )
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_draw_mode(qtbot: QtBot, positive_stims: StimArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_widget.set_mode(InteractMode.DRAW)
    assert stim_widget.n_neurons == 0
    n_timestamps = positive_stims.values.shape[0]
    n_neurons = positive_stims.values.shape[1]
    controller.set_limits(n_timestamps * positive_stims.dt, n_neurons)
    stim_widget.set_data(positive_stims)
    assert stim_widget.n_neurons == n_neurons
    assert stim_widget.dt_ms == 1000 * positive_stims.dt
    x0 = stim_widget.size().width() // 4
    y0 = stim_widget.size().height() // 4
    x1 = x0 * 2
    y1 = y0 * 2
    x2 = x0 * 3
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None
    wait_for_canvas(
        qtbot,
        stim_widget,
        QtCore.QPoint(x0, y0),
        QtCore.QPoint(x1, y1),
        QtCore.QPoint(x2, y0),
    )

    # mouse clicks
    stim_widget.setFocus()
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(qtbot, stim_widget.figure.canvas, QtCore.QPoint(x0, y0))
    neuron_index_offset, time_index, mouse_button = signal.args
    assert 0 < time_index / n_timestamps < 0.4
    assert 0 < neuron_index_offset / n_neurons < 1.0
    assert mouse_button == 1
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(
            qtbot,
            stim_widget.figure.canvas,
            QtCore.QPoint(x1, y0),
            QtCore.Qt.MouseButton.RightButton,
        )
    neuron_index_offset, time_index, mouse_button = signal.args
    assert 0.3 < time_index / n_timestamps < 0.7
    assert 0 < neuron_index_offset / n_neurons < 1.0
    assert mouse_button == 2
    with qtbot.waitSignal(stim_widget.mouse_clicked) as signal:
        mouse_click(qtbot, stim_widget.figure.canvas, QtCore.QPoint(x2, y1))
    neuron_index_offset, time_index, mouse_button = signal.args
    assert 0.6 < time_index / n_timestamps < 1.0
    assert 0 < neuron_index_offset / n_neurons < 1.0
    assert mouse_button == 1
    assert stim_widget.selected_time_slice is None
    assert stim_widget.selected_neuron_slice is None

    # mouse drag
    drag_signals = []
    stim_widget.mouse_dragged.connect(
        lambda n0, n1, t0, t1, btn: drag_signals.append((n0, n1, t0, t1, btn))
    )
    mouse_click_and_drag(
        qtbot, stim_widget, QtCore.QPoint(x0, y0), QtCore.QPoint(x2, y0)
    )
    assert len(drag_signals) > 2
    # first signal on start of drag
    n0, n1, t0, t1, btn = drag_signals[0]
    assert btn == 1
    assert n0 == n1
    assert t1 == t0
    assert 0 < t0 / n_timestamps < 0.4
    assert 0 < n0 / n_neurons < 1.0
    # last signal at end of drag
    n0, n1, t0, t1, btn = drag_signals[-1]
    assert btn == 1
    assert n0 == n1
    assert t1 > t0
    assert 0.6 < t1 / n_timestamps < 1.0
    assert 0 < n0 / n_neurons < 1.0
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_ext_stims(
    qtbot: QtBot, negative_and_positive_stims: StimArray
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    # default stim type and colormap settings
    assert stim_widget.stim_type == StimType.Excitation
    assert stim_widget.vmin == pytest.approx(0.0)
    assert stim_widget.vmax == pytest.approx(1.0)
    # set external stim type with negative and positive values
    stim_widget.set_data(negative_and_positive_stims)
    assert stim_widget.stim_type == StimType.External
    # vmin/vmax should be symmetric around zero and include the largest absolute value
    assert stim_widget.vmin == pytest.approx(-3.0)
    assert stim_widget.vmax == pytest.approx(3.0)
    # update data slice with larger (negative) values than current max
    stim_widget.update_data_slice(
        neuron_slice=slice(0, 2),
        time_slice=slice(0, 2),
        values=np.array([[2.0, -4.0], [2.0, -2.0]]),
    )
    assert stim_widget.vmin == pytest.approx(-4.0)
    assert stim_widget.vmax == pytest.approx(4.0)
    # update data slice with larger (positive) values than current max
    stim_widget.update_data_slice(
        neuron_slice=slice(1, 2), time_slice=slice(1, 2), values=12
    )
    assert stim_widget.vmin == pytest.approx(-12.0)
    assert stim_widget.vmax == pytest.approx(12.0)
    # update data slice by overwriting the current largest (positive) values with zero
    stim_widget.update_data_slice(
        neuron_slice=slice(1, 2), time_slice=slice(1, 2), values=0
    )
    assert stim_widget.vmin == pytest.approx(-4.0)
    assert stim_widget.vmax == pytest.approx(4.0)
    # update data slice by overwriting the current largest (negative) values with zero
    stim_widget.update_data_slice(
        neuron_slice=slice(0, 2), time_slice=slice(0, 2), values=np.zeros((2, 2))
    )
    assert stim_widget.vmin == pytest.approx(-3.0)
    assert stim_widget.vmax == pytest.approx(3.0)
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())


def test_stim_widget_minimum_size_overlay_not_rebuilt_while_hidden(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((1000, 2))
    values[500, 1] = 1.0
    stim_widget.set_data(
        StimArray("short", StimType.Excitation, values, "mV", dt=0.001)
    )

    stim_widget.set_keep_short_stimuli_visible(False)
    hidden_graphic = stim_widget.minimum_size_graphic
    # An edit while the overlay is hidden must not rebuild the invisible scatter.
    stim_widget.update_data_slice(slice(0, 2), slice(0, 1), np.zeros((1, 2)))
    assert stim_widget.minimum_size_graphic is hidden_graphic

    # Once visible again, an edit refreshes the overlay (a new graphic object).
    stim_widget.set_keep_short_stimuli_visible(True)
    stim_widget.update_data_slice(slice(0, 2), slice(10, 11), np.ones((1, 2)))
    assert stim_widget.minimum_size_graphic is not hidden_graphic


def test_stim_widget_hover_preview_stays_snapped_within_cell(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible())
    stim_array = StimArray(
        "stim", StimType.Excitation, np.random.random((10, 8)), "mV", dt=0.0005
    )
    stim_widget.set_data(stim_array)
    stim_widget.set_hover_slice(np.ones((2, 1)))
    assert stim_widget.hover_slice_graphic is not None

    # First move: pointer at world x=1.3 -> time cell 2 (centre 1.25).
    monkeypatch.setattr(
        stim_widget.subplot, "map_screen_to_world", lambda ev: np.array([1.3, 2.2, 0.0])
    )
    stim_widget.pointer_move(None)
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)

    # Second move within the SAME cell (x=1.45): the preview must stay snapped to
    # the cell centre rather than drifting to the raw pointer position, and keep
    # its hover z-order (regression: the base class used to overwrite both).
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([1.45, 2.2, 0.0]),
    )
    stim_widget.pointer_move(None)
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)
    assert stim_widget.hover_slice_graphic.offset[2] == pytest.approx(
        stim_widget.HOVER_Z
    )
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible())
