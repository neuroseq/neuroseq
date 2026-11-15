from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui

from neuroseq.utils import InteractMode
from neuroseq.visualization.stim_widget import StimArray
from neuroseq.visualization.stim_widget import StimType
from neuroseq.visualization.stim_widget import StimWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from tests.wait_timeout import WAIT_TIMEOUT_MS


def test_stim_preview_keeps_editor_menu(qtbot, monkeypatch):
    from qtpy import QtWidgets

    menu = QtWidgets.QMenu()
    edit_action = menu.addAction("Edit stim array")
    widget = StimWidget(ConstrainedPanZoomController(), menu=menu)
    qtbot.addWidget(widget)
    shown = []
    monkeypatch.setattr(menu, "exec", lambda pos: shown.append(pos))
    widget.mode = InteractMode.SELECT
    event = QtGui.QContextMenuEvent(
        QtGui.QContextMenuEvent.Reason.Mouse, QtCore.QPoint(5, 5)
    )
    widget.contextMenuEvent(event)
    assert shown
    assert edit_action in menu.actions()
    assert widget.preview_action in menu.actions()


@pytest.mark.parametrize("stim_type", list(StimType))
@pytest.mark.parametrize("selected", [False, True])
def test_stim_vector_preview(qtbot, tmp_path, stim_type, selected):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    from neuroseq.utils import get_cmap

    controller = ConstrainedPanZoomController()
    widget = StimWidget(controller)
    qtbot.addWidget(widget)
    assert widget.preview_stimuli() is None
    assert not widget.preview_action.isEnabled()
    values = np.array([[0, 1], [2, 1], [2, 0], [0, 0]], dtype=np.float32)
    if stim_type == StimType.External:
        values[:, 1] *= -1
    widget.set_data(StimArray("Pulses", stim_type, values, "1", 0.002))
    widget.set_background_data(StimArray("Other", stim_type, values, "1", 0.001))
    widget.set_background_data_visible(True)
    controller.focus_time_range(0, 0.008)
    controller.focus_neuron_range(0, 2)
    state = controller._get_camera_state()
    if selected:
        widget.select_neurons(slice(1, 2), slice(1, 3))
    assert widget.preview_action.isEnabled()
    widget.preview_action.trigger()
    from qtpy import QtWidgets

    dialog = widget.findChild(QtWidgets.QDialog)
    qtbot.addWidget(dialog)
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog.findChild(NavigationToolbar2QT) is not None
    canvas = dialog.findChild(FigureCanvasQTAgg)
    ax = canvas.figure.axes[0]
    assert ax.get_title() == "Pulses"
    assert len(ax.collections) == 2
    blocks = ax.collections[-1]
    assert len(blocks.get_paths()) == 2
    np.testing.assert_allclose(
        blocks.get_paths()[0].vertices[:4],
        [[2, -0.5], [6, -0.5], [6, 0.5], [2, 0.5]],
    )
    normalized = np.array([1, -0.5 if stim_type == StimType.External else 0.5])
    if stim_type == StimType.External:
        normalized = (normalized + 1) / 2
    np.testing.assert_allclose(blocks.get_facecolors(), get_cmap(stim_type)(normalized))
    cx, cy = state["position"][:2]
    assert ax.get_xlim() == pytest.approx(
        (2, 6) if selected else (cx - state["width"] / 2, cx + state["width"] / 2)
    )
    assert ax.get_ylim() == pytest.approx(
        (0.5, 1.5) if selected else (cy - state["height"] / 2, cy + state["height"] / 2)
    )
    output = tmp_path / "stim.svg"
    from matplotlib import rc_context

    with rc_context({"savefig.facecolor": "white", "savefig.edgecolor": "white"}):
        canvas.figure.savefig(output)
    svg = output.read_text(encoding="utf-8")
    assert "<image" not in svg
    assert 'id="patch_1"' in svg  # Axis spines remain visible.
    assert "fill: #ffffff" not in svg
    assert not canvas.figure.patch.get_visible()
    assert not ax.patch.get_visible()
    dialog.close()
    widget.set_background_data_visible(False)
    dialog = widget.preview_stimuli()
    qtbot.addWidget(dialog)
    assert len(dialog.findChild(FigureCanvasQTAgg).figure.axes[0].collections) == 1
    dialog.close()
    widget.clear_data()
    assert not widget.preview_action.isEnabled()


def get_event_target(widget):
    # return a windowHandle() for the widget if it exists, otherwise return the widget itself
    # note that rendercanvas Qt backend may wrap the real interactive widget inside `canvas._subwidget`
    # so we first check for that and use the subwidget instead if present
    widget = getattr(widget, "_subwidget", widget)
    handle = widget.windowHandle()
    return handle if handle is not None else widget


def wait_for_canvas_events(qtbot: QtBot, widget: QtGui.QWidget) -> None:
    # rendercanvas queues events (merging consecutive moves) and only
    # dispatches them once per frame, which can take a while (e.g. on
    # Windows): wait until it has done so
    events = getattr(widget, "_subwidget", widget)._events
    qtbot.waitUntil(lambda: not events._pending_events, timeout=WAIT_TIMEOUT_MS)


def mouse_click(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    pos: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
) -> None:
    wait_for_canvas_events(qtbot, widget)
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
        ),
        timeout=WAIT_TIMEOUT_MS,
    )


def mouse_click_and_drag(
    qtbot: QtBot,
    widget: QtGui.QWidget,
    start: QtCore.QPoint,
    end: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
    intermediate_points=4,
) -> None:
    wait_for_canvas_events(qtbot, widget)
    qtbot.mouseMove(get_event_target(widget), pos=start)
    wait_for_canvas_events(qtbot, widget)
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
        wait_for_canvas_events(qtbot, widget)
        qtbot.mouseMove(
            get_event_target(widget),
            pos=QtCore.QPoint(int(x), int(y)),
        )
    wait_for_canvas_events(qtbot, widget)
    qtbot.mouseRelease(
        get_event_target(widget),
        button,
        pos=end,
    )
    wait_for_canvas_events(qtbot, widget)


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
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_stim_widget_background_stims_use_their_own_dt(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_stim_widget_hover_uses_background_values_in_show_all_mode(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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

    def capture_text(pos: np.ndarray, additional_text: str, *_: str) -> None:
        captured["additional_text"] = additional_text

    monkeypatch.setattr(stim_widget.text, "set_pos", capture_text)
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([2.0, 1.0, 0.0]),
    )

    stim_widget.pointer_move(SimpleNamespace(x=0.0))

    assert len(stim_widget._background_stim_arrays) == 1
    assert captured["additional_text"] == "Neuron 1\n-4.5 pA\nStart: 2 ms\nEnd: 3 ms"
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_stim_widget_clear_data_resets_background_stim_arrays(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


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


def test_stim_widget_set_data_keeps_greyed_neurons(
    qtbot: QtBot,
    positive_stims: StimArray,
) -> None:
    """Hidden neurons follow the cell selection: switching arrays or undo
    (both call set_data) must not bring them back, as the traces keep them."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(positive_stims)
    stim_widget.set_neuron_greyed(1, True)

    stim_widget.set_data(positive_stims)

    assert stim_widget.is_neuron_greyed(1)
    assert stim_widget.n_neurons == positive_stims.values.shape[1] - 1
    assert stim_widget.graphic is not None
    rendered = np.asarray(stim_widget.graphic.data.value)
    assert rendered.shape[0] == positive_stims.values.shape[1] - 1
    np.testing.assert_allclose(rendered[1], positive_stims.values[:, 2])


def test_stim_widget_greying_hides_and_compacts_rows(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    controller.set_limits(max_time_secs=1.0, max_neurons=3)
    values = np.zeros((5, 3))  # (n_timestamps, n_neurons)
    values[:, 0] = 0.1
    values[:, 1] = 0.5
    values[:, 2] = 0.9
    stim_array = StimArray("", StimType.Excitation, values, "mV", dt=0.001)
    original_values = stim_array.values.copy()
    stim_widget.set_data(stim_array)
    assert stim_widget.graphic is not None
    assert stim_widget.data_values is not None
    assert stim_widget.grid_lines is not None

    stim_widget.set_neuron_greyed(1, True)

    rendered = np.asarray(stim_widget.graphic.data.value)
    assert rendered.shape[0] == 2  # the hidden row is gone, not just blanked
    assert np.allclose(rendered[0, :], 0.1)
    assert np.allclose(rendered[1, :], 0.9)  # neuron 2 shifted up to fill the gap
    assert stim_widget.n_neurons == 2
    assert stim_widget.n_neurons_total == 3
    assert np.allclose(stim_widget.data_values, original_values)  # untouched
    # the view is refit to exactly the 2 remaining rows
    assert stim_widget.grid_lines.n_neurons == 2
    assert controller._get_camera_state()["height"] == pytest.approx(2)

    stim_widget.set_neuron_greyed(1, False)
    rendered = np.asarray(stim_widget.graphic.data.value)
    assert rendered.shape[0] == 3
    assert np.allclose(rendered[0, :], 0.1)
    assert np.allclose(rendered[1, :], 0.5)
    assert np.allclose(rendered[2, :], 0.9)
    assert stim_widget.n_neurons == 3
    assert stim_widget.grid_lines.n_neurons == 3
    assert controller._get_camera_state()["height"] == pytest.approx(3)


def test_stim_widget_hover_shows_real_neuron_id_when_only_one_visible(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, positive_stims: StimArray
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(positive_stims)
    # hide every neuron except 14 (simulating "preferential selection" of just
    # neuron 14, then Apply greying out the rest)
    for neuron_id in range(positive_stims.values.shape[1]):
        if neuron_id != 14:
            stim_widget.set_neuron_greyed(neuron_id, True)
    assert stim_widget.n_neurons == 1

    captured: dict[str, str] = {}
    monkeypatch.setattr(
        stim_widget.text,
        "set_pos",
        lambda pos, text, *_: captured.__setitem__("t", text),
    )
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([2.0, 0.0, 0.0]),  # the only (display row 0) visible row
    )

    stim_widget.pointer_move(SimpleNamespace(x=0.0))

    assert captured["t"].startswith("Neuron 14\n")


def test_stim_widget_y_tick_labels_stay_on_their_own_row(qtbot: QtBot) -> None:
    """Regression: when there isn't room for a tick per row, pygfx halves
    its step and asks for labels at half-integer y's too (e.g. 1.5, not just
    1 and 2). Rounding those to the nearest row drew that row's number half
    a row off from its actual position - and Python's round-half-to-even
    then hid every other integer row's own label outright as a same-value
    duplicate. Every label pygfx does ask for here must land exactly on the
    row it names, with no misplaced or missing ones in between."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_array = StimArray("", StimType.Excitation, np.zeros((10, 10)), "mV", dt=0.001)
    stim_widget.set_data(stim_array)

    format_tick = stim_widget._create_y_tick_formatter()
    for half_step in range(19):  # 0, 0.5, 1, ..., 9
        y = half_step * 0.5
        label = format_tick(y, 0, 9)
        if label:
            assert y == pytest.approx(round(y)), (
                f"label {label!r} drawn off-row at y={y}"
            )
            assert label == str(round(y))


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
    with qtbot.waitSignal(
        stim_widget.mouse_dragged, timeout=WAIT_TIMEOUT_MS
    ) as blocker:
        stim_widget.pointer_down(SimpleNamespace(button=1))

    assert blocker.args[0] == 2
    assert blocker.args[1] == 2


def test_stim_widget_gridlines_follow_stim_dt(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


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
    assert graphic.marker_line is not None
    assert graphic.marker_line.material.thickness == pytest.approx(4.0)
    assert graphic.marker_line.material.thickness_space == "screen"
    assert stim_widget.graphic is not None
    assert stim_widget.graphic.offset[2] > 199.0
    assert np.all(graphic.marker_line.geometry.positions.data[:, 2] > 199.0)
    assert graphic.fill_line is not None
    assert np.all(graphic.fill_line.geometry.positions.data[:, 2] > 199.0)
    # One vertical marker segment (2 endpoints) for the single nonzero
    # sample, spanning the full row height around neuron index 1.
    positions = graphic.marker_line.geometry.positions.data
    assert positions.shape[0] == 2
    assert sorted(positions[:, 1].tolist()) == pytest.approx([0.5, 1.5])
    assert graphic.visible

    stim_widget.set_keep_short_stimuli_visible(False)
    assert not graphic.visible
    assert values[50_000, 1] == 1.0
    stim_widget.set_keep_short_stimuli_visible(True)
    assert graphic.visible


def test_stim_widget_minimum_size_markers_stay_opaque_regardless_of_amplitude(
    qtbot: QtBot,
) -> None:
    """A low-amplitude pulse's "keep visible" markers must render at full
    alpha, just like a high-amplitude pulse's - the whole point of the
    minimum-size overlay is to guarantee visibility regardless of
    amplitude. Coloring markers by normalized magnitude (as the main
    heatmap image legitimately does) silently defeats that guarantee for
    anything far below the array's peak amplitude, exactly like
    _stim_pulse_rgba was already fixed to avoid for the closed-loop
    routines' live heatmap."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((200, 1))
    values[10:60, 0] = 100.0  # high amplitude, wide
    values[150:160, 0] = 1.0  # low amplitude (1% of peak), narrow
    stim_widget.set_data(
        StimArray("mixed", StimType.Excitation, values, "mV", dt=0.001)
    )

    graphic = stim_widget.minimum_size_graphic
    assert graphic is not None
    assert graphic.fill_line is not None
    assert graphic.marker_line is not None
    fill_colors = graphic.fill_line.geometry.colors.data
    fill_positions = graphic.fill_line.geometry.positions.data
    marker_colors = graphic.marker_line.geometry.colors.data
    marker_positions = graphic.marker_line.geometry.positions.data
    # Two contiguous runs (10:60 and 150:160), one fill + one marker segment
    # (2 endpoints each) per run, regardless of amplitude.
    assert fill_positions.shape[0] == 4
    assert marker_positions.shape[0] == 4
    assert np.all(fill_colors[:, 3] == pytest.approx(1.0))
    assert np.all(marker_colors[:, 3] == pytest.approx(1.0))
    # Marker segments span the full row height (neuron index 0: -0.5 to 0.5),
    # not just a fixed small screen-space square.
    assert np.all(np.isin(np.round(marker_positions[:, 1], 3), [-0.5, 0.5]))
    # Fill segments sit at the row center and span each run's true extent -
    # this is what keeps a wide low-amplitude pulse looking like one
    # continuous block instead of a train of short separate pulses.
    assert np.all(fill_positions[:, 1] == pytest.approx(0.0))
    assert sorted(fill_positions[:, 0].tolist()) == pytest.approx(
        [10.0, 60.0, 150.0, 160.0]
    )


def test_stim_widget_minimum_size_markers_keep_sign_for_external_stim(
    qtbot: QtBot,
) -> None:
    """External (bidirectional) stim markers must still distinguish
    positive from negative samples by color, even though magnitude no
    longer affects opacity - confirms per-segment (not just per-graphic)
    coloring works for the line-segment overlay."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((10, 1))
    values[2, 0] = 5.0
    values[7, 0] = -5.0
    stim_widget.set_data(StimArray("ext", StimType.External, values, "mV", dt=0.001))

    graphic = stim_widget.minimum_size_graphic
    assert graphic is not None
    assert graphic.fill_line is not None
    positions = graphic.fill_line.geometry.positions.data
    colors = graphic.fill_line.geometry.colors.data
    assert positions.shape[0] == 4  # 2 runs x 2 endpoints
    # Runs are ordered by time, so the positive sample (time index 2) comes
    # before the negative one (time index 7).
    positive_color = colors[0]
    negative_color = colors[2]
    assert not np.allclose(positive_color, negative_color)
    assert np.all(colors[:, 3] == pytest.approx(1.0))


def test_stim_widget_minimum_size_overlay_merges_contiguous_run_into_one_fill(
    qtbot: QtBot,
) -> None:
    """A single wide contiguous pulse must produce one fill segment for its
    whole duration, not one per sample. Building one segment per sample (an
    earlier version of this overlay did) left visible gaps between adjacent
    samples whenever they render narrower than their on-screen spacing,
    breaking a single continuous low-amplitude pulse into what looks like a
    rapid train of short separate pulses."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((200, 1))
    values[10:160, 0] = 1.0  # one 150-sample-wide contiguous pulse
    stim_widget.set_data(StimArray("wide", StimType.Excitation, values, "mV", dt=0.001))

    graphic = stim_widget.minimum_size_graphic
    assert graphic is not None
    assert graphic.fill_line is not None
    positions = graphic.fill_line.geometry.positions.data
    assert positions.shape[0] == 2  # one run -> one fill segment, not 150
    assert sorted(positions[:, 0].tolist()) == pytest.approx([10.0, 160.0])


def test_stim_widget_minimum_size_overlay_keeps_separate_runs_apart(
    qtbot: QtBot,
) -> None:
    """Two pulses separated by a real zero gap must stay two distinct fill
    segments, not merge into one."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((200, 1))
    values[10:60, 0] = 1.0
    values[70:120, 0] = 1.0
    stim_widget.set_data(StimArray("two", StimType.Excitation, values, "mV", dt=0.001))

    graphic = stim_widget.minimum_size_graphic
    assert graphic is not None
    assert graphic.fill_line is not None
    positions = graphic.fill_line.geometry.positions.data
    assert positions.shape[0] == 4  # two runs -> two fill segments
    assert sorted(positions[:, 0].tolist()) == pytest.approx([10.0, 60.0, 70.0, 120.0])


def test_stim_widget_hover_preview_snaps_to_sample_on_pointer_move(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    stim_widget.pointer_move(SimpleNamespace(x=0.0))

    assert stim_widget.hover_slice_graphic is not None
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)
    assert stim_widget.hover_slice_graphic.offset[1] == pytest.approx(2.0)
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


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


def test_stim_widget_select_mode(qtbot: QtBot, positive_stims: StimArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    with qtbot.waitExposed(stim_widget, timeout=WAIT_TIMEOUT_MS):
        stim_widget.show()
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
    qtbot.waitUntil(
        lambda: stim_widget.selected_neuron_slice == slice(1, 3),
        timeout=WAIT_TIMEOUT_MS,
    )
    assert stim_widget.selected_time_slice == slice(10, 20)
    stim_widget.unselect_neurons()
    qtbot.waitUntil(
        lambda: stim_widget.selected_neuron_slice is None, timeout=WAIT_TIMEOUT_MS
    )
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
    mouse_click_and_drag(qtbot, stim_widget.figure.canvas, start, end)
    assert stim_widget.selected_neuron_slice is not None
    assert stim_widget.selected_time_slice is not None
    # A quarter-to-three-quarters drag should land roughly in the middle
    # neuron/time band - bounds are inclusive since pixel-to-index rounding
    # can legitimately land exactly on a boundary (e.g. 0.35 * 20 == 7).
    assert (
        0.15 * n_neurons <= stim_widget.selected_neuron_slice.start <= 0.35 * n_neurons
    )
    assert (
        0.65 * n_neurons <= stim_widget.selected_neuron_slice.stop <= 0.90 * n_neurons
    )
    assert (
        0.15 * n_timestamps
        <= stim_widget.selected_time_slice.start
        <= 0.35 * n_timestamps
    )
    assert (
        0.65 * n_timestamps
        <= stim_widget.selected_time_slice.stop
        <= 0.90 * n_timestamps
    )
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_stim_widget_draw_mode(qtbot: QtBot, positive_stims: StimArray) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    with qtbot.waitExposed(stim_widget, timeout=WAIT_TIMEOUT_MS):
        stim_widget.show()
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
    with qtbot.waitSignal(stim_widget.mouse_clicked, timeout=WAIT_TIMEOUT_MS) as signal:
        mouse_click(qtbot, stim_widget.figure.canvas, QtCore.QPoint(x0, y0))
    neuron_index_offset, time_index, mouse_button = signal.args
    assert 0 < time_index / n_timestamps < 0.4
    assert 0 < neuron_index_offset / n_neurons < 1.0
    assert mouse_button == 1
    with qtbot.waitSignal(stim_widget.mouse_clicked, timeout=WAIT_TIMEOUT_MS) as signal:
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
    with qtbot.waitSignal(stim_widget.mouse_clicked, timeout=WAIT_TIMEOUT_MS) as signal:
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
        qtbot, stim_widget.figure.canvas, QtCore.QPoint(x0, y0), QtCore.QPoint(x2, y0)
    )
    # moves between start and end may be merged into one event
    assert len(drag_signals) >= 2
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_stim_widget_ext_stims(
    qtbot: QtBot, negative_and_positive_stims: StimArray
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


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


def test_background_short_input_indicators_obey_show_all_input_arrays(
    qtbot: QtBot,
) -> None:
    """#585: with "Keep short inputs visible" on and "Show all input arrays"
    off, other arrays' short-input indicators must stay hidden - including
    ones added after the toggle (i.e. when switching input arrays)."""
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((1000, 2))
    values[500, 1] = 1.0
    current = StimArray("current", StimType.Excitation, values, "mV", dt=0.001)
    other = StimArray("other", StimType.Excitation, values.copy(), "mV", dt=0.001)
    stim_widget.set_data(current)
    stim_widget.set_keep_short_stimuli_visible(True)
    stim_widget.set_background_data_visible(False)

    stim_widget.set_background_data(other)

    assert len(stim_widget._background_minimum_size_graphics) == 1
    assert not stim_widget._background_minimum_size_graphics[0].visible
    assert stim_widget.minimum_size_graphic is not None
    assert stim_widget.minimum_size_graphic.visible

    stim_widget.set_background_data_visible(True)
    assert stim_widget._background_minimum_size_graphics[0].visible


def test_stim_widget_hover_preview_stays_snapped_within_cell(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    stim_widget.show()
    qtbot.waitUntil(lambda: stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    stim_widget.pointer_move(SimpleNamespace(x=0.0))
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)

    # Second move within the SAME cell (x=1.45): the preview must stay snapped to
    # the cell centre rather than drifting to the raw pointer position, and keep
    # its hover z-order (regression: the base class used to overwrite both).
    monkeypatch.setattr(
        stim_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([1.45, 2.2, 0.0]),
    )
    stim_widget.pointer_move(SimpleNamespace(x=0.0))
    assert stim_widget.hover_slice_graphic.offset[0] == pytest.approx(1.25)
    assert stim_widget.hover_slice_graphic.offset[2] == pytest.approx(
        stim_widget.HOVER_Z
    )
    stim_widget.close()
    qtbot.waitUntil(lambda: not stim_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_active_span_finds_bounds_of_contiguous_nonzero_run() -> None:
    from neuroseq.visualization.stim_widget import _active_span

    column = np.array([0, 1, 2, 2, 0, 0, 3, 0], dtype=float)
    assert _active_span(column, 0) is None
    assert _active_span(column, 1) == (1, 3)
    assert _active_span(column, 3) == (1, 3)
    assert _active_span(column, 6) == (6, 6)
    assert _active_span(np.ones(4), 2) == (0, 3)


def test_stim_widget_hover_shows_start_and_end_of_stimulus(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #590: hovering a stimulus reports its start and end time."""
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    values = np.zeros((10, 2), dtype=float)
    values[3:7, 1] = 5.0
    stim_widget.set_data(StimArray("s", StimType.Excitation, values, "pA", dt=0.002))

    captured: dict[str, str] = {}
    monkeypatch.setattr(
        stim_widget.text,
        "set_pos",
        lambda pos, text, *_: captured.__setitem__("t", text),
    )
    world = {"pos": np.array([8.0, 1.0, 0.0])}  # 2 ms samples -> time index 4
    monkeypatch.setattr(
        stim_widget.subplot, "map_screen_to_world", lambda ev: world["pos"]
    )

    stim_widget.pointer_move(SimpleNamespace(x=0.0))
    assert captured["t"] == "Neuron 1\n5 pA\nStart: 6 ms\nEnd: 14 ms"

    # Empty space: no start/end lines.
    world["pos"] = np.array([1.0, 1.0, 0.0])
    stim_widget.pointer_move(SimpleNamespace(x=0.0))
    assert captured["t"] == "Neuron 1\n0 pA"


def test_stim_widget_set_data_with_every_neuron_hidden(qtbot: QtBot) -> None:
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    stim_array = StimArray("", StimType.Excitation, np.ones((5, 2)), "mV", dt=0.001)
    stim_widget.set_data(stim_array)
    stim_widget.set_neurons_greyed([0, 1], [])

    stim_widget.set_data(stim_array)

    assert stim_widget.graphic is not None
    assert not stim_widget.graphic.visible
    stim_widget.set_neurons_greyed([], [0])
    assert stim_widget.graphic.visible
    assert np.asarray(stim_widget.graphic.data.value).shape[0] == 1


def test_stim_widget_hiding_every_neuron_hides_the_drawn_stimuli(
    qtbot: QtBot,
) -> None:
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    stim_array = StimArray("", StimType.Excitation, np.ones((5, 2)), "mV", dt=0.001)
    stim_widget.set_data(stim_array)
    assert stim_widget.graphic is not None
    assert stim_widget.minimum_size_graphic is not None

    stim_widget.set_neurons_greyed([0, 1], [])
    assert not stim_widget.graphic.visible
    assert stim_widget.minimum_size_graphic is None

    stim_widget.set_neurons_greyed([], [1])
    assert stim_widget.graphic.visible
    assert np.asarray(stim_widget.graphic.data.value).shape[0] == 1
    assert stim_widget.minimum_size_graphic is not None
    assert stim_widget.minimum_size_graphic.visible


def test_stim_widget_background_arrays_follow_hidden_neurons(qtbot: QtBot) -> None:
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    foreground = np.zeros((4, 3))
    background = np.zeros((4, 3))
    background[:, 0] = 1.0
    background[:, 2] = 3.0
    stim_widget.set_data(StimArray("fg", StimType.Excitation, foreground, "1", 0.001))
    stim_widget.set_background_data(
        StimArray("bg", StimType.Excitation, background, "1", 0.001)
    )

    stim_widget.set_neuron_greyed(1, True)

    assert len(stim_widget.background_graphics) == 1
    rendered = np.asarray(stim_widget.background_graphics[0].data.value)
    assert rendered.shape[0] == 2  # neuron 1's row is gone
    np.testing.assert_allclose(rendered[0], 1.0)
    np.testing.assert_allclose(rendered[1], 3.0)  # neuron 2 on display row 1

    # A background array added while neurons are hidden is compacted too.
    stim_widget.set_background_data(
        StimArray("bg2", StimType.Excitation, background, "1", 0.001)
    )
    assert np.asarray(stim_widget.background_graphics[1].data.value).shape[0] == 2


def test_stim_widget_hover_ignores_background_without_hovered_neuron(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(
        StimArray("fg", StimType.Excitation, np.zeros((4, 3)), "1", 0.001)
    )
    # One neuron only: hovering neuron 2 must not show neuron 0's value.
    stim_widget.set_background_data(
        StimArray("bg", StimType.Excitation, np.full((4, 1), 5.0), "1", 0.001)
    )
    captured: list[str] = []
    monkeypatch.setattr(
        stim_widget.text, "set_pos", lambda pos, text, *_: captured.append(text)
    )
    monkeypatch.setattr(
        stim_widget.subplot, "map_screen_to_world", lambda ev: np.array([1.0, 2.0, 0])
    )

    stim_widget.pointer_move(SimpleNamespace(x=0.0))

    assert captured[-1].startswith("Neuron 2\n0 ")


def test_stim_preview_draws_visible_neurons_at_display_rows(qtbot: QtBot) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    controller = ConstrainedPanZoomController()
    stim_widget = StimWidget(controller)
    qtbot.addWidget(stim_widget)
    values = np.zeros((4, 3))
    values[1:3, 2] = 1.0
    stim_widget.set_data(StimArray("s", StimType.Excitation, values, "1", 0.001))
    stim_widget.set_neuron_greyed(1, True)
    stim_widget.select_neurons(slice(1, 2), None)  # display row 1 = neuron 2

    dialog = stim_widget.preview_stimuli()
    assert dialog is not None
    qtbot.addWidget(dialog)
    ax = dialog.findChild(FigureCanvasQTAgg).figure.axes[0]

    # Neuron 2's pulse sits on display row 1, inside the selected y range.
    np.testing.assert_allclose(
        ax.collections[-1].get_paths()[0].vertices[:4],
        [[1, 0.5], [3, 0.5], [3, 1.5], [1, 1.5]],
    )
    assert ax.get_ylim() == pytest.approx((0.5, 1.5))
    assert ax.yaxis.get_major_formatter()(1.0, 0) == "2"
    dialog.close()


def test_stim_widget_draw_click_on_empty_row_emits_nothing(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    stim_widget = StimWidget(ConstrainedPanZoomController())
    qtbot.addWidget(stim_widget)
    stim_widget.set_data(
        StimArray("s", StimType.Excitation, np.zeros((100, 4)), "1", 0.001)
    )
    stim_widget.set_mode(InteractMode.DRAW)
    clicks: list[tuple] = []
    drags: list[tuple] = []
    stim_widget.mouse_clicked.connect(lambda *args: clicks.append(args))
    stim_widget.mouse_dragged.connect(lambda *args: drags.append(args))
    event = SimpleNamespace(button=1, x=0.0, y=0.0)

    def click_at(position) -> None:
        monkeypatch.setattr(
            stim_widget.subplot, "map_screen_to_world", lambda ev: position
        )
        stim_widget.pointer_down(event)
        stim_widget.pointer_move(event)
        stim_widget.pointer_up(event)

    click_at(np.array([5.0, -3.0, 0.0]))  # below row 0: was int(None) error
    click_at(np.array([10.0, 2.0, 0.0]))
    click_at(np.array([50.0, 7.0, 0.0]))  # no row 7: re-emitted the last click
    click_at(None)  # outside the plot

    assert clicks == [(2, 10, 1)]
    assert drags == [(2, 2, 10, 10, 1)]
