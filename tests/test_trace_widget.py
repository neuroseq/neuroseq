from __future__ import annotations

import brian2.numpy_ as np
import pygfx
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import TraceData
from neuroseq.region_of_interest import RegionOfInterest
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_widget import StimWidget
from neuroseq.trace_widget import TraceNormDialog
from neuroseq.trace_widget import TraceWidget
from neuroseq.trace_widget_state import TraceWidgetState
from neuroseq.utils import neuron_indices_to_colors


def test_timeseries_overlays_follow_theme(qtbot: QtBot, make_trace_data) -> None:
    trace_widget = TraceWidget(ConstrainedPanZoomController())
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data())
    assert trace_widget.grid_lines is not None

    trace_widget.apply_theme(theme.RETRO)

    colors = theme.plot_overlay_colors(theme.RETRO)
    assert tuple(trace_widget.roi.fill.material.color) == pytest.approx(
        tuple(pygfx.Color(colors["roi_fill"]))
    )
    assert tuple(trace_widget.roi.text.material.color) == pytest.approx(
        tuple(pygfx.Color(colors["roi_text"]))
    )
    assert tuple(
        trace_widget.grid_lines.gridline_10samples.material.color
    ) == pytest.approx(tuple(pygfx.Color(colors["grid"])))


@pytest.mark.parametrize("clip,center", [(True, True), (False, False)])
def test_trace_preview_preserves_arrangement(qtbot, clip, center, tmp_path) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.colors import to_rgba

    controller = ConstrainedPanZoomController()
    widget = TraceWidget(controller)
    qtbot.addWidget(widget)
    assert widget.preview_traces() is None
    data = TraceData(
        name="preview source",
        traces=np.array([[1, 2, 5, 1], [-2, 0, 2, 0]], dtype=np.float32),
        indices=np.array([3, 7], dtype=np.uint32),
        dt=0.002,
        units="mV",
    )
    widget.set_data(data)
    widget.trace_norm_percent = 300
    widget.clip = clip
    widget.offset_by_median = center
    widget.replot_traces()
    controller.focus_neuron_range(2, 9)
    controller.focus_time_range(0.001, 0.005)
    state = controller._get_camera_state()
    dialog = widget.preview_traces()
    assert dialog is not None
    qtbot.addWidget(dialog)
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.findChild(NavigationToolbar2QT) is not None
    canvas = dialog.findChild(FigureCanvasQTAgg)
    ax = canvas.figure.axes[0]
    assert to_rgba(canvas.figure.get_facecolor())[3] == 0
    assert to_rgba(ax.get_facecolor())[3] == 0
    assert ax.get_title() == "preview source"
    expected = (
        widget.get_normalized_trace_data() + np.arange(len(data.indices))[:, None]
    )
    for i, line in enumerate(ax.lines):
        np.testing.assert_allclose(line.get_xdata(), [0, 2, 4, 6])
        np.testing.assert_allclose(line.get_ydata(), expected[i])
        np.testing.assert_allclose(
            line.get_color(), neuron_indices_to_colors(data.indices)[i]
        )
        assert line.get_gid() == f"trace_{data.indices[i]}"
    assert ax.get_xlim() == pytest.approx((1, 5))
    cy = state["position"][1]
    assert ax.get_ylim() == pytest.approx(
        (cy - state["height"] / 2, cy + state["height"] / 2)
    )
    widget.trace_norm_percent = 50
    widget.replot_traces()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected[0])
    output = tmp_path / "preview.svg"
    canvas.figure.savefig(output)
    assert 'id="trace_3"' in output.read_text(encoding="utf-8")
    dialog.close()


@pytest.mark.parametrize(
    "neuron_range,time_range", [(slice(0, 2), slice(1, 3)), (slice(1, 2), slice(2, 3))]
)
def test_trace_preview_uses_selected_region(
    qtbot, make_trace_data, neuron_range, time_range
):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    controller = ConstrainedPanZoomController()
    widget = TraceWidget(controller)
    qtbot.addWidget(widget)
    widget.set_data(
        make_trace_data(
            n_neurons=2,
            n_timestamps=10,
            indices=np.array([3, 7], dtype=np.uint32),
            dt=0.002,
        )
    )
    state = controller._get_camera_state()
    widget.select_neurons(neuron_range, time_range)
    dialog = widget.preview_traces()
    assert dialog is not None
    qtbot.addWidget(dialog)
    ax = dialog.findChild(FigureCanvasQTAgg).figure.axes[0]
    assert ax.get_xlim() == pytest.approx((time_range.start * 2, time_range.stop * 2))
    assert ax.get_ylim() == pytest.approx(
        (neuron_range.start - 0.5, neuron_range.stop - 0.5)
    )
    for key in ("position", "width", "height"):
        np.testing.assert_allclose(controller._get_camera_state()[key], state[key])
    dialog.close()

    widget.unselect_neurons()
    dialog = widget.preview_traces()
    assert dialog is not None
    qtbot.addWidget(dialog)
    ax = dialog.findChild(FigureCanvasQTAgg).figure.axes[0]
    cx, cy = state["position"][:2]
    assert ax.get_xlim() == pytest.approx(
        (cx - state["width"] / 2, cx + state["width"] / 2)
    )
    assert ax.get_ylim() == pytest.approx(
        (cy - state["height"] / 2, cy + state["height"] / 2)
    )
    dialog.close()


def test_trace_widget_context_menu_offers_vector_export(
    qtbot: QtBot, make_trace_data
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget._rebuild_custom_analysis_menu()
    export_action = next(
        action
        for action in trace_widget.context_menu.actions()
        if action.text() == "Preview vector graphic…"
    )
    assert not export_action.isEnabled()

    trace_widget.set_data(make_trace_data())
    trace_widget._rebuild_custom_analysis_menu()
    export_action = next(
        action
        for action in trace_widget.context_menu.actions()
        if action.text() == "Preview vector graphic…"
    )
    assert export_action.isEnabled()
    with qtbot.waitSignal(trace_widget.vector_export_requested):
        export_action.trigger()


def test_trace_widget_context_menu_offers_normalization(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget._rebuild_custom_analysis_menu()
    norm_action = next(
        action
        for action in trace_widget.context_menu.actions()
        if action.text() == "Normalization…"
    )
    norm_action.trigger()
    assert trace_widget._trace_norm_dialog is not None
    assert trace_widget._trace_norm_dialog.isVisible()
    trace_widget._trace_norm_dialog.close()


@pytest.mark.parametrize("widget_cls", [TraceWidget, SpikeWidget])
def test_timeseries_context_menu_offers_detach_actions(
    qtbot: QtBot, widget_cls
) -> None:
    controller = ConstrainedPanZoomController()
    widget = widget_cls(controller)
    qtbot.addWidget(widget)

    widget._rebuild_custom_analysis_menu()
    action_texts = [action.text() for action in widget.context_menu.actions()]

    assert "Detach panel" in action_texts
    assert "Detach copy of panel" in action_texts


def test_stim_widget_context_menu_has_no_detach_actions(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    widget = StimWidget(controller)
    qtbot.addWidget(widget)

    widget._rebuild_custom_analysis_menu()
    action_texts = [action.text() for action in widget.context_menu.actions()]

    assert "Detach panel" not in action_texts
    assert "Detach copy of panel" not in action_texts


def test_trace_norm_dialog_minimum_is_50_percent(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)

    assert trace_norm_dialog.slider_norm.minimum() == 50


def test_trace_norm_dialog_slider_and_spinbox_stay_in_sync(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)

    trace_norm_dialog.slider_norm.setValue(250)
    assert trace_norm_dialog.spin_norm.value() == 250
    assert trace_widget.trace_norm_percent == 250

    # typing a value into the spin box should drive the slider too
    trace_norm_dialog.spin_norm.setValue(400)
    assert trace_norm_dialog.slider_norm.value() == 400
    assert trace_widget.trace_norm_percent == 400


def test_trace_norm_dialog_is_modeless_and_reused(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget.trace_norm_dialog()
    dialog = trace_widget._trace_norm_dialog

    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.isModal()
    # No Qt parent - see _SelfRaisingDialog's docstring for why a shared
    # WM_TRANSIENT_FOR chain with a sibling popup breaks click-to-front.
    assert dialog.parent() is None

    trace_widget.trace_norm_dialog()
    assert trace_widget._trace_norm_dialog is dialog


def test_trace_widget_zero_traces(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)
    trace_widget.show()
    trace_norm_dialog.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 0.0991
    data = TraceData(
        name="traces",
        traces=np.zeros(shape=(n_neurons, n_timestamps), dtype=np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=dt,
        units="mV",
    )
    trace_widget.set_data(data)
    # if all traces are zero we don't normalize by the max value (which is zero),
    # so the plotted values should still be zero
    # (as we are testing values from the gpu allow for some floating point rounding errors)
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.0, abs=0.01
    )
    assert np.min(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.0, abs=0.01
    )
    # changing the normalization doesn't change zero
    trace_norm_dialog.update_norm_input(200)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # nor does disabling the offset, since the median is also zero
    trace_norm_dialog.chk_offset_clicked(False)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # nor does disabling clipping
    trace_norm_dialog.chk_clip_clicked(False)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    trace_widget.close()


def test_trace_widget_median_offset(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)
    trace_widget.show()
    trace_norm_dialog.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    data = TraceData(
        name="traces",
        traces=np.array(
            [np.zeros(100), np.ones(100), -2 * np.ones(100)], dtype=np.float32
        ),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    # default normalisation with median offset: all three constant traces should be 0
    assert trace_widget.offset_by_median is True
    assert np.array(trace_widget.graphic.data[:]).shape == (3, 100, 3)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # disable median offset: traces should now be 0.0, 0.25, -0.5
    trace_norm_dialog.chk_offset_clicked(False)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.25, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-0.5, abs=0.01)
    # increase normalization to 200%: traces should now be 0.0, 0.5, -1.0 (clipped to -0.5)
    trace_norm_dialog.update_norm_input(200)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.5, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-0.5, abs=0.01)
    # disable clipping: traces should now be 0.0, 0.5, -1.0
    trace_norm_dialog.chk_clip_clicked(False)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.5, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-1.0, abs=0.01)
    # re-enable median offset: all three constant traces should be 0
    trace_norm_dialog.chk_offset_clicked(True)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    trace_widget.close()


def test_median_offset_normalization_bounds_negative_excursions_too(
    qtbot: QtBot,
) -> None:
    """Regression for issue #533: a neuron whose deviation *below* its
    median is bigger than its deviation above it (e.g. a calcium baseline
    with an occasional deep dip but only modest transients) must still be
    scaled to fit within +/-0.5 at 100% normalization - not just the side
    above the median."""
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    traces = np.full((2, 5), 100.0, dtype=np.float32)
    traces[0, 2] = 110.0  # +10 above median
    traces[0, 3] = -900.0  # -1000 below median - the binding excursion
    traces[1, 2] = 105.0
    traces[1, 3] = 95.0
    data = TraceData(
        name="trace",
        traces=traces,
        indices=np.array([0, 1], dtype=np.uint32),
        dt=0.01,
        units="a.u.",
    )
    trace_widget.set_data(data)
    trace_widget.offset_by_median = True
    trace_widget.clip = False
    trace_widget.trace_norm_percent = 100

    normalized = trace_widget.get_normalized_trace_data()

    assert normalized[0].min() == pytest.approx(-0.5)
    assert normalized[0].max() == pytest.approx(0.005)
    trace_widget.close()


def _make_scaled_trace_data(n_neurons: int, n_timesteps: int = 50) -> TraceData:
    """neuron i's amplitude is (i + 1)x neuron 0's, so at normalization
    percentages above 100% only the low-index neurons would clip."""
    t = np.linspace(0, 2 * np.pi, n_timesteps, dtype=np.float32)
    traces = np.stack([np.sin(t) * (i + 1) for i in range(n_neurons)], axis=0).astype(
        np.float32
    )
    return TraceData(
        name="traces",
        traces=traces,
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=0.001,
        units="mV",
    )


def test_auto_extend_vertical_space_grows_rows_without_clipping(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 5)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = _make_scaled_trace_data(5)
    trace_widget.set_data(data)

    trace_widget.trace_norm_percent = 500
    trace_widget.auto_extend_vertical_space = True
    trace_widget.replot_traces()

    # every row's *unclipped* excursion actually reaches +/-(i+1)*0.5 at 500%
    # (neuron 0 fits its default 0.5 half-row; neurons 1-4 need more)
    assert trace_widget._row_half_heights == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    normalized = trace_widget.get_normalized_trace_data()
    for i in range(5):
        assert np.max(np.abs(normalized[i])) == pytest.approx(0.5 * (i + 1), rel=0.05)

    # rows must not overlap: each row's own band stays within its boundaries
    centers = trace_widget._row_centers
    half_heights = trace_widget._row_half_heights
    for i in range(len(centers) - 1):
        this_row_top = centers[i] + half_heights[i]
        next_row_bottom = centers[i + 1] - half_heights[i + 1]
        assert this_row_top == pytest.approx(next_row_bottom)

    # the controller's height limit grows to fit the new total, and the
    # camera actually gets refocused onto the larger extent
    total_height = sum(2 * h for h in half_heights)
    assert controller.max_height == pytest.approx(total_height)
    trace_widget.close()


def test_auto_extend_vertical_space_leaves_uniform_rows_alone_when_off(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 5)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = _make_scaled_trace_data(5)
    trace_widget.set_data(data)

    trace_widget.trace_norm_percent = 500
    trace_widget.replot_traces()

    assert trace_widget._row_half_heights == []
    assert trace_widget._row_centers == []
    assert controller.max_height == pytest.approx(5.0)
    # clipping (the default) still applies when auto-extend is off
    normalized = trace_widget.get_normalized_trace_data()
    assert np.max(np.abs(normalized)) == pytest.approx(0.5)
    trace_widget.close()


def test_auto_extend_vertical_space_restores_controller_height_when_turned_off(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 5)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = _make_scaled_trace_data(5)
    trace_widget.set_data(data)

    trace_widget.trace_norm_percent = 500
    trace_widget.auto_extend_vertical_space = True
    trace_widget.replot_traces()
    assert controller.max_height > 5.0

    trace_widget.auto_extend_vertical_space = False
    trace_widget.replot_traces()
    assert controller.max_height == pytest.approx(5.0)
    assert trace_widget._row_half_heights == []
    trace_widget.close()


def test_auto_extend_vertical_space_hover_and_tick_labels_track_warped_rows(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 5)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = _make_scaled_trace_data(5)
    trace_widget.set_data(data)
    trace_widget.trace_norm_percent = 500
    trace_widget.auto_extend_vertical_space = True
    trace_widget.replot_traces()

    centers = trace_widget._row_centers
    formatter = trace_widget._create_y_tick_formatter()
    for neuron_id, center in enumerate(centers):
        assert formatter(center, 0, 0) == str(neuron_id)
    # a position past the last row's extent has no real row behind it
    assert formatter(centers[-1] + 100, 0, 0) == ""

    for neuron_id, center in enumerate(centers):
        assert trace_widget._display_row_for_mouse_y(center) == neuron_id
    # hovering past the last row clamps to it rather than raising/wrapping
    assert trace_widget._display_row_for_mouse_y(centers[-1] + 100) == len(centers) - 1
    trace_widget.close()


def test_auto_extend_vertical_space_does_not_move_camera_on_replot(
    qtbot: QtBot,
) -> None:
    """Normalization tweaks while auto-extend is on must only grow the view's
    extent (so the user *can* zoom out to see a taller row) - not snap the
    camera to it. Forcing a camera refocus on every replot made unrelated
    traces look visually rescaled/recentered even though their data never
    changed (issue #551 review)."""
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 5)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = _make_scaled_trace_data(5)
    trace_widget.set_data(data)

    trace_widget.trace_norm_percent = 500
    trace_widget.auto_extend_vertical_space = True
    trace_widget.replot_traces()

    # simulate the user manually zooming/panning to their own view
    controller.focus_neuron_range(1, 3)
    camera_state_before = dict(controller._get_camera_state())

    # a no-op toggle under auto-extend (clip is bypassed whenever
    # auto-extend is on) must not disturb that camera state at all
    trace_widget.clip = not trace_widget.clip
    trace_widget.replot_traces()
    camera_state_after = dict(controller._get_camera_state())
    assert camera_state_after["height"] == pytest.approx(camera_state_before["height"])
    assert camera_state_after["position"] == pytest.approx(
        camera_state_before["position"]
    )

    # a genuine normalization change (growing a row further) still must not
    # move the camera - only the extent it's allowed to grow into
    trace_widget.trace_norm_percent = 900
    trace_widget.replot_traces()
    camera_state_after = dict(controller._get_camera_state())
    assert camera_state_after["height"] == pytest.approx(camera_state_before["height"])
    assert camera_state_after["position"] == pytest.approx(
        camera_state_before["position"]
    )
    assert controller.max_height >= sum(2 * h for h in trace_widget._row_half_heights)
    trace_widget.close()


def test_trace_norm_dialog_auto_extend_checkbox_updates_trace_widget(
    qtbot: QtBot,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)

    assert trace_widget.auto_extend_vertical_space is False
    trace_norm_dialog.chk_auto_extend_clicked(True)
    assert trace_widget.auto_extend_vertical_space is True

    trace_norm_dialog.reset_defaults()
    assert trace_widget.auto_extend_vertical_space is False
    assert trace_norm_dialog.chk_auto_extend.isChecked() is False
    trace_widget.close()


def test_trace_widget_indices(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    data = TraceData(
        name="traces",
        traces=np.random.random_sample((5, 100)).astype(np.float32),
        indices=np.array([0, 1, 6, 11, 15], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    colormap = {i: neuron_indices_to_colors(np.array([i]))[0] for i in data.indices}
    data.traces[0, :] = 0.0
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    for i, line in enumerate(trace_widget.graphic.graphics):
        idx = int(data.indices[i])
        assert line.offset[1] == pytest.approx(
            trace_widget._neuron_id_to_display_row[idx]
        )
        assert np.allclose(line.colors, colormap[data.indices[i]])
    # now set same data with same indices in a different order
    data.indices = np.array([15, 6, 11, 1, 0], dtype=np.uint32)
    trace_widget.set_data(data)
    for i, line in enumerate(trace_widget.graphic.graphics):
        idx = int(data.indices[i])
        assert line.offset[1] == pytest.approx(
            trace_widget._neuron_id_to_display_row[idx]
        )
        assert np.allclose(line.colors, colormap[data.indices[i]])


def test_trace_widget_y_tick_labels_show_real_neuron_id(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = TraceData(
        name="traces",
        traces=np.zeros((5, 10), dtype=np.float32),
        indices=np.array([0, 1, 6, 11, 15], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    trace_widget.set_data(data)

    # sparse ids: display row 2 (compacted rank) is real id 6, not "2"
    format_tick = trace_widget._create_y_tick_formatter()
    assert format_tick(2, 0, 5) == "6"

    trace_widget.set_neuron_greyed(6, True)
    # after hiding id 6, display row 2 now corresponds to real id 11
    format_tick = trace_widget._create_y_tick_formatter()
    assert format_tick(2, 0, 5) == "11"
    trace_widget.close()


def test_trace_widget_y_tick_formatter_blanks_rows_beyond_visible_count(
    qtbot: QtBot,
) -> None:
    """Regression: the axis can be asked for tick candidates beyond the
    currently visible row count (the broader pannable extent doesn't shrink
    just because some neurons are hidden). Such a position used to fall back
    to labeling the raw row index as if it were a real neuron id, drawing a
    "ghost" numbered tick with no grid line or data behind it."""
    controller = ConstrainedPanZoomController()
    controller.set_limits(max_time_secs=1.0, max_neurons=10)
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = TraceData(
        name="traces",
        traces=np.zeros((3, 10), dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    trace_widget.set_data(data)

    format_tick = trace_widget._create_y_tick_formatter()

    assert format_tick(5, 0, 10) == ""
    trace_widget.close()


def test_trace_widget_hover_shows_real_neuron_id_when_only_one_visible(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=20, n_timestamps=10))
    for neuron_id in range(20):
        if neuron_id != 14:
            trace_widget.set_neuron_greyed(neuron_id, True)
    assert trace_widget.n_neurons == 1

    captured: dict[str, str] = {}
    monkeypatch.setattr(
        trace_widget.text, "set_pos", lambda pos, text: captured.__setitem__("t", text)
    )
    monkeypatch.setattr(
        trace_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),  # the only (display row 0) visible row
    )

    trace_widget.pointer_move(None)

    assert captured["t"].startswith("Neuron 14\n")
    trace_widget.close()


def test_pointer_move_emits_neuron_hovered_only_on_change(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    emitted: list[int | None] = []
    trace_widget.neuron_hovered.connect(emitted.append)

    monkeypatch.setattr(
        trace_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 1.0, 0.0]),  # row 1
    )
    trace_widget.pointer_move(None)
    trace_widget.pointer_move(None)  # same row - no duplicate emission

    assert emitted == [1]

    monkeypatch.setattr(
        trace_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 2.0, 0.0]),  # row 2
    )
    trace_widget.pointer_move(None)

    assert emitted == [1, 2]

    monkeypatch.setattr(
        trace_widget.subplot, "map_screen_to_world", lambda ev: None
    )  # pointer left the plot area
    trace_widget.pointer_move(None)

    assert emitted == [1, 2, None]
    trace_widget.close()


def test_pointer_leave_clears_hover_state(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    make_trace_data,
) -> None:
    """The canvas gets a pointer_leave event (not a pointer_move) when the

    mouse leaves the widget bounds directly, e.g. by moving quickly onto a
    neighboring panel. Highlighting driven by the last hover must not stick.
    """
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    emitted: list[int | None] = []
    trace_widget.neuron_hovered.connect(emitted.append)

    monkeypatch.setattr(
        trace_widget.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 1.0, 0.0]),  # row 1
    )
    trace_widget.pointer_move(None)
    assert emitted == [1]
    assert trace_widget.text.visible

    trace_widget.pointer_leave(None)

    assert emitted == [1, None]
    assert trace_widget.mouse_pos is None
    assert not trace_widget.text.visible
    trace_widget.close()


def test_trace_widget_clear_data_releases_plot_buffers(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    data = TraceData(
        name="traces",
        traces=np.ones((2, 100), dtype=np.float32),
        indices=np.array([0, 1], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )

    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    assert trace_widget.data_values is not None
    assert trace_widget.grid_lines is not None

    trace_widget.clear_data()

    assert trace_widget.graphic is None
    assert trace_widget.data is None
    assert trace_widget.data_values is None
    assert trace_widget.grid_lines is None


def test_trace_widget_set_neuron_greyed_adds_and_removes_overlay(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    assert not trace_widget.is_neuron_greyed(2)
    trace_widget.set_neuron_greyed(2, True)
    assert trace_widget.is_neuron_greyed(2)
    trace_widget.set_neuron_greyed(2, True)
    assert trace_widget._greyed_neuron_ids == {2}
    trace_widget.set_neuron_greyed(2, False)
    assert not trace_widget.is_neuron_greyed(2)
    trace_widget.close()


def test_trace_widget_greying_hides_and_compacts_lines(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    controller.set_limits(max_time_secs=1.0, max_neurons=3)
    data = make_trace_data(n_neurons=3, n_timestamps=10)
    original_traces = data.traces.copy()
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    assert isinstance(trace_widget.data, TraceData)
    assert trace_widget.grid_lines is not None

    trace_widget.set_neuron_greyed(1, True)

    lines = trace_widget.graphic.graphics
    assert lines[0].visible
    assert lines[0].offset[1] == pytest.approx(0)
    assert not lines[1].visible
    assert lines[2].visible
    assert lines[2].offset[1] == pytest.approx(1)  # moved up to fill the gap
    assert trace_widget.n_neurons == 2
    assert trace_widget.n_neurons_total == 3
    # underlying data is untouched
    assert np.allclose(trace_widget.data.traces, original_traces)
    # the view is refit to exactly the 2 remaining rows, not left showing dead
    # space where the whole original 3-row extent used to be
    assert trace_widget.grid_lines.n_neurons == 2
    camera_state = controller._get_camera_state()
    assert camera_state["height"] == pytest.approx(2)

    trace_widget.set_neuron_greyed(1, False)
    assert lines[1].visible
    assert lines[1].offset[1] == pytest.approx(1)
    assert lines[2].offset[1] == pytest.approx(2)  # restored to its original slot
    assert trace_widget.n_neurons == 3
    assert trace_widget.grid_lines.n_neurons == 3
    camera_state = controller._get_camera_state()
    assert camera_state["height"] == pytest.approx(3)
    trace_widget.close()


def test_two_trace_views_sharing_state_stay_in_sync(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    state = TraceWidgetState()
    controller_a = ConstrainedPanZoomController()
    controller_b = ConstrainedPanZoomController()
    view_a = TraceWidget(controller_a, state=state)
    view_b = TraceWidget(controller_b, state=state)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)
    controller_a.set_limits(max_time_secs=1.0, max_neurons=3)
    controller_b.set_limits(max_time_secs=1.0, max_neurons=3)

    view_a.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    # data set via one view is visible in the other immediately
    assert view_b.data is view_a.data
    assert view_b.graphic is not None

    view_a.set_neuron_greyed(1, True)

    assert view_a.n_neurons == 2
    assert view_b.n_neurons == 2
    assert view_b.graphic is not None
    assert not view_b.graphic.graphics[1].visible

    view_b.set_neuron_greyed(1, False)

    assert view_a.n_neurons == 3
    assert view_a.graphic is not None
    assert view_a.graphic.graphics[1].visible
    view_a.close()
    view_b.close()


def test_two_trace_views_sharing_state_stay_in_sync_for_batched_greying(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    state = TraceWidgetState()
    controller_a = ConstrainedPanZoomController()
    controller_b = ConstrainedPanZoomController()
    view_a = TraceWidget(controller_a, state=state)
    view_b = TraceWidget(controller_b, state=state)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)
    controller_a.set_limits(max_time_secs=1.0, max_neurons=3)
    controller_b.set_limits(max_time_secs=1.0, max_neurons=3)

    view_a.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    view_a.set_neurons_greyed([0, 1], [])

    assert view_a.n_neurons == 1
    assert view_b.n_neurons == 1
    assert view_b.is_neuron_greyed(0)
    assert view_b.is_neuron_greyed(1)

    view_b.set_neurons_greyed([2], [0, 1])

    assert view_a.n_neurons == 2
    assert not view_a.is_neuron_greyed(0)
    assert not view_a.is_neuron_greyed(1)
    assert view_a.is_neuron_greyed(2)
    view_a.close()
    view_b.close()


def test_trace_widget_with_no_state_arg_gets_a_private_state(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    view_a = TraceWidget(ConstrainedPanZoomController())
    view_b = TraceWidget(ConstrainedPanZoomController())
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    view_a.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    assert view_a.data is not None
    assert view_b.data is None
    view_a.close()
    view_b.close()


def test_trace_widget_clear_data_clears_greyed_neurons(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    trace_widget.set_neuron_greyed(1, True)

    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    assert not trace_widget.is_neuron_greyed(1)
    trace_widget.close()


def test_trace_widget(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)
    trace_widget.show()
    trace_norm_dialog.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 0.0991
    data = TraceData(
        name="traces",
        traces=np.random.rand(n_neurons, n_timestamps),
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=dt,
        units="mV",
    )
    trace_widget.offset_by_median = False
    trace_widget.set_data(data)
    assert trace_widget.n_neurons == n_neurons
    assert trace_widget.n_timestamps == n_timestamps
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    # default normalisation with median offset disabled: all traces should be in range [-0.5, 0.5]
    # and the max abs value should be 0.5
    # (as we are testing values from the gpu allow for some floating point rounding errors)
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.51
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.51
    # update normalisation using the dialog to 500% with clipping, no offset
    trace_norm_dialog.clip = True
    trace_norm_dialog.offset_by_median = False
    trace_norm_dialog.update_norm_input(500)
    assert trace_widget.trace_norm_percent == 500
    assert trace_widget.clip is True
    assert trace_widget.offset_by_median is False
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # diable clipping
    trace_norm_dialog.chk_clip_clicked(False)
    assert trace_widget.clip is False
    assert trace_widget.trace_norm_percent == 500
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        2.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 2.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -2.50001
    # reset to default
    trace_norm_dialog.reset_defaults()
    assert trace_widget.trace_norm_percent == 100
    assert trace_widget.clip is True
    assert trace_widget.offset_by_median is True
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # initially no neurons are selected
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
    trace_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice == slice(1, 3)
    assert trace_widget.selected_time_slice == slice(10, 20)
    # de-selecting neurons reverts to displaying all of them as selected
    trace_widget.unselect_neurons()
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
    trace_widget.close()


def test_trace_widget_gridlines_use_controller_limits(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 5
    n_timestamps = 50
    dt = 0.001
    data = TraceData(
        name="traces",
        traces=np.random.rand(n_neurons, n_timestamps).astype(np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=dt,
        units="mV",
    )
    controller.set_limits(max_time_secs=1.0, max_neurons=n_neurons)
    trace_widget.set_data(data)
    assert trace_widget.grid_lines is not None
    assert trace_widget.grid_lines.time_ms == pytest.approx(controller.max_width)
    positions = trace_widget.grid_lines.gridline_10samples.geometry.positions.data
    assert np.max(positions[:, 0]) == pytest.approx(controller.max_width)
    trace_widget.close()


def test_trace_widget_gridlines_use_controller_neuron_limit(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 3
    n_timestamps = 20
    dt = 0.001
    data = TraceData(
        name="traces",
        traces=np.random.rand(n_neurons, n_timestamps).astype(np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=dt,
        units="mV",
    )
    controller.set_limits(max_time_secs=0.1, max_neurons=10)
    trace_widget.set_data(data)
    assert trace_widget.grid_lines is not None
    assert trace_widget.grid_lines.n_neurons == 10
    positions = trace_widget.grid_lines.gridline_10samples.geometry.positions.data
    assert np.max(positions[:, 1]) == pytest.approx(controller.max_height - 0.5)
    trace_widget.close()


def test_trace_widget_gridlines_follow_trace_dt(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    dt = 0.0991
    data = TraceData(
        name="traces",
        traces=np.random.rand(3, 20).astype(np.float32),
        indices=np.arange(3, dtype=np.uint32),
        dt=dt,
        units="mV",
    )

    trace_widget.set_data(data)

    assert trace_widget.grid_lines is not None
    assert trace_widget.grid_lines.dt_ms == pytest.approx(99.1)
    positions = trace_widget.grid_lines.gridline_1sample.geometry.positions.data[
        0::2, 0
    ]
    assert positions[1] - positions[0] == pytest.approx(99.1)
    trace_widget.close()


def test_trace_widget_set_grid_visible(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    data = TraceData(
        name="traces",
        traces=np.random.rand(4, 20).astype(np.float32),
        indices=np.arange(4, dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    trace_widget.set_data(data)
    assert trace_widget.grid_lines is not None
    assert trace_widget.grid_lines.visible
    trace_widget.set_grid_visible(False)
    assert not trace_widget.show_grid
    assert trace_widget.grid_lines is not None
    assert not trace_widget.grid_lines.visible
    trace_widget.set_data(data)
    assert trace_widget.grid_lines is not None
    assert not trace_widget.grid_lines.visible
    trace_widget.set_grid_visible(True)
    assert trace_widget.grid_lines is not None
    assert trace_widget.grid_lines.visible
    trace_widget.close()


def test_trace_widget_shrinks_line_thickness_when_zoomed_out(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    n_neurons = 300
    data = TraceData(
        name="traces",
        traces=np.random.rand(n_neurons, 20).astype(np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    controller.set_limits(max_time_secs=1.0, max_neurons=n_neurons)
    trace_widget.set_data(data)
    # pin the render area so the test doesn't depend on real canvas resize timing
    monkeypatch.setattr(
        trace_widget.subplot.frame,
        "get_render_rect",
        lambda: (0.0, 0.0, 800.0, 600.0),
    )

    # 300 neurons in a 600px view is 2px/band; 0.7 * 2px clamps into (1, 2)
    controller.focus_neuron_range(0, n_neurons)
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.graphics[0].thickness == pytest.approx(1.4)

    # a handful of neurons have plenty of room, so thickness reverts to the default
    controller.focus_neuron_range(0, 3)
    assert trace_widget.graphic.graphics[0].thickness == pytest.approx(
        trace_widget._MAX_LINE_THICKNESS
    )

    controller.focus_neuron_range(0, n_neurons)
    assert trace_widget.graphic.graphics[0].thickness == pytest.approx(1.4)
    trace_widget.close()


def test_trace_widget_x_lock_allows_independent_and_linked_scrolling(
    qtbot: QtBot,
) -> None:
    default_controller = ConstrainedPanZoomController()
    default_x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
    default_y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    default_controller.connect_x_scroll_bar(default_x_scrollbar)
    default_controller.connect_y_scroll_bar(default_y_scrollbar)

    default_trace_widget = TraceWidget(controller=default_controller)
    qtbot.addWidget(default_trace_widget)
    default_trace_widget.show()
    qtbot.waitUntil(lambda: default_trace_widget.isVisible())

    local_controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(
        controller=local_controller,
        default_controller=default_controller,
        show_axis_link_controls=True,
    )
    trace_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())

    default_controller.set_limits(5.0, 20)
    local_controller.set_limits(5.0, 20)

    assert trace_widget.lock_x_checkbox is not None
    assert trace_widget.lock_x_checkbox.isChecked()
    assert trace_widget.x_scrollbar is not None
    assert trace_widget.y_scrollbar is not None

    default_state = default_controller._get_camera_state().copy()
    default_state["width"] = default_controller.max_width / 2
    default_state["position"] = (
        default_controller.max_width / 4,
        default_state["position"][1],
        default_state["position"][2],
    )
    default_controller._set_camera_state(default_state)
    default_controller._update_cameras()

    local_before = local_controller._get_camera_state().copy()
    trace_widget.lock_x_checkbox.setChecked(False)

    default_state["position"] = (
        default_controller.max_width * 0.75,
        default_state["position"][1],
        default_state["position"][2],
    )
    default_controller._set_camera_state(default_state)
    default_controller._update_cameras()

    local_after_unlock = local_controller._get_camera_state().copy()
    assert local_after_unlock["position"][0] == pytest.approx(
        local_before["position"][0]
    )

    trace_widget.lock_x_checkbox.setChecked(True)

    default_state["position"] = (
        default_controller.max_width * 0.6,
        default_state["position"][1],
        default_state["position"][2],
    )
    default_controller._set_camera_state(default_state)
    default_controller._update_cameras()

    local_after_lock = local_controller._get_camera_state().copy()
    default_after_lock = default_controller._get_camera_state().copy()
    assert local_after_lock["position"][0] == pytest.approx(
        default_after_lock["position"][0],
        abs=5.0,
    )

    trace_widget.close()
    default_trace_widget.close()


def test_zooming_the_default_controller_propagates_to_a_locked_widget(
    qtbot: QtBot,
) -> None:
    """Regression test for #564: zooming (e.g. the +/- buttons, or the mouse
    wheel) used to silently drop the new width/position for any widget
    locked to the controller being zoomed, because
    ConstrainedPanZoomController._set_camera_state() mutated the connected
    scrollbars - which synchronously and reentrantly triggers the locked
    widget's axis sync - before the new state was actually committed/pushed
    to the camera, so that reentrant read saw stale data and propagated it
    (a no-op) instead of the just-applied zoom."""
    default_controller = ConstrainedPanZoomController()
    default_x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
    default_y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    default_controller.connect_x_scroll_bar(default_x_scrollbar)
    default_controller.connect_y_scroll_bar(default_y_scrollbar)

    default_trace_widget = TraceWidget(controller=default_controller)
    qtbot.addWidget(default_trace_widget)
    default_trace_widget.show()
    qtbot.waitUntil(lambda: default_trace_widget.isVisible())

    local_controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(
        controller=local_controller,
        default_controller=default_controller,
        show_axis_link_controls=True,
    )
    trace_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())

    default_controller.set_limits(5.0, 20)
    local_controller.set_limits(5.0, 20)
    assert trace_widget.x_locked

    before_width = float(default_controller._get_camera_state()["width"])
    # Same call the X zoom-in button makes (timeseries_widget.py's
    # x_btn_zoom_in), i.e. a real zoom rather than a direct _set_camera_state.
    default_controller.zoom((0.5, 0.0), ())
    after_width = float(default_controller._get_camera_state()["width"])
    assert after_width < before_width  # sanity check: the zoom actually applied

    local_width = float(local_controller._get_camera_state()["width"])
    assert local_width == pytest.approx(after_width, rel=0.01)

    trace_widget.close()
    default_trace_widget.close()


def test_region_of_interest_select_slice_label_uses_override_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roi = RegionOfInterest(fill_color=(0.5, 0.5, 0.5, 0.2))
    captured: list[str] = []
    monkeypatch.setattr(roi.text, "set_text", captured.append)

    roi.select_slice(slice(1, 3), slice(10, 20), dt_ms=1.0, label_neuron_range=(5, 8))

    assert captured == ["Neurons 5-7\n10ms-20ms"]


def test_region_of_interest_select_slice_label_falls_back_to_slice_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roi = RegionOfInterest(fill_color=(0.5, 0.5, 0.5, 0.2))
    captured: list[str] = []
    monkeypatch.setattr(roi.text, "set_text", captured.append)

    roi.select_slice(slice(1, 3), slice(10, 20), dt_ms=1.0)

    assert captured == ["Neurons 1-2\n10ms-20ms"]


def test_trace_widget_select_neurons_translates_display_rows_when_greyed(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=5, n_timestamps=10))
    trace_widget.set_neuron_greyed(1, True)

    # display rows 1-2 now correspond to real neurons 2-3
    trace_widget.select_neurons(slice(1, 3), slice(0, 5))

    assert trace_widget.selected_neuron_slice == slice(2, 4)


def _stim_array(n_timesteps: int = 10, n_neurons: int = 3) -> StimArray:
    values = np.zeros((n_timesteps, n_neurons), dtype=np.float32)
    values[2, :] = 1.0
    return StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="mV",
        dt=0.001,
    )


def test_stim_overlay_hidden_until_enabled(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    trace_widget.set_stim_overlay_data(_stim_array())
    assert trace_widget._stim_overlay_graphic is None

    trace_widget.set_stim_overlay_enabled(True)
    assert trace_widget._stim_overlay_graphic is not None
    rendered = np.asarray(trace_widget._stim_overlay_graphic.data.value)
    assert rendered.shape[0] == 3

    trace_widget.set_stim_overlay_enabled(False)
    assert trace_widget._stim_overlay_graphic is None
    trace_widget.close()


def test_stim_overlay_compacts_with_greyed_neurons(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    trace_widget.set_stim_overlay_data(_stim_array())
    trace_widget.set_stim_overlay_enabled(True)

    trace_widget.set_neuron_greyed(1, True)

    assert trace_widget._stim_overlay_graphic is not None
    rendered = np.asarray(trace_widget._stim_overlay_graphic.data.value)
    assert rendered.shape[0] == 2  # hidden neuron's row is gone, not just blank

    trace_widget.set_neuron_greyed(1, False)
    rendered = np.asarray(trace_widget._stim_overlay_graphic.data.value)
    assert rendered.shape[0] == 3
    trace_widget.close()


def test_stim_overlay_aligns_rows_by_same_neuron_id(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    """Deferred cross-neuron pairing (issue #530): for now a trace row only
    ever overlays the stim column for that same real neuron id."""
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    stim = _stim_array()
    stim.values[:, 1] = 5.0
    trace_widget.set_stim_overlay_data(stim)
    trace_widget.set_stim_overlay_enabled(True)

    assert trace_widget._stim_overlay_graphic is not None
    rendered = np.asarray(trace_widget._stim_overlay_graphic.data.value)
    assert np.allclose(rendered[1, :], 5.0)
    assert np.allclose(rendered[0, :], stim.values[:, 0])
    trace_widget.close()


def test_stim_overlay_survives_data_refresh(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    trace_widget.set_stim_overlay_data(_stim_array())
    trace_widget.set_stim_overlay_enabled(True)
    assert trace_widget._stim_overlay_graphic is not None

    # re-setting trace data (e.g. selecting a different trace) tears down and
    # rebuilds every graphic - the overlay setting must survive that refresh
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))

    assert trace_widget._stim_overlay_graphic is not None
    trace_widget.close()


def test_stim_overlay_cleared_when_trace_data_cleared(
    qtbot: QtBot,
    make_trace_data,
) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    trace_widget.set_stim_overlay_data(_stim_array())
    trace_widget.set_stim_overlay_enabled(True)
    assert trace_widget._stim_overlay_graphic is not None

    trace_widget.set_data(None)

    assert trace_widget._stim_overlay_graphic is None
    trace_widget.close()


def test_calling_clear_data_directly_with_greyed_neuron_and_overlay_does_not_warn(
    qtbot: QtBot,
    make_trace_data,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: clear_data() used to null the stim overlay graphic
    reference *after* calling super().clear_data(), which wipes the whole
    subplot and, at its own tail end, calls clear_greyed_neurons() - if a
    neuron was already greyed, that re-enters _rebuild_stim_overlay()
    *before* the null happened, so it tried to remove a graphic the subplot
    wipe had already discarded (pygfx logs a warning for this, it doesn't
    raise, so `contextlib.suppress` around the removal call didn't help).

    set_data() itself resets the greyed set to empty before clear_data()
    even runs, so it never reproduced this - gui.py's update_project() calls
    clear_data() directly (`for widget in self.plot_widgets: widget.clear_data()`)
    without going through set_data() first, which is what actually hit it.
    """
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.set_data(make_trace_data(n_neurons=3, n_timestamps=10))
    trace_widget.set_stim_overlay_data(_stim_array())
    trace_widget.set_stim_overlay_enabled(True)
    trace_widget.set_neuron_greyed(1, True)

    with caplog.at_level("WARNING"):
        trace_widget.clear_data()

    assert "not a child" not in caplog.text
    trace_widget.close()


def test_close_fully_disconnects_from_shared_state(qtbot: QtBot) -> None:
    """Regression: repeated detach/close cycles of a view sharing state left
    stale connections behind (disconnecting by re-passing the callable
    doesn't sever a connection reliably in every case - only disconnecting
    the exact handle .connect() returned does), eventually invoking a
    slot on a closed, GPU-resource-less view and crashing."""
    state = TraceWidgetState()
    controller_a = ConstrainedPanZoomController()
    controller_b = ConstrainedPanZoomController()
    view_a = TraceWidget(controller_a, state=state)
    qtbot.addWidget(view_a)
    baseline_data = state.receivers(state.data_changed)
    baseline_greyed = state.receivers(state.greyed_changed)

    view_b = TraceWidget(controller_b, state=state)
    assert state.receivers(state.data_changed) == baseline_data + 1
    assert state.receivers(state.greyed_changed) == baseline_greyed + 1

    view_b.close()

    assert state.receivers(state.data_changed) == baseline_data
    assert state.receivers(state.greyed_changed) == baseline_greyed
    view_a.close()
