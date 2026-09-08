from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import TraceData
from neuroseq.region_of_interest import RegionOfInterest
from neuroseq.spike_widget import SpikeWidget
from neuroseq.stim_widget import StimWidget
from neuroseq.trace_widget import TraceNormDialog
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import neuron_indices_to_colors


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
        if action.text() == "Export as vector graphic…"
    )
    assert not export_action.isEnabled()

    trace_widget.set_data(make_trace_data())
    trace_widget._rebuild_custom_analysis_menu()
    export_action = next(
        action
        for action in trace_widget.context_menu.actions()
        if action.text() == "Export as vector graphic…"
    )
    assert export_action.isEnabled()
    with qtbot.waitSignal(trace_widget.vector_export_requested):
        export_action.trigger()


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


def test_trace_norm_dialog_is_modeless_and_reused(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)

    trace_widget.trace_norm_dialog()
    dialog = trace_widget._trace_norm_dialog

    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.isModal()

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
    data = make_trace_data(n_neurons=3, n_timestamps=10)
    original_traces = data.traces.copy()
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    assert isinstance(trace_widget.data, TraceData)

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

    trace_widget.set_neuron_greyed(1, False)
    assert lines[1].visible
    assert lines[1].offset[1] == pytest.approx(1)
    assert lines[2].offset[1] == pytest.approx(2)  # restored to its original slot
    assert trace_widget.n_neurons == 3
    trace_widget.close()


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
