from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.utils import neuron_indices_to_colors
from neuroseq.visualization.spike_widget import SpikeWidget
from neuroseq.visualization.trace_widget import TraceWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from tests.wait_timeout import WAIT_TIMEOUT_MS


@pytest.mark.parametrize("selected", [False, True])
def test_spike_vector_preview(qtbot: QtBot, tmp_path, selected) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    controller = ConstrainedPanZoomController()
    widget = SpikeWidget(controller)
    qtbot.addWidget(widget)
    assert widget.preview_spikes() is None
    widget._rebuild_custom_analysis_menu()
    assert not widget.context_menu.actions()[0].isEnabled()
    data = SpikeData(
        name="preview source",
        spike_indices=np.array([3, 7, 3]),
        spike_times=np.array([0.001, 0.004, 0.008]),
    )
    widget.set_data(data)
    controller.focus_time_range(0.002, 0.006)
    controller.focus_neuron_range(2, 9)
    state = controller._get_camera_state().copy()
    if selected:
        widget.select_neurons(slice(3, 4), slice(1, 5))
    widget._rebuild_custom_analysis_menu()
    action = widget.context_menu.actions()[0]
    assert action.text() == "Preview vector graphic…"
    assert action.isEnabled()
    assert "Detach panel" in [a.text() for a in widget.context_menu.actions()]
    action.trigger()
    dialog = widget.findChild(QtWidgets.QDialog)
    assert dialog is not None
    qtbot.addWidget(dialog)
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.findChild(NavigationToolbar2QT) is not None
    canvas = dialog.findChild(FigureCanvasQTAgg)
    ax = canvas.figure.axes[0]
    assert canvas.figure.get_facecolor()[3] == 0
    assert ax.get_facecolor()[3] == 0
    assert ax.get_title() == data.name
    assert ax.get_xlabel() == "Time (ms)"
    assert ax.get_ylabel() == "Neuron index"
    points = ax.collections[0]
    np.testing.assert_allclose(points.get_offsets(), [[1, 3], [4, 7], [8, 3]])
    np.testing.assert_allclose(
        points.get_facecolors(), neuron_indices_to_colors(data.spike_indices)
    )
    cx, cy = state["position"][:2]
    assert ax.get_xlim() == pytest.approx(
        (1, 5) if selected else (cx - state["width"] / 2, cx + state["width"] / 2)
    )
    assert ax.get_ylim() == pytest.approx(
        (2.5, 3.5) if selected else (cy - state["height"] / 2, cy + state["height"] / 2)
    )
    for key in ("position", "width", "height"):
        np.testing.assert_allclose(controller._get_camera_state()[key], state[key])
    widget.set_data(None)
    np.testing.assert_allclose(points.get_offsets(), [[1, 3], [4, 7], [8, 3]])
    output = tmp_path / "spikes.svg"
    canvas.figure.savefig(output)
    svg = output.read_text(encoding="utf-8")
    assert 'id="spikes"' in svg
    assert "<image" not in svg
    dialog.close()
    widget._rebuild_custom_analysis_menu()
    assert not widget.context_menu.actions()[0].isEnabled()
    widget.set_data(
        SpikeData(
            name="empty",
            spike_indices=np.array([], dtype=np.uint32),
            spike_times=np.array([], dtype=np.float32),
        )
    )
    assert widget.preview_spikes() is None


def test_spike_widget(qtbot: QtBot, make_spike_data) -> None:
    controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    assert spike_widget.n_neurons == 0
    spike_data = make_spike_data(n_neurons=10, n_spikes=50, max_time_secs=1.0)
    spike_widget.set_data(spike_data)
    assert spike_widget.n_neurons == max(spike_data.spike_indices) + 1


def test_spike_widget_populates_visible_neuron_ids_for_y_axis_labels(
    qtbot: QtBot, make_spike_data
) -> None:
    """Every row needs a real neuron id behind it so the y-axis tick
    formatter (which looks up ``_visible_neuron_ids``) can label it -
    otherwise the Spikes panel silently draws no neuron numbers at all."""
    controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    spike_data = make_spike_data(n_neurons=10, n_spikes=50, max_time_secs=1.0)
    spike_widget.set_data(spike_data)
    assert spike_widget._all_neuron_ids == list(range(spike_widget.n_neurons))
    assert spike_widget._visible_neuron_ids == list(range(spike_widget.n_neurons))


def test_spike_widget_neuron_count_falls_back_to_controller_limit(
    qtbot: QtBot,
) -> None:
    """A neuron that never spikes has no entry in spike_indices at all, so
    the highest spiking index alone would undercount it - and any trailing
    silent neurons above it. Since the shared controller already knows the
    project's true neuron count (kept in sync by Gui.update_limits()), it
    is used as a floor so this panel's rows stay aligned with the
    Traces/Stimuli panels even when some neurons never fire."""
    controller = ConstrainedPanZoomController()
    controller.set_limits(1.0, 10)
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)

    import brian2.numpy_ as np

    from neuroseq.data import SpikeData

    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 2, 4], dtype=np.uint32),
        spike_times=np.array([0.001, 0.002, 0.003], dtype=np.float32),
    )
    spike_widget.set_data(spike_data)
    assert spike_widget.n_neurons == 10
    assert spike_widget._all_neuron_ids == list(range(10))


def test_spike_widget_x_lock_allows_independent_and_linked_scrolling(
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
    qtbot.waitUntil(lambda: default_trace_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)

    local_controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(
        controller=local_controller,
        default_controller=default_controller,
        show_axis_link_controls=True,
    )
    spike_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)

    default_controller.set_limits(5.0, 20)
    local_controller.set_limits(5.0, 20)

    assert spike_widget.lock_x_checkbox is not None
    assert spike_widget.lock_x_checkbox.isChecked()
    assert spike_widget.x_scrollbar is not None
    assert spike_widget.y_scrollbar is not None

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
    spike_widget.lock_x_checkbox.setChecked(False)

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

    spike_widget.lock_x_checkbox.setChecked(True)

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


def test_spike_widget_hiding_neurons_compacts_spike_rows(qtbot: QtBot) -> None:
    widget = SpikeWidget(ConstrainedPanZoomController())
    qtbot.addWidget(widget)
    widget.set_data(
        SpikeData(
            name="s",
            spike_indices=np.array([0, 1, 2, 2]),
            spike_times=np.array([0.001, 0.002, 0.003, 0.004]),
        )
    )

    widget.set_neurons_greyed([0], [])

    assert widget.graphic is not None
    points = np.asarray(widget.graphic.data[:])
    # Neuron 0's spike is gone; neurons 1 and 2 move up to rows 0 and 1,
    # where the y tick labels now say "1" and "2".
    np.testing.assert_allclose(points[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(points[:, 1], [0.0, 1.0, 1.0])
    assert widget._visible_neuron_ids == [1, 2]

    widget.set_neurons_greyed([1, 2], [])
    assert widget.graphic is not None
    assert not widget.graphic.visible

    widget.set_neurons_greyed([], [0, 1, 2])
    points = np.asarray(widget.graphic.data[:])
    np.testing.assert_allclose(points[:, 1], [0.0, 1.0, 2.0, 2.0])
    assert widget.graphic.visible


def test_spike_widget_keeps_hidden_neurons_across_set_data(qtbot: QtBot) -> None:
    widget = SpikeWidget(ConstrainedPanZoomController())
    qtbot.addWidget(widget)
    data = SpikeData(
        name="s", spike_indices=np.array([0, 1]), spike_times=np.array([0.001, 0.002])
    )
    widget.set_data(data)
    widget.set_neurons_greyed([0], [])

    widget.set_data(data)

    assert widget.is_neuron_greyed(0)
    assert widget.graphic is not None
    np.testing.assert_allclose(np.asarray(widget.graphic.data[:])[:, 1], [0.0])


def test_empty_spike_widget_drag_selects_without_error(qtbot: QtBot) -> None:
    widget = SpikeWidget(ConstrainedPanZoomController())
    qtbot.addWidget(widget)
    widget.set_data(None)
    positions = iter([np.array([1.0, 1.0, 0.0]), np.array([5.0, 2.0, 0.0])])
    widget.subplot.map_screen_to_world = lambda ev: next(positions)
    event = SimpleNamespace(button=1, x=0.0, y=0.0)

    widget.pointer_down(event)
    widget.pointer_move(event)  # previously ZeroDivisionError (dt_ms == 0)

    assert widget.selected_time_slice == slice(1, 5)


def test_spike_preview_labels_rows_with_neuron_ids(qtbot: QtBot) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    widget = SpikeWidget(ConstrainedPanZoomController())
    qtbot.addWidget(widget)
    widget.set_data(
        SpikeData(
            name="s",
            spike_indices=np.array([0, 1, 2]),
            spike_times=np.array([0.001, 0.002, 0.003]),
        )
    )
    widget.set_neuron_greyed(1, True)

    dialog = widget.preview_spikes()
    assert dialog is not None
    qtbot.addWidget(dialog)
    ax = dialog.findChild(FigureCanvasQTAgg).figure.axes[0]
    formatter = ax.yaxis.get_major_formatter()
    assert [formatter(y, 0) for y in (0.0, 1.0, 2.0)] == ["0", "2", ""]
    dialog.close()
