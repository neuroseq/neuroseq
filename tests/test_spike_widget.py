from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.spike_widget import SpikeWidget
from neuroseq.trace_widget import TraceWidget


def test_spike_widget(qtbot: QtBot, make_spike_data) -> None:
    controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(controller=controller)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    assert spike_widget.n_neurons == 0
    spike_data = make_spike_data(n_neurons=10, n_spikes=50, max_time_secs=1.0)
    spike_widget.set_data(spike_data)
    assert spike_widget.n_neurons == max(spike_data.spike_indices) + 1
    spike_widget.close()


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
    qtbot.waitUntil(lambda: spike_widget.isVisible())
    spike_data = make_spike_data(n_neurons=10, n_spikes=50, max_time_secs=1.0)
    spike_widget.set_data(spike_data)
    assert spike_widget._all_neuron_ids == list(range(spike_widget.n_neurons))
    assert spike_widget._visible_neuron_ids == list(range(spike_widget.n_neurons))
    spike_widget.close()


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
    qtbot.waitUntil(lambda: spike_widget.isVisible())

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
    spike_widget.close()


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
    qtbot.waitUntil(lambda: default_trace_widget.isVisible())

    local_controller = ConstrainedPanZoomController()
    spike_widget = SpikeWidget(
        controller=local_controller,
        default_controller=default_controller,
        show_axis_link_controls=True,
    )
    spike_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    qtbot.addWidget(spike_widget)
    spike_widget.show()
    qtbot.waitUntil(lambda: spike_widget.isVisible())

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

    spike_widget.close()
    default_trace_widget.close()
