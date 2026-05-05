from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data_widget import DataWidget
from neuroseq.trace_widget import TraceWidget


def test_data_widget(
    qtbot: QtBot, make_trace_data, make_spike_data, make_synapse_data
) -> None:
    default_controller = ConstrainedPanZoomController()
    default_x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
    default_y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    default_controller.connect_x_scroll_bar(default_x_scrollbar)
    default_controller.connect_y_scroll_bar(default_y_scrollbar)

    data_widget = DataWidget(controller=default_controller)
    data_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    qtbot.addWidget(data_widget)
    data_widget.show()
    qtbot.waitUntil(lambda: data_widget.isVisible())
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_trace_data(n_neurons=3))
    assert data_widget.trace_widget.n_neurons == 3
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_synapse_data(n_i=6, n_j=4))
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.set_data(make_spike_data(n_neurons=5))
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons <= 5
    data_widget.clear_data()
    assert data_widget.trace_widget.n_neurons == 0
    assert data_widget.spike_widget.n_neurons == 0
    data_widget.close()


def test_data_widget_x_lock_allows_independent_and_linked_scrolling(
    qtbot: QtBot,
) -> None:
    default_controller = ConstrainedPanZoomController()
    default_x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
    default_y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    default_controller.connect_x_scroll_bar(default_x_scrollbar)
    default_controller.connect_y_scroll_bar(default_y_scrollbar)

    # Attach a camera so controller state is available in this unit test.
    default_trace_widget = TraceWidget(controller=default_controller)
    qtbot.addWidget(default_trace_widget)
    default_trace_widget.show()
    qtbot.waitUntil(lambda: default_trace_widget.isVisible())

    default_controller.set_limits(5.0, 20)

    data_widget = DataWidget(controller=default_controller)
    data_widget.set_default_scroll_bars(default_x_scrollbar, default_y_scrollbar)
    data_widget.set_limits(5.0, 20)
    qtbot.addWidget(data_widget)

    state = default_controller._get_camera_state().copy()
    state["width"] = default_controller.max_width / 2
    state["position"] = (
        default_controller.max_width / 4,
        state["position"][1],
        state["position"][2],
    )
    default_controller._set_camera_state(state)
    default_controller._update_cameras()

    initial_default_value = default_x_scrollbar.value()
    assert data_widget.x_locked

    data_widget.lock_x_checkbox.setChecked(False)
    data_widget.x_scrollbar.setValue(min(data_widget.x_scrollbar.maximum(), 200))
    assert default_x_scrollbar.value() == initial_default_value

    data_widget.lock_x_checkbox.setChecked(True)
    data_widget.x_scrollbar.setValue(min(data_widget.x_scrollbar.maximum(), 300))
    assert default_x_scrollbar.value() == data_widget.x_scrollbar.value()

    default_trace_widget.close()
