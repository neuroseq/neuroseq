from __future__ import annotations

from typing import Any

import brian2 as b2
import pygfx

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController


def test_set_limits_casts_quantity_to_float(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    dummy_state = {
        "width": 1.0,
        "height": 1.0,
        "position": (0.5, 0.0, 1.0),
    }

    monkeypatch.setattr(controller, "_get_camera_state", lambda: dummy_state.copy())
    monkeypatch.setattr(controller, "_set_camera_state", lambda state: None)
    monkeypatch.setattr(controller, "_update_cameras", lambda: None)

    controller.set_limits(1.0 * b2.second, 10)

    assert isinstance(controller.max_width, float)
    assert isinstance(controller.max_height, float)
    assert controller.max_width == 1000.0
    assert controller.max_height == 10.0


def test_set_camera_state_accepts_quantity_bounds(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    controller.min_width = 1.0
    controller.min_height = 1.0
    controller.max_width = 1.0 * b2.ksecond
    controller.max_height = 10.0

    dummy_state = {
        "width": 1000.0,
        "height": 10.0,
        "position": (500.0, 4.5, 1.0),
    }
    monkeypatch.setattr(controller, "_get_camera_state", lambda: dummy_state.copy())
    monkeypatch.setattr(
        pygfx.PanZoomController, "_set_camera_state", lambda *args: None
    )

    controller._set_camera_state(dummy_state.copy())


def test_focus_neuron_range_updates_height_and_y_position(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    controller.min_height = 1.0
    controller.max_height = 50.0
    dummy_state = {
        "width": 1000.0,
        "height": 50.0,
        "position": (500.0, 24.5, 1.0),
    }
    captured_state = {}

    monkeypatch.setattr(controller, "_get_camera_state", lambda: dummy_state.copy())
    monkeypatch.setattr(
        controller,
        "_set_camera_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(controller, "_update_cameras", lambda: None)

    controller.focus_neuron_range(0, 1)

    assert captured_state["height"] == 1.0
    assert captured_state["position"] == (500.0, 0.0, 1.0)


def test_focus_time_range_updates_width_and_x_position(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    controller.min_width = 1.0
    controller.max_width = 10_000.0
    dummy_state = {
        "width": 1000.0,
        "height": 10.0,
        "position": (500.0, 4.5, 1.0),
    }
    captured_state = {}
    monkeypatch.setattr(controller, "_get_camera_state", lambda: dummy_state.copy())
    monkeypatch.setattr(
        controller,
        "_set_camera_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(controller, "_update_cameras", lambda: None)

    controller.focus_time_range(2.0, 2.25)

    assert captured_state["width"] == 250.0
    assert captured_state["position"] == (2125.0, 4.5, 1.0)


def test_focus_time_range_can_include_empty_future_space(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    controller.min_width = 1.0
    controller.max_width = 100.0
    dummy_state = {
        "width": 100.0,
        "height": 10.0,
        "position": (50.0, 4.5, 1.0),
    }
    captured_state = {}
    monkeypatch.setattr(controller, "_get_camera_state", lambda: dummy_state.copy())
    monkeypatch.setattr(
        controller,
        "_set_camera_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(controller, "_update_cameras", lambda: None)

    controller.focus_time_range(0.0, 0.3)

    assert controller.max_width == 300.0
    assert captured_state["width"] == 300.0
    assert captured_state["position"] == (150.0, 4.5, 1.0)

    controller.set_limits(0.1, 10)

    assert controller.max_width == 300.0


def test_current_time_range_reports_camera_interval(monkeypatch) -> None:
    controller = ConstrainedPanZoomController()
    monkeypatch.setattr(
        controller,
        "_get_camera_state",
        lambda: {
            "width": 300.0,
            "height": 10.0,
            "position": (250.0, 4.5, 1.0),
        },
    )

    assert controller.current_time_range() == (0.1, 0.4)


def test_small_pan_steps_are_not_snapped_back_by_the_scroll_bar(
    qtbot, monkeypatch
) -> None:
    from qtpy import QtCore
    from qtpy import QtWidgets

    controller = ConstrainedPanZoomController()
    controller.min_height = 1.0
    controller.max_height = 1000.0
    controller.min_width = 1.0
    controller.max_width = 1000.0
    scroll_bar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
    qtbot.addWidget(scroll_bar)
    controller.connect_y_scroll_bar(scroll_bar)

    # Stand-in camera: the state the controller reads back is the one it last set.
    camera: dict[str, Any] = {
        "width": 1000.0,
        "height": 10.0,
        "position": (500.0, 500.0, 1.0),
    }
    monkeypatch.setattr(controller, "_get_camera_state", lambda: dict(camera))
    monkeypatch.setattr(
        pygfx.PanZoomController,
        "_set_camera_state",
        lambda self, state: camera.update(state),
    )
    monkeypatch.setattr(controller, "_update_cameras", lambda: None)

    controller._set_camera_state(dict(camera))
    previous_y = camera["position"][1]
    # Scrolling down moves the view by 0.2 rows per step; the scroll bar only has
    # ~1 step per row at this zoom, so rounding must not pull the view back.
    for _ in range(40):
        state = dict(camera)
        state["position"] = (500.0, camera["position"][1] - 0.2, 1.0)
        controller._set_camera_state(state)
        y = camera["position"][1]
        assert y < previous_y
        previous_y = y
