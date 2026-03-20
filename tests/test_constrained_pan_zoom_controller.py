from __future__ import annotations

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
