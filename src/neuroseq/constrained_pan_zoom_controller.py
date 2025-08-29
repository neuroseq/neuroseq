from __future__ import annotations

import brian2.numpy_ as np
import pygfx


class ConstrainedPanZoomController(pygfx.PanZoomController):
    _default_controls = {
        "control+mouse1": ("pan", "drag", (1, 1)),
        "control+mouse2": ("zoom", "drag", (0.005, -0.005)),
        "wheel": ("zoom_to_point", "push", -0.001),
    }

    def __init__(self, *args, **kwargs):
        self.min_width = 1.0
        self.max_width = 1.0
        self.min_height = 1.0
        self.max_height = 1.0
        super().__init__(*args, **kwargs)

    def _set_camera_state(self, new_state):
        """Override the _set_camera_state method to limit the width, height and position to keep the data visible."""
        if "position" in new_state:
            current_state = self._get_camera_state()
            half_width = 0.5 * current_state["width"]
            half_height = 0.5 * current_state["height"]
            new_state["position"][0] = np.clip(
                new_state["position"][0], half_width, self.max_width - half_width
            )
            new_state["position"][1] = np.clip(
                new_state["position"][1],
                half_height - 0.5,
                self.max_height - half_height - 0.5,
            )
        if "width" in new_state:
            new_state["width"] = np.clip(
                new_state["width"], self.min_width, self.max_width
            )
        if "height" in new_state:
            new_state["height"] = np.clip(
                new_state["height"], self.min_height, self.max_height
            )
        super()._set_camera_state(new_state)
