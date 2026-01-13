from __future__ import annotations

import contextlib

import brian2.numpy_ as np
import pygfx
from qtpy import QtWidgets

from neuroseq.gridlines import GridLines


class ConstrainedPanZoomController(pygfx.PanZoomController):
    _default_controls = {
        "mouse3": ("pan", "drag", (1, 1)),  # pan with middle button drag
        "wheel": ("pan", "push", (0, -0.5)),  # pan y-axis on mouse scroll
        "control+wheel": (
            "zoom",
            "push",
            (-0.001, 0),
        ),  # zoom x-axis on mouse scroll with ctrl
        "shift+wheel": (
            "zoom",
            "push",
            (0, -0.001),
        ),  # zoom y-axis on mouse scroll with shift
    }
    scroll_bar_steps = 999

    def __init__(self, *args, **kwargs) -> None:
        self.min_width = 1.0
        self.max_width = 1.0
        self.min_height = 1.0
        self.max_height = 1.0
        self.initialized = False
        self.x_scroll_bar: QtWidgets.QScrollBar | None = None
        self.y_scroll_bar: QtWidgets.QScrollBar | None = None
        self.grid_lines: list[GridLines] = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.max_width = 1.0
        self.max_height = 1.0
        self.grid_lines.clear()
        self.initialized = False

    def set_limits(self, max_time_secs: float, max_neurons: int) -> None:
        if max_time_secs <= 0.0 or max_neurons <= 0:
            return
        self.max_width = 1000.0 * max_time_secs
        self.max_height = float(max_neurons)
        state = self._get_camera_state()
        if not self.initialized:
            # first use of controller: set initial camera state to show all data
            state["width"] = self.max_width
            state["height"] = self.max_height
            state["position"] = (0.5 * self.max_width, 0.5 * self.max_height - 0.5, 1.0)
            # ensure sufficient depth such that all graphics are visible
            state["depth"] = 1000.0
            state["zoom"] = 1.0
            # allow independent x and y zoom
            state["maintain_aspect"] = False
            state["depth_range"] = None
            self.initialized = True
        self._set_camera_state(state)
        self._update_cameras()

    def _set_camera_state(self, new_state):
        """Override the _set_camera_state method
        - limit the width, height and position to prevent scrolling or zooming beyond the data
        - update the scroll bars if they are connected
        - hide or show sub-gridlines based on the current zoom level
        - Clamps the width/height and position.
        """
        current_state = self._get_camera_state()
        # width constraints
        if "width" in new_state:
            new_state["width"] = float(
                np.clip(
                    new_state["width"],
                    self.min_width,
                    self.max_width,
                )
            )
            self._set_scroll_bar_size(
                self.x_scroll_bar, new_state["width"] / self.max_width
            )
            for grid_line in self.grid_lines:
                if grid_line is not None:
                    grid_line.set_current_time_extent(new_state["width"])
        else:
            new_state["width"] = current_state["width"]
        # height constraints
        if "height" in new_state:
            new_state["height"] = float(
                np.clip(
                    new_state["height"],
                    self.min_height,
                    self.max_height,
                )
            )
            self._set_scroll_bar_size(
                self.y_scroll_bar, new_state["height"] / self.max_height
            )
        else:
            new_state["height"] = current_state["height"]
        # position constraints
        if "position" not in new_state:
            new_state["position"] = current_state["position"]
        half_width = 0.5 * new_state["width"]
        half_height = 0.5 * new_state["height"]
        x = np.clip(new_state["position"][0], half_width, self.max_width - half_width)
        y = np.clip(
            new_state["position"][1],
            half_height - 0.5,
            self.max_height - half_height - 0.5,
        )
        new_state["position"] = (x, y, new_state["position"][2])
        self._set_scroll_bar_position(
            self.x_scroll_bar, (x - half_width) / self.max_width
        )
        self._set_scroll_bar_position(
            self.y_scroll_bar, 1 - (y + 0.5 + half_height) / self.max_height
        )
        super()._set_camera_state(new_state)

    def _set_scroll_bar_position(
        self, scroll_bar: QtWidgets.QScrollBar | None, relative_position: float
    ) -> None:
        if scroll_bar is None:
            return
        scroll_bar.setValue(int(self.scroll_bar_steps * relative_position))

    def _set_scroll_bar_size(
        self, scroll_bar: QtWidgets.QScrollBar | None, relative_size: float
    ) -> None:
        if scroll_bar is None:
            return
        old_page_step = scroll_bar.pageStep()
        old_value = scroll_bar.value()
        # rescale the scrollbar page step to the new relative size
        new_page_step = int(self.scroll_bar_steps * relative_size)
        # center the scrollbar handle on the same position as before
        new_value = int(old_value + old_page_step / 2 - new_page_step / 2)
        scroll_bar.setPageStep(new_page_step)
        scroll_bar.setMinimum(0)
        scroll_bar.setMaximum(self.scroll_bar_steps - new_page_step)
        scroll_bar.setValue(new_value)

    def _init_scroll_bar(self, scroll_bar: QtWidgets.QScrollBar):
        scroll_bar.setMinimum(0)
        scroll_bar.setMaximum(0)
        scroll_bar.setValue(0)
        scroll_bar.setSingleStep(1)
        scroll_bar.setPageStep(self.scroll_bar_steps)

    def add_grid_lines(self, gridlines: GridLines):
        self.grid_lines.append(gridlines)

    def remove_grid_lines(self, gridlines: GridLines):
        if not self.grid_lines:
            return
        with contextlib.suppress(ValueError):
            self.grid_lines.remove(gridlines)

    def connect_x_scroll_bar(self, scroll_bar: QtWidgets.QScrollBar):
        self.x_scroll_bar = scroll_bar
        self._init_scroll_bar(scroll_bar)
        scroll_bar.valueChanged.connect(self.x_scroll_event)

    def connect_y_scroll_bar(self, scroll_bar: QtWidgets.QScrollBar):
        self.y_scroll_bar = scroll_bar
        self._init_scroll_bar(scroll_bar)
        scroll_bar.valueChanged.connect(self.y_scroll_event)

    def x_scroll_event(self, value: int):
        state = self._get_camera_state()
        half_width = 0.5 * state["width"]
        # get fractional position of lower edge of the scroll bar
        fractional_pos = value / self.scroll_bar_steps
        # convert to center position of the new view
        pos = fractional_pos * self.max_width + half_width
        # clip to valid range to account for rounding errors due to integer scroll bar values
        x = np.clip(pos, half_width, self.max_width - half_width)
        state["position"] = (x, state["position"][1], state["position"][2])
        super()._set_camera_state(state)
        self._update_cameras()

    def y_scroll_event(self, value: int):
        assert self.y_scroll_bar is not None
        state = self._get_camera_state()
        half_height = 0.5 * state["height"]
        # get fractional position of lower edge of the scroll bar (y scroll bar is inverted so subtract from max)
        fractional_pos = (self.y_scroll_bar.maximum() - value) / self.scroll_bar_steps
        # convert to center position of the new view (subtract 0.5 to account for 0.5 offset of neuron positions)
        pos = fractional_pos * self.max_height + half_height - 0.5
        # clip to valid range to account for rounding errors due to integer scroll bar values
        y = np.clip(pos, half_height - 0.5, self.max_height - half_height - 0.5)
        state["position"] = (state["position"][0], y, state["position"][2])
        super()._set_camera_state(state)
        self._update_cameras()
