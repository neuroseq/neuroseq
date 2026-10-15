from __future__ import annotations

import contextlib
from collections.abc import Callable

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
        # A focus request (e.g. the live Design dynamic input view) may pin the
        # scrollable extent wider than the current data so the view keeps a
        # stable width while blocks are appended. This intentionally persists
        # across set_max_extents() and only clears on reset() (project load);
        # the trade-off is that the camera can scroll into empty space until
        # then. See test_focus_time_range_can_include_empty_future_space.
        self._minimum_time_extent = 0.0
        self.initialized = False
        self._updating_scroll_bars = False
        self.x_scroll_bar: QtWidgets.QScrollBar | None = None
        self.y_scroll_bar: QtWidgets.QScrollBar | None = None
        self.grid_lines: list[GridLines] = []
        self._state_change_listeners: list[Callable[[], None]] = []
        super().__init__(*args, **kwargs)

    def add_state_change_listener(self, listener: Callable[[], None]) -> None:
        if listener not in self._state_change_listeners:
            self._state_change_listeners.append(listener)

    def remove_state_change_listener(self, listener: Callable[[], None]) -> None:
        with contextlib.suppress(ValueError):
            self._state_change_listeners.remove(listener)

    def _notify_state_change(self) -> None:
        for listener in list(self._state_change_listeners):
            with contextlib.suppress(Exception):
                listener()

    def reset(self):
        self.max_width = 1.0
        self.max_height = 1.0
        self._minimum_time_extent = 0.0
        self.grid_lines.clear()
        self.initialized = False

    def set_limits(self, max_time_secs: float, max_neurons: int) -> None:
        max_time_secs = float(max_time_secs)
        max_neurons = int(max_neurons)
        if max_time_secs <= 0.0 or max_neurons <= 0:
            return
        self.max_width = max(1000.0 * max_time_secs, self._minimum_time_extent)
        self.max_height = float(max_neurons)
        for grid_line in self.grid_lines:
            if grid_line is not None:
                grid_line.update_time_extent(self.max_width)
                grid_line.update_neuron_extent(int(self.max_height))
        try:
            state = self._get_camera_state()
        except ValueError:
            # Limits can be configured before any camera is attached.
            return
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
        # _set_camera_state() already pushes the new state to the camera
        # itself (see its docstring) - an extra _update_cameras() here would
        # just be a redundant, wasted re-render on every call.
        self._set_camera_state(state)

    def focus_neuron_range(self, start_index: int, stop_index: int) -> None:
        start_index = int(start_index)
        stop_index = int(stop_index)
        if stop_index <= start_index:
            return
        state = self._get_camera_state()
        visible_neurons = float(
            np.clip(
                stop_index - start_index,
                float(self.min_height),
                float(self.max_height),
            )
        )
        state["height"] = visible_neurons
        state["position"] = (
            float(state["position"][0]),
            float(start_index) + 0.5 * visible_neurons - 0.5,
            float(state["position"][2]),
        )
        # See set_limits() - _set_camera_state() already pushes to the camera.
        self._set_camera_state(state)

    def focus_time_range(self, start_seconds: float, stop_seconds: float) -> None:
        """Move the shared plot camera to a time interval, in seconds."""
        start_ms = 1000.0 * float(start_seconds)
        stop_ms = 1000.0 * float(stop_seconds)
        if stop_ms <= start_ms:
            return
        # Explicit focus requests may intentionally include empty timeline space,
        # for example to keep a live view at a stable width while data is appended.
        requested_extent = max(stop_ms - min(0.0, start_ms), stop_ms)
        self._minimum_time_extent = max(self._minimum_time_extent, requested_extent)
        self.max_width = max(self.max_width, self._minimum_time_extent)
        for grid_line in self.grid_lines:
            if grid_line is not None:
                grid_line.update_time_extent(self.max_width)
        state = self._get_camera_state()
        visible_time = float(
            np.clip(stop_ms - start_ms, float(self.min_width), float(self.max_width))
        )
        state["width"] = visible_time
        state["position"] = (
            start_ms + 0.5 * visible_time,
            float(state["position"][1]),
            float(state["position"][2]),
        )
        # See set_limits() - _set_camera_state() already pushes to the camera.
        self._set_camera_state(state)

    def current_time_range(self) -> tuple[float, float] | None:
        """Return the visible time interval in seconds, if a camera is attached."""
        try:
            state = self._get_camera_state()
        except ValueError:
            return None
        half_width = 0.5 * float(state["width"])
        center = float(state["position"][0])
        return ((center - half_width) / 1000.0, (center + half_width) / 1000.0)

    def vertical_view(self) -> tuple[float, float] | None:
        """Return the visible height and the camera's y position, if a camera
        is attached."""
        try:
            state = self._get_camera_state()
        except ValueError:
            return None
        return float(state["height"]), float(state["position"][1])

    def set_vertical_view(self, view: tuple[float, float]) -> None:
        """Restore a view returned by vertical_view(), keeping the time axis."""
        try:
            state = self._get_camera_state()
        except ValueError:
            return
        height, y = view
        self._set_camera_state(
            {
                "height": height,
                "position": (
                    float(state["position"][0]),
                    y,
                    float(state["position"][2]),
                ),
            }
        )

    def _set_camera_state(self, new_state):
        """Override the _set_camera_state method
        - limit the width, height and position to prevent scrolling or zooming beyond the data
        - update the scroll bars if they are connected
        - hide or show sub-gridlines based on the current zoom level
        - Clamps the width/height and position.
        - Pushes the result to the actual camera (calls _update_cameras()
          itself - see below) - callers should NOT call _update_cameras()
          again afterward, that would just re-render for nothing.
        """
        current_state = self._get_camera_state()
        min_width = float(self.min_width)
        max_width = float(self.max_width)
        min_height = float(self.min_height)
        max_height = float(self.max_height)
        # width constraints
        if "width" in new_state:
            new_state["width"] = float(
                np.clip(
                    float(new_state["width"]),
                    min_width,
                    max_width,
                )
            )
            for grid_line in self.grid_lines:
                if grid_line is not None:
                    grid_line.set_current_time_extent(new_state["width"])
        else:
            new_state["width"] = float(current_state["width"])
        # height constraints
        if "height" in new_state:
            new_state["height"] = float(
                np.clip(
                    float(new_state["height"]),
                    min_height,
                    max_height,
                )
            )
        else:
            new_state["height"] = float(current_state["height"])
        # position constraints
        if "position" not in new_state:
            new_state["position"] = current_state["position"]
        pos_x = float(new_state["position"][0])
        pos_y = float(new_state["position"][1])
        pos_z = float(new_state["position"][2])
        half_width = 0.5 * new_state["width"]
        half_height = 0.5 * new_state["height"]
        x = float(np.clip(pos_x, half_width, max_width - half_width))
        y = float(
            np.clip(
                pos_y,
                half_height - 0.5,
                max_height - half_height - 0.5,
            )
        )
        new_state["position"] = (x, y, pos_z)
        # Commit the clamped state - and push it all the way to the real
        # camera - *before* touching the scrollbars below.
        # QScrollBar.setValue()/setPageStep()/setMaximum() emit valueChanged/
        # rangeChanged synchronously, and any axis locked to this controller
        # (TimeseriesWidget/DataWidget's _sync_axis) reacts to that signal by
        # reentrantly reading this very controller's state back via
        # _get_camera_state(). Outside of an active pygfx gesture (e.g. a
        # scrollbar drag, or this method being called directly) that read
        # goes straight to the camera, not the pending state dict, so it
        # only sees the new value once _update_cameras() has actually run.
        # Mutating the scrollbars first - before either had happened - meant
        # that reentrant read saw stale pre-update data, so a locked axis
        # silently kept the previous width/position instead of picking up
        # the zoom/pan that was just applied (issue #564).
        super()._set_camera_state(new_state)
        self._update_cameras()
        # The scrollbars are integer-valued (``scroll_bar_steps`` steps over the whole
        # extent). Writing the camera position into them emits valueChanged, which
        # must not feed the rounded value back into the camera: when zoomed in a
        # wheel step is far below one scrollbar step, so that round trip snapped the
        # view back against the scroll direction.
        self._updating_scroll_bars = True
        try:
            self._set_scroll_bar_size(self.x_scroll_bar, new_state["width"] / max_width)
            self._set_scroll_bar_size(
                self.y_scroll_bar, new_state["height"] / max_height
            )
            self._set_scroll_bar_position(
                self.x_scroll_bar, (x - half_width) / max_width
            )
            self._set_scroll_bar_position(
                self.y_scroll_bar, 1 - (y + 0.5 + half_height) / max_height
            )
        finally:
            self._updating_scroll_bars = False
        self._notify_state_change()

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
        current_state = self._get_camera_state()
        if "width" in current_state:
            gridlines.set_current_time_extent(current_state["width"])

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
        if self._updating_scroll_bars:
            return
        state = self._get_camera_state()
        width = float(state["width"])
        max_width = float(self.max_width)
        half_width = 0.5 * width
        # get fractional position of lower edge of the scroll bar
        fractional_pos = value / self.scroll_bar_steps
        # convert to center position of the new view
        pos = fractional_pos * max_width + half_width
        # clip to valid range to account for rounding errors due to integer scroll bar values
        x = float(np.clip(pos, half_width, max_width - half_width))
        state["position"] = (
            x,
            float(state["position"][1]),
            float(state["position"][2]),
        )
        super()._set_camera_state(state)
        self._update_cameras()
        self._notify_state_change()

    def y_scroll_event(self, value: int):
        assert self.y_scroll_bar is not None
        if self._updating_scroll_bars:
            return
        state = self._get_camera_state()
        height = float(state["height"])
        max_height = float(self.max_height)
        half_height = 0.5 * height
        # get fractional position of lower edge of the scroll bar (y scroll bar is inverted so subtract from max)
        fractional_pos = (self.y_scroll_bar.maximum() - value) / self.scroll_bar_steps
        # convert to center position of the new view (subtract 0.5 to account for 0.5 offset of neuron positions)
        pos = fractional_pos * max_height + half_height - 0.5
        # clip to valid range to account for rounding errors due to integer scroll bar values
        y = float(
            np.clip(
                pos,
                half_height - 0.5,
                max_height - half_height - 0.5,
            )
        )
        state["position"] = (
            float(state["position"][0]),
            y,
            float(state["position"][2]),
        )
        super()._set_camera_state(state)
        self._update_cameras()
        self._notify_state_change()
