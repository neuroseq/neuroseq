from __future__ import annotations

import pyqtgraph as pg


class Timeline(pg.InfiniteLine):
    """A timeline GraphicsObject that can be moved by the user."""

    def __init__(self) -> None:
        super().__init__(
            pos=0, angle=90, movable=True, pen={"color": (200, 200, 100), "width": 3}
        )
        self.addMarker("v", 1.0)
        self.addMarker("^", 0.0)
