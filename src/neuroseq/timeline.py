import pygfx
from fastplotlib import Graphic


class Timeline(Graphic):
    def __init__(self):
        """Create a vertical line that displays the current time in a timeseries plot."""
        super().__init__()

        self._t = 0.0
        group = pygfx.Group()
        self.line = pygfx.Line(
            geometry=pygfx.Geometry(positions=[[0, 0, 0], [0, 1e6, 0]]),
            material=pygfx.LineMaterial(thickness=2.0, color=pygfx.Color(1, 0, 0, 0.7)),
        )
        self.line.world.position = (0, self._t, 200)
        group.add(self.line)
        self._set_world_object(group)

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, t: float):
        self._t = t
        self.line.world.position = (self._t, -1, 200)
