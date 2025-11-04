import numpy as np
import pygfx
from fastplotlib import Graphic


class Text(Graphic):
    def __init__(self):
        """Create a text graphic to display as part of a plot."""
        super().__init__()

        group = pygfx.Group()
        self.text = pygfx.Text(screen_space=True, anchor="top-left", anchor_offset=10.0)
        self.text.world.position = (0, 0, 200)
        self.text.material.color = pygfx.Color(0, 0, 0)
        group.add(self.text)
        self._set_world_object(group)

    def set_pos(self, pos: np.ndarray):
        self.text.world.position = (pos[0], pos[1], 200)
        self.text.set_text(f"Neuron {int(pos[1] + 0.5)}\n{int(pos[0])}ms")
