import numpy as np
import pygfx
from fastplotlib import Graphic


class Text(Graphic):
    def __init__(self):
        """Create a text graphic to display as part of a plot."""
        super().__init__()

        group = pygfx.Group()
        self.text = pygfx.Text(
            screen_space=True, anchor="middle-left", anchor_offset=20.0
        )
        self.text.world.position = (0, 0, 200)
        self.text.material.color = pygfx.Color(0, 0, 0)
        self.text.material.outline_color = pygfx.Color(1, 1, 1)
        self.text.material.outline_thickness = 0.5
        group.add(self.text)
        self._set_world_object(group)
        # Screen pixels between the cursor and the plot's right border, set by
        # the widget on each pointer move.
        self.room_right_px: float | None = None

    def set_pos(self, pos: np.ndarray, additional_text: str):
        # `additional_text` carries the neuron label itself (real id, not the raw
        # display-row position `pos[1]`) - callers know the real id after any
        # display-row translation, this class doesn't.
        text = f"{int(pos[0])}ms\n{additional_text}"
        self.text.world.position = (pos[0], pos[1], 200)
        self.text.set_text(text)
        # Near the right border the text would be cut off, so show it on the
        # left of the cursor instead (issue #602).
        fits_right = self.room_right_px is None or self.room_right_px >= self._width_px(
            text
        )
        self.text.anchor = "middle-left" if fits_right else "middle-right"

    def _width_px(self, text: str) -> float:
        # The laid-out size is only known after rendering; estimate it from
        # the longest line, a little wider than a typical glyph.
        longest = max(len(line) for line in text.split("\n"))
        return self.text.anchor_offset + 0.65 * self.text.font_size * longest
