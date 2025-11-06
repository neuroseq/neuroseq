from collections.abc import Sequence

import pygfx
from fastplotlib import Graphic
from fastplotlib.utils import mesh_masks


class RegionOfInterest(Graphic):
    def __init__(self, fill_color: Sequence[float]):
        """Create a rectangle that can be used to highlight a selection in the plot."""
        super().__init__()

        group = pygfx.Group()
        self.fill = pygfx.Mesh(
            pygfx.box_geometry(100, 10, 1),
            pygfx.MeshBasicMaterial(
                color=pygfx.Color(pygfx.Color(fill_color)), pick_write=True
            ),
        )
        self.fill.world.position = (0, 0, 200)
        group.add(self.fill)
        self.text = pygfx.Text(screen_space=True)
        self.text.world.position = (0, 0, 200)
        self.text.material.color = pygfx.Color(0, 0, 0)
        group.add(self.text)
        self._set_world_object(group)

    def select(self, x0: float, x1: float, y0: float, y1: float):
        # update the rectangle's position based on the new selection
        self.fill.geometry.positions.data[mesh_masks.x_left] = x0
        self.fill.geometry.positions.data[mesh_masks.x_right] = x1
        self.fill.geometry.positions.data[mesh_masks.y_bottom] = y0
        self.fill.geometry.positions.data[mesh_masks.y_top] = y1
        # send the changes to the GPU
        self.fill.geometry.positions.update_range()

    def select_slice(self, neuron_index_slice: slice, time_index_slice: slice):
        neuron_text = (
            f"Neurons {neuron_index_slice.start}-{neuron_index_slice.stop - 1}"
            if neuron_index_slice.stop > neuron_index_slice.start + 1
            else f"Neuron {neuron_index_slice.start}"
        )
        self.text.set_text(
            f"{neuron_text}\n{time_index_slice.start}ms-{time_index_slice.stop}ms"
        )
        self.text.world.position = (
            (time_index_slice.stop + time_index_slice.start) / 2,
            (neuron_index_slice.stop + neuron_index_slice.start) / 2 - 0.5,
            200,
        )
        self.select(
            time_index_slice.start,
            time_index_slice.stop,
            neuron_index_slice.start - 0.5,
            neuron_index_slice.stop - 0.5,
        )
