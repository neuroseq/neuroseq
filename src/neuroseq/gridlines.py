import numpy as np
import pygfx
import pylinalg
from fastplotlib import Graphic

DTYPE = "f4"


def generate_gridlines(
    n_neurons: int, time: float, dt: float, thickness: float
) -> pygfx.Line:
    """Generate vertical gridlines for a timeseries plot."""
    n_x = int(time / dt) + 1
    coords = np.zeros((2 * n_x, 3), dtype=DTYPE)

    # time and dt are in seconds, but the plots use milliseconds
    max_timestamp = time * 1000.0
    # each line segment is represented by two consecutive points in 3D space
    x_coords = np.linspace(0, max_timestamp, num=n_x, dtype=DTYPE)
    coords[0 : 2 * n_x : 2, 0] = x_coords
    coords[1 : 2 * n_x : 2, 0] = x_coords
    coords[0 : 2 * n_x : 2, 1] = -0.5
    coords[1 : 2 * n_x : 2, 1] = n_neurons - 0.5
    coords[:, 2] = 199.0

    color = pygfx.Color((0.6, 0.6, 0.6, 0.3))
    geometry = pygfx.Geometry(positions=coords)
    material = pygfx.LineSegmentMaterial(thickness=thickness, aa=True, color=color)
    return pygfx.Line(geometry, material)


def generate_stripes(n_neurons: int, time: float) -> pygfx.InstancedMesh:
    """Generate horizontal zebra-stripes for a timeseries plot."""
    color = pygfx.Color((0.0, 0.0, 0.4, 0.2))
    material = pygfx.MeshBasicMaterial(color=color)
    # time is in seconds, but the plots use milliseconds
    width = time * 1000.0
    geometry = pygfx.plane_geometry(width, 1)
    n_stripes = n_neurons // 2
    # create an instanced mesh with an instance for each stripe
    rect = pygfx.InstancedMesh(geometry, material, n_stripes)
    # translate each rectangle to the correct position
    for i_neuron in range(n_stripes):
        rect.set_matrix_at(
            i_neuron, pylinalg.mat_from_translation((0, i_neuron * 2, 0))
        )
    rect.world.position = [0.5 * width, 0, 199.0]
    return rect


class GridLines(Graphic):
    def __init__(self, n_neurons: int, time: float):
        """Create gridlines for a timeseries plot."""
        super().__init__()
        group = pygfx.Group()
        group.add(generate_stripes(n_neurons, time))
        for (
            dt,
            thickness,
        ) in [
            (0.001, 0.2),
            (0.01, 1),
            (0.1, 3),
        ]:
            group.add(generate_gridlines(n_neurons, time, dt, thickness))
        self._set_world_object(group)
