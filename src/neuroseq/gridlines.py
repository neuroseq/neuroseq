import numpy as np
import pygfx
from fastplotlib import Graphic

DTYPE = "f4"


def generate_gridlines(
    n_neurons: int, time: float, dt: float, draw_neuron_lines: bool
) -> np.ndarray:
    """Generate gridlines for a timeseries plot.

    Returns an array of shape (2*N, 3) where N is the number of line segments,
    and each line segment is represented by two consecutive points in 3D space.
    """
    n_x = int(time / dt) + 1
    n_y = n_neurons + 1 if draw_neuron_lines else 0
    coords = np.zeros((2 * (n_x + n_y), 3), dtype=DTYPE)

    # time and dt are in seconds, but the plots use milliseconds
    max_timestamp = time * 1000.0
    # add vertical lines
    x_coords = np.linspace(0, max_timestamp, num=n_x, dtype=DTYPE)
    coords[0 : 2 * n_x : 2, 0] = x_coords
    coords[1 : 2 * n_x : 2, 0] = x_coords
    coords[0 : 2 * n_x : 2, 1] = -0.5
    coords[1 : 2 * n_x : 2, 1] = n_neurons

    if draw_neuron_lines:
        # add horizontal lines
        y_coords = np.linspace(-0.5, n_neurons - 0.5, num=n_y, dtype=DTYPE)
        coords[2 * n_x :: 2, 1] = y_coords
        coords[2 * n_x + 1 :: 2, 1] = y_coords
        coords[2 * n_x + 1 :: 2, 0] = max_timestamp

    # set z-coordinates
    coords[:, 2] = 199.0

    return coords


class GridLines(Graphic):
    def __init__(self, n_neurons: int, time: float):
        """Create gridlines for a timeseries plot."""
        super().__init__()
        group = pygfx.Group()
        for dt, thickness, draw_neuron_lines in [
            (0.001, 0.2, False),
            (0.01, 1, True),
            (0.1, 3, False),
        ]:
            color = pygfx.Color((0.6, 0.6, 0.6, 0.3))
            geometry = pygfx.Geometry(
                positions=generate_gridlines(n_neurons, time, dt, draw_neuron_lines)
            )
            material = pygfx.LineSegmentMaterial(
                thickness=thickness, aa=True, color=color
            )
            lines = pygfx.Line(geometry, material)
            group.add(lines)
        self._set_world_object(group)
