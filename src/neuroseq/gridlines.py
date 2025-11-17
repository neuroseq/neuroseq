import numpy as np
import pygfx
import pylinalg
from fastplotlib import Graphic

DTYPE = "f4"


def generate_gridlines(
    n_neurons: int, time_ms: float, dt_ms: float, thickness: float
) -> pygfx.Line:
    """Generate vertical gridlines for a timeseries plot."""
    n_x = int(time_ms / dt_ms) + 1
    coords = np.zeros((2 * n_x, 3), dtype=DTYPE)

    # each line segment is represented by two consecutive points in 3D space
    x_coords = np.linspace(0, (n_x - 1) * dt_ms, num=n_x, dtype=DTYPE)
    coords[0 : 2 * n_x : 2, 0] = x_coords
    coords[1 : 2 * n_x : 2, 0] = x_coords
    coords[0 : 2 * n_x : 2, 1] = -0.5
    coords[1 : 2 * n_x : 2, 1] = n_neurons - 0.5
    coords[:, 2] = 199.0

    color = pygfx.Color((0.6, 0.6, 0.6, 0.3))
    geometry = pygfx.Geometry(positions=coords)
    material = pygfx.LineSegmentMaterial(thickness=thickness, aa=True, color=color)
    return pygfx.Line(geometry, material)


def generate_stripes(n_neurons: int, time_ms: float) -> pygfx.InstancedMesh:
    """Generate horizontal zebra-stripes for a timeseries plot."""
    color = pygfx.Color((0.0, 0.0, 0.0, 0.05))
    material = pygfx.MeshBasicMaterial(color=color)
    geometry = pygfx.plane_geometry(time_ms, 1)
    n_stripes = n_neurons // 2
    # create an instanced mesh with an instance for each stripe
    rect = pygfx.InstancedMesh(geometry, material, n_stripes)
    # translate each rectangle to the correct position
    for i_neuron in range(n_stripes):
        rect.set_matrix_at(
            i_neuron, pylinalg.mat_from_translation((0, i_neuron * 2, 0))
        )
    rect.world.position = [0.5 * time_ms, 0, 199.0]
    return rect


class GridLines(Graphic):
    def __init__(self, n_neurons: int, time_ms: float):
        """Create gridlines for a timeseries plot."""
        super().__init__()
        group = pygfx.Group()
        group.add(generate_stripes(n_neurons, time_ms))
        self.gridline_1ms = generate_gridlines(n_neurons, time_ms, 1, 0.2)
        group.add(self.gridline_1ms)
        self.gridline_10ms = generate_gridlines(n_neurons, time_ms, 10, 1)
        group.add(self.gridline_10ms)
        self.gridline_100ms = generate_gridlines(n_neurons, time_ms, 100, 2)
        group.add(self.gridline_100ms)
        self.gridline_1000ms = generate_gridlines(n_neurons, time_ms, 1000, 3)
        group.add(self.gridline_1000ms)
        self._set_world_object(group)
        self.set_current_time_extent(time_ms)

    def set_current_time_extent(self, time_ms: float):
        """Set visibility of gridlines based on current time extent."""
        self.gridline_1ms.visible = time_ms < 250.0
        self.gridline_10ms.visible = time_ms < 2500.0
        self.gridline_100ms.visible = time_ms < 25000.0
        self.gridline_1000ms.visible = time_ms < 250000.0
