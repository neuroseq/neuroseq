import numpy as np
import pygfx
import pylinalg
from fastplotlib import Graphic

DTYPE = "f4"


def _gridline_coords(n_neurons: int, time_ms: float, dt_ms: float) -> np.ndarray:
    n_x = int(time_ms / dt_ms) + 1
    coords = np.zeros((2 * n_x, 3), dtype=DTYPE)

    # each line segment is represented by two consecutive points in 3D space
    x_coords = np.linspace(0, (n_x - 1) * dt_ms, num=n_x, dtype=DTYPE)
    coords[0 : 2 * n_x : 2, 0] = x_coords
    coords[1 : 2 * n_x : 2, 0] = x_coords
    coords[0 : 2 * n_x : 2, 1] = -0.5
    coords[1 : 2 * n_x : 2, 1] = n_neurons - 0.5
    coords[:, 2] = 199.0
    return coords


def generate_gridlines(
    n_neurons: int, time_ms: float, dt_ms: float, thickness: float
) -> pygfx.Line:
    """Generate vertical gridlines for a timeseries plot."""
    coords = _gridline_coords(n_neurons, time_ms, dt_ms)

    color = pygfx.Color((0.6, 0.6, 0.6, 0.3))
    geometry = pygfx.Geometry(positions=coords)
    material = pygfx.LineSegmentMaterial(thickness=thickness, aa=True, color=color)
    return pygfx.Line(geometry, material)


def _stripe_geometry(time_ms: float) -> pygfx.Geometry:
    return pygfx.plane_geometry(time_ms, 1)


def generate_stripes(n_neurons: int, time_ms: float) -> pygfx.InstancedMesh:
    """Generate horizontal zebra-stripes for a timeseries plot."""
    color = pygfx.Color((0.0, 0.0, 0.0, 0.05))
    material = pygfx.MeshBasicMaterial(color=color)
    geometry = _stripe_geometry(time_ms)
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
        self.n_neurons = int(n_neurons)
        self.time_ms = float(time_ms)
        self._group = pygfx.Group()
        self._stripes = None
        if n_neurons > 1:
            self._stripes = generate_stripes(n_neurons, self.time_ms)
            self._group.add(self._stripes)
        self.gridline_1ms = generate_gridlines(n_neurons, self.time_ms, 1, 0.2)
        self._group.add(self.gridline_1ms)
        self.gridline_10ms = generate_gridlines(n_neurons, self.time_ms, 10, 1)
        self._group.add(self.gridline_10ms)
        self.gridline_100ms = generate_gridlines(n_neurons, self.time_ms, 100, 2)
        self._group.add(self.gridline_100ms)
        self.gridline_1000ms = generate_gridlines(n_neurons, self.time_ms, 1000, 3)
        self._group.add(self.gridline_1000ms)
        self._set_world_object(self._group)
        self.set_current_time_extent(time_ms)

    def update_time_extent(self, time_ms: float) -> None:
        if time_ms <= 0.0:
            return
        time_ms = float(time_ms)
        if time_ms == self.time_ms:
            return
        self.time_ms = time_ms
        if self._stripes is not None:
            self._stripes.geometry = _stripe_geometry(self.time_ms)
            self._stripes.world.position = [0.5 * self.time_ms, 0, 199.0]
        self.gridline_1ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 1)
        )
        self.gridline_10ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 10)
        )
        self.gridline_100ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 100)
        )
        self.gridline_1000ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 1000)
        )

    def update_neuron_extent(self, n_neurons: int) -> None:
        n_neurons = int(n_neurons)
        if n_neurons <= 0 or n_neurons == self.n_neurons:
            return
        self.n_neurons = n_neurons
        if self._stripes is not None:
            self._group.remove(self._stripes)
            self._stripes = None
        if self.n_neurons > 1:
            self._stripes = generate_stripes(self.n_neurons, self.time_ms)
            self._group.add(self._stripes)
        self.gridline_1ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 1)
        )
        self.gridline_10ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 10)
        )
        self.gridline_100ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 100)
        )
        self.gridline_1000ms.geometry = pygfx.Geometry(
            positions=_gridline_coords(self.n_neurons, self.time_ms, 1000)
        )

    def set_current_time_extent(self, time_ms: float):
        """Set visibility of gridlines based on current time extent."""
        self.gridline_1ms.visible = time_ms < 250.0
        self.gridline_10ms.visible = time_ms < 2500.0
        self.gridline_100ms.visible = time_ms < 25000.0
        self.gridline_1000ms.visible = time_ms < 250000.0
