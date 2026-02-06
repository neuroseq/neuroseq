import numpy as np

from neuroseq.gridlines import GridLines


def _max_x(gridlines: GridLines) -> float:
    positions = gridlines.gridline_10ms.geometry.positions.data
    return float(np.max(positions[:, 0]))


def _max_y(gridlines: GridLines) -> float:
    positions = gridlines.gridline_10ms.geometry.positions.data
    return float(np.max(positions[:, 1]))


def test_gridlines_update_time_extent() -> None:
    gridlines = GridLines(n_neurons=4, time_ms=1000.0)
    assert _max_x(gridlines) == 1000.0
    gridlines.update_time_extent(2000.0)
    assert _max_x(gridlines) == 2000.0


def test_gridlines_update_neuron_extent() -> None:
    gridlines = GridLines(n_neurons=4, time_ms=1000.0)
    assert _max_y(gridlines) == 3.5
    gridlines.update_neuron_extent(7)
    assert _max_y(gridlines) == 6.5
