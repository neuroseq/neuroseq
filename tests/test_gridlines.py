import brian2 as b2
import numpy as np
import pytest

from neuroseq.gridlines import GridLines


def _max_x(gridlines: GridLines) -> float:
    positions = gridlines.gridline_10samples.geometry.positions.data
    return float(np.max(positions[:, 0]))


def _max_y(gridlines: GridLines) -> float:
    positions = gridlines.gridline_10samples.geometry.positions.data
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


def test_gridlines_update_time_extent_accepts_quantity() -> None:
    gridlines = GridLines(n_neurons=4, time_ms=1000.0)
    gridlines.update_time_extent(1.27 * b2.ksecond)
    assert _max_x(gridlines) == pytest.approx(1270.0)


def test_gridlines_set_current_time_extent_accepts_quantity() -> None:
    gridlines = GridLines(n_neurons=4, time_ms=1000.0)
    gridlines.set_current_time_extent(1.0 * b2.ksecond)
    assert not gridlines.gridline_1sample.visible
    assert gridlines.gridline_10samples.visible
    assert gridlines.gridline_100samples.visible
    assert gridlines.gridline_1000samples.visible


def test_gridlines_update_dt_ms_updates_sample_spacing() -> None:
    gridlines = GridLines(n_neurons=4, time_ms=10.0)

    gridlines.update_dt_ms(0.5208125)

    positions = gridlines.gridline_1sample.geometry.positions.data[0::2, 0]
    assert positions[1] - positions[0] == pytest.approx(0.5208125)
    positions = gridlines.gridline_10samples.geometry.positions.data[0::2, 0]
    assert positions[1] - positions[0] == pytest.approx(5.208125)
