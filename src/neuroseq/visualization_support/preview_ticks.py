"""Neuron-id y-axis ticks for the matplotlib vector previews of the
timeseries panels (traces, spikes, stimuli)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import Locator
from matplotlib.ticker import MaxNLocator


class _RowCenterLocator(Locator):
    """Tick at the centers of a subset of rows, picked like MaxNLocator
    picks integers, for rows of differing heights."""

    def __init__(self, row_centers: Sequence[float]) -> None:
        self._centers = np.asarray(row_centers, dtype=float)
        self._rows = MaxNLocator(integer=True)

    def __call__(self) -> Sequence[float]:
        assert self.axis is not None
        low, high = sorted(self.axis.get_view_interval())
        inside = np.flatnonzero((self._centers >= low) & (self._centers <= high))
        if inside.size == 0:
            return []
        first, last = int(inside[0]), int(inside[-1])
        rows = [
            int(row)
            for row in self._rows.tick_values(first, last)
            if first <= row <= last
        ]
        return self._centers[rows].tolist()


def set_neuron_id_ticks(
    axis: Axis,
    visible_neuron_ids: Sequence[int],
    row_centers: Sequence[float] | None = None,
) -> None:
    """Label ``axis`` with the neuron id drawn at each display row.

    Rows are one unit apart (row i centred at y=i) unless ``row_centers``
    gives each row's center, as for auto-extended trace rows.
    """
    ids = list(visible_neuron_ids)
    if row_centers:
        centers = np.asarray(row_centers, dtype=float)
        axis.set_major_locator(_RowCenterLocator(centers))

        def row_at(y: float) -> int:
            row = int(np.argmin(np.abs(centers - y)))
            return row if np.isclose(centers[row], y) else -1

    else:
        axis.set_major_locator(MaxNLocator(integer=True))

        def row_at(y: float) -> int:
            return round(y)

    def label(y: float, _pos) -> str:
        row = row_at(y)
        return str(ids[row]) if 0 <= row < len(ids) else ""

    axis.set_major_formatter(FuncFormatter(label))
