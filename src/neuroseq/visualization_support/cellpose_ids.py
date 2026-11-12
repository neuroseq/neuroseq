"""Widget-free helpers for Cellpose cell ids, stimulus ids and centroids."""

from __future__ import annotations

from collections.abc import Iterable

import brian2.numpy_ as np
from scipy import ndimage

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping


def cellpose_to_stim_ids(
    cellpose_ids: Iterable[int], cellpose_data: CellposeImportResult | None
) -> dict[int, int]:
    """Neuron (stimulus) id of each non-negative Cellpose id that drives one.

    Uses the mapping in effect (``resolve_stimulus_to_cellpose_mapping``), the
    one the trace importers and footprint colours use, so a Linear offset
    mapping, which is never stored as a dict, is honoured too. Without a
    mapping Cellpose ids are neuron ids. With one, a cell the mapping leaves
    out drives no neuron and is omitted: its number is not a neuron id and
    could alias an unrelated neuron. The mapping is resolved once for all ids.
    """
    ids = {int(i) for i in cellpose_ids if int(i) >= 0}
    mapping = resolve_stimulus_to_cellpose_mapping(cellpose_data)
    if not mapping:
        return {cell: cell for cell in ids}
    to_stim = {int(cell): int(stim) for stim, cell in mapping.items()}
    return {cell: to_stim[cell] for cell in ids if cell in to_stim}


def cellpose_to_stim_id(
    cellpose_id: int, cellpose_data: CellposeImportResult | None
) -> int | None:
    """See :func:`cellpose_to_stim_ids`; ``None`` if the cell drives no neuron."""
    return cellpose_to_stim_ids([cellpose_id], cellpose_data).get(int(cellpose_id))


def stim_id_to_cellpose_id(
    stim_id: int | None, cellpose_data: CellposeImportResult | None
) -> int | None:
    """The Cellpose cell driving neuron ``stim_id`` (the inverse of
    :func:`cellpose_to_stim_id`), or ``None`` if no cell does."""
    if stim_id is None:
        return None
    mapping = resolve_stimulus_to_cellpose_mapping(cellpose_data)
    if not mapping:
        return stim_id
    cell = mapping.get(int(stim_id))
    return None if cell is None else int(cell)


def compute_cell_centroids(
    segmentation: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Return cell centroids as (column, row) image/world coordinates."""
    centroids: dict[int, tuple[float, float]] = {}
    for cell_id in np.unique(segmentation):
        if cell_id < 0:
            continue
        row, col = ndimage.center_of_mass(segmentation == cell_id)
        centroids[int(cell_id)] = (float(col), float(row))
    return centroids
