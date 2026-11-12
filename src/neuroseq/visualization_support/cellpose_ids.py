"""Widget-free helpers for Cellpose cell ids, stimulus ids and centroids."""

from __future__ import annotations

from collections.abc import Iterable

import brian2.numpy_ as np
from scipy import ndimage

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping
from neuroseq.importers.importer_suite2p import cellpose_to_stimulus_permutation


def _permutation(cellpose_data: CellposeImportResult | None) -> dict[int, int]:
    """Cellpose -> neuron id entries of the mapping in effect (identity for
    every other id); identity when the mapping cannot be resolved, which is
    what the importers and footprint colours then fall back to as well."""
    try:
        return cellpose_to_stimulus_permutation(
            resolve_stimulus_to_cellpose_mapping(cellpose_data)
        )
    except ValueError:
        return {}


def cellpose_to_stim_ids(
    cellpose_ids: Iterable[int], cellpose_data: CellposeImportResult | None
) -> dict[int, int]:
    """Neuron (stimulus) id of each non-negative Cellpose id.

    Resolved exactly as the trace importers and the footprint colours resolve
    it (``cellpose_to_stimulus_permutation`` over the mapping in effect, so a
    Linear offset mapping, which is never stored as a dict, is honoured too).
    A cell the mapping leaves out gets the same fallback id its traces get;
    the resolution is a permutation, so no two cells share a neuron id. The
    mapping is resolved once for all ids.
    """
    permutation = _permutation(cellpose_data)
    ids = {int(i) for i in cellpose_ids if int(i) >= 0}
    return {cell: permutation.get(cell, cell) for cell in ids}


def cellpose_to_stim_id(
    cellpose_id: int, cellpose_data: CellposeImportResult | None
) -> int | None:
    """See :func:`cellpose_to_stim_ids`; ``None`` for a negative (background) id."""
    return cellpose_to_stim_ids([cellpose_id], cellpose_data).get(int(cellpose_id))


def stim_ids_to_cellpose_ids(
    stim_ids: Iterable[int], cellpose_data: CellposeImportResult | None
) -> dict[int, int]:
    """The Cellpose cell whose traces get each non-negative neuron id (the
    inverse of :func:`cellpose_to_stim_ids`); the mapping is resolved once."""
    inverse = {neuron: cell for cell, neuron in _permutation(cellpose_data).items()}
    ids = {int(i) for i in stim_ids if int(i) >= 0}
    return {stim: inverse.get(stim, stim) for stim in ids}


def stim_id_to_cellpose_id(
    stim_id: int | None, cellpose_data: CellposeImportResult | None
) -> int | None:
    """See :func:`stim_ids_to_cellpose_ids`; ``None`` for ``None``/negative ids."""
    if stim_id is None:
        return None
    return stim_ids_to_cellpose_ids([stim_id], cellpose_data).get(int(stim_id))


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
