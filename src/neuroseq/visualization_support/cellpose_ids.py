"""Widget-free helpers for Cellpose cell ids, stimulus ids and centroids."""

from __future__ import annotations

import brian2.numpy_ as np
from scipy import ndimage

from neuroseq.importers.importer_cellpose import CellposeImportResult


def cellpose_to_stim_id(
    cellpose_id: int, cellpose_data: CellposeImportResult | None
) -> int | None:
    mapping = (
        cellpose_data.stimulus_to_cellpose_mapping
        if cellpose_data is not None
        else None
    )
    if not mapping:
        return cellpose_id
    return {value: key for key, value in mapping.items()}.get(cellpose_id)


def stim_id_to_cellpose_id(
    stim_id: int | None, cellpose_data: CellposeImportResult | None
) -> int | None:
    if stim_id is None:
        return None
    mapping = (
        cellpose_data.stimulus_to_cellpose_mapping
        if cellpose_data is not None
        else None
    )
    if not mapping:
        return stim_id
    return mapping.get(stim_id)


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
