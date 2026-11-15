"""Widget-free colouring of Cellpose footprints to match their neurons' traces.

Trace, spike and stim views colour a neuron by its neuron (stimulus) id via
``neuron_indices_to_colors``. Footprints are labelled by Cellpose id in the
segmentation, and the two differ whenever the Cellpose mapping is not the
identity: a Linear offset, a Custom mapping, or Cellpose ids with gaps (cells
removed by an area of interest or deleted), which the mapping compacts into
consecutive neuron ids. Colour each footprint by the neuron id its traces get,
resolved exactly as the trace importers resolve it (``remap_raw_cellpose_ids``).
"""

from __future__ import annotations

import brian2.numpy_ as np

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping
from neuroseq.importers.importer_suite2p import remap_raw_cellpose_ids
from neuroseq.utils import neuron_indices_to_colors


def footprint_neuron_ids(
    cellpose_ids: np.ndarray, cellpose_data: CellposeImportResult | None
) -> np.ndarray:
    """Neuron id whose colour each (non-negative) Cellpose id takes.

    That is the id trace/spike rows extracted for that cell get. Falls back to
    the Cellpose ids themselves when there is no Cellpose data or the mapping
    cannot be resolved (it then cannot have been applied to traces either).
    """
    ids = np.asarray(cellpose_ids, dtype=np.int64)
    if ids.size == 0 or cellpose_data is None:
        return ids
    mapping = resolve_stimulus_to_cellpose_mapping(cellpose_data)
    try:
        return remap_raw_cellpose_ids(ids, mapping).astype(np.int64)
    except ValueError:
        return ids


def footprint_legend(
    segmentation: np.ndarray, cellpose_data: CellposeImportResult | None = None
) -> list[tuple[int, tuple[int, int, int]]]:
    """``(cellpose_id, rgb)`` per segmented cell, coloured like its traces."""
    unique_ids = np.unique(segmentation)
    cellpose_ids = unique_ids[unique_ids >= 0]
    if cellpose_ids.size == 0:
        return []
    colors = neuron_indices_to_colors(footprint_neuron_ids(cellpose_ids, cellpose_data))
    return [
        (int(cell_id), (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
        for cell_id, c in zip(cellpose_ids, colors, strict=True)
    ]


def compose_footprint_overlay(
    background: np.ndarray,
    segmentation: np.ndarray,
    legend: list[tuple[int, tuple[int, int, int]]],
    opacity: float,
) -> np.ndarray:
    """Blend each cell's legend colour over ``background`` inside the cell.

    Same blend as ``CellPoseImporter.generate_overlay`` (which bakes colours
    keyed by Cellpose id into the stored overlay), but with ``legend``'s
    colours. ``background`` is an RGB(A) uint8 image of the segmentation's
    shape.
    """
    overlay = np.asarray(background, dtype=np.float32).copy()
    if not legend:
        return overlay.astype(np.uint8)
    channels = min(3, overlay.shape[-1])
    segmentation = np.asarray(segmentation)
    max_id = max(int(cell_id) for cell_id, _ in legend)
    # Lookup table by Cellpose id: one vectorised blend instead of one full
    # image comparison per cell.
    colors = np.zeros((max_id + 1, channels), dtype=np.float32)
    known = np.zeros(max_id + 1, dtype=bool)
    for cell_id, rgb in legend:
        colors[cell_id] = rgb[:channels]
        known[cell_id] = True
    in_cell = (segmentation >= 0) & (segmentation <= max_id)
    in_cell[in_cell] = known[segmentation[in_cell]]
    overlay[in_cell, :channels] = (1 - opacity) * overlay[
        in_cell, :channels
    ] + opacity * colors[segmentation[in_cell]]
    return overlay.astype(np.uint8)
