from __future__ import annotations

from collections.abc import Iterable

import brian2.numpy_ as np
from qtpy import QtCore

from neuroseq.importer_cellpose import CellposeImportResult


class SpatialFootprintState(QtCore.QObject):
    """Canonical, shareable state for the Spatial Footprints panel.

    Multiple `SpatialFootprintWidget` views can be bound to the same
    `SpatialFootprintState` instance (see `SpatialFootprintWidget.__init__`'s
    `state` parameter) and stay in sync: a mutation via any one view (or via
    `gui.py`) updates this object and emits a signal, which every connected
    view uses to refresh its own rendering. No widget/rendering logic lives
    here - only data and the operations that change it.

    Clicking a cell selects it (a ring marks it); every trace/stim row not
    backed by a currently selected cell is hidden - this applies live, there
    is no separate "apply" step. An empty selection means no filtering is in
    effect (every row shown).
    """

    data_changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.cellpose_data: CellposeImportResult | None = None
        self.selected_cellpose_ids: set[int] = set()

    def _all_cellpose_ids(self) -> set[int]:
        if self.cellpose_data is None:
            return set()
        return {
            int(cellpose_id)
            for cellpose_id in np.unique(self.cellpose_data.segmentation_array)
            if cellpose_id >= 0
        }

    @property
    def greyed_cellpose_ids(self) -> set[int]:
        """Cells currently hidden from trace/stim: everything not selected,
        or none at all while nothing is selected (no filtering in effect)."""
        if not self.selected_cellpose_ids:
            return set()
        return self._all_cellpose_ids() - self.selected_cellpose_ids

    def _cellpose_to_stim_id(self, cellpose_id: int) -> int | None:
        """Resolve a segmented cell to the trace/stim row it drives, or
        `None` if it doesn't drive one at all.

        When there is no recorded mapping, cellpose ids are used directly as
        neuron/stim indices (identity), so `cellpose_id` itself is correct.
        But once a (possibly partial) mapping exists - e.g. Cellpose found
        more cells than were mapped to simulated/imported neurons - falling
        back to identity for an *unmapped* id would be wrong: that id's
        number doesn't refer to a stim/trace row at all, and could easily
        collide with the real row of some other, unrelated neuron (both id
        spaces are small ints starting at 0). Such an id must resolve to
        `None` (and callers must skip emitting for it), not to itself.
        """
        mapping = (
            self.cellpose_data.stimulus_to_cellpose_mapping
            if self.cellpose_data is not None
            else None
        )
        if not mapping:
            return cellpose_id
        cellpose_to_stim = {v: k for k, v in mapping.items()}
        return cellpose_to_stim.get(cellpose_id)

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self.cellpose_data = cellpose_data
        self.selected_cellpose_ids = set()
        self.data_changed.emit()

    def _set_selected(self, new_selected: set[int]) -> None:
        old_greyed = self.greyed_cellpose_ids
        self.selected_cellpose_ids = new_selected
        new_greyed = self.greyed_cellpose_ids
        self.selection_changed.emit()
        for cellpose_id in new_greyed - old_greyed:
            stim_id = self._cellpose_to_stim_id(cellpose_id)
            if stim_id is not None:
                self.neuron_greyed_out_changed.emit(stim_id, True)
        for cellpose_id in old_greyed - new_greyed:
            stim_id = self._cellpose_to_stim_id(cellpose_id)
            if stim_id is not None:
                self.neuron_greyed_out_changed.emit(stim_id, False)

    def toggle_selected(self, cellpose_id: int) -> None:
        """Select the cell, or deselect it if it was already selected -
        deselecting the last selected cell clears the whole selection."""
        new_selected = set(self.selected_cellpose_ids)
        if cellpose_id in new_selected:
            new_selected.discard(cellpose_id)
        else:
            new_selected.add(cellpose_id)
        self._set_selected(new_selected)

    def add_selected(self, cellpose_ids: Iterable[int]) -> None:
        """Add every id to the selection (additive - a lasso never
        deselects an id that was already selected)."""
        new_ids = set(cellpose_ids) - self.selected_cellpose_ids
        if not new_ids:
            return
        self._set_selected(self.selected_cellpose_ids | new_ids)

    def clear_selection(self) -> None:
        if not self.selected_cellpose_ids:
            return
        self._set_selected(set())
