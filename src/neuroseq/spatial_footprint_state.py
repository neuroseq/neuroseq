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
    """

    data_changed = QtCore.Signal()
    greyed_changed = QtCore.Signal()
    preferential_selection_changed = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.cellpose_data: CellposeImportResult | None = None
        self.greyed_cellpose_ids: set[int] = set()
        self.preferential_selected_cellpose_ids: set[int] = set()

    def _cellpose_to_stim_id(self, cellpose_id: int) -> int:
        mapping = (
            self.cellpose_data.stimulus_to_cellpose_mapping
            if self.cellpose_data is not None
            else None
        )
        cellpose_to_stim = {v: k for k, v in (mapping or {}).items()}
        return cellpose_to_stim.get(cellpose_id, cellpose_id)

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self.cellpose_data = cellpose_data
        self.greyed_cellpose_ids = set()
        self.preferential_selected_cellpose_ids = set()
        self.data_changed.emit()

    def toggle_greyed(self, cellpose_id: int) -> None:
        if cellpose_id in self.greyed_cellpose_ids:
            self.greyed_cellpose_ids.discard(cellpose_id)
            greyed = False
        else:
            self.greyed_cellpose_ids.add(cellpose_id)
            greyed = True
        self.greyed_changed.emit()
        self.neuron_greyed_out_changed.emit(
            self._cellpose_to_stim_id(cellpose_id), greyed
        )

    def add_greyed(self, cellpose_ids: Iterable[int]) -> None:
        """Add every id to the greyed set (additive - never un-greys an id
        already in the set, unlike the single-toggle `toggle_greyed`)."""
        new_ids = set(cellpose_ids) - self.greyed_cellpose_ids
        if not new_ids:
            return
        self.greyed_cellpose_ids |= new_ids
        self.greyed_changed.emit()
        for cellpose_id in new_ids:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), True
            )

    def toggle_preferential_selection(self, cellpose_id: int) -> None:
        if cellpose_id in self.preferential_selected_cellpose_ids:
            self.preferential_selected_cellpose_ids.discard(cellpose_id)
        else:
            self.preferential_selected_cellpose_ids.add(cellpose_id)
        self.preferential_selection_changed.emit()

    def add_preferential_selection(self, cellpose_ids: Iterable[int]) -> None:
        """Add every id to the pending preferential selection (additive, same
        rationale as `add_greyed`)."""
        new_ids = set(cellpose_ids) - self.preferential_selected_cellpose_ids
        if not new_ids:
            return
        self.preferential_selected_cellpose_ids |= new_ids
        self.preferential_selection_changed.emit()

    def clear_preferential_selection(self) -> None:
        if not self.preferential_selected_cellpose_ids:
            return
        self.preferential_selected_cellpose_ids = set()
        self.preferential_selection_changed.emit()

    def apply_preferential_selection(self) -> None:
        """Grey out every cell not marked as preferentially selected, and
        un-grey any previously-greyed cell that was marked - an absolute
        "only the selected cells stay visible" reset."""
        if self.cellpose_data is None or not self.preferential_selected_cellpose_ids:
            return
        all_ids = {
            int(cellpose_id)
            for cellpose_id in np.unique(self.cellpose_data.segmentation_array)
            if cellpose_id >= 0
        }
        old_greyed = self.greyed_cellpose_ids
        new_greyed = all_ids - self.preferential_selected_cellpose_ids
        self.preferential_selected_cellpose_ids = set()
        self.greyed_cellpose_ids = new_greyed
        self.preferential_selection_changed.emit()
        self.greyed_changed.emit()
        for cellpose_id in new_greyed - old_greyed:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), True
            )
        for cellpose_id in old_greyed - new_greyed:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), False
            )

    def clear_all(self) -> None:
        """Un-grey every currently greyed cell and drop any pending (unapplied)
        preferential selection.

        Emits the same neuron_greyed_out_changed(stim_id, False) signal a
        toggle would for each greyed cell, so anything listening (trace/stim
        rows) stays in sync without duplicating that propagation logic here.
        Pending preferential picks were never greyed, so no signal is emitted
        for those.
        """
        if not self.greyed_cellpose_ids and not self.preferential_selected_cellpose_ids:
            return
        cellpose_ids = list(self.greyed_cellpose_ids)
        had_preferential = bool(self.preferential_selected_cellpose_ids)
        self.greyed_cellpose_ids = set()
        self.preferential_selected_cellpose_ids = set()
        if cellpose_ids:
            self.greyed_changed.emit()
        if had_preferential:
            self.preferential_selection_changed.emit()
        for cellpose_id in cellpose_ids:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), False
            )
