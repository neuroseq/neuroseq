from __future__ import annotations

from collections.abc import Iterable

import brian2.numpy_ as np
from qtpy import QtCore

from neuroseq.cell_overlay_widget import cellpose_to_stim_id
from neuroseq.cell_overlay_widget import stim_id_to_cellpose_id
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import load_background_image


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
    # Batched equivalent of neuron_greyed_out_changed: (newly_hidden_stim_ids,
    # newly_shown_stim_ids), emitted once per selection change regardless of
    # how many rows it affects. A selection change on a segmentation with N
    # cells can touch up to N-1 rows at once (selecting the first cell hides
    # everyone else); consumers that rebuild a whole graphic per row (see
    # TimeseriesWidget.set_neuron_greyed) should use this instead of
    # neuron_greyed_out_changed to do that rebuild once, not N times.
    neurons_greyed_out_changed = QtCore.Signal(list, list)
    hovered_stim_id_changed = QtCore.Signal()
    active_stim_ids_changed = QtCore.Signal()
    display_range_changed = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.cellpose_data: CellposeImportResult | None = None
        self.selected_cellpose_ids: set[int] = set()
        # The neuron/stim row currently hovered in the Trace/Stim panels, if
        # any - lets the Spatial Footprints panel highlight where that
        # neuron sits in the FOV. Not derived from a click, so it's tracked
        # separately from selection.
        self.hovered_stim_id: int | None = None
        # Stim/neuron rows currently receiving nonzero stimulation at the
        # time slider's current position - driven by gui.py from the active
        # stim array, not by any interaction with this panel itself.
        self.active_stim_ids: frozenset[int] = frozenset()

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
        return cellpose_to_stim_id(cellpose_id, self.cellpose_data)

    def _stim_id_to_cellpose_id(self, stim_id: int | None) -> int | None:
        """The reverse of `_cellpose_to_stim_id`: resolve a trace/stim row
        to the segmented cell that drives it, or `None` if `stim_id` is
        `None` or there is no mapping entry for it."""
        return stim_id_to_cellpose_id(stim_id, self.cellpose_data)

    @property
    def hovered_cellpose_id(self) -> int | None:
        return self._stim_id_to_cellpose_id(self.hovered_stim_id)

    def set_hovered_stim_id(self, stim_id: int | None) -> None:
        if stim_id == self.hovered_stim_id:
            return
        self.hovered_stim_id = stim_id
        self.hovered_stim_id_changed.emit()

    @property
    def active_cellpose_ids(self) -> set[int]:
        cellpose_ids = (
            self._stim_id_to_cellpose_id(stim_id) for stim_id in self.active_stim_ids
        )
        return {cellpose_id for cellpose_id in cellpose_ids if cellpose_id is not None}

    def set_active_stim_ids(self, stim_ids: Iterable[int]) -> None:
        new_ids = frozenset(stim_ids)
        if new_ids == self.active_stim_ids:
            return
        self.active_stim_ids = new_ids
        self.active_stim_ids_changed.emit()

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        if cellpose_data is not None and cellpose_data.background_image is None:
            # Projects saved before `background_image` existed have none
            # baked into the archive - best-effort reload it from the
            # original file so "Show segmentation" has a raw background to
            # fall back to here too, matching what the Cell Segmentation
            # import dialog already does via
            # `CellPoseImporter.restore_from_result`.
            cellpose_data.background_image = load_background_image(
                cellpose_data.image_path
            )
        self.cellpose_data = cellpose_data
        self.selected_cellpose_ids = set()
        self.hovered_stim_id = None
        self.active_stim_ids = frozenset()
        self.data_changed.emit()

    @property
    def display_range(self) -> tuple[int, int] | None:
        """The FOV brightness/contrast window, persisted with the project
        via `CellposeImportResult.display_range` (None = unadjusted)."""
        if self.cellpose_data is None:
            return None
        return self.cellpose_data.display_range

    def set_display_range(self, display_range: tuple[int, int] | None) -> None:
        if self.cellpose_data is None or display_range == self.display_range:
            return
        self.cellpose_data.display_range = display_range
        self.display_range_changed.emit()

    def _set_selected(self, new_selected: set[int]) -> None:
        old_greyed = self.greyed_cellpose_ids
        self.selected_cellpose_ids = new_selected
        new_greyed = self.greyed_cellpose_ids
        self.selection_changed.emit()

        newly_hidden_stim_ids: list[int] = []
        for cellpose_id in new_greyed - old_greyed:
            stim_id = self._cellpose_to_stim_id(cellpose_id)
            if stim_id is not None:
                newly_hidden_stim_ids.append(stim_id)
                self.neuron_greyed_out_changed.emit(stim_id, True)

        newly_shown_stim_ids: list[int] = []
        for cellpose_id in old_greyed - new_greyed:
            stim_id = self._cellpose_to_stim_id(cellpose_id)
            if stim_id is not None:
                newly_shown_stim_ids.append(stim_id)
                self.neuron_greyed_out_changed.emit(stim_id, False)

        if newly_hidden_stim_ids or newly_shown_stim_ids:
            self.neurons_greyed_out_changed.emit(
                newly_hidden_stim_ids, newly_shown_stim_ids
            )

    def toggle_selected(self, cellpose_id: int) -> None:
        """Select the cell, or deselect it if it was already selected -
        deselecting the last selected cell clears the whole selection."""
        new_selected = set(self.selected_cellpose_ids)
        if cellpose_id in new_selected:
            new_selected.discard(cellpose_id)
        else:
            new_selected.add(cellpose_id)
        self._set_selected(new_selected)

    def toggle_selected_many(self, cellpose_ids: Iterable[int]) -> None:
        """Toggle every id's selection state - each one already selected is
        deselected, each one not yet selected is added, exactly like
        `toggle_selected` applied to each id at once. Lets a lasso be used
        to both grow a selection (enclose new cells) and shrink it (enclose
        already-selected ones to drop them), same as a plain click does for
        one cell at a time."""
        ids = set(cellpose_ids)
        if not ids:
            return
        self._set_selected(self.selected_cellpose_ids ^ ids)

    def clear_selection(self) -> None:
        if not self.selected_cellpose_ids:
            return
        self._set_selected(set())

    def set_selected(self, cellpose_ids: Iterable[int]) -> None:
        """Replace the selection outright with `cellpose_ids`, unlike
        `toggle_selected(_many)` which flips each id's own state. Used to
        restore a project's remembered selection when switching back to its
        tab - see `Gui._activate_tab_index`/`update_project`."""
        self._set_selected(set(cellpose_ids))
