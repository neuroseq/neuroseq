# Spatial Footprint Viewer — toggleable panel with click-to-grey linking to trace/stim rows

## Overview

A new **Spatial Footprints** panel lives in the same splitter as the
Heatmap panel, toggled from the top bar's View menu/toolbar. It shows the
imported Cellpose segmentation (colour-coded cells, ID-labelled) for the
current project's `Project.cellpose_data`, reusing `CellOverlayWidget` —
previously used only inside the Cellpose import dialog. Clicking a cell
toggles a striped grey/white overlay on that cell in the picture, and a
solid grey overlay on the corresponding row in both `trace_widget` and
`stim_editor.stim_widget`. Clicking the same cell again removes all three.

Unlike every other optional panel (Heatmap, Traces, Spikes, Data, Custom
Analysis), **Spatial Footprints starts visible** — this is a deliberate
choice, confirmed with the user, not an oversight: when no cellpose data is
loaded the panel shows an empty state with an **Import Cell Segmentation**
button, so new users see an actionable prompt on first launch instead of
another panel to discover.

Applies to:

- `src/neuroseq/cell_overlay_widget.py` — striped-cell rendering, click
  detection (opt-in, gated behind a flag), legend-colour extraction
- `src/neuroseq/timeseries_widget.py` — `set_neuron_greyed` /
  `is_neuron_greyed` / `clear_greyed_neurons` on the shared base class
- `src/neuroseq/spatial_footprint_widget.py` (new) — `SpatialFootprintWidget`
- `src/neuroseq/gui.py` — panel wiring, toolbar toggle, refresh call sites
- `tests/test_spatial_footprint_widget.py` (new), plus additions to
  `tests/test_trace_widget.py`, `tests/test_stim_widget.py`,
  `tests/test_gui.py`

---

## Neuron-index conventions

`TraceData.indices[i]` / a `StimArray` column index and
`CellposeImportResult.stimulus_to_cellpose_mapping`'s **key** (`stim_id`)
are the same integer space — confirmed via `importer_suite2p.py`'s
`_resolve_suite2p_indices`, which inverts the same dict the same way:

```python
cellpose_to_stim = {v: k for k, v in (mapping or {}).items()}
stim_id = cellpose_to_stim.get(cellpose_id, cellpose_id)
```

`SpatialFootprintWidget._cellpose_to_stim_id` does exactly this. When
`stimulus_to_cellpose_mapping` is `None` (no custom mapping was set up at
import time), `stim_id == cellpose_id` — the identity fallback.

`OverlayView.segmentation` is always keyed by raw Cellpose id, regardless of
the `show_stimulus_ids` display toggle (`label_segmentation=self.segmentation`
is passed separately from the possibly stim-id-remapped
`display_segmentation` in `CellOverlayWidget._update_overlay_display`), so a
pixel lookup into it always yields a Cellpose id to invert.

---

## `cell_overlay_widget.py` changes

- **`compute_segmentation_legend(segmentation) -> list[(cellpose_id, rgb)]`**
  extracted to module level from `CellOverlayWidget._compute_segementation_colors`
  (which now delegates to it), so `SpatialFootprintWidget` can compute its
  own base legend without reaching into another class's private method.

- **`OverlayView.enable_cell_selection`** (default `False`) gates a new
  click-to-select path, added alongside — not replacing — the existing AOI
  drag handlers (`_aoi_pointer_down`/`_aoi_pointer_up`); pygfx dispatches an
  event to every handler registered for that type, so both coexist safely.
  With the flag off (the Cellpose import dialog's default, untouched), the
  new handlers are a no-op — verified by
  `test_overlay_view_ignores_click_when_selection_disabled`.
  - `_cell_selection_pointer_down` / `_cell_selection_pointer_up`: records
    the world position on press, and on release only treats it as a click
    (not the end of a pan) if the release position is within
    `_world_tolerance()` of the press position.
  - `_segmentation_id_at(world_x, world_y) -> int | None`: bounds-checked
    lookup into `self.segmentation`, returns `None` for out-of-bounds or
    background (`< 0`) pixels.
  - `neuron_clicked = QtCore.Signal(int)` emits the raw Cellpose id; a
    background click emits nothing (`test_overlay_view_ignores_background_click`).

- **Striped-cell rendering** (`STRIPED_CELL_GREY`, `STRIPED_CELL_WHITE`,
  `STRIPED_CELL_ALPHA = 1.0`, `STRIPED_CELL_STRIPE_WIDTH`): cells in
  `striped_cell_ids` get a diagonal grey/white stripe pattern instead of
  their normal legend colour, drawn fully opaque (`alpha=1.0`, vs. the
  normal `0.4`) in `_build_segmentation_colors`. This was chosen over a flat
  grey recolour (the original plan) because it reads more clearly as "this
  cell is excluded" than a merely-dim colour would. `striped_label_ids`
  independently darkens the ID label text/outline for the same cells
  (`_update_label_graphics`) — a small scope addition over the original
  plan (which called out label styling as a non-goal), justified because a
  cell that's visually opaque-grey but still carries a bright, unmodified ID
  label reads inconsistently.

  Both `striped_cell_ids` and `striped_label_ids` are correctly translated
  by `CellOverlayWidget._get_display_striped_cell_ids` to whichever id-space
  `_get_display_segmentation`/`_get_display_legend` are already using
  (stim-id when `show_stimulus_ids` is on and a mapping exists) — mirroring
  the pre-existing remapping pattern instead of introducing a new one.

- **Incidental fix**: `_build_segmentation_colors` filtered
  `neuron_ids[neuron_ids > 0]`, excluding cell id `0` from the coloured
  overlay even though `compute_segmentation_legend`/`_compute_segementation_colors`
  already included it (`>= 0`) and the rest of the codebase's cell-counting
  convention (`Project._cellpose_neuron_count`, `original_cellpose_ids`)
  also treats `0` as a real cell (only negative ids are background). Fixed
  to `>= 0` to match — otherwise a segmentation with a real cell at id 0
  (the common case for 0-indexed exports) would never render a colour for
  it, in both this new panel and the existing Cellpose import dialog.

- **`resizeEvent` override removed** from `CellOverlayWidget`. It previously
  forced `setFixedSize` on the embedded `OverlayView`, which is incompatible
  with `CellOverlayWidget` now also living inside a `QSplitter` panel that
  must resize with the window — a fixed-size child fights the splitter's
  layout. Removing it lets Qt's layout system size the view naturally in
  both contexts; `test_cell_overlay_resize_keeps_embedded_view_layout_driven`
  and the full `test_cellpose_import_dialog.py` suite confirm the import
  dialog is unaffected.

---

## `timeseries_widget.py` changes

Shared base class for `TraceWidget` and `StimWidget` (not `DataWidget`,
which is a plain `QWidget`) gained:

```python
GREYED_NEURON_Z = 250.0
GREYED_NEURON_FILL_COLOR = (0.5, 0.5, 0.5, 0.75)

def set_neuron_greyed(self, neuron_index: int, greyed: bool) -> None: ...
def is_neuron_greyed(self, neuron_index: int) -> bool: ...
def clear_greyed_neurons(self) -> None: ...
```

`set_neuron_greyed` is **idempotent** — it doesn't assume anything about
prior state, so gui.py can call it from an external source of truth
(the panel's click handler) without first checking `is_neuron_greyed`.
Implemented by reusing `RegionOfInterest` (the same primitive behind the
drag-select highlight) as a persistent full-width overlay spanning
`y ∈ [neuron_index - 0.5, neuron_index + 0.5]`, at `z=250` — between grid
lines (`z=199`) / the stim image (`z=200-210`) and the hover
crosshair/axes (`z=300`).

`clear_data()` now also calls `clear_greyed_neurons()`, so switching the
displayed `TraceData`/`StimArray` drops stale neuron-indexed overlays —
matching how it already drops the unrelated ROI-selection state.

**Incidental fix**: `_setup_scroll_bars` was refactored to narrow
`self.controller`/`self.layout()` into local variables before use. This
fixes 5 pre-existing mypy errors in that method (confirmed by diffing mypy
output against the pre-change tree) unrelated to the new feature, cleaned
up as part of the same verification pass rather than left failing.

---

## `spatial_footprint_widget.py` (new)

```python
class SpatialFootprintWidget(QtWidgets.QWidget):
    import_requested = QtCore.Signal()             # empty-state button clicked
    neuron_greyed_out_changed = QtCore.Signal(int, bool)  # (stim_id, greyed)
```

- Owns a `CellOverlayWidget(title="Spatial Footprints")` with
  `show_cellpose_ids=False, show_stimulus_ids=True` and
  `set_cell_selection_enabled(True)` — this is the only place in the
  codebase that turns cell-click-selection on.
- **Empty state**: when `set_data(None)` (no cellpose data), shows a
  centred label + "Import Cell Segmentation" button instead of the overlay
  widget, and emits `import_requested` on click — wired in gui.py directly
  to `btn_import_cellpose_data_clicked`, so clicking it opens the same
  import dialog as the toolbar action. Not part of the original plan; added
  because the panel is visible by default and would otherwise show nothing
  actionable on a fresh project.
- `set_data(cellpose_data)` computes `self._base_legend` once via
  `compute_segmentation_legend` and stores it; `_on_neuron_clicked` only
  toggles membership in `self._greyed_cellpose_ids` and re-calls
  `overlay_widget.set_overlay(..., legend_data=self._base_legend,
striped_cell_ids=self._greyed_cellpose_ids)` — the legend itself never
  changes, only which cells are drawn striped.
- `_refresh_overlay` passes `self._cellpose_data.overlay_image` through
  unmodified. An earlier version of this method also pre-striped the base
  image pixel-by-pixel before handing it to `set_overlay`, but that's
  always invisible: `OverlayView._build_segmentation_colors` already draws
  a fully opaque (`alpha=1.0`) striped colour layer on top of the base
  image for the same cells, and `SpatialFootprintWidget` never disables
  `show_segmentation` to reveal the layer underneath. Removed as dead
  computation during review; `test_clicking_cell_greys_rendered_spatial_footprint`
  now explicitly asserts the base image is untouched by a click.

---

## `gui.py` wiring

- `self.spatial_footprint_widget` constructed next to `heatmap_widget`,
  added to the same `split_left_right` splitter, **not** added to
  `self.plot_widgets` (no time axis, so `TimeSliderWidget` doesn't need it)
  but explicitly `.close()`'d in `closeEvent` (it owns a fastplotlib figure
  via `CellOverlayWidget` → `OverlayView`, same reason `plot_widgets` are
  explicitly closed there).
- Toolbar/menu entry `"&Spatial Footprints"` in `_create_actions_view`,
  `is_checked=True` (see Overview) — everything else about the toggle is
  the generic `_set_widget_visible` machinery, no widget-specific toggle
  code needed.
- `Project.cellpose_data` has three write sites in gui.py (`update_project`,
  the non-resize tail of `_apply_cellpose_import_result`, and
  `_delete_cellpose_data`); each now also calls
  `self.spatial_footprint_widget.set_data(self.project.cellpose_data)`.
  `_update_data()` (a narrower refresh path used after undo/redo and
  stim-editing operations, none of which touch `cellpose_data`) also got
  the call — harmless (idempotent re-application of unchanged data) even
  though not strictly necessary there.
- `btn_import_cellpose_data_clicked` auto-shows the panel
  (`_auto_show_widget`) after a successful import, matching how
  `set_synapse` auto-shows the Heatmap panel.
- `_on_neuron_greyed_out_changed(neuron_index, greyed)` forwards 1:1 to
  `self.trace_widget.set_neuron_greyed(...)` and
  `self.stim_editor.stim_widget.set_neuron_greyed(...)`.

---

## Non-goals (intentionally not implemented)

- **Synchronising greyed state across a dataset switch.** If a neuron is
  greyed and the user picks a different entry in `cmb_trace`,
  `TraceWidget.clear_data()` drops that neuron's overlay (same as it
  already drops the unrelated ROI selection). The panel's own greyed set is
  untouched — clicking the same cell again still works correctly
  (`set_neuron_greyed` is idempotent) — but the trace/stim panels won't
  "remember" the mark across a dataset switch without a fresh click.
- **Multi-select gestures.** One click toggles one cell; no shift-click
  ranges, drag-select, or "grey all"/"clear all" buttons.
- **AOI interaction.** The new panel never sets `_aoi_config`; no AOI
  shapes are ever drawn in it.
- **Persisting the greyed set** into the `.nseq` project file — view-only
  interaction state, resets on reload/restart, same as ROI selection.
- **A combo box / multi-dataset picker.** `Project.cellpose_data` is a
  single optional slot, not a named collection like `TraceData`.

---

## Verification

```
ruff check src/neuroseq/cell_overlay_widget.py src/neuroseq/timeseries_widget.py \
           src/neuroseq/spatial_footprint_widget.py src/neuroseq/gui.py \
           tests/test_spatial_footprint_widget.py tests/test_trace_widget.py \
           tests/test_stim_widget.py tests/test_gui.py
ruff format --check <same files>
mypy --ignore-missing-imports src/neuroseq/cell_overlay_widget.py \
           src/neuroseq/timeseries_widget.py src/neuroseq/spatial_footprint_widget.py \
           src/neuroseq/gui.py
uv run pytest tests/test_spatial_footprint_widget.py tests/test_trace_widget.py \
           tests/test_stim_widget.py tests/test_cellpose_import_dialog.py \
           tests/test_cellpose_aoi_mapping.py tests/test_gui.py   # 154 passed
uv run pytest   # full suite: 1267 passed, 2 skipped
```

All clean. No regressions in the Cellpose import dialog's own test suite
despite `cell_overlay_widget.py` being shared code.
