# Cellpose import — "Custom with areas of interest" mapping strategy

Adds a third mapping strategy to the Cellpose import dialog. The user defines
two on-FOV shapes — a **square** (well-behaved laser power region) and a
**circle** (region where power cannot be generated properly) — and assigns
each a role: _preferential_ (cells inside get the lowest stim IDs) or
_exclusion_ (cells inside get the highest stim IDs). Exclusion wins on
overlap. The resulting mapping autofills the existing custom mapping line
edit, which already drives the rest of the dialog and downstream consumers.

## Files

| Path                                     | Change                                                                                                                                                                                                                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/neuroseq/aoi_shapes.py`             | NEW — pure-geometry helpers (coord conversion, polylines, handles)                                                                                                                                                                                                             |
| `src/neuroseq/importer_cellpose.py`      | dataclasses (`AOIRole`, `SquareAOI`, `CircleAOI`, `AreasOfInterestConfig`), `default_aoi_config`, `point_in_square`, `point_in_circle`, `generate_aoi_stimulus_mapping`, `clone_aoi_config`; `CellposeImportResult.aoi_config`; `CellPoseImporter.create_result(aoi_config=…)` |
| `src/neuroseq/cellpose_import_dialog.py` | new radio "Custom with areas of interest", AOI sub-group UI, "Show shapes" checkbox, autofill + restore logic                                                                                                                                                                  |
| `src/neuroseq/cell_overlay_widget.py`    | render shape outlines + handles on the pygfx subplot, pointer-down/up/move drag-and-resize, Qt signal `aoi_geometry_changed`                                                                                                                                                   |
| `src/neuroseq/io.py`                     | `_aoi_to_json` / `_aoi_from_json`; new `CellposeImportResult` match branch with `aoi_config`, older two retained for back-compat                                                                                                                                               |
| `tests/test_cellpose_aoi_mapping.py`     | NEW — 8 pure-logic + 1 io round-trip + 4 dialog UI tests (13 total; full suite adds 16 existing → 29 in cellpose path)                                                                                                                                                         |

## Data model

`AOIRole` is `StrEnum` with values `"preferential"` / `"exclusion"`.
`SquareAOI` carries `enabled`, `role`, top-left `x`, `y`, and `side`.
`CircleAOI` carries `enabled`, `role`, centre `cx`, `cy`, and `diameter`.
`AreasOfInterestConfig` bundles a square + circle + `show_shapes` flag.

Defaults — `default_aoi_config(image_shape)`:

- circle diameter = `min(H, W) / 8`, square side = `3 × circle_d`
- both centred on the FOV
- square role = preferential, circle role = exclusion

## Coordinate convention

The importer flips both image and segmentation via `np.flipud` so pygfx's
y-up camera renders the FOV right-way-up (Fiji style). All conversions live
in `aoi_shapes.user_to_world_point` / `world_to_user_point`:

- **User coords** (what the spinboxes and the persisted config use):
  origin top-left of the displayed image, y grows down.
- **World coords** (pygfx + `_label_centers`): origin bottom-left, y grows up.
- Conversion: `world_y = H − user_y`. x is unchanged.

`generate_aoi_stimulus_mapping` receives `_label_centers` in world coords and
converts each centroid to user space before testing membership.

## Mapping algorithm

`generate_aoi_stimulus_mapping(original_ids, label_centers, config, image_height)`:

1. For each cellpose id with a known centroid:
   - test in_square (only if `square.enabled`)
   - test in_circle (only if `circle.enabled`)
2. Tier assignment (in order):
   - **exclusion** if `(square.role == EXCLUSION and in_square) or (circle.role == EXCLUSION and in_circle)`
   - **preferential** else if equivalent with `PREFERENTIAL`
   - **neutral** otherwise
3. Sort each tier by cellpose id ascending.
4. Concatenate `[pref…, neutral…, excl…]`; assign stim id = position.

Returns `{stim_id: cellpose_id}` — the same shape `_parse_custom_mapping`
already accepts.

## Dialog wiring

- New constant `CUSTOM_AOI_STRATEGY = "Custom with areas of interest"` in
  `cellpose_import_dialog.py`.
- Three radio buttons: `Linear`, `Custom`, `Custom with areas of interest`.
- AOI subgroup (`self.aoi_group`) hidden unless the AOI radio is active.
  Each shape row has: enable checkbox, mutually-exclusive preferential /
  exclusion radio pair (`QButtonGroup`), and three `QDoubleSpinBox`es
  (range `[0, 1e6]`, 1 decimal, step 1).
- `self.show_shapes_checkbox` in `checkbox_container` under the preview;
  hidden unless AOI mode.
- In AOI mode the existing `custom_mapping_line` is shown but **read-only**
  (its content is regenerated on every shape change).
- Any spinbox / checkbox / radio change calls `_regenerate_aoi_mapping`,
  which:
  1. reads `_current_aoi_config()` from the controls,
  2. computes the mapping via `generate_aoi_stimulus_mapping`,
  3. writes the serialised mapping into `custom_mapping_line`
     (`blockSignals` during the write),
  4. pushes the config to the overlay via `set_aoi_overlay`,
  5. refreshes the colour preview via `_update_preview_mapping`.
- A `_suppress_aoi_signals` flag guards programmatic updates
  (`_set_aoi_controls`) from re-entering `_regenerate_aoi_mapping`.
- `_ensure_aoi_defaults(force=…)` populates the spinboxes from
  `default_aoi_config(image.shape)` once per loaded image (or always when
  restoring existing data).

## Overlay interaction (`OverlayView`)

- New pygfx graphics: `_square_line` + 4 corner `_square_handles`;
  `_circle_line` (72-pt polyline) + 4 cardinal `_circle_handles`. Lines use
  `size_space="model"`; handles use `size_space="screen"` with `sizes=8`.
- Colours: preferential = green `(0.3, 0.85, 0.3, 0.9)`,
  exclusion = red `(0.95, 0.3, 0.3, 0.9)`.
- New pointer handlers `_aoi_pointer_down` / `_aoi_pointer_up` registered on
  `figure.renderer`. The existing `pointer_move` short-circuits to
  `_apply_drag` when a drag is in progress.
- `_aoi_hit_test` checks (in order) square corners → circle handles →
  circle body → square body, with a screen-aware tolerance from
  `_world_tolerance()` (`max(2.0, (camera_width / widget_width) * 10)`).
- Square corner resize keeps the opposite corner fixed and forces a square
  (`side = max(|dx|, |dy|)`, side clamped ≥ 4 px). Circle cardinal resize
  computes `radius = distance(centre, handle)` and clamps against the four
  image edges so the circle can't escape the FOV.
- Body drags clamp the shape to the image bounds. Pan/zoom is disabled
  during a drag via `self.subplot.controller.enabled = False`, restored on
  pointer-up.
- Shape geometry changes propagate via
  `OverlayView.on_geometry_changed` → `CellOverlayWidget.aoi_geometry_changed`
  Qt signal → dialog's `_on_overlay_aoi_geometry_changed`, which calls
  `_set_aoi_controls` (signal-suppressed) and re-runs
  `_regenerate_aoi_mapping`.

## Persistence (io.py)

`_aoi_to_json` flattens the config to plain dicts (role serialised as the
enum value string). The encoder adds `aoi_config` to the
`CellposeImportResult` payload.

The decoder has three `CellposeImportResult` match branches:

1. with `mapping_strategy`, `linear_offset`, **and** `aoi_config` (new)
2. with `mapping_strategy` and `linear_offset` (back-compat — original)
3. without either (back-compat — earliest archives)

`linear_offset` is no longer required to be int — None survives the round
trip (the dataclass already allowed it).

## Tests — `tests/test_cellpose_aoi_mapping.py`

Pure-logic (no Qt):

- `test_default_aoi_config_centred_and_3x`
- `test_point_in_square_inclusive_edges`
- `test_point_in_circle`
- `test_aoi_mapping_preferential_only`
- `test_aoi_mapping_exclusion_only`
- `test_aoi_mapping_overlap_exclusion_wins`
- `test_aoi_mapping_disabled_shape_ignored`
- `test_aoi_mapping_sorted_by_cellpose_id_within_tier`

io round-trip:

- `test_aoi_config_round_trips_through_io` — writes a
  `CellposeImportResult` (with an AOI config) into a zip and reloads,
  asserting full equality of the restored config.

Dialog UI (qtbot, mirrors existing `test_cellpose_import_dialog.py`):

- `test_dialog_selects_aoi_strategy_shows_subgroup`
- `test_dialog_aoi_autofills_custom_mapping_line`
- `test_dialog_aoi_changes_remap_on_spinbox_edit`
- `test_dialog_aoi_persists_through_reopen`

## Out of scope

- Rotating shapes, multiple shapes of a kind, polygons / ellipses /
  annuli, automatic FOV detection. The hardware export and other
  downstream consumers see only the final stim-to-cellpose mapping; the
  AOI config travels with the import result purely for round-tripping
  the dialog UI on reopen.
