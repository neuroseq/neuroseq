# TIFF import / Spatial Footprints consistency

## Overview

The TIFF calcium-import dialog (`TifImportDialog`) can produce or load a cell
segmentation two ways that never reached `Project.cellpose_data` — browsing a
`_seg.npy` file, or (after the suite2p_pipeline_enhancement change)
Suite2p auto-detection — so neither ever showed up in the main window's
Spatial Footprints panel. Only the third mask source ("use masks already
loaded in the project") was consistent, because it reads its masks FROM that
project state in the first place. Separately, running the pipeline in
auto-detect mode (no segmentation at all) had no confirmation step.

## Change

- `src/neuroseq/tif_import_dialog.py` — a new `_build_cellpose_result_from_pipeline`
  helper wraps a `TifPipelineResult`'s `masks`/`mean_image` into a
  `CellposeImportResult`, reusing `CellPoseImporter`'s existing
  image/segmentation conventions (background = -1, vertically flipped) so
  colours and rendering match the main Cellpose dialog. `_start_pipeline`
  captures, once at launch time, whether the mask source produced something
  new (`_propagate_new_segmentation = not self._use_loaded_radio.isChecked()`)
  and which segmentation file (if any) backs it (`_new_segmentation_path`) —
  captured up front rather than re-read from the radios at finish time, since
  the radios stay interactive while the background pipeline thread runs.
  `_finalize_import` builds `self.new_cellpose_result` from that captured
  state, for both pipelines (`run_suite2p_pipeline` and
  `run_roi_average_pipeline`). Reusing already-loaded masks (`_use_loaded_radio`)
  deliberately leaves `new_cellpose_result` as `None` — rebuilding there would
  silently drop any existing `stimulus_to_cellpose_mapping`/`aoi_config` on
  the current `Project.cellpose_data`.
- `_start_pipeline` also now asks for confirmation
  (`QMessageBox.question`, Yes/No, default No) before running with no
  segmentation at all (`masks is None`, i.e. auto-detect mode), warning that
  Suite2p's own detector is typically less reliable than a Cellpose-verified
  segmentation.
- `src/neuroseq/gui.py` — `btn_import_tif_clicked` now routes
  `dialog.new_cellpose_result` (when not `None`) through the existing
  `Gui._apply_cellpose_import_result`, the same entry point the main Cellpose
  import dialog uses — gets stim-array-resize prompts and the Spatial
  Footprints panel refresh for free, no new panel-refresh logic needed.
- Unrelated but verified in the same diff: `spatial_footprint_widget.py`'s
  greyed-cell rendering now also recolours the base overlay image (grey/white
  stripe pattern) in addition to the existing semi-transparent stripe layer,
  since the stripe-only overlay wasn't visually distinct enough against a
  plain grayscale background (as produced by the new TIFF-dialog segmentation
  path, which has no pre-existing cell colouring baked into its image).

## Non-goals

No propagation for the "use already-loaded masks" branch (see above — would
regress existing AOI/mapping metadata). No AOI/custom-mapping UI added to
`TifImportDialog` itself — propagated results always use `create_result`'s
neutral defaults. No change to the "Import successful" message box text.

## Verification

`ruff check` + `ruff format --check` + `mypy --ignore-missing-imports` clean
on all four touched source files. Targeted suites (`test_gui.py`,
`test_tif_import_dialog.py`, `test_importer_tif.py`,
`test_spatial_footprint_widget.py`, `test_cellpose_aoi_mapping.py`,
`test_cellpose_import_dialog.py`) — 188 tests, all passing. Full suite: 1286
passed, 2 skipped, 0 failed.
