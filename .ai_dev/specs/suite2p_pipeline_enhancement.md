# Suite2p pipeline enhancement (issue #506)

## Overview

Issue #506 asked for three things: quick analysis without Cellpose masks,
more detailed Suite2p logs during import, and a percent-of-FOV option for
AOI exclusion/inclusion criteria. Investigation found the third item already
shipped in an earlier, unlinked commit (`AreasOfInterestConfig.coordinate_mode`

- the "% of FOV" combo in `cellpose_import_dialog.py`), so this change covers
  the first two plus one missing regression test for the third.

## Change

- `src/neuroseq/importer_tif.py` — `run_suite2p_pipeline`'s `masks` parameter
  is now optional (`npt.NDArray[Any] | None`). When `None`, ROIs are detected
  via `suite2p.detection.detection_wrapper` (Suite2p's own `sparsery`
  algorithm) using a new `TifPipelineParams.diameter` field instead of
  `_masks_to_stat`'s Cellpose-mask conversion. A new `_stat_to_masks` helper
  rebuilds a Cellpose-style label image from the detected `stat` array so
  downstream code (overlay display, re-import) sees the same shape/convention
  regardless of ROI source. `iscell` is only force-accepted (`= 1`) when
  Cellpose masks were supplied — auto-detected ROIs keep Suite2p's own
  classifier verdict.
- `src/neuroseq/tif_import_dialog.py` — a third mask-source radio
  ("Auto-detect ROIs with Suite2p (no Cellpose)") resolves to `masks=None`;
  selecting it forces and disables the pipeline combo to "Suite2p" (ROI
  averaging cannot run maskless) and reveals a new diameter spinbox. A
  `_QtLogHandler(logging.Handler)` bridges Python logging to a new
  `_PipelineWorker.log` Qt signal, attached to both the `suite2p` and
  `neuroseq.importer_tif` loggers (with their levels temporarily raised to
  INFO so `logger.info(...)` calls actually reach the handler) for the
  duration of a pipeline run. The former `QProgressDialog` is replaced by a
  new `_PipelineProgressDialog` with the same label/value/cancel surface plus
  a collapsible "Details" `QPlainTextEdit` fed by that signal.
- Beyond the handoff's scope, the implementation also wrapped the dialog's
  body in a `QScrollArea` with wheel-event passthrough on every parameter
  spin box (`_install_wheel_scroll_passthrough`/`eventFilter`), so scrolling
  over a spin box scrolls the dialog instead of changing its value — a
  reasonable, self-contained addition now that the dialog has more rows
  (third radio, diameter spinbox, log panel) and no other code reaches into
  the dialog's internal layout.
- `tests/test_cellpose_aoi_mapping.py` — one new test,
  `test_dialog_aoi_percent_mode_round_trips_through_set_aoi_controls`, closing
  the only real gap in item 3 (only pixel-mode round-trip was tested before).

## Non-goals

No Suite2p algorithm-choice UI (sparsery only, per the library default), no
second-channel red-cell detection in the no-mask path, no log persistence/
filtering/export, no change to `run_roi_average_pipeline` (stays
mask-required) or to item 3's existing implementation.

## Verification

`ruff check` + `ruff format --check` + `mypy --ignore-missing-imports` clean
on both touched source files. Targeted suites
(`test_importer_tif.py`, `test_tif_import_dialog.py`,
`test_cellpose_aoi_mapping.py`, `test_cellpose_import_dialog.py`) — 83 tests,
all passing. Full suite: 1279 passed, 2 skipped, 0 failed.
