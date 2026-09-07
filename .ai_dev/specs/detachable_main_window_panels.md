# Detachable main-window panels

## Overview

Right-click any of 7 main-window panels (Traces, Spikes, Data, Heatmap,
Spatial Footprints, Custom Analysis, Console, Chat) for "Detach panel" (pop
into its own interactive window, hide the original) or "Detach copy of
panel" (same, but the original stays visible). Stimuli is deliberately
excluded — its widget is entangled with a combo box and toolbar living in
separate parts of the main window, making a clean detach structurally
awkward; documented as a follow-up rather than solved here.

## Change

- New `src/neuroseq/detached_panel_window.py` — `DetachedPanelWindow`, a
  generic top-level `QWidget` (mirrors the one existing precedent,
  `CustomAnalysisWindow`) wrapping any panel widget in a `QScrollArea`, with
  `closeEvent` explicitly closing the embedded widget (GPU/fastplotlib
  cleanup) before an optional `on_close` callback runs.
- `src/neuroseq/timeseries_widget.py` — `TraceWidget`/`SpikeWidget` (which
  already build their own right-click menu) get the two new actions added to
  `_rebuild_custom_analysis_menu`, gated by a new `SUPPORTS_DETACH` class
  attribute (`True` on the base class, overridden to `False` on `StimWidget`
  in `stim_widget.py` — avoids an import cycle an `isinstance` check would
  have needed).
- `src/neuroseq/gui.py` — the other 6 panels (none of which had a context
  menu before) get one via Qt's `customContextMenuRequested` policy,
  installed on the panel container AND its nested canvas/view widgets so a
  right-click lands correctly regardless of which visual layer the cursor is
  over (`_install_detach_context_menus`). `Gui.detach_panel(source,
keep_original)` is the single mechanism behind both actions — it builds a
  fresh instance via a per-panel-type `_build_fresh_*_widget` factory (never
  reparents the live widget, avoiding untested cross-window GPU-canvas
  reparenting), tracks it in `self._detached_panel_windows`, and — when
  `keep_original` is `False` — hides the original via `QAction.trigger()`
  (not `setChecked()`, so the normal `_set_widget_visible` side effects still
  run). `Gui.closeEvent` sweeps `findChildren(DetachedPanelWindow)` the same
  way it already sweeps `CustomAnalysisWindow`. Each factory seeds from the
  canonical source (`Project`/comboboxes), not from transformed/lost widget
  state (`HeatmapWidget` stores a transformed ndarray, not the original
  `SynapseData`; `CustomAnalysisWidget` doesn't retain its inputs at all — a
  new `Gui._last_custom_analysis` field, set whenever the docked panel is
  updated, plugs that gap).

## Bug found and fixed during review

`_build_fresh_console_widget`'s initial version reconnected the fresh
console's `close_requested` signal (emitted by an embedded close button
inside the widget) to the _original_ docked console panel's view-toggle
action — copy-pasted from `Gui.__init__`'s wiring for the original instance.
Reproduced: clicking that button on a **detached copy** hid the main
window's docked Console panel while the detached window itself stayed open.
Fixed by wiring `close_requested` generically in `detach_panel()` instead,
connecting it to the newly-created `DetachedPanelWindow.close` — so a
panel's own embedded close control now always closes the window it's
actually in. Added a regression test,
`test_detached_console_copy_close_button_closes_its_own_window_only`.

## Non-goals

Stimuli panel (see Overview). Live-mirroring later main-window changes into
an already-detached/copied panel. True widget reparenting. Persisting
detached-window layout across restarts.

## Verification

`ruff check` + `ruff format --check` + `mypy --ignore-missing-imports` clean
on all touched source files. Targeted suites (`test_gui.py`,
`test_trace_widget.py`, `test_spike_widget.py`, `test_stim_widget.py`,
`test_detached_panel_window.py`) — 132 tests, all passing (including the new
regression test for the fix above). Full suite: 1299 passed, 2 skipped, 0
failed.
