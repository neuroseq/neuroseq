# Igor2 Import Dialog — Waveform Preview Plot

## Overview

Both Igor2 import dialogs display an embedded matplotlib line plot of the selected
wave immediately below the text metadata block. The plot updates every time a
wave is selected (whether from a PXP list or a direct .ibw/.csv load) and is
hidden when no wave is loaded.

Applies to:

- `src/neuroseq/igor_import_dialog.py` — `IgorImportDialog` (stim import)
- `src/neuroseq/igor_trace_import_dialog.py` — `IgorTraceImportDialog` (trace import)

---

## UI changes

Inside the **"Wave Preview"** `QGroupBox`:

| Element             | Detail                                                                          |
| ------------------- | ------------------------------------------------------------------------------- |
| `self.preview_text` | `setMaximumHeight(120)` added to keep the text block compact                    |
| `self.wave_plot`    | `MatplotlibWidget`, `minimumHeight=200`, `maximumHeight=260`, hidden by default |

Both are stacked vertically in `preview_layout`. The plot is invisible until a
wave is loaded, so the dialog height is unchanged on first open.

---

## New methods

### `_update_wave_plot(wave: IgorWave | None) -> None`

Called from `_update_preview()` as its first statement.

- **`wave is None`** — clears the figure, hides `wave_plot`.
- **wave provided** — builds a downsampled (max 4 000 pts, step = `len // 4000`)
  line plot:
  - X axis: `np.arange(0, len, step) * source_dt_spin.value()` in seconds
  - Y axis: `wave.units` or `"a.u."` if absent
  - Title: `wave.name`
  - Fonts: labels 8 pt, tick labels 7 pt
  - `fig.tight_layout(pad=0.5)` before `draw()`
  - Shows `wave_plot`.

### `_cleanup_wave_plot() -> None`

Resets `canvas._draw_pending = False` and `canvas._is_drawing = False` to prevent
a stale Qt draw callback from firing after the dialog is destroyed.
Called from both `reject()` (via `_cleanup_wave_plot` before `super().reject()`)
and `closeEvent()`.

---

## Defensive `_safe_draw_idle` guard

At construction time, the original `canvas._draw_idle` is wrapped in a closure
that swallows `RuntimeError` containing `"deleted"`. This prevents a crash if
the Qt widget is destroyed while a deferred draw is still pending.

```python
original_draw_idle = self.wave_plot.canvas._draw_idle

def _safe_draw_idle(canvas) -> None:
    try:
        original_draw_idle()
    except RuntimeError as exc:
        if "deleted" not in str(exc).lower():
            raise

self.wave_plot.canvas._draw_idle = MethodType(_safe_draw_idle, self.wave_plot.canvas)
```

---

## Tests

| File                                     | Tests                                                             |
| ---------------------------------------- | ----------------------------------------------------------------- |
| `tests/test_igor_import_dialog.py`       | hidden-initially, visible-after-.ibw-load (IgorImportDialog)      |
| `tests/test_igor_trace_import_dialog.py` | hidden-initially, visible-after-.ibw-load (IgorTraceImportDialog) |

32 tests pass across both new test files; full suite 547 passed.
