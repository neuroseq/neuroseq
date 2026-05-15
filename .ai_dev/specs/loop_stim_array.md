# Loop Stim Array — tile + optional periodicity-detect trim

## Overview

The stim editor has a **Loop stim array** toolbar action next to
`Pad / Trim / Clear stim array`. It opens a small dialog where the user
enters an integer loop count `N` and optionally clicks a
**Detect period and trim** button before looping. On OK the dialog
multiplies every project stim array's time axis by `N` (literal
`np.tile`-style repetition), keeping every stim array time-aligned the way
`Pad` and `Trim` already do.

Loop is a **reshape** operation — it changes the time-axis length of the
values array. The per-cell `StimArrayModifier` undo machinery
(`stim_array_modifier.py:14-42`) cannot represent reshapes (its
`ReversibleAction` requires a fixed shape), so Loop follows the same
pattern as Pad/Trim: it mutates the stim array values in place via
`Project.loop_stim_arrays(n_loops)` and then calls
`self.set_project(self.project, keep_undo_history=False)` to refresh the
view. Loop is therefore **not Ctrl-Z-undoable** — same as Pad/Trim.

Applies to:

- `src/neuroseq/project.py` — `Project.loop_stim_arrays` method
- `src/neuroseq/stim_editor.py` — `StimEditor.loop_stim_array` method + toolbar action
- `src/neuroseq/stim_loop_dialog.py` (new) — `StimLoopDialog`, `_detect_period`
- `tests/test_stim_loop.py` (new) — unit + qtbot tests

---

## Project API — `Project.loop_stim_arrays`

Sibling of `pad_stim_arrays` (`project.py:673`) and `trim_stim_arrays`
(`project.py:691`):

```python
def loop_stim_arrays(self, n_loops: int) -> None:
    """Tile every stim array's values along the time axis."""
    if n_loops < 1:
        raise ValueError(f"n_loops must be >= 1, got {n_loops}")
    if n_loops == 1:
        return
    for stim_array in self.stim_arrays.values():
        stim_array.values = np.tile(stim_array.values, (n_loops, 1))
    self.update_limits()
```

- `n_loops == 1` is a documented no-op.
- `n_loops < 1` raises `ValueError`.
- Operation is uniform across all stim arrays — there's no per-stim
  selection. This keeps time-alignment that the simulator and stim widgets
  rely on.

---

## Dialog — `StimLoopDialog`

Three rows in a `QVBoxLayout`:

1. **Status label** (read-only, top). Shows current length and the looped
   length, refreshed live on spinbox changes and after Detect-period trims:

   > `Current stim length: 2000.0 ms (2000 samples) -> after x 5 = 10000.0 ms`

2. **Detect-period row.** `QPushButton` "Detect period and trim" plus an
   adjacent caption that's initially empty. Tooltip on the button:
   _"Find the smallest repeating block in the current stim and trim the
   array to one period. Loop then composes that with the multiplier."_

   On click: runs `_detect_period` on the first stim array
   (`next(iter(project.stim_arrays.values()))`), and if a usable period is
   found, calls `project.trim_stim_arrays(period * dt)` and updates the
   caption to `"Trimmed to {ms} ms ({samples} samples) - 1 period"`. If no
   period is found, the caption says
   `"No periodic block detected; array unchanged."` and nothing is mutated.

3. **Spinbox row.** `QSpinBox` "Number of repetitions", range 1..1000,
   default 2. `QDialogButtonBox.StandardButton.Ok | Cancel`.

A `loop_count` property returns `int(self.spinbox.value())`.

---

## Periodicity detection — `_detect_period(values) -> int | None`

```python
def _detect_period(values: np.ndarray) -> int | None:
    """Return the smallest exact period of `values`, if one exists."""
    n_timestamps = int(values.shape[0])
    if n_timestamps < 2:
        return None
    for period in range(1, n_timestamps // 2 + 1):
        if n_timestamps % period != 0:
            continue
        if np.allclose(
            values[: n_timestamps - period],
            values[period:],
            rtol=0.0,
            atol=1e-6,
        ):
            return period
    return None
```

The `n_timestamps % period != 0` filter is deliberate: we only declare
periodicity when the cycle tiles cleanly. The default 2-second Für Elise
stim `(2000, 13)` (8 cycles of 240 samples + 80 trailing padding) returns
`None` because 2000 % 240 ≠ 0. After trimming to 1920 samples, the same
detector returns 240. If users want fuzzier detection, they Trim first.

---

## Toolbar wiring

Added in `stim_editor.py` right after the "Trim stim array
(Ctrl+Shift+Minus)" `add_action` block:

```python
add_action(
    "Loop stim array",
    self.loop_stim_array,
    menu,
    toolbar,
    self,
    icon=qta.icon("mdi6.repeat-variant"),
)
```

No keyboard shortcut. The `StimEditor.loop_stim_array` method opens the
dialog, runs `loop_stim_arrays` on accept, and always refreshes the view
via `set_project(..., keep_undo_history=False)` — even on cancel, because
the **Detect period and trim** button may have shrunk the stim before the
user pressed Cancel.

---

## Tests

`tests/test_stim_loop.py` — 12 tests, all green:

| Test                                                               | Asserts                                                                                                          |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `test_loop_stim_arrays_tiles_values`                               | A `(100, 4)` stim looped 3× has shape `(300, 4)` and equals `np.tile(orig, (3, 1))`.                             |
| `test_loop_stim_arrays_loops_all_stims_simultaneously`             | Two stims of length 100 → both length 200 after `loop_stim_arrays(2)`.                                           |
| `test_loop_stim_arrays_n_equals_one_is_noop`                       | Shapes and values unchanged.                                                                                     |
| `test_loop_stim_arrays_rejects_zero_and_negative`                  | `loop_stim_arrays(0)` and `loop_stim_arrays(-3)` raise `ValueError`.                                             |
| `test_detect_period_finds_minimum`                                 | A `(7, 3)` block tiled 5× → period 7. Same with one mutated sample → `None`.                                     |
| `test_detect_period_rejects_non_dividing`                          | `(2000, 13)` Für Elise-shaped stim with 80 padding → `None`. Trimmed to `(1920, 13)` → period 240.               |
| `test_detect_period_handles_short_arrays`                          | Empty or single-sample arrays return `None`.                                                                     |
| `test_stim_loop_dialog_accept_returns_loop_count` (qtbot)          | Spinbox set to 4, accept → `loop_count == 4` and `result() == Accepted`.                                         |
| `test_stim_loop_dialog_detect_period_trims_when_periodic` (qtbot)  | Periodic `(1920, 2)` stim → click Detect → project stim shrinks to 240, caption mentions "1 period".             |
| `test_stim_loop_dialog_detect_period_no_op_when_aperiodic` (qtbot) | Random aperiodic stim → click Detect → unchanged, caption matches "No periodic block detected; array unchanged." |
| `test_stim_editor_loop_stim_array_accepts_and_tiles` (qtbot)       | Mocked dialog returns Accepted with `loop_count=3` → `(4, 3)` stim becomes `(12, 3)`.                            |
| `test_stim_editor_loop_stim_array_rejected_refreshes_trim` (qtbot) | Mocked dialog trims the project then returns Rejected → editor still refreshes; final shape `(2, 3)`.            |

CI gates: ruff lint + format clean, mypy clean, full pytest 639 passed /
2 skipped, total coverage 78.63%, diff-coverage 89% on the new code.

The four diff-cover misses in `stim_editor.py` (lines 311, 319-320, 325)
are defensive guards — the
`current_stim_modifier is None / project is None` early `return` and the
`try / except / QMessageBox.critical` path inside `loop_stim_array`.
Neither is reachable in normal use; both kept as safety nets.

---

## Non-goals (intentionally not implemented)

- **Per-stim looping.** All stim arrays loop together. Per-stim counts
  would require rethinking how the simulator handles unequal-length stims.
- **Fuzzy / autocorrelation-based detection.** The detector requires
  `T % p == 0` and uses `np.allclose(rtol=0, atol=1e-6)`. Approximate
  detection (autocorrelation with a tolerance) is a separate feature.
- **Undo of Loop.** Loop is a reshape, so undo history is cleared via
  `keep_undo_history=False` — same as Pad/Trim today. Re-enabling undo
  for reshape ops would mean redesigning `StimArrayModifier` across the
  whole Pad/Trim/Loop family, not retrofitting Loop alone.
- **Looping a selected time range.** Loop tiles the entire values array.
  Users who want to loop a slice should Trim first.
