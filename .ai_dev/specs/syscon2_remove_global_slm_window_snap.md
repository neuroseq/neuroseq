# SysCon2 Sanity Check — Remove "Snap to Allowed Windows"

## Overview

The SysCon2 sanity-check flow in the stim-generator dialog used to run a
"global 16.666 ms SLM window" alignment check as its fifth step. When raw stim
starts didn't land on the 32-tick grid, a dialog prompted the user with three
buttons: **Snap To Allowed Windows** (auto-moved the raw drawing), **Continue
Without Fix** (no-op), or **Cancel**. With the spiralling + preblank-insertion
pipeline now landing every pulse on the global grid automatically, the upfront
warn-and-snap was obsolete and confusing — it never produced a different export
than letting the pipeline run.

This change is a pure deletion: 350 lines removed across three files, zero
added.

Applies to:

- `src/neuroseq/exporter_syscon2.py` — `GlobalSlmWindowViolation`,
  `find_global_slm_window_violations`, `snap_stim_values_to_global_slm_windows`,
  `_nearest_global_slm_window_start_tick`
- `src/neuroseq/stimulation_generator_dialog.py` — three imports, the
  `_GlobalWindowFixChoice` inner enum, step 5 of `_run_sanity_check`, and the
  `_prompt_global_slm_window_fix_choice` method
- `tests/test_stimulation_generator_dialog.py` — one import, one
  `_patch_no_slm_violations` monkeypatch line, the `_fake_global_window_violation`
  helper, and two tests
  (`...cancels_on_global_slm_window_violation`,
  `...snaps_to_global_slm_windows`)

---

## Why the upfront snap was redundant

The exporter's `_apply_preblank_insertions_to_raw_segments` (in
`exporter_syscon2.py`) shifts every raw stim segment forward by one full
16.666 ms SLM window before scheduling. That guarantees every
`adjusted_raw_segment.start_idx` lands on the global grid (modulo sub-window
resolution, which is fine for TTL pulses and irrelevant for shape windows). The
"Snap To Allowed Windows" auto-fix mutated the user's drawn stim to align with
the same grid before that automatic shift happened — the two operations never
diverged on output. The "Continue Without Fix" branch was a no-op falling
through to the same downstream pipeline.

---

## What was removed

| File                                         | Item                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `exporter_syscon2.py`                        | `@dataclass GlobalSlmWindowViolation` (7 fields)                                                     |
| `exporter_syscon2.py`                        | `find_global_slm_window_violations(stim_array, *, active_threshold, global_window_origin_tick)`      |
| `exporter_syscon2.py`                        | `snap_stim_values_to_global_slm_windows(stim_array, *, active_threshold, global_window_origin_tick)` |
| `exporter_syscon2.py`                        | `_nearest_global_slm_window_start_tick(...)` (helper used only by the above two)                     |
| `stimulation_generator_dialog.py`            | Three `from neuroseq.exporter_syscon2 import …` lines                                                |
| `stimulation_generator_dialog.py`            | Inner `class _GlobalWindowFixChoice(Enum)` (SNAP/CONTINUE/CANCEL)                                    |
| `stimulation_generator_dialog.py`            | Step 5 block in `_run_sanity_check` (detection + dispatch on SNAP/CONTINUE/CANCEL)                   |
| `stimulation_generator_dialog.py`            | `_prompt_global_slm_window_fix_choice(violations) -> _GlobalWindowFixChoice` method                  |
| `tests/test_stimulation_generator_dialog.py` | `from neuroseq.exporter_syscon2 import GlobalSlmWindowViolation`                                     |
| `tests/test_stimulation_generator_dialog.py` | `_patch_no_slm_violations` monkeypatch line for `find_global_slm_window_violations`                  |
| `tests/test_stimulation_generator_dialog.py` | `_fake_global_window_violation()` helper                                                             |
| `tests/test_stimulation_generator_dialog.py` | `test_stimulation_generator_dialog_cancels_on_global_slm_window_violation`                           |
| `tests/test_stimulation_generator_dialog.py` | `test_stimulation_generator_dialog_snaps_to_global_slm_windows`                                      |

A repo-wide grep for any of the deleted symbols after the patch returns zero
matches.

---

## Tests

| Before | After | Delta                                |
| ------ | ----- | ------------------------------------ |
| 83     | 81    | exactly the two deleted dialog tests |

All remaining 81 tests in `tests/test_exporter_syscon2.py` and
`tests/test_stimulation_generator_dialog.py` pass with no other modifications.
The exporter unit tests were unaffected because the global-window detection
never had its own exporter-level coverage.

---

## Behaviour change for the user

When opening the stim-generator dialog and clicking "Check sanity":

- **Before**: if any drawn stim start was off the global SLM grid, a modal
  appeared asking the user to snap / continue / cancel. Picking "Snap"
  rewrote the user's stim values; picking "Continue" did nothing extra.
- **After**: the detection and modal are gone. The sanity check proceeds
  directly through the remaining steps; the exporter's internal preblank
  insertion handles grid alignment at export time without ever mutating the
  user's drawing.
