# SysCon2 Import — Auto-Align Project dt to Tick Resolution

## Overview

When the user imports a SysCon2 `.seq` file via **Import Data → Light Stimulus →
SysCon2**, the project's canonical sample interval (`project.dt`) is automatically
aligned to the SysCon2 tick resolution chosen in the dialog
(`tick_resolution_spin`). All existing `StimArray`s and `TraceData`s are
ZOH-remapped to the new dt before the SysCon2 sequence is imported, so no timing
precision is silently lost.

This mirrors the existing pattern in `IgorImportDialog._import_file` and
`IgorTraceImportDialog._import_file`.

Applies to:

- `src/neuroseq/syscon2_import_dialog.py` — `Syscon2ImportDialog._import_file`

---

## Two "ticks" reconciled

| Concept                 | Field                                      | Unit | Default                   |
| ----------------------- | ------------------------------------------ | ---- | ------------------------- |
| NeuroSeq timeline ticks | `project.dt`                               | s    | 0.001 (1 ms)              |
| SysCon ticks            | `Syscon2ImportDialog.tick_resolution_spin` | µs   | 500 (Holo4D); 50 (UGA-42) |

Conversion: `target_dt_seconds = tick_resolution_us / 1e6`.

---

## Flow inside `_import_file`

Inserted between the early-return guards (`self.sequence is None`,
`stim_array is None`) and the existing `try:` block that calls
`import_sequence_into_stim_array`:

1. Compute `target_dt = float(self.tick_resolution_spin.value()) / 1e6`.
2. If `not stim_dt_matches(target_dt, self.project.dt)`:
   - Build `affected = self.project.remappable_data_description(target_dt)`.
   - If `affected` is non-empty, prompt the user with `QMessageBox.question`
     showing both dts (formatted via `format_milliseconds`) and the affected
     data description. **No** → return early without importing or remapping.
   - Call `self.project.remap_all_data_to_dt(target_dt)`. This updates
     `project.dt`, ZOH-remaps every `StimArray.values`/`StimArray.dt` and every
     `TraceData.traces`/`TraceData.dt`, and calls `update_limits()`.
3. Continue with the unchanged `import_sequence_into_stim_array(...)` call.

The local `stim_array` reference captured before the remap is still valid —
`remap_all_data_to_dt` mutates each `StimArray` in place.

---

## Imports added

```python
from neuroseq.stimulation.stim_timing import format_milliseconds
from neuroseq.stimulation.stim_timing import stim_dt_matches
```

---

## Tests

In `tests/test_importer_syscon2.py`:

| Test                                                                  | Asserts                                                                                                           |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `test_syscon2_import_dialog_imports_into_project` (existing, updated) | After 500 µs import: `project.dt == 0.0005`, `stim.dt == 0.0005`, sample indices doubled (12, 13 instead of 6, 7) |
| `test_syscon2_import_dialog_remaps_existing_trace_data`               | `TraceData([1,2,3])` at 1 ms is ZOH-remapped to `[1,1,2,2,3,3]` at 500 µs; question prompt shown exactly once     |
| `test_syscon2_import_dialog_skips_remap_prompt_when_dt_matches`       | When project already at 500 µs, no `QMessageBox.question` call is made                                            |
| `test_syscon2_import_dialog_aborts_when_user_declines_remap`          | "No" → `project.dt` unchanged, `StimArray.values` unchanged, dialog not accepted                                  |

All 21 tests in `tests/test_importer_syscon2.py` pass; ruff and mypy clean on
the modified file.
