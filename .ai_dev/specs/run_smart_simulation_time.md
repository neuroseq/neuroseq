# Run dialog — Smart default + auto-tile on overshoot

## Overview

When the user clicks **Run simulation**, the simulation-time prompt
pre-fills with the **full duration** of the project's longest stim
array — for the default ELISE example with its 1.92 s Für Elise stim,
that's 1.92 s rather than the historical hard-coded "1.0". The same
single-field `getSimulationTime` dialog is used for both Brian2 and
external-simulation paths; there is no separate "ring-out" field. To
ring the network out at the end of a run, the user pads every stim
array with zeros (via the toolbar Pad action) and then runs — the
suggested time already reflects the padded length, so the user just
clicks OK.

On accept, the external-simulation dispatch tiles every stim array
(`np.tile` + partial tail) so each covers the absolute-time window the
simulator will read (`cumulative_so_far + simulation_time`). Tiling
replaces the previous `pad_stim_arrays` zero-padding inside
`_do_external_simulation`, which produced an unintended teaching nudge
past the end of the prepared pattern. The tile-target uses cumulative
simulated time, so multi-run continuation extends the stim across runs
rather than clamping to the last sample of the original.

The Brian2 dispatch path keeps its existing `Project.pad_stim_arrays`
zero-pad semantics inside `Project.run_simulation` — that's the
historical Brian2 contract and is unrelated to the external-sim flow.

### Why ring-out is implemented as Pad + Run, not as a dialog field

An earlier iteration added a "ring-out duration" field to the Run
dialog that, internally, called `Project.pad_stim_arrays` before
dispatching the simulation. That worked but introduced two artefacts of
the same effect — Pad + Run and Run-with-ring-out — that mutated the
stim arrays identically. They converged once the suggested time was
changed to use the full stim length: after a Pad, the dialog's default
already includes the trailing zeros, so clicking OK runs the
ring-out without any second field. The dialog stays single-field, the
Run-button code path is the smaller one it was before the ring-out
prototype, and Pad becomes a fully sufficient ring-out affordance.

Applies to:

- `src/neuroseq/project.py` — `Project.suggested_simulation_time_seconds`,
  `Project.tile_stim_arrays_to_length`
- `src/neuroseq/utils.py` — `getSimulationTime` gains a `default_seconds`
  kwarg
- `src/neuroseq/gui.py` — both Run-button paths pass the suggested
  default into `getSimulationTime`; `_do_external_simulation` swaps
  pad → tile with a cumulative-time offset
- `tests/test_project_suggested_sim_time.py` (new) — 7 unit tests
- `tests/test_project_tile_to_length.py` (new) — 6 unit tests
- `tests/test_gui_external_simulate_dispatch.py` (extended) — 2 new
  qtbot tests covering the default-prefill and the auto-tile dispatch

---

## `Project.suggested_simulation_time_seconds`

```python
def suggested_simulation_time_seconds(self) -> float:
    """Return the longest stim-array duration across the project.

    The full stim length is used as the suggested simulation time so
    that any pre-Run modification (Pad, Loop, Trim, manual editing)
    is faithfully reflected in the Run dialog's default. In particular,
    padding a stim with trailing zeros to create a "ring-out" period
    produces the same end state as the dialog would: the simulator
    reads the entire stim, including the trailing zeros, and produces
    no teaching current during that interval.
    """
```

Behaviour:

- Walks every stim array, multiplies its row count by `dt`, returns
  the max.
- Empty stim array contributes nothing (skipped).
- Empty project (no stim arrays) returns `self.max_time_secs` if
  positive, else `1.0` (matches the historical default).

The suggestion is **independent of cumulative simulated time** —
clicking Run a third time on a Für Elise project still suggests the
full stim duration, not "remaining time." Easier mental model than
"remaining"; the dispatch-side tile handles whatever absolute-time
window the user actually requests.

There is no "meaningful length" / trim-trailing-zeros heuristic.
Anything the user has put in the stim — including a deliberate
zero-padded ring-out tail — is part of the default. Anything the user
wants to _skip_ can be removed via the toolbar Trim action.

---

## `Project.tile_stim_arrays_to_length`

```python
def tile_stim_arrays_to_length(self, target_seconds: float) -> None:
    """Extend every stim array via tiling so each covers `target_seconds`."""
```

For each stim array `s`:

- If `s` already covers `target_seconds` (`s.values.shape[0] * s.dt >= target_seconds`), it is left alone.
- Otherwise the values are tiled along axis 0 to the smallest number of full repetitions that covers the target, then truncated to exactly `ceil(target_seconds / dt)` samples — so the last tile is partial when the target isn't a multiple of the original length.

Calls `self.update_limits()` at the end. Raises `ValueError` for
negative targets. No-op (early `continue`) for empty stim arrays.

---

## GUI dispatch

`gui.py` keeps a single `getSimulationTime` call for both paths,
sharing the same `default_seconds` argument:

```python
def btn_continue_simulation_clicked(self) -> None:
    simulation_time = getSimulationTime(
        self,
        default_seconds=self.project.suggested_simulation_time_seconds(),
    )
    if simulation_time is None:
        return
    if self.project.external_simulate_path is not None:
        self._do_external_simulation(simulation_time)
        return
    # ... Brian2 path unchanged ...
```

`getSimulationTime` pre-fills with `f"{x:g}"` (so 1.92 renders as
`"1.92"`, not `"1.920000"`); values that are non-finite or non-positive
fall back to the historical `"1.0"`.

`btn_do_external_simulation_clicked` (the standalone toolbar action)
uses the same dialog with the same default.

`_do_external_simulation(simulation_time)` replaces the previous

```python
self.project.pad_stim_arrays(total_time_seconds=simulation_time)
```

with the tile-only version:

```python
cumulative_so_far = 0.0
sim = getattr(self.project, "external_simulator", None)
if sim is not None and hasattr(sim, "simulated_time_seconds"):
    cumulative_so_far = float(sim.simulated_time_seconds)
target_seconds = cumulative_so_far + simulation_time
self.project.tile_stim_arrays_to_length(target_seconds)
```

The `getattr/hasattr` guard handles the `runpy.run_path` legacy case
where `external_simulator` is `None` — in that case `cumulative_so_far
= 0.0` and the tile target is just `simulation_time`.

The Brian2 path is unchanged (still calls `project.run_simulation`,
which internally uses `pad_stim_arrays`).

---

## Tests

`tests/test_project_suggested_sim_time.py` — 7 tests:

| Test                                                                       | Asserts                                                                                |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `test_suggested_time_for_fully_filled_stim`                                | `arange(300).reshape(100, 3)` → 0.1 s (full length)                                    |
| `test_suggested_time_includes_constant_trailing_tail`                      | Distinct rows 0-49, uniform-zero rows 50-99 → 0.1 s (no trimming; Pad+Run equivalence) |
| `test_suggested_time_full_length_when_all_rows_identical`                  | All zeros → 0.1 s                                                                      |
| `test_suggested_time_takes_max_across_arrays`                              | Two stims, lengths 80 and 120 → 0.12 s (longest wins)                                  |
| `test_suggested_time_no_stim_arrays_defaults_to_one_second`                | Empty project → 1.0 s                                                                  |
| `test_suggested_time_no_stim_arrays_uses_positive_max_time_secs`           | Empty project with `max_time_secs = 2.5` → 2.5 s                                       |
| `test_suggested_time_for_fuer_elise_default_is_one_point_nine_two_seconds` | Default Für Elise opsin stims (1920 samples each) → 1.92 s                             |
| `test_suggested_time_includes_single_channel_uniform_tail`                 | `[1.0]*30 + [0.0]*70` (single channel, uniform tail) → 0.1 s (no trimming)             |

`tests/test_project_tile_to_length.py` — 6 tests:

| Test                                                   | Asserts                                                             |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `test_tile_to_length_extends_short_stim`               | `(100, 3)` → target 0.25 s → `(250, 3)`, two full tiles + half tile |
| `test_tile_to_length_is_noop_when_already_long_enough` | Stim already longer than target → unchanged                         |
| `test_tile_to_length_handles_exact_multiple`           | `(100, 3)` → target 0.2 s → `(200, 3)` exact tile                   |
| `test_tile_to_length_applies_to_all_stims`             | Two stims of length 100 → both extended to length 250               |
| `test_tile_to_length_rejects_negative_target`          | `tile_stim_arrays_to_length(-1.0)` raises `ValueError`              |
| `test_tile_to_length_skips_empty_arrays`               | `(0, 3)` stim is skipped via early `continue`                       |

`tests/test_gui_external_simulate_dispatch.py` — 2 qtbot tests:

| Test                                                   | Asserts                                                                                                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_run_dialog_default_is_suggested_simulation_time` | Monkeypatched `QInputDialog.getText` captures its 5th positional arg (the default text). With suggested time 1.92, the default is `"1.92"`. |
| `test_run_with_overshoot_tiles_stim_before_dispatch`   | `(100, 2)` stim + sim_time = 0.25 s → before the simulate function fires, the stim is `(250, 2)`. Confirms tile happens before dispatch.    |

---

## Non-goals (intentionally not implemented)

- **A separate "ring-out" field on the Run dialog.** Explored, then
  removed (see Overview "Why ring-out is implemented as Pad + Run"
  above). The Pad toolbar action plus the full-length default suggestion
  cover the same workflow with one less UI control.
- **A "meaningful length" / trim-trailing-zeros heuristic.** Trailing
  uniform rows are kept in the suggested time. Users who want to skip
  them call the toolbar Trim action. The earlier heuristic
  (`_meaningful_row_count`) was deleted along with the ring-out dialog.
- **In-wrapper stim modulo.** The clamp-to-last behaviour in
  `Elise.simulate` is left as-is; tiling-before-dispatch makes it
  irrelevant in practice (the simulator never reads past `stim_T - 1`).
  A modulo fallback inside the wrapper would be redundant.
- **Changing `pad_stim_arrays`.** Its signature, callers in the Brian2
  path, and behaviour stay unchanged. Only `_do_external_simulation`
  swaps from pad to tile; the standalone Pad toolbar action still uses
  it.
- **Loop-forever stim semantics.** The tile call materialises the
  loops into the values array — the stim editor shows what was
  actually presented, not a one-cycle preview that the simulator
  internally loops. Different feature, separate spec.
- **Trimming the stim after a shorter-than-prepared Run.** If
  `sim_time < current_length`, the stim is left intact; the simulator
  just reads the prefix. The user's authored pattern is preserved.
