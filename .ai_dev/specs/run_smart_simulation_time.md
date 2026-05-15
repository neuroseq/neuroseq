# Run dialog — Smart default + auto-tile on overshoot

## Overview

When the user clicks **Run simulation**, the simulation-time prompt
pre-fills with the length of the **meaningfully-filled** portion of the
project's stim arrays — for the default ELISE example with its 1.92 s
Für Elise stim, that's 1.92 s rather than the historical hard-coded
"1.0". If the user accepts the suggestion (or types any value), the
external-simulation dispatch tiles every stim array (`np.tile` + partial
tail) so each one exactly covers the absolute-time window the simulator
will read. Tiling replaces the previous `pad_stim_arrays` zero-padding,
which produced an unintended 0 mV teaching nudge past the end of the
prepared pattern.

The tile-target uses cumulative simulated time
(`external_simulator.simulated_time_seconds + this_run_seconds`), so
multi-run continuation extends the stim across runs rather than
clamping to the last sample of the original. This applies to **external
simulations only** — the Brian2 dispatch path keeps its existing
`Project.pad_stim_arrays` zero-pad semantics, which is the historical
Brian2 contract.

The "meaningful" length is computed by a small helper that trims any
**uniform-and-constant** trailing block — both conditions are required.
A multi-channel last row with only one channel active (e.g., the final
pitch-5 event in the default Für Elise stim) is signal, not rest, and
is preserved.

Applies to:

- `src/neuroseq/project.py` — `_meaningful_row_count` helper,
  `Project.suggested_simulation_time_seconds`,
  `Project.tile_stim_arrays_to_length`
- `src/neuroseq/utils.py` — `getSimulationTime` gains a `default_seconds`
  kwarg
- `src/neuroseq/gui.py` — both Run-button paths pass the suggested
  default into the dialog; `_do_external_simulation` swaps pad → tile
  with a cumulative-time offset
- `tests/test_project_suggested_sim_time.py` (new) — 8 unit tests
- `tests/test_project_tile_to_length.py` (new) — 6 unit tests
- `tests/test_gui_external_simulate_dispatch.py` (extended) — 2 new
  qtbot tests

---

## The "meaningful length" rule — `_meaningful_row_count`

A stim's "meaningfully filled" length is the index of the last row that
is _not_ part of a uniform-and-constant trailing tail:

```python
def _meaningful_row_count(values: np.ndarray) -> int:
    """Return n such that values[n:] is a uniform-and-constant tail.

    A "tail" must satisfy both:
    * the tail row is uniform across channels (all elements equal), so it
      represents "every neuron at the same rest value", and
    * the row is repeated identically from some index onward.

    Multi-channel patterns whose last event has only one channel active are
    not uniform and are left intact.
    """
    n_rows = int(values.shape[0])
    if n_rows == 0:
        return 0
    tail = values[-1]
    if tail.size > 0 and not np.all(tail == tail.flat[0]):
        return n_rows
    tail_len = 0
    for i in range(n_rows - 1, -1, -1):
        if np.array_equal(values[i], tail):
            tail_len += 1
            continue
        break
    if tail_len == n_rows:
        return 0
    return n_rows - tail_len
```

The two-condition rule (uniform row AND repeated tail) is what makes
the Für Elise default produce 1920, not 1910: its final event spans 10
identical rows but each row has one channel active and the others at
rest, so the row is not uniform and the trim never starts. A genuine
all-rest tail (`np.pad` zeros, or every channel at `E_l`) is uniform
across channels and gets trimmed correctly.

Pure helper, no GUI dependency; lives next to `Project` in
`project.py`.

---

## `Project.suggested_simulation_time_seconds`

```python
def suggested_simulation_time_seconds(self) -> float:
    """Return the longest meaningfully-filled duration across stim arrays."""
```

- Walks every stim array, applies `_meaningful_row_count`, multiplies
  by that stim's `dt`, returns the max.
- A stim array whose rows are all identical (degenerate "no signal")
  contributes its full duration so the user can still run something
  for that length — never a zero suggestion.
- Empty project (no stim arrays) returns `self.max_time_secs` if
  positive, else `1.0` (matches the historical default).

The suggestion is **independent of cumulative simulated time** —
clicking Run a third time on a Für Elise project still suggests 1.92 s,
not "remaining time." Easier mental model than "remaining"; the
dispatch-side tile handles whatever absolute-time window the user
actually requests.

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

`gui.py` changes are all confined to existing handlers. No new
toolbar action, no new dialogs.

**`btn_continue_simulation_clicked`** (the main "Run simulation"
action) and **`btn_do_external_simulation_clicked`** (the standalone
"Run external stimulation function" action) both pass
`project.suggested_simulation_time_seconds()` as the default to
`getSimulationTime`. The dialog pre-fills with `f"{x:g}"` (so 1.92
renders as `"1.92"`, not `"1.920000"`); values that are non-finite or
non-positive fall back to the historical `"1.0"`.

**`_do_external_simulation`** replaces the previous

```python
self.project.pad_stim_arrays(total_time_seconds=simulation_time)
```

with

```python
cumulative_so_far = 0.0
sim = getattr(self.project, "external_simulator", None)
if sim is not None and hasattr(sim, "simulated_time_seconds"):
    cumulative_so_far = float(sim.simulated_time_seconds)
target_seconds = cumulative_so_far + simulation_time
self.project.tile_stim_arrays_to_length(target_seconds)
```

The `getattr/hasattr` guard handles the `runpy.run_path` legacy case
where `external_simulator` is `None` (an external sim loaded as a raw
function, not an `NN` instance) — in that case `cumulative_so_far = 0.0`
and the tile target is just `simulation_time`.

The Brian2 path is unchanged.

---

## Tests

`tests/test_project_suggested_sim_time.py` — 8 tests:

| Test                                                                       | Asserts                                                                                                                |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `test_suggested_time_for_fully_filled_stim`                                | `arange(300).reshape(100, 3)` (no constant tail) → 0.1 s                                                               |
| `test_suggested_time_trims_constant_trailing_tail`                         | Distinct rows 0-49, uniform-zero rows 50-99 → 0.05 s                                                                   |
| `test_suggested_time_full_length_when_all_rows_identical`                  | All zeros → 0.1 s (degenerate fallback to full length, not 0)                                                          |
| `test_suggested_time_takes_max_across_arrays`                              | Two stims with different meaningful lengths → max wins                                                                 |
| `test_suggested_time_no_stim_arrays_defaults_to_one_second`                | Empty project → 1.0 s                                                                                                  |
| `test_suggested_time_no_stim_arrays_uses_positive_max_time_secs`           | Empty project with `max_time_secs = 2.5` → 2.5 s                                                                       |
| `test_suggested_time_for_fuer_elise_default_is_one_point_nine_two_seconds` | The user-stated requirement: default Für Elise pattern → exactly 1.92 s, not 1.91. Multi-channel last event preserved. |
| `test_suggested_time_trims_single_channel_uniform_tail`                    | `[1.0]*30 + [0.0]*70` (single channel, uniform tail) → 0.03 s                                                          |

`tests/test_project_tile_to_length.py` — 6 tests:

| Test                                                   | Asserts                                                             |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `test_tile_to_length_extends_short_stim`               | `(100, 3)` → target 0.25 s → `(250, 3)`, two full tiles + half tile |
| `test_tile_to_length_is_noop_when_already_long_enough` | Stim already longer than target → unchanged                         |
| `test_tile_to_length_handles_exact_multiple`           | `(100, 3)` → target 0.2 s → `(200, 3)` exact tile                   |
| `test_tile_to_length_applies_to_all_stims`             | Two stims of length 100 → both extended to length 250               |
| `test_tile_to_length_rejects_negative_target`          | `tile_stim_arrays_to_length(-1.0)` raises `ValueError`              |
| `test_tile_to_length_skips_empty_arrays`               | `(0, 3)` stim is skipped via early `continue`                       |

`tests/test_gui_external_simulate_dispatch.py` — 2 new qtbot tests:

| Test                                                   | Asserts                                                                                                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_run_dialog_default_is_suggested_simulation_time` | Monkeypatched `QInputDialog.getText` captures its 5th positional arg (the default text). With suggested time 1.92, the default is `"1.92"`. |
| `test_run_with_overshoot_tiles_stim_before_dispatch`   | `(100, 2)` stim + sim_time = 0.25 s → before the simulate function fires, the stim is `(250, 2)`. Confirms tile happens before dispatch.    |

---

## Non-goals (intentionally not implemented)

- **In-wrapper stim modulo.** The clamp-to-last behaviour in
  `Elise.simulate` is left as-is; tiling-before-dispatch makes it
  irrelevant in practice (the simulator never reads past `stim_T - 1`).
  A modulo fallback inside the wrapper would be redundant.
- **Changing `pad_stim_arrays`.** Its signature, callers in the Brian2
  path, and behaviour stay unchanged. Only `_do_external_simulation`
  swaps from pad to tile.
- **Loop-forever stim semantics.** The tile call materialises the
  loops into the values array — the stim editor shows what was
  actually presented, not a one-cycle preview that the simulator
  internally loops. Different feature, separate spec.
- **Trimming the stim after a shorter-than-prepared Run.** If
  `sim_time < current_length`, the stim is left intact; the simulator
  just reads the prefix. The user's authored pattern is preserved.
- **Semantic "rest" detection.** The meaningful-length rule trims any
  uniform-and-constant tail without knowing what "rest" means for the
  simulator (e.g., E_l). A multi-channel non-uniform tail is signal,
  full stop. A user with an unusual all-active uniform tail (every
  channel at -50 mV) will see it trimmed; they can override the
  suggestion in the dialog.
