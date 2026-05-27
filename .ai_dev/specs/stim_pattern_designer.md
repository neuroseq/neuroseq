# Design stimulation pattern — guided pattern generator dialog

A new non-modal dialog launched from the stim editor's toolbar that guides
the user through building common stimulus patterns and writes them into the
currently selected stim array. Each Apply is fully undoable as a single
step — including the project pad that grows the stim arrays when the
pattern would otherwise overrun. The dialog stays open between Applies so
the user can stack patterns continuously; a cursor auto-advances to the
end of the last placed pattern (which already carries a trailing IPI) so
the next Apply lands on a clean canvas.

## Files

| Path                                  | Change                                                                                                                                                                                                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/neuroseq/stim_patterns.py`       | NEW — pure-math generators (sequential / simultaneous / Für Elise / ramp / per-neuron with intensity profile) plus `_per_pulse_magnitudes` and `pattern_total_samples` helpers                                                                                 |
| `src/neuroseq/stim_pattern_dialog.py` | NEW — non-modal `StimPatternDialog` with scope + pattern radios, parameter spins, Apply / Reset start time / Clear / Close buttons, and a `StimPatternContext` dataclass                                                                                       |
| `src/neuroseq/stim_array_modifier.py` | New `ResizableReversibleAction` (one-step undo for pad + slice overwrite); bounds guard added to `ReversibleAction._asign_values` so a future undo that shrinks an array silently no-ops stale slice actions                                                   |
| `src/neuroseq/stim_editor.py`         | new `Design stimulation pattern` toolbar action, `_pattern_dialog` singleton, `open_pattern_designer`, `apply_pattern`, `_current_pattern_context`, controller-limits refresh on undo/redo, dialog context refresh on stim selection and on clear              |
| `src/neuroseq/example_projects.py`    | Getting Started reverted to the 10-neuron sequential stim (100 ms / 18 ms / 1.0 / "siemens" / dt=1 ms) — restores it as the canonical reference for the dialog's defaults                                                                                      |
| `tests/test_stim_patterns.py`         | NEW — 10 pure-math tests (sequential layout + trailing IPI, simultaneous, Für Elise with first-13 pitches + chord + clip + multi-repeat IPI, per-neuron Constant / Linear / Log / Random with input validation)                                                |
| `tests/test_stim_pattern_dialog.py`   | NEW — 8 qtbot integration tests (Apply + cursor advance, double-Apply stacking, undo/redo, pad-with-undo as one step, Clear via undoable path, specific-neuron Constant, Für Elise clipped to 6 neurons, context update on stim switch, invalid-pulse warning) |
| `tests/test_example_projects.py`      | New test asserting Getting Started's 10-neuron sequential layout                                                                                                                                                                                               |

## Pattern math (`stim_patterns.py`)

Every generator returns an `(n_samples, n_neurons)` `np.ndarray` of dtype
`float`, with a **trailing IPI** built into the last `ipi_n` samples so
consecutive Applies stack cleanly. All generators share a
`_validate_common(n_neurons, pulse_ms, ipi_ms, repeats, dt_seconds)`
helper that:

- raises `ValueError` for `n_neurons < 1`, `pulse_ms <= 0`, `ipi_ms < 0`,
  `repeats < 1`, or `dt_seconds <= 0`
- converts ms → samples via `int(round(ms / 1000 / dt_seconds))`

Generators:

- `generate_sequential(n_neurons, pulse_ms, ipi_ms, magnitude, dt_seconds)`
  — Each neuron fires alone in turn for `pulse_n` samples, then `ipi_n`
  zeros. Shape `(n_neurons * (pulse_n + ipi_n), n_neurons)`.
- `generate_simultaneous(n_neurons, pulse_ms, ipi_ms, magnitude, repeats, dt_seconds)`
  — All neurons fire together. Shape `(repeats * (pulse_n + ipi_n), n_neurons)`.
- `generate_fuer_elise(n_neurons, pulse_ms, ipi_ms, magnitude, repeats, dt_seconds)`
  — Walks `_FUER_ELISE_INTRO` (reused from `example_projects.py`, 24
  events including chords). Per event, writes `magnitude` to every pitch
  in the event whose index is `< n_neurons`; pitches outside that range
  are silently dropped. Phrase repeated `repeats` times. Shape
  `(repeats * 24 * (pulse_n + ipi_n), n_neurons)`.
- `generate_ramp(n_neurons, ramp_ms, ipi_ms, start_magnitude, end_magnitude, repeats, dt_seconds, neuron_index=None)`
  — Continuous linear ramp from `start_magnitude` to `end_magnitude` over
  `ramp_ms`. Repeated `repeats` times separated by `ipi_ms` of zeros, with
  a trailing IPI. When `neuron_index is None`, every column gets the same
  ramp (all-neurons scope). When set, only that column gets the ramp
  (specific-neuron scope) — `neuron_index` outside `[0, n_neurons)` raises
  `ValueError`. Shape `(repeats * (pulse_n + ipi_n), n_neurons)`.
- `generate_per_neuron(neuron_index, n_neurons, pulse_ms, ipi_ms, profile, value1, value2, repeats, dt_seconds)`
  — Single-column train of `repeats` pulses on `neuron_index`. Per-pulse
  magnitudes are produced by `_per_pulse_magnitudes`:

  | profile     | per-pulse values                                                       |
  | ----------- | ---------------------------------------------------------------------- |
  | Constant    | `np.full(repeats, value1)`                                             |
  | Linear      | `np.linspace(value1, value2, repeats)`                                 |
  | Logarithmic | `np.geomspace(value1, value2, repeats)` (requires both positive)       |
  | Random      | `np.random.uniform(min(value1, value2), max(value1, value2), repeats)` |

  Only Constant and Linear are surfaced in the dialog; Logarithmic and
  Random remain available as library entry points (and stay covered by
  `tests/test_stim_patterns.py`).

## Compound undo (`stim_array_modifier.py`)

`ResizableReversibleAction` wraps a pad-and-overwrite as one undoable
step:

- Snapshots `before_lengths: dict[str, int]` for every stim array in the
  project before the action runs.
- `__call__` (do):
  1. `project.pad_stim_arrays(self.new_length * target_dt)` — pads every
     array independently using each array's own `dt`.
  2. snapshots `before_target_values = target.values[time_slice, neuron_slice].copy()`
     for the overwrite undo.
  3. writes `new_values` into the target slice.
  4. refreshes the stim widget and calls `project.update_limits()`.
- `undo`: restores the target slice values, then truncates every array
  back to its `before_lengths[name]` (in samples) regardless of dt
  mismatches.

`ReversibleAction._asign_values` gained a small bounds guard: if the
recorded slice now extends past the array's current shape (because a
later undo of a `ResizableReversibleAction` shrunk the array), the action
silently no-ops instead of crashing. This keeps prior slice-based actions
in the same modifier's stack safe across pad/shrink cycles.

## Editor wiring (`stim_editor.py`)

- New toolbar action `Design stimulation pattern` (placed between Loop
  stim array and Show all stim arrays), with icon lookup fallback chain
  `mdi6.creation` → `mdi6.auto-fix` → `mdi6.gesture-tap-button` →
  `mdi6.pencil`.
- `StimEditor._pattern_dialog` is a singleton (lazily created in
  `open_pattern_designer`). The dialog is non-modal; on open it raises
  itself and gets a fresh `StimPatternContext`.
- `select_stim` and `clear_stim_array` push updated context / reset the
  start-time cursor on the dialog when present, so the dialog tracks
  whichever stim array the user is viewing.
- `apply_pattern(neuron_slice, time_slice, new_values)`:
  1. Builds a `ResizableReversibleAction` from the current modifier's
     stim array.
  2. Truncates the modifier's action stack to `current_index + 1`,
     appends the new action, runs it, advances `current_index`.
  3. **Clears every OTHER modifier's `action_stack`** (and resets
     `current_index = -1`) — their slice-based actions would be invalidated
     by future undo of this compound action shrinking the project.
  4. Refreshes `controller.set_limits` and emits `limits_modified`.
- `undo` / `redo` now also refresh the controller's `set_limits` and emit
  `limits_modified` so the project-length-dependent UI stays in sync
  when a `ResizableReversibleAction` is popped or replayed.

## Dialog (`stim_pattern_dialog.py`)

Non-modal `QDialog` (`setModal(False)`). Layout, top to bottom:

1. **Target labels** — stim array name, units, `dt` (read-only).
2. **Scope** group — "All neurons" vs. "Specific neuron" (with neuron-index
   spinbox).
3. **Pattern type** group — radios filtered by scope:
   - All neurons: Sequential / Simultaneous / Für Elise / Ramp stimulus
   - Specific neuron: Constant / Linear gradient / Ramp stimulus

   "Ramp stimulus" lives in both scopes and keeps its selection across a
   scope toggle. Picking it routes through `generate_ramp` with
   `neuron_index=None` (all) or `neuron_index=int` (specific).

4. **Parameters form** (rows shown/hidden by `_pattern_changed`):
   - Pulse duration (ms) — default 100.0 (also used as the ramp duration when Ramp is selected)
   - Inter-pulse interval (ms) — default 18.0
   - Magnitude — default 1.0 (visible for All-neurons non-Ramp patterns + Specific-Constant)
   - Initial / Final magnitude — defaults 1.0 / 2.0 (visible for Specific-Linear and Ramp in both scopes)
   - Repeats — default 1, range `[0, 10000]` (visible for every pattern except All-Sequential)
   - Start time (ms) — default 0.0, range `[0, +∞)`, auto-advances after Apply
5. **Buttons** — Apply / Reset start time / Clear stim array / Close.

`apply_pattern` validates inputs (positive pulse, non-negative IPI,
`repeats >= 1` when visible, all finite, Log requires positive
magnitudes), generates the array via the appropriate `generate_*`
function, then calls `self.editor.apply_pattern(slice(0, n_neurons),
slice(start_samples, end_samples), pattern)` and advances the Start time
spin to `end_samples * dt * 1000` ms. The trailing IPI inside the
pattern itself is the safety gap — the cursor lands at exactly the
position right after that IPI, ready for the next pulse.

Reset start time → set spin to 0.0. Clear → delegates to
`editor.clear_stim_array()` (already an undoable path via
`replace_stim_values_with_undo`). Close → `self.hide()` (state preserved
for next open).

## Getting Started realignment (`example_projects.py`)

The Getting Started example was reset to the 10-neuron sequential
pattern (100 ms pulses, 18 ms gaps, `amp = 1.0`, `units = "siemens"`,
`dt = 1 ms`) so the dialog's defaults stay anchored to a project the
user can reach from the GUI. `tests/test_example_projects.py` gained
`test_default_project_uses_10_neurons_and_sequential_stimulus`, which
walks each neuron's pulse window and asserts the expected magnitudes
and silences across the off-target columns.

## Tests

`tests/test_stim_patterns.py` (pure math, no Qt):

- `test_sequential_default_matches_getting_started`
- `test_sequential_includes_trailing_ipi`
- `test_simultaneous_all_columns_fire_together`
- `test_fuer_elise_uses_first_13_neurons`
- `test_fuer_elise_clips_to_available_neurons`
- `test_fuer_elise_repeats_with_trailing_ipi_between`
- `test_per_neuron_constant`
- `test_per_neuron_linear_intensity_ramps`
- `test_per_neuron_logarithmic_requires_positive`
- `test_per_neuron_random_seeded_reproducible` (range check only — random is not seeded inside the generator)
- `test_invalid_inputs_raise`
- `test_ramp_all_neurons_broadcasts_same_ramp`
- `test_ramp_specific_neuron_only_target_column`
- `test_ramp_includes_trailing_ipi`
- `test_ramp_invalid_neuron_index_raises`

`tests/test_stim_pattern_dialog.py` (qtbot integration):

- `test_dialog_apply_sequential_writes_pattern_and_advances_cursor`
- `test_dialog_apply_twice_stacks_two_patterns`
- `test_dialog_apply_undo_redo`
- `test_dialog_apply_with_padding_is_one_undo`
- `test_dialog_clear_button_clears_via_undoable_path`
- `test_dialog_specific_neuron_constant_writes_only_that_column`
- `test_dialog_fuer_elise_clipped_to_six_neurons`
- `test_dialog_context_updates_when_user_switches_stim_array`
- `test_dialog_invalid_pulse_duration_shows_warning_and_no_change`
- `test_dialog_apply_ramp_all_neurons`
- `test_dialog_apply_ramp_specific_neuron`
- `test_dialog_ramp_visible_in_both_scopes`

## Known minor — set_data on undo-after-switch

`ResizableReversibleAction._refresh_stim_widget` calls
`stim_widget.set_data(target)` unconditionally. If the user applies a
pattern, switches the stim combo to a different array, then undoes, the
widget force-refreshes to display the action's _target_ array instead of
the user's current selection. This is non-crashing and only matters in
that specific sequence; tests don't exercise it. A cleaner fix is to
lift the refresh out of the action and into `StimEditor.undo / redo`,
keying off `self.current_stim_modifier.stim_array`. Out of scope here.

## Out of scope

- Mid-array editing (the cursor advances; previous patterns aren't
  re-mixed with new ones).
- Multi-array atomic patterns (Für Elise stays single-array; no
  exc / inh opsin pair generation).
- Preview before Apply (the user can Apply → eyeball → undo).
- A keyboard shortcut for the toolbar button.
- Auto-shrinking the project when an Apply leaves trailing padded space.
