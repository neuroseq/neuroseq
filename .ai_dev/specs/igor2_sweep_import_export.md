# Igor2 multi-sweep IBW: import (concatenate) + export (auto-detect)

## Overview

The Igor Binary Wave format permits 2-D waves of shape `(samples, sweeps)`
where each column is one trial of a protocol. SutterPatch files like
`R5_S1_IVstep.ibw` (30000 samples × 7 sweeps, dt 50 µs, `dataUnits='A'`)
are the canonical example.

This change does two related things in neuroseq:

1. **Import**. Multi-sweep IBW files load successfully by **concatenating
   sweeps column-major into a single 1-D wave** (sweep 0 then sweep 1 …
   then sweep N−1). The rest of the importer (resampler, single-column
   StimArray/TraceData builders, resample-and-place machinery) is
   unchanged — concatenation happens once at parse time. Inter-sweep gaps
   from the recording's `SweepStartTimes` are **not** reconstructed; the
   sweeps abut. Both import dialogs (`igor_import_dialog`,
   `igor_trace_import_dialog`) surface a non-blocking info label
   ("Multi-sweep IBW: 7 sweeps × 30000 samples → concatenated to 210 000
   samples. Inter-sweep recording gaps are not preserved.") whenever a
   loaded wave has `n_sweeps > 1`.

2. **Export**. The single-wave export dialog
   (`Igor2SingleWaveExportDialog`) grows an opt-in checkbox **"Export as
   sweeps (auto-detect period)"**. When enabled, the exporter runs
   `detect_period` on the post-scaling column. If a period is found that
   splits the column evenly into ≥2 sweeps, the IBW is written as 2-D
   `(period, n_sweeps)` in Igor's column-major wire format. Otherwise the
   exporter falls back to 1-D silently (and `gui.py` logs an info-level
   line). CSV exports ignore the checkbox by design — the format has no
   sweep concept.

`detect_period` is now a public helper in `src/neuroseq/utils.py`, shared
between the loop dialog (its original caller) and the new export
checkbox. No behaviour change for the loop dialog.

Applies to:

- `src/neuroseq/utils.py` — new public `detect_period`
- `src/neuroseq/stim_loop_dialog.py` — imports `detect_period` from utils
- `src/neuroseq/importer_igor2.py` — 2-D concat path in
  `parse_igor_binary_wave_data`; `n_sweeps`/`samples_per_sweep` plumbed
  through `IgorWave`, `PreparedIgorWave`, `IgorImportReport`,
  `IgorTraceImportReport`
- `src/neuroseq/igor_import_dialog.py`,
  `src/neuroseq/igor_trace_import_dialog.py` — `multi_sweep_info` QLabel
- `src/neuroseq/exporter_igor2.py` — 2-D code path in `_write_ibw_v5`;
  `as_sweeps` keyword in `export_igor2_ibw`; `n_sweeps` field on
  `Igor2SingleWaveExport`
- `src/neuroseq/igor2_single_wave_export_dialog.py` — checkbox +
  `sweep_detection_status` live label
- `src/neuroseq/gui.py` — threads `as_sweeps`, logs CSV/fallback cases,
  augments success message with sweep split
- `tests/test_utils.py`, `tests/test_importer_igor2.py`,
  `tests/test_igor_import_dialog.py`,
  `tests/test_igor_trace_import_dialog.py`,
  `tests/test_exporter_igor2.py`,
  `tests/test_igor2_single_wave_export_dialog.py`,
  `tests/test_gui.py` — full coverage incl. real-file test guarded by
  `pytest.mark.skipif(not REAL_SUTTERPATCH_IBW.exists())`

---

## Import path

`parse_igor_binary_wave_data` now branches on
`np.squeeze(wData).ndim`:

| Squeezed ndim | Behaviour                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------- |
| 1             | Pass through; `n_sweeps=1`, `samples_per_sweep=N`                                               |
| 2             | `values = squeezed.ravel(order="F")`; `n_sweeps=shape[1]`, `samples_per_sweep=shape[0]`         |
| ≥3            | `ValueError("Igor wave has K dimensions after squeezing; only 1-D and 2-D … can be imported.")` |

A TODO is parked in the file pointing at SutterPatch's `note` text
(`ScaleFactor`, `ScaleOffset`, `SweepStartTimes`) as a future
improvement — none of those are applied yet. For
`R5_S1_IVstep.ibw` specifically the note has `ScaleFactor=1,
ScaleOffset=0`, so the lack of parsing is a no-op there.

The two new fields (`n_sweeps`, `samples_per_sweep`) flow through:

- `IgorWave` (defaults 1, None)
- `PreparedIgorWave` (copied from the source wave)
- `IgorImportReport` and `IgorTraceImportReport` (copied from prepared)

Downstream signature checks are unchanged because the new fields have
sensible defaults; existing call sites that build `IgorWave(...)`
manually (e.g. PXP-loaded waves, CSV-loaded waves) continue to compile
and report `n_sweeps=1`.

### Dialog info label

In both import dialogs a `QLabel` with `objectName="multi_sweep_info"`
is created hidden in `_setup_ui` and shown only when the loaded wave
has `n_sweeps > 1`. Its text:

```
Multi-sweep IBW: 7 sweeps × 30000 samples -> concatenated to
210 000 samples. Inter-sweep recording gaps are not preserved.
```

The light-stim dialog also surfaces the same fact in the resampled
post-import summary block, prefixed with "Source wave contained N
sweeps × M samples, concatenated on import."

---

## Export path

### `_write_ibw_v5` 2-D code path

The writer now accepts a 1-D or 2-D `ndarray`. For 2-D
`(samples, sweeps)`:

- `npnts = samples * sweeps`
- `nDim = (samples, sweeps, 0, 0)` (trailing zeros = "axis absent")
- `sfA = (dt, 1.0, 1.0, 1.0)` (only axis 0 carries a meaningful step)
- `dimUnits[0]` keeps the `x_units` argument; `dimUnits[1..3]` empty
- data block written column-major:
  `np.asarray(data, order="F").tobytes(order="F")`

3-D+ inputs raise `ValueError("IBW export expects a 1-D or 2-D …")`.

Round-trip verified by `test_write_ibw_v5_round_trips_two_d`: load via
`igor2.binarywave.load`, assert `wave_data.shape == (5, 3)` and per-cell
equality with the input.

### `export_igor2_ibw(as_sweeps=False)`

When `as_sweeps=True`, after the existing scaling step, the exporter
runs:

```python
period = detect_period(values)
if period is not None and period > 0 and values.size % period == 0:
    sweep_count = values.size // period
    if sweep_count > 1:
        write_values = values.reshape(sweep_count, period).T
        write_values = np.ascontiguousarray(write_values)
        n_sweeps_out = sweep_count
```

If any guard fails, `write_values` stays 1-D and `n_sweeps_out` stays 1.
The return value `Igor2SingleWaveExport` gained an `n_sweeps` field
(default 1) that the GUI uses to format the post-export message.

**Detection runs on the scaled column**, so a stim loop with
`scaling_factor=0.5` (or any uniform scale) still detects the same
period — `detect_period` is scale-invariant by construction. This is
exercised by `test_export_ibw_as_sweeps_applies_scaling_before_detection`.

### Dialog: checkbox + status

`Igor2SingleWaveExportDialog` adds two form-row widgets after the
scaling-factor row:

- A `QCheckBox` "Export as sweeps (auto-detect period)" with tooltip
  noting CSV ignores it.
- A `QLabel` (`objectName="sweep_detection_status"`, wordWrap on) that
  updates whenever:
  - the checkbox is toggled, or
  - the user picks a different neuron in the combo (the dialog's
    existing `_update_suggested_experiment_name` callback gained a
    tail call into `_update_sweep_status`).

The label texts are:

| Condition               | Text                                                          |
| ----------------------- | ------------------------------------------------------------- |
| Checkbox off            | (empty)                                                       |
| No period detected      | "No periodic block detected — will export as a single sweep." |
| Period == column length | "Only one repetition — will export as a single sweep."        |
| Useful period found     | "Period detected: P samples × N sweeps (T total)."            |

The dialog exposes the new state via the `as_sweeps` property; the
existing `scaling_factor` / `unit_override` / `experiment_name` /
`selected_neuron_index` properties are unchanged.

### GUI wiring

`btn_export_igor2_single_wave_clicked`:

- Forwards `dialog.as_sweeps` into `export_igor2_ibw`.
- If `as_sweeps=True` and the report comes back with `n_sweeps == 1`,
  logs an info-level fallback line.
- If `as_sweeps=True` and the user picked `.csv`, logs an info-level
  CSV-ignores-sweeps line.
- Augments the success `QMessageBox.information` body with a
  " Split into N sweeps × M samples." suffix when `n_sweeps > 1`. M is
  computed as `sample_count // n_sweeps`, which equals the detected
  period (sample_count is the original column length).

---

## Test inventory

`tests/test_utils.py` — 4 new tests for `detect_period`:

| Test                                            | Asserts                      |
| ----------------------------------------------- | ---------------------------- |
| test_detect_period_returns_period_for_repeat    | `[1,2,3]*4` → period 3       |
| test_detect_period_returns_none_for_non_repeat  | `[1,2,3,4,5,6,7]` → None     |
| test_detect_period_returns_none_for_short_array | `[1]` → None                 |
| test_detect_period_handles_2d_arrays            | `[[1,2],[3,4]]*3` → period 2 |

`tests/test_importer_igor2.py` — 5 new tests:

| Test                                                    | Asserts                                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| test_parse_binary_wave_accepts_two_d_and_concatenates   | (4×3) → 12-sample 1-D, sweep-by-sweep order, metadata set                |
| test_parse_binary_wave_rejects_three_d                  | (4×3×2) → ValueError                                                     |
| test_parse_binary_wave_keeps_one_d_unchanged            | 1-D input untouched, metadata defaults set                               |
| test_real_sutterpatch_file_imports (skipif not present) | Real R5_S1_IVstep.ibw → n_sweeps=7, samples_per_sweep=30000, dt=5e-5     |
| test_prepared_wave_carries_sweep_metadata               | n_sweeps/samples_per_sweep flow through `prepare_igor_wave_for_neuroseq` |
| test_load_igor_pxp_file_imports_two_d_waves             | PXP-embedded 2-D wave loads with concatenation                           |

`tests/test_igor_import_dialog.py` + `tests/test_igor_trace_import_dialog.py`
— 1 test each: load a synthetic multi-sweep `IgorWave`, open the dialog,
assert `findChild(QLabel, "multi_sweep_info")` shows the
"3 sweeps × N samples" line.

`tests/test_exporter_igor2.py` — 8 new tests for the writer/exporter:

| Test                                                       | Asserts                                                                                       |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| test_write_ibw_v5_two_d_emits_nDim_correctly               | nDim=(5,3,0,0), npnts=15                                                                      |
| test_write_ibw_v5_two_d_data_is_column_major               | (3,2) input → byte order [1,3,5,2,4,6]                                                        |
| test_write_ibw_v5_round_trips_two_d                        | (5,3) input round-trips through igor2.binarywave.load                                         |
| test_write_ibw_v5_rejects_three_d                          | (2,2,2) input → ValueError                                                                    |
| test_export_ibw_as_sweeps_writes_2d_when_period_detected   | `[1,2,3]*5` + as_sweeps → loaded shape (3,5), report.n_sweeps=5                               |
| test_export_ibw_as_sweeps_falls_back_to_1d_when_no_period  | Aperiodic input + as_sweeps → 1-D output, report.n_sweeps=1                                   |
| test_export_ibw_as_sweeps_falls_back_when_single_sweep     | period == full length → 1-D                                                                   |
| test_export_ibw_as_sweeps_applies_scaling_before_detection | `[1,2,3]*4 * 0.5` + scaling_factor=2 → detects period, sweep_count=4, columns equal `[1,2,3]` |

`tests/test_igor2_single_wave_export_dialog.py` — 4 new tests for the
checkbox + status label.

`tests/test_gui.py` — 2 new tests + extended MockDialog stubs:

| Test                                                                        | Asserts                                                                                                              |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| test_gui_export_igor2_single_wave_ibw_with_as_sweeps_succeeds               | Mocked dialog `as_sweeps=True` → loaded IBW shape (3,4), success message contains "Split into 4 sweeps × 3 samples." |
| test_gui_export_igor2_single_wave_ibw_as_sweeps_falls_back_for_non_periodic | Monotonic ramp + as_sweeps=True → 1-D output, fallback log emitted                                                   |

---

## Verification

- `uv run pytest` across the touched test files: **205 passed, 2 skipped**
  (both skips are unrelated PXP "real-world broken file" tests).
- `tests/test_importer_igor2.py::test_real_sutterpatch_file_imports`
  runs and passes against the actual `R5_S1_IVstep.ibw` file.
- `uv run pre-commit run --files <all touched files>` is clean.

---

## Non-goals (explicitly out of scope)

- **PXP multi-sweep export.** PXP already produces natural per-array
  waves; loop-detect-and-split doesn't map cleanly.
- **Per-sweep selective import.** The only import mode is "concatenate
  all sweeps".
- **SutterPatch `note` text parsing.** ScaleFactor, ScaleOffset, and
  SweepStartTimes remain unparsed; a TODO is parked in
  `parse_igor_binary_wave_data` for a future change.
- **Inter-sweep gap reconstruction.** Sweeps abut on concatenation.
- **CSV sweep export.** CSV ignores `as_sweeps` and writes a flat
  column; the GUI logs an info line when the combination is requested.

---

## Implementation notes (for future maintainers)

- `_write_ibw_v5`'s 2-D byte-write path does `np.asarray(data,
order="F").tobytes(order="F")`. The `tobytes(order='F')` is the
  operative call — the surrounding `asarray(..., order='F')` is
  defensive and could be dropped once the round-trip test confirms it
  isn't needed. Leaving it as-is for safety.
- `export_igor2_ibw`'s reshape uses `.reshape(sweep_count, period).T`
  followed by `np.ascontiguousarray`. Equivalent shorthands exist
  (`values.reshape(period, sweep_count, order='F')`), but the current
  form is explicit about the input layout (sweep-major concatenation
  from the import side) → output layout (`(period, sweeps)` column =
  one sweep) mapping.
- Cross-feature symmetry: the import side concatenates with
  `.ravel(order='F')`, and the export side undoes this with
  `.reshape(sweep_count, period).T`. A round trip (import multi-sweep
  IBW → re-export with `as_sweeps=True`) recovers the original
  `(samples, sweeps)` structure exactly, modulo resampling.
- `detect_period` lives in `utils.py` alongside other lightweight
  array helpers. It's `O(n²)` in the worst case (tries every divisor
  of `n`); for the ELISE Für-Elise stims (~2000 samples) and typical
  patch-clamp recordings (~30000 samples) this is fine. If a much
  larger stim ever needs sweep detection, an FFT-autocorrelation
  variant would be the standard upgrade — but until then, the simple
  exact-match implementation is what both callers want.
