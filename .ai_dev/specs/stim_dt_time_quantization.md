# Stim-dt time quantization — report effective (dt-snapped) times

## Overview

All three Design dynamic input routines — Targeted STDP synaptic strength
(`InputDesignerDialog`), Minimal stimulus for spiking / rheobase
(`RheobaseDialog`), and STDP timing window (`StdpWindowDialog`) — turned
millisecond timing inputs into integer stim samples with `round(ms/1000/dt)` but
recorded/plotted the **requested** millisecond value, not the quantized one that
was simulated. When the stim `dt` is coarse, distinct requests collapse to the
same sample count → identical stimulus and result, yet appeared at different x
positions (the "vertical slices at nearby times with the same point
distribution"). The fix records the **effective** (dt-snapped) time everywhere it
is plotted/logged/interpolated, so identical-pulse requests collapse onto one
honest grid point.

## Change

- `src/neuroseq/stim_timing.py` — new pure helper
  `snap_milliseconds_to_dt(time_ms, dt_seconds, minimum_samples=1) -> (n_samples,
effective_ms)`, built on the existing `milliseconds_to_samples_nearest` /
  `samples_to_milliseconds`.
- `rheobase_dialog.py` — `_probe` snaps recovery/pulse/tail and rewrites
  `_current_duration_ms` to the effective value before it is recorded in
  `_samples`/`_thresholds` (so plot x, logs, and chronaxie interpolation are
  grid-aligned). The duration-bisection tolerance is floored at one sample
  (`_duration_tolerance_ms`), and a bisection midpoint that snaps to an
  already-measured duration ends that branch instead of re-simulating it.
- `stdp_window_dialog.py` — `_append_pairing` snaps the gap and records the
  effective signed Δt (`_pending_delta_t = copysign(gap_ms, Δt)`);
  `_current_delta_t_values` snaps every grid Δt and drops duplicates that map to
  the same signed sample count; probe/pairing sample counts snapped.
- `input_designer_dialog.py` — `_append_pairing` snaps the gap and records the
  effective signed `pairing_time_ms` in `_pairing_times_ms` (grid-aligned
  pairing-time axis); probe sample counts snapped. Search/convergence logic
  unchanged.
- All three log a one-time resolution warning when the requested resolution
  (rheobase duration tolerance, STDP Δt spacing, adaptive pairing increment) is
  finer than one sample.

## Non-goals

No resampling of the stim `dt`, no change to search/convergence/readout semantics
(reset vs continuous, EPSP vs weight, the adaptive STDP search) — only which time
value is recorded and plotted. Recovery/response windows are snapped for their
sample counts but are not reported quantities.

## Verification

`ruff` + `ruff format --check` + `mypy` clean on all four source files. Targeted
suites (`test_stim_timing`, rheobase, stdp_window, input_designer, both
integration tests) pass — 112 tests. The integration tests keep their LTD/LTP
signs, confirming the effective-time snap did not change behaviour, only the
recorded/plotted resolution. New tests: `snap_milliseconds_to_dt` rounding/
effective value; rheobase points snap to the grid + bisection stops at one
sample; STDP Δt recorded as the effective grid value + finer-than-dt warning;
targeted-STDP pairing time recorded as effective grid value + snapped warning.
