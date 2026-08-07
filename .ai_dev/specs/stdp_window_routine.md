# STDP timing window routine — "STDP timing window (LTP/LTD vs Δt)"

## Overview

A third Design dynamic input routine, alongside "Targeted STDP synaptic
strength" (`InputDesignerDialog`) and "Minimal stimulus for spiking (rheobase)"
(`RheobaseDialog`). It traces the functional STDP learning window for one
synapse: the percentage change in the postsynaptic EPSP caused by a pre/post
pairing, as a function of the spike-timing difference Δt = t_post − t_pre.
Δt > 0 (pre before post) is potentiation (LTP), Δt < 0 (post before pre) is
depression (LTD).

Unlike the other two routines it is **not** a converging search — it is a
**deterministic sweep** over a fixed Δt grid. State handling is **append +
run-forward with a recovery lead, no `store`/`restore`** (the model standardised
on for rheobase).

Two selectable **readouts**:

- **EPSP amplitude change (%)** (default) — functional; reuses the five-pulse
  probe + `detect_epsps` the STDP routine uses. Needs a clean postsynaptic EPSP,
  so it suits an isolated pre→post pair. On recurrent networks the EPSP is
  shunted by feedback inhibition and becomes unmeasurable.
- **Synaptic weight change (%)** — reads the synapse weight `we` directly from a
  synapse `StateMonitor` (`SynapseData.values[pre, post, -1]`, `[i, j]` = source→
  target per `construct_synapse_matrix`). Immune to shunting/inhibition, needs no
  probe pulses. Requires a synapse weight monitor in the project.

**Robustness:** an undetectable EPSP is a _skipped_ point (recorded NaN, warned,
sweep continues) via a dedicated `_NoEpspDetected` exception — only a missing
baseline EPSP or a structural error (missing trace row, trace too short) aborts,
and the baseline-abort message points at the weight readout.

**Target suitability:** a clean single-synapse STDP window needs an isolated
pre→post pair firing one well-timed spike each (e.g. `two_hh_neurons_with_
plasticity`; ~100 amplitude, ~2 ms pulse). Recurrent all-to-all networks such as
the _Getting Started_ project entangle the target synapse with its reciprocal and
recurrent partners, so neither readout yields a physically meaningful window
there — the weight readout completes without error but the numbers are not
interpretable.

Entry point: Stim editor → "Design dynamic input" → pick
`StimEditor.STDP_WINDOW_ROUTINE`.

## Files

- `src/neuroseq/stdp_window.py` — pure helpers (no Qt): `delta_t_sweep`,
  `percent_change`, `classify_window`.
- `src/neuroseq/stdp_window_dialog.py` — `StdpWindowDialog(QtWidgets.QDialog)`.
- `src/neuroseq/stim_editor.py` — routine constant, dispatch, and fan-out
  (`_stdp_window_dialog` added to `_show_routine_dialog`, `_dynamic_input_dialogs`,
  `discard_input_designer`; `DYNAMIC_INPUT_ROUTINES` extended). No gui.py change.
- Tests: `tests/test_stdp_window.py` (pure), `tests/test_stdp_window_dialog.py`
  (faked closed loop), `tests/test_stdp_window_integration.py` (real Brian2),
  plus additions to `tests/test_stim_editor.py`.

## What it measures

For each Δt in the sweep the routine records
`ΔEPSP%ₖ = 100·(EPSPₖ − EPSPₖ₋₁)/EPSP₀` (successive differences normalised by the
initial baseline). A single baseline probe establishes `EPSP₀`; then for each Δt
point it appends the pairing(s) and one probe, measures the EPSP, and records the
per-point change. The pre/post pulses must be **suprathreshold** so both neurons
spike and drive the synapse's `on_pre`/`on_post`; the parameter note points users
at the rheobase routine to find an amplitude.

## State handling (why no restore)

Synaptic weight does not passively recover, so a recovery lead cannot restore a
fixed baseline **weight**. The routine instead measures the per-pairing change
from a rested **eligibility** state: each probe/pairing block starts with a
`Recovery time` lead (default 150 ms ≫ `taupre`/`taupost`) so the `Apre`/`Apost`
traces decay to rest between blocks and the probe's own pre-spikes do not
contaminate the next pairing. Blocks are appended (`_append_block`, start =
`max(len, ceil(sim_time/dt))`) and the network runs forward — no checkpoints.
The absolute weight drifts across the sweep; when the running EPSP leaves the
0.25×–4× baseline band a one-line saturation warning is logged (drift is
reported, not corrected).

## Closed-loop contract

`simulation_requested(float)` / `simulation_finished()` / `simulation_failed()` /
`apply_theme()` / `refresh()`, guarded by `_awaiting_simulation`; `closeEvent`
stops a running routine. Phases: `initialize_trace` → `baseline_probe` → per point
`pair` (×`pairings per point`) → `probe`. Only `baseline_probe`/`probe` runs
measure EPSP; `pair` runs advance the pairing count. Estimated run count
(`1 + n_points·(pairings + 1)`) is shown live in `lbl_estimate` and logged at
start.

## Parameters

Readout mode (EPSP / weight); trace selector (EPSP mode) or synapse-weight
selector (weight mode), toggled by mode; presynaptic / postsynaptic neuron IDs;
pulse amplitude
(suprathreshold) + unit; pulse duration (default 5 ms); Δt min/max (default
±100 ms); number of Δt points (default 21); include Δt = 0 (default off);
pairings per point (default 1); recovery time (default 150 ms); post-pulse window
(default 50 ms); live-view follow + visible history. Validation: pre ≠ post,
Δt max > Δt min, non-empty sweep. On start, if any stim array is non-empty a
warning offers **Clear all stim arrays** / **Append anyway** / **Cancel** (reuses
`StimEditor.clear_all_stim_arrays`).

## Plot

Separate plot window, two axes: (1) the STDP window — ΔEPSP% vs Δt, LTP points
green / LTD red, zero lines and LTP/LTD annotations; (2) the EPSP trajectory over
probes (drift/saturation view) with a dashed baseline. Theme-synced via
`theme.mpl_colors`.

## Reuse

Copies the routine-agnostic block/probe/measurement bodies and the module-level
`_WheelToScrollFilter` / `_set_native_minimized_position` / `EPSPMeasurement`
from `input_designer_dialog.py`, and the existing-input warning + run-estimate
pattern from `RheobaseDialog`. `InputDesignerDialog` and `RheobaseDialog` are
unchanged and their suites stay green.

## Verification

`ruff` + `ruff format --check` + `mypy` clean. Full suite: all STDP-window,
rheobase, STDP-targeting, and stim-editor tests pass. The real-model integration
test (`two_hh_neurons_with_plasticity`, deterministic init) confirms Δt = −30 ms
yields LTD (< 0), Δt = +30 ms yields LTP (> 0), the trace is a real
`Simulation :: … / v`, the stim array grew (append), `clear_all_stim_arrays` ran,
and `b2_model.checkpoints == ["Initial state"]` (no `store`/`restore`).
