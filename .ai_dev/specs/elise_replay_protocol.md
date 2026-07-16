# ELiSe train/validation/replay protocol (paper Fig. 2 in the GUI)

## Summary

A one-click GUI feature that reproduces the learning + replay curves of
Kriener et al. (2024), "ELiSe", Fig. 2b/c, entirely inside neuroseq — without
running the authors' external `scripts/train.py`.

A new toolbar action **"Run ELiSe Protocol (learning curves)"** trains the
rate-coded ELiSe network on the teaching pattern, measures a teacher-off
validation replay after every training epoch, then runs a free replay phase,
and plots per-epoch **MSE** and **Pearson correlation** against the paper
target `y = φ(u_tgt)` (Eq. 6). Results are exported to CSV and rendered by a
dedicated analysis plugin.

The replay _metric_ already existed (`replay_metrics.compute_performance_against_target`,
see `elise_target_mse.md` / Eq. 22–26). What was missing was the _protocol_ that
drives training/validation/replay and produces the curves. This spec adds that,
plus the correct target for the opsin teacher.

Implemented on branch `elise_replay_fix` (commits `1d747cc`, `ae28fb0`).

## Motivation

- The existing "Analyse ELiSe Replay (target MSE)" button post-processes one
  completed simulation trace (the _right_ panels of Fig. 2, replay cycles). It
  cannot produce the training curve (left panels) because that needs
  interleaved teacher-off validation, and it cannot separate a free-replay
  phase (its target is built from a _non-zero_ `opsin_exc`, but free replay
  zeroes the stim).
- The paper's full protocol (train → interleaved validation → replay) lives in
  the private `elise` package's `scripts/train.py`, which requires a heavy
  install (mlflow, coolname, …) and bypasses neuroseq's Project/stim/param
  model.
- `Elise.run_protocol` collapses that protocol into the neuroseq wrapper,
  reusing the existing metric and the GUI's worker/progress infrastructure.

## Architecture

```
gui.py
  _create_actions_replay_analysis()   → adds "Run ELiSe Protocol" action
  _elise_protocol_inputs()            → (opsin_exc, opsin_inh, stim_dt_s) | None
  _run_elise_protocol()               → dialog → EliseProtocolWorker(thread)
  _on_elise_protocol_finished()       → CSV + CustomAnalysisWindow(plugin)
  EliseProtocolWorker(QObject)        → runs run_protocol off the UI thread

elise_protocol_dialog.py
  EliseProtocolDialog                 → teacher mode, epochs/cycles, replay plasticity

example_models/elise.py
  Elise.run_protocol(...)             → the orchestrator (pure compute, no Qt)
  _matching_potential(...)            → conductance-based u_tgt (shared)
  _opsin_u_tgt_cycle(...)             → steady-state opsin teacher trajectory

resources/analysis_plugins/elise_learning_curves.py
  render_analysis(...)                → 4-panel Fig. 2b/c plot from the curves dict

replay_metrics.py
  save_learning_curves_csv(...)       → per-epoch CSV (phase, epoch, mse, corr)
  compute_performance_against_target  → the metric (unchanged, reused)
```

## `Elise.run_protocol`

Signature (module-level `run_protocol(...)` delegates to the singleton):

```python
run_protocol(
    opsin_exc, opsin_inh, stim_dt_s, *,
    base_params=None, teacher_mode="direct",
    training_epochs=10, training_cycles=25, validation_cycles=2,
    replay_epochs=50, replay_cycles=2, replay_learning=True,
    u_active_offset=20.0, cycle_length=None, progress=None,
) -> dict
```

It **steps the ELiSe network directly** (via `net(u_inp=…, learn=…)`), not
through `Elise.simulate`. This is deliberate:

- It needs to toggle the teacher on/off and plasticity on/off per phase, and to
  free-run with `u_inp=None`. Going through `simulate` would fight its
  cumulative-time stim indexing and allocate full `u/v/r` traces every phase.
- Direct stepping records only the visible firing rate during scored phases,
  which is all the metric needs.

Phase loop (state — voltages, rates, weights — carries across all phases, as in
the paper; no reset between cycles):

```
initialise a fresh network from base_params
for epoch in range(training_epochs):
    train training_cycles           # teacher on, learn=True
    r = free_run(validation_cycles) # teacher off (u_inp=None), learn=False
    val_mse[e], val_corr[e] = score(r vs target_rate)
for epoch in range(replay_epochs):
    r = free_run(replay_cycles)     # teacher off, learn=replay_learning
    rep_mse[e], rep_corr[e] = score(r vs target_rate)
```

`score` is `compute_performance_against_target` (shift-aligned MSE + Pearson,
Eq. 22–26), averaged over its evaluated windows. `validation_cycles` and
`replay_cycles` are clamped to ≥ 2 (the metric needs two whole cycles).

Returns a dict: `val_mse`, `val_corr`, `rep_mse`, `rep_corr` (per epoch),
`target_rate`, `cycle_trace_samples`, `trace_dt_s`, `num_vis`,
`final_replay_rate`, `experiment`, and `config`.

### Recorded replay Experiment

After training/replay, `run_protocol` records a **teacher-off replay of the
trained network** over the full opsin-stim timeline (zeroed stims, plasticity
off) via `Elise.simulate` (`_record_replay_experiment`). This returns a standard
neuroseq `Experiment` — `somatic_potential_u`, `dendritic_potential_v`,
`firing_rate_r` (all `num_all × T_stim`), and `dendritic_weights` snapshots —
at the trace dt, so its duration and sampling match the project's `opsin_exc` /
`opsin_inh` stim arrays. The GUI imports it (below) so the learned activity
lines up with the stimuli in the trace/stim widgets, exactly like any other
simulation result. Weights are frozen during this recording so it reflects the
trained attractor without replay-time drift.

### Cycle detection and resampling

The teaching cycle length is autodetected with `build_target_rate_pattern`
(binary footprint of `opsin_exc`, trailing zeros trimmed). Stim resolution
(`stim_dt_s`, e.g. 1 ms) is upsampled to trace resolution
(`dt_ms/1000`, e.g. 0.1 ms); `stim_dt_s` must be an integer multiple.

## Teacher modes

The metric target is `y = φ(u_tgt)` where `u_tgt` is the voltage the teacher
**actually** nudges the soma toward — matching the paper's own definition. The
two modes differ only in how `u_tgt` is produced.

### `direct` (default, paper-faithful)

`u_tgt = E_l + u_active_offset · pattern` (crisp binary, `1 → E_l+20 = −50 mV`,
`0 → E_l = −70 mV`), exactly as `scripts/train.py` (`r_target = eq_phi(u_target)`).
Reproduces Fig. 2 with the crisp Für-Elise pattern.

Verified (40 epochs × 25 cycles, `replay_learning=False`):
val MSE 0.037 → **0.0054**, corr 0.80 → **0.96**, replay flat 0.005 / 0.96 —
matching the external `train.py` run (10 000 cycles → val MSE 0.0059).

### `opsin` (biologically realistic)

`u_tgt` is the 3-state opsin **matching potential** (as `Elise.simulate`;
see `elise_opsin_kinetics.md`). This is the holographic-stimulation-faithful
teacher. The metric target is `φ` of the **steady-state** opsin `u_tgt`
(`_opsin_u_tgt_cycle`, warmed up 3 cycles so the slow tails reach their periodic
steady state), **not** the crisp binary.

Verified (40 × 25, `replay_learning=False`):
val MSE 0.0088 → **0.0012**, corr 0.94 → **0.99**, replay 0.0013 / 0.99.

## The opsin metric-mismatch fix (commit `ae28fb0`)

The first implementation scored **both** modes against the crisp binary target.
Opsin mode then plateaued at val MSE ≈ 0.047 and looked like it "could not
learn". Characterizing the opsin teacher over one steady Für-Elise cycle
(default params) showed why:

| quantity               | value                            | meaning                                                        |
| ---------------------- | -------------------------------- | -------------------------------------------------------------- |
| active-step `u_tgt`    | mean **−55.6 mV** (target −50)   | activation ramp never cleanly reaches −50 within a 10 ms event |
| inactive-step `u_tgt`  | max −51.4 mV; **16 %** above −65 | tails bleed active drive into "off" slots                      |
| off-tail τ             | exc **20 ms**, inh **100 ms**    | inh tail ≫ 10 ms event ⇒ heavy smear                           |
| corr(φ_opsin, φ_crisp) | **0.72** (min 0.63)              | delivered teacher only 72 % like the crisp pattern             |
| MSE(φ_opsin, φ_crisp)  | **0.042**                        | a hard error floor vs the crisp target                         |

The 0.042 mismatch floor was the entire "issue". Training with the opsin
teacher and scoring against the **delivered** teacher instead drops MSE to
**0.0015** — the network learns the smeared pattern perfectly:

```
epoch   MSE vs opsin-delivered   MSE vs crisp
start   0.068                    0.082
20      0.0015                   0.039 (stuck at floor)
```

**Fix:** in `teacher_mode="opsin"`, build the target from the steady-state
opsin `u_tgt` (`_opsin_u_tgt_cycle`) rather than the crisp binary. This is the
paper-consistent choice (`y = φ(u_tgt)` for the applied `u_tgt`) and turns the
opsin path into a meaningful experiment: _"can the network replay what the
holographic stimulation actually delivers?"_ — yes. `direct` mode is unchanged.

Note this is **not** a learning-rate problem: the wrapper's `eta` handling
(`SimpleUpdater(eta/dt)` × `+= dwdt·dt` in the step) cancels to an effective
`eta` identical to `train.py`.

## `replay_learning` stability caveat

The paper keeps plasticity on during replay to show the network "does not
unlearn", but that only holds for a _converged_ network. At modest training
(~750 cycles) with `replay_learning=True`, replay-time plasticity reinforces the
net's own errors and replay diverges (e.g. 0.005 → 0.08 over 60 teacher-off
cycles); the external 10 000-cycle `train.py` run shows the same drift, milder
(0.006 → 0.025). With `replay_learning=False` replay is flat at the trained
quality. For clean Fig. 2 replay at short training, turn replay plasticity off
or train longer. The dialog defaults it **on** (paper-faithful); users can
uncheck it.

## GUI behaviour

- The action is enabled only for `external_simulate_path == "neuroseq:elise"`
  projects that have both `opsin_exc` and `opsin_inh` stims (does not require a
  prior simulation — it runs one). Guard: `_refresh_elise_replay_action`.
- `EliseProtocolWorker` runs on a `QThread`; the shared non-modal
  `progress_dialog` shows progress and a **Cancel** button (cancellation raises
  through the progress callback, as `ExternalSimWorker` does).
- On completion: import the recorded replay `Experiment` via
  `_append_experimental_data` (the same `ImportExperimentDialog` flow as
  external-sim results — offset / append-vs-new), so the activity traces line up
  with the opsin stim arrays. Then prompt for a CSV path (skippable) and open a
  `CustomAnalysisWindow` whose `elise_learning_curves` plugin draws the four
  panels. The curves dict is attached to the widget via `CURVES_ATTR`; the
  final replay trace is passed as the `set_data` payload.

## How to run it in the GUI (reproduce Fig. 2)

0. **One-time:** install the private `elise` package into the same venv, else
   the action errors with a "requires the private 'elise' package" message:

   ```
   uv pip install --no-deps -e /mnt/live1/svintila/sequence-learning
   ```

1. Launch (`uv run neuroseq`) and open **File → Open example project → Network
   → "ELISE Sequence Learning Network"**. Its defaults already match the
   reference (`num_lat=50`, `lam=0.7`, `seed=2`, `a=0.3`, `b=-58`, plus the
   `opsin_exc`/`opsin_inh` stims); edit via the parameter tree if desired.

2. On the stim-editor toolbar click **"Run ELiSe Protocol (learning curves)"**
   (graduation-cap icon) — distinct from "Analyse ELiSe Replay (target MSE)",
   which only post-processes one finished simulation. Set the dialog:

   | Field                            | Match reference  | Quick demo (~1 min) |
   | -------------------------------- | ---------------- | ------------------- |
   | Teacher mode                     | Direct           | Direct              |
   | Training epochs                  | 100              | 30                  |
   | Training cycles / epoch          | 100 (10 000 cyc) | 25 (750 cyc)        |
   | Validation cycles / epoch        | 2                | 2                   |
   | Replay epochs                    | 100              | 20                  |
   | Replay cycles / epoch            | 2                | 2                   |
   | Keep plasticity on during replay | checked          | unchecked for flat  |

3. It trains on a background thread (responsive, cancellable). On finish it
   prompts for a CSV (`phase, epoch, mse, corr_coeff`; skippable) and opens the
   four-panel analysis window — the same Fig. 2b/c layout as
   `elise_reference_fig2.png`.

Expectation: **Direct** + 100×100 + replay-plasticity-on closely tracks the
reference (val MSE ~0.07 → ~0.006, val corr ~0.22 → ~0.95, replay MSE with a
mild upward creep, replay corr ~0.9). It is not bit-identical — the wrapper's
RNG streams and its opsin-free direct drive differ from `train.py`. For the
cleanest flat-replay Fig. 2, use **Direct** with replay plasticity **off**
(val/replay MSE ~0.005, corr ~0.96). See `replay_learning` caveat above and the
teacher-mode notes for the `opsin` variant.

Reproducing the raw ground-truth curves outside the GUI (the authors' own
`train.py`) is a separate offline driver, documented in the results folder
(`.../random/elise_neuroseq_results/`), not part of neuroseq.

## Files

New:

- `src/neuroseq/elise_protocol_dialog.py`
- `src/neuroseq/resources/analysis_plugins/elise_learning_curves.py`
- `tests/test_elise_protocol.py`

Modified:

- `src/neuroseq/example_models/elise.py` — `run_protocol`, `_matching_potential`,
  `_opsin_u_tgt_cycle`, module-level `run_protocol` delegate.
- `src/neuroseq/gui.py` — `EliseProtocolWorker`, action + guard, dispatch and
  finished/cancelled/failed/progress slots, `_elise_protocol_inputs`.
- `src/neuroseq/replay_metrics.py` — `save_learning_curves_csv`.

## Tests (`tests/test_elise_protocol.py`)

Skip-guarded on the private `elise` package where needed:

- `run_protocol` direct mode learns (MSE ↓, corr ↑; shapes; rates in [0,1]).
- `run_protocol` opsin mode learns the delivered teacher (MSE ↓, corr ↑).
- teacher-mode / shape validation errors.
- `save_learning_curves_csv` round-trip.
- plugin discovery.
- GUI: action enablement + `_elise_protocol_inputs`; dialog values; the
  finished/failed/cancelled/progress slots; and a full threaded dispatch
  (fake dialog → worker → analysis window).

## Non-goals (intentionally not implemented)

- Reproducing Fig. 3 (ablation), Fig. 4 (p/q/r sweeps), Fig. 5 (disruptions).
- Multi-pattern training / catastrophic-forgetting experiments.
- Persisting the curves into the `.nseq` archive (results live in the CSV and
  the analysis window only).
- Tuning the opsin kinetics to track the crisp pattern (see continuation).

## Proposed continuation

Ordered roughly by value / effort.

1. **Opsin fidelity to the crisp pattern.** The opsin teacher blurs Für-Elise
   to corr 0.72 (MSE 0.042) — a quantitative statement about how faithfully
   holographic stimulation can deliver a fast sequence. Explore the levers that
   sharpen it and re-measure corr(φ_opsin, φ_crisp): (a) inter-event gaps /
   silence so the 20 ms (exc) and 100 ms (inh) tails decay before the next
   event; (b) shorter or higher-flux pulses (faster activation); (c) kinetic
   parameters. Deliverable: a small sweep + a figure of crisp-corr vs gap
   length, so the lab can pick a stimulation regime. Keep the biophysical
   defaults intact; expose overrides via `base_params`.

2. **Persist protocol results in the project.** Store the curves dict (and
   `config`) on the `Experiment`/`Project` so they survive save/load and can be
   re-plotted, instead of living only in the CSV + ephemeral window. Would let
   the `elise_learning_curves` plugin be re-triggered from the analysis menu.

3. **Unify the single-run "Analyse ELiSe Replay" button with free replay.**
   That button still needs a teacher-on trace (its target is built from a
   non-zero `opsin_exc`). `run_protocol` already resolves the target/free-replay
   tension; factor a shared helper so the single-run analysis can score a
   released (teacher-off) trace against the delivered teacher, matching the
   protocol's semantics.

4. **`n_lat` sweep (rest of Fig. 2).** Loop `run_protocol` over
   `num_lat ∈ {30, 50, 100}` (and a few seeds) and overlay the curves with
   variance bands. Small orchestration on top of the existing runner.

5. **Fig. 3 — ablation.** Add a "random-init latent weights" mode (the paper's
   reservoir-mimicking ablation) and compare full vs ablated replay MSE. Needs a
   weight-init switch analogous to the private package's `weight_type`.

6. **Fig. 5 — disruptions.** Clamp visible-neuron voltages during a replay
   window (`u_disrupt = E_l` or `E_l+15`) and show recovery. `run_protocol`'s
   free-run loop is the natural hook for an optional per-step clamp.

7. **(Done)** The protocol now records the trained network's teacher-off replay
   as a standard `Experiment` (`u/v/r` + weight snapshots) and imports it, so the
   trace/heatmap widgets show the learned attractor aligned to the stim. Possible
   extensions: also expose a teacher-on recording, or the per-epoch scalar curves
   as project data so they survive save/load.

8. **Performance.** For very large runs, the Python per-step loop dominates;
   consider batching the free-run scoring or reusing the (numba) private-package
   simulator for the training phase while keeping neuroseq's metric.
