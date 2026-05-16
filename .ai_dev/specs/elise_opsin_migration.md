# ELISE — Virtual-opsin migration: two stim arrays replace `u_inp` + `clamp_visible`

## Overview

The ELISE wrapper consumes **two stim arrays** — `opsin_exc` and
`opsin_inh` — instead of the previous single `u_inp` stim, and the
`clamp_visible` parameter is gone. The two stims are interpreted as
fractional activations of a pair of "virtual opsins" with reversal
potentials matched to ELISE's teaching synapses (`E_exc = 0 mV`,
`E_inh = −75 mV`). For each pitch at each step, the wrapper derives the
upstream `u_inp` from the ratio of the two activations via the
matching-potential formula (Kriener et al., 2024 §5.3.3, eq. 13):

```
u_tgt_k = (g_exc_k · E_exc + g_inh_k · E_inh) / (g_exc_k + g_inh_k)
```

Only the ratio matters per pitch; magnitudes are normalised out by the
division. Overall teaching strength is still controlled by the `lam`
parameter (upstream's eq. 16 budget).

The migration resolves two problems with the previous design:

1. **`u_inp` cannot express "no teaching".** No value of `u_inp`
   produces zero teaching conductance under upstream's eq. 17/18, so
   the only release path was the `if u_inp is not None:` gate, which
   the wrapper hit via the `clamp_visible` boolean. With opsin
   activations, "release" is just "both stims zero" — natively
   expressible in the stim editor.

2. **Stim widget and simulator could disagree.** `clamp_visible` was
   orthogonal to the stim contents, so a vibrant Für Elise heatmap
   could be playing while the simulator was actually evolving freely.
   With the migration, the stim widget IS the source of truth: zero
   stim = zero teaching, non-zero stim = teaching active.

The upstream `elise` package is untouched. The wrapper still calls
`Network(...)` with a single `u_inp` (or `None`); it just derives that
`u_inp` from the two stim arrays internally.

Applies to:

- `src/neuroseq/example_models/elise.py` — DEFAULT_PARAMS, `_PHASE_PARAMS`,
  `update_params` filter, the simulate-loop rewrite
- `src/neuroseq/example_projects.py` — `_build_fuer_elise_opsin_stims`
  helper, updated `elise_project` factory
- `src/neuroseq/project.py` — `_matching_potential_to_opsins` helper,
  `Project._migrate_legacy_elise_stim`, hook in `Project.__init__`
- `tests/test_elise_clamp_visible.py` (deleted) → `tests/test_elise_opsin_release.py` (new)
- `tests/test_elise_smoke.py`, `tests/test_example_project_elise.py`,
  `tests/test_example_models_backend.py`, `tests/test_project_external_simulate.py`,
  `tests/test_project_suggested_sim_time.py` — updated for the new stim API
- `.ai_dev/specs/elise_clamp_visible_toggle.md` — gets a "Superseded by"
  header note pointing here

---

## The plugin contract change

`Elise.simulate(stims_dict, ...)` now requires:

```python
stims_dict["opsin_exc"]["array"]   # ndarray[T, num_vis], float, [0, ∞)
stims_dict["opsin_exc"]["dt"]      # seconds
stims_dict["opsin_inh"]["array"]   # ndarray[T, num_vis], same shape
stims_dict["opsin_inh"]["dt"]      # seconds (must equal opsin_exc's dt)
```

Validation enforces matching shapes and that both arrays are 2D with
non-zero time axis. A missing key, or a shape mismatch, raises a clear
`ValueError`. The wrapper trusts `opsin_exc["dt"]` as the time
resolution (both stims share neuroseq's project time axis).

---

## Phase params — `clamp_visible` removed

`Elise._PHASE_PARAMS` is now `frozenset({"learn"})`. `clamp_visible` was
deleted from `DEFAULT_PARAMS`, the `_PHASE_PARAMS` set, the class
docstring, and every reading site in `Elise.simulate`. The boolean is
gone from the user-facing model editor entirely.

### Legacy-archive compatibility

`.nseq` files saved before this migration may still carry
`clamp_visible=True/False` in `external_simulate_params`. `Elise.update_params`
now filters merged params against `self._default_params_schema.keys()`
before storing:

```python
def update_params(self, params):
    merged = self._flat_defaults()
    if params:
        # Drop unknown keys (legacy archives may still carry
        # `clamp_visible`); they're no longer meaningful.
        known = set(self._default_params_schema.keys())
        merged.update({k: v for k, v in params.items() if k in known})
    # ... trajectory-vs-phase diff check as before ...
```

This is general: any future schema removal (not just `clamp_visible`)
benefits from the same filter.

---

## Per-step simulate-loop math

The simulate body of `Elise.simulate` (excerpted):

```python
E_exc = _float_param(self._params, "E_exc")
E_inh = _float_param(self._params, "E_inh")
E_l = _float_param(self._params, "E_l")
...
for step in range(n_steps):
    t_abs_s = self._simulated_time_s + step * dt_ms / 1000.0
    stim_idx = min(int(t_abs_s / stim_dt_s), stim_T - 1)
    g_exc_k = g_exc_full[stim_idx]
    g_inh_k = g_inh_full[stim_idx]
    sum_k = g_exc_k + g_inh_k
    if not np.any(sum_k > 0):
        u_inp_step = None
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            u_tgt = (g_exc_k * E_exc + g_inh_k * E_inh) / sum_k
        u_tgt = np.where(sum_k > 0, u_tgt, E_l)
        u_inp_step = u_tgt.astype(np.float64)
    self._network(u_inp=u_inp_step, learn=learn)
    ...
```

The two-condition behaviour:

- **All pitches have zero opsin activation** → `u_inp = None`. Upstream
  takes the `if u_inp is not None:` gate and produces zero teaching
  current — the real "release" condition. This replaces what
  `clamp_visible = False` used to do.
- **Some pitches have non-zero activation** → upstream gets a per-pitch
  `u_inp` vector. Pitches with `g_exc + g_inh > 0` get their
  matching-potential `u_tgt`; pitches with both opsins at zero get
  `u_inp[k] = E_l` (the `np.where` fallback after the `np.errstate`).

The asymmetry between "all pitches zero" and "selectively zero" is by
design: upstream doesn't expose a per-pitch teaching gate, only the
global `None` gate. Holding released pitches at `E_l` is the closest
local approximation when teaching is still active for other pitches.

---

## Für Elise example constants

The default `_build_fuer_elise_opsin_stims` reproduces the previous
single-stim pattern's per-pitch targets (`u_tgt = −50 mV` for active
pitches, `u_tgt = −70 mV` for rest) via normalised ratios that keep
both arrays in `[0, 1]` and visually informative:

```python
# Normalised so g_exc + g_inh = 1 in each state; ratios match
# paper §5.3.3:
#   active (u_tgt = -50 mV): exc/inh = 1/2  → 0.333 : 0.667
#   rest   (u_tgt = -70 mV): exc/inh = 1/14 → 0.067 : 0.933
_FUER_ELISE_OPSIN_EXC_ACTIVE: float = 1.0 / 3.0
_FUER_ELISE_OPSIN_INH_ACTIVE: float = 2.0 / 3.0
_FUER_ELISE_OPSIN_EXC_REST:   float = 1.0 / 15.0
_FUER_ELISE_OPSIN_INH_REST:   float = 14.0 / 15.0
```

Resulting stims:

| Stim        | Rest value | Active value | Contrast |
| ----------- | ---------- | ------------ | -------- |
| `opsin_exc` | 0.067      | 0.333        | 5×       |
| `opsin_inh` | 0.933      | 0.667        | 1.4×     |

Both arrays show the Für Elise pattern in their heatmaps; neither is
a flat block. The normalisation also matches what
`_matching_potential_to_opsins` produces for migrated archives, so
fresh-built and migrated projects look consistent.

The choice of normalisation `g_exc + g_inh = 1` per state was
deliberate. Other scalings (e.g., `(1, 2)` active and `(1, 14)` rest)
would preserve the same per-pitch `u_tgt` — only the ratio matters
under the ratio-only semantics — but yield asymmetric visual contrast
in the two arrays.

---

## Legacy archive migration

`Project._migrate_legacy_elise_stim` runs immediately after
`Project.load_NN()` in `Project.__init__`, gated on
`external_simulate_path == "neuroseq:elise"` and the presence of a
single `u_inp` stim alongside the absence of either opsin stim:

```python
def _migrate_legacy_elise_stim(self) -> None:
    if self.external_simulate_path != "neuroseq:elise":
        return
    if "u_inp" not in self.stim_arrays:
        return
    if "opsin_exc" in self.stim_arrays or "opsin_inh" in self.stim_arrays:
        return
    # ... read E_exc, E_inh from external_simulate_params (defaults 0, -75)
    converted = _matching_potential_to_opsins(u_inp, E_exc, E_inh)
    # ... replace u_inp with opsin_exc + opsin_inh in stim_arrays ...
```

The conversion (in the top-level helper `_matching_potential_to_opsins`):

```python
g_exc = clip((E_inh − u_inp) / (E_inh − E_exc), 0, None)
g_inh = clip((u_inp − E_exc) / (E_inh − E_exc), 0, None)
```

These satisfy `g_exc + g_inh = 1` for `u_inp ∈ [E_exc, E_inh]` — same
normalisation as the fresh-built Für Elise default. Values outside that
range get clipped to zero on the violating side, which matches
upstream's preference for physically meaningful inputs.

Migration is **idempotent** (second call is a no-op because `u_inp` is
gone), **gated** (other external_simulate_paths like `neuroseq:sinus_offset`
leave `u_inp` alone), and **load-time only** (no runtime overhead).

---

## Tests

`tests/test_elise_opsin_release.py` — 7 tests:

| Test                                                              | Asserts                                                                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `test_opsin_exc_inh_not_in_default_params_under_simulation_group` | `DEFAULT_PARAMS` has neither `opsin_exc`/`opsin_inh` (they're stim names, not params) nor `clamp_visible`          |
| `test_phase_params_no_longer_contains_clamp_visible`              | `_PHASE_PARAMS == {"learn"}`                                                                                       |
| `test_clamp_visible_in_legacy_archive_is_dropped`                 | `update_params({"clamp_visible": False, "num_lat": 4})` keeps `num_lat` and drops `clamp_visible`                  |
| `test_simulate_both_stims_zero_passes_none_to_network` (mocked)   | Both arrays all-zero → fake network's `__call__` receives `u_inp is None` every step                               |
| `test_simulate_mixed_stim_drives_per_pitch_u_tgt` (mocked)        | Per-pitch (1, 2) → `u_tgt = -50`; per-pitch (0, 0) → `u_tgt = -70` (E_l fallback)                                  |
| `test_simulate_preserves_network_across_releases` (mocked)        | Train then release: same network instance, accumulated `simulated_time_s`, `u_inp = None` calls in the release run |
| `test_nonzero_opsins_drive_visible_neurons` (elise-gated)         | Non-zero stims for 100 ms → at least one visible neuron's final `u` deviates from `E_l` by > 1 mV                  |
| `test_all_zero_opsins_release_visible_neurons` (elise-gated)      | Zero stims for 100 ms → all visible `u` within 5 mV of `E_l`                                                       |

Migration tests in `tests/test_project_external_simulate.py` — 4 new tests:

| Test                                                    | Asserts                                                                                             |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `test_legacy_u_inp_stim_migrated_to_opsin_pair_on_load` | `_migrate_legacy_elise_stim()` removes `u_inp`, adds both opsins with correct values                |
| `test_legacy_u_inp_archive_migrated_on_reload`          | End-to-end `Project.save` → `Project(filepath)` triggers the `__init__` hook and migrates correctly |
| `test_migration_is_idempotent`                          | Second call is a no-op                                                                              |
| `test_migration_only_for_elise_external_path`           | `neuroseq:sinus_offset` projects do NOT get their `u_inp` migrated                                  |

Example-project tests in `tests/test_example_project_elise.py` — updated assertions on stim names (`opsin_exc`/`opsin_inh`), shapes `(1920, 13)`, and first-event ratios. Period detection now expects `_detect_period == 240` for **both** arrays — the previous `_detect_period(stim_exc) == 1` codified the uniform-stim bug that the constant tuning resolved.

Smoke tests in `tests/test_elise_smoke.py` rewritten to author non-uniform two-stim inputs. Fake-network scaffolding in `tests/test_example_models_backend.py` updated to handle `u_inp = None`.

---

## Non-goals

- **Independent teaching strength per pitch.** Upstream's eq. 16
  constraint stays global — `lam` controls the overall budget.
  Per-pitch strength would require forking `elise.model`.
- **Real opsin kinetics.** ChR2 / halorhodopsin time constants and
  saturation are out of scope; these are instantaneous activation →
  conductance mappings.
- **φ(u) broadening.** The supervisor's suggestion to shift `b → 0`
  and broaden the sigmoid is a separate modelling experiment that
  doesn't touch wrapper architecture.
- **GUI badge for "teaching active".** The two stim widgets ARE the
  visualisation — non-zero values = teaching active for that pitch.
  No separate indicator needed.
- **Mid-run release.** The two stims are read per-step from the user's
  authored arrays. To release mid-run, author the appropriate zero
  block in the stim editor; no special parameter.
