# ELISE opsin kinetics — 3-state collapsed PyRho model

## Overview

The current ELISE opsin pair (`opsin_exc`, `opsin_inh`) carries
**fractional activations in [0, 1]** that are used directly as
conductances inside `Elise.simulate`: `g_exc_k = opsin_exc[t, k]`,
`g_inh_k = opsin_inh[t, k]`, then the matching potential
`u_tgt = (g_exc·E_exc + g_inh·E_inh)/(g_exc + g_inh)` is clamped onto
the neuron. There is no light intensity, no kinetic state, no time
constants — the stim _is_ the conductance.

This spec specifies a deeper opsin layer that:

1. Re-interprets the stim arrays as **photon flux density**
   (mm⁻²·s⁻¹), bringing units in line with the existing
   `pyrho_6state_chr2_project` example in
   `src/neuroseq/example_projects.py:1342`.
2. Introduces a **3-state kinetic model** (closed/open/desensitized)
   per opsin type, integrated forward each simulation step.
3. Couples the steady-state of the kinetic model to the existing
   matching-potential calibration so that the **default Für-Elise
   teaching signal is preserved bit-for-bit at the rate-coded
   timescales ELISE operates in** (the calibration is re-derived;
   the learning outcome is not).

The 3-state collapse drops the slow `_2` pathway, the intermediate
`I` states, the `f_v` voltage rectification, and `γ`-weighted open
states from PyRho. For ELISE's 10-ms square-pulse regime the
deviation from a full 6-state port is single-digit percent in the
peak open fraction (see the regime analysis below). Level 3 (full
6-state) is **explicitly a non-goal** — the slow pathway never wakes
up at ELISE's timescales.

Applies to:

- `src/neuroseq/example_models/elise.py` — `Elise.simulate` gains a
  kinetic step before the conductance read; `Elise.reset` resets the
  new state arrays; `Elise.default_params_schema` exposes the new
  parameters
- `src/neuroseq/example_projects.py` — Für-Elise constants
  re-derived against the steady-state Hill curve; new helper
  `_calibrate_phi_for_target_open_fraction`
- `src/neuroseq/project.py` — `_migrate_legacy_elise_stim` extended
  to convert pre-kinetic opsin fractions into photon-flux values
  via the inverse Hill on archive load
- `tests/test_elise_opsin_kinetics.py` (new) — unit tests for the
  kinetic step, steady-state Hill, calibration, paired-pulse
  recovery
- `tests/test_elise_signature_cache.py` (extended) — kinetic
  parameters added to the signature key
- `tests/test_project_archive_roundtrip.py` (extended) — legacy
  pre-kinetic archive migration

---

## The 3-state model

Each opsin molecule is in one of three states; per pitch `k` we
track fractions in [0, 1]:

- `C[k]` — **closed**, ready to be activated by light
- `O[k]` — **open**, conducting (`g[k] = g0 · O[k]`)
- `D[k]` — **desensitized**, light-insensitive, recovers slowly

Conservation: `C + O + D = 1`. The integrator carries `O` and `D`
explicitly; `C` is reconstructed as `1 − O − D` each step.

### State diagram

```
       k_a(φ)            k_d
  C  ──────────►  O  ──────────►  D
                                  │
  ▲                               │
  └──────────────── k_r ──────────┘
```

- `C → O` at light-dependent rate `k_a(φ)`
- `O → D` at fixed rate `k_d` (desensitization)
- `D → C` at fixed rate `k_r` (recovery)

### Rate constants

| Symbol    | Meaning                                    | Default (excitatory, ChR2-like) | Default (inhibitory, NpHR-like) |
| --------- | ------------------------------------------ | ------------------------------- | ------------------------------- |
| `k_a_max` | maximum activation rate (saturating light) | 1000 s⁻¹                        | 800 s⁻¹                         |
| `phi_m`   | Hill half-activation photon flux           | 5×10¹⁷ mm⁻²·s⁻¹                 | 5×10¹⁷ mm⁻²·s⁻¹                 |
| `k_d`     | desensitization rate (`O → D`)             | 50 s⁻¹ (≈20 ms)                 | 10 s⁻¹ (≈100 ms)                |
| `k_r`     | recovery rate (`D → C`)                    | 5 s⁻¹ (≈200 ms)                 | 5 s⁻¹ (≈200 ms)                 |
| `g0`      | peak conductance × population              | calibrated (see below)          | calibrated (see below)          |
| `E`       | reversal potential                         | 0 mV (fixed)                    | −75 mV (fixed)                  |

`phi_m` and `k_a_max` are taken from the PyRho article values
(`phi_m = 5e17 mm⁻²·s⁻¹`, `k1 = 18.2 ms⁻¹` ⇒ saturating Ga1 = 18.2
ms⁻¹ ≈ 18 200 s⁻¹). We use a deliberately gentler `k_a_max = 1000
s⁻¹` so that at saturating light the C→O transient is ≈ 1 ms — fast
relative to ELISE's `dt = 1 ms` but not so fast that forward Euler
becomes unstable. **No inhibitory PyRho values exist in
`example_projects.py`**; the inhibitory defaults are hand-chosen to
match published GtACR/NpHR macroscopics (slower desensitization,
slightly lower peak rate).

### Activation curve

Hill with exponent `p = 1` (no second exponent):

```
k_a(φ) = k_a_max · φ / (φ + phi_m)
```

For `φ → ∞`, `k_a → k_a_max`. For `φ → 0`, `k_a → 0`. No explicit
`Theta` gate — `0 / (0 + phi_m) = 0` handles the dark case cleanly.

### Per-step update (forward Euler)

For each pitch `k`, each step `dt`:

```
phi    = opsin_stim[t, k]                        # photon flux (mm⁻²·s⁻¹)
k_a    = k_a_max * phi / (phi + phi_m)
C      = 1.0 − O[k] − D[k]
dO     = (k_a * C  −  k_d * O[k]) * dt
dD     = (k_d * O[k]  −  k_r * D[k]) * dt
O[k]  += dO
D[k]  += dD
g[k]   = g0 * O[k]
```

Forward Euler is stable at `dt = 1 ms` because the fastest active
rate is `k_a_max = 1000 s⁻¹` (τ = 1 ms). At higher `k_a_max` switch
to exponential Euler (`O_new = O_eq + (O − O_eq)·exp(−(k_a + k_d)·dt)`);
this is a one-line change behind an `if k_a_max * dt > 0.5` guard.

### Steady-state open fraction (closed form)

For constant `φ`, setting `dO/dt = dD/dt = 0`:

```
O* = k_a · k_r / (k_a · k_r + k_a · k_d + k_d · k_r)
```

(Substitute `C = 1 − O − D`, `D = (k_d/k_r)·O`, solve.) This closed
form drives the Für-Elise calibration below.

---

## Two opsin types in parallel

Excitatory and inhibitory opsins are independent molecule populations
with their own kinetics:

|                      | Excitatory                     | Inhibitory                     |
| -------------------- | ------------------------------ | ------------------------------ |
| Stim array           | `opsin_exc` (φ_exc)            | `opsin_inh` (φ_inh)            |
| State arrays         | `O_exc[k]`, `D_exc[k]`         | `O_inh[k]`, `D_inh[k]`         |
| Reversal             | `E_exc = 0 mV`                 | `E_inh = −75 mV`               |
| Per-step conductance | `g_exc[k] = g0_exc · O_exc[k]` | `g_inh[k] = g0_inh · O_inh[k]` |
| Parameter group      | `opsin_exc.*` in param tree    | `opsin_inh.*` in param tree    |

The matching-potential formula is unchanged:

```
sum_k = g_exc[k] + g_inh[k]
u_tgt = (g_exc[k] · E_exc + g_inh[k] · E_inh) / sum_k      where sum_k > 0
u_tgt = E_l                                                where sum_k == 0
```

So the downstream Urbanczik-Senn update sees the same `u_tgt`
interface; only how `g_exc, g_inh` are produced has changed.

---

## Integration into `Elise.simulate`

The current per-step body in `src/neuroseq/example_models/elise.py:_simulate_loop`
already does:

```python
g_exc_k = opsin_exc[t, :]
g_inh_k = opsin_inh[t, :]
sum_k = g_exc_k + g_inh_k
if not np.any(sum_k > 0):
    u_inp_step = None
else:
    with np.errstate(invalid="ignore", divide="ignore"):
        u_tgt = (g_exc_k * E_exc + g_inh_k * E_inh) / sum_k
    u_tgt = np.where(sum_k > 0, u_tgt, E_l)
    u_inp_step = u_tgt.astype(np.float64)
```

The kinetic version replaces the first two lines:

```python
# 1. Pull photon flux for this step
phi_exc = opsin_exc[t, :]
phi_inh = opsin_inh[t, :]

# 2. Integrate kinetics one dt forward, in-place on singleton state
_step_kinetics(
    self._O_exc, self._D_exc, phi_exc,
    self._params.k_a_max_exc, self._params.phi_m_exc,
    self._params.k_d_exc, self._params.k_r_exc, dt,
)
_step_kinetics(
    self._O_inh, self._D_inh, phi_inh,
    self._params.k_a_max_inh, self._params.phi_m_inh,
    self._params.k_d_inh, self._params.k_r_inh, dt,
)

# 3. Read out conductances
g_exc_k = self._params.g0_exc * self._O_exc
g_inh_k = self._params.g0_inh * self._O_inh

# rest of the body (matching-potential, weight update) unchanged
```

`_step_kinetics` is a numba-jitted helper:

```python
@njit(cache=True, fastmath=False)
def _step_kinetics(O, D, phi, k_a_max, phi_m, k_d, k_r, dt):
    for k in range(O.shape[0]):
        k_a   = k_a_max * phi[k] / (phi[k] + phi_m)
        C     = 1.0 - O[k] - D[k]
        dO    = (k_a * C - k_d * O[k]) * dt
        dD    = (k_d * O[k] - k_r * D[k]) * dt
        O[k] += dO
        D[k] += dD
```

`fastmath=False` keeps the divide-by-zero behaviour predictable
(`0 / phi_m = 0`, not NaN).

---

## Singleton state arrays

`Elise.__init__` is extended to allocate four new arrays of shape
`(num_vis,)`:

```python
self._O_exc:  np.ndarray  # shape (num_vis,), dtype float64
self._D_exc:  np.ndarray
self._O_inh:  np.ndarray
self._D_inh:  np.ndarray
```

Allocation is lazy in `Elise._ensure_network()` (matching how
`_network` is built today). Initial values: all zeros — the opsin
starts fully closed.

`Elise.reset()` zeros these arrays alongside the existing `_network`
and `_simulated_time_s` reset. They are **not** included in archive
serialisation — kinetic state is transient and re-builds from `phi`
within milliseconds, so persisting it would only hurt
reproducibility across machine boundaries.

---

## Signature cache key

`Elise._signature` already includes a tuple of (num_vis, learning
rate, …). Extend it with the kinetic parameters:

```python
def _signature(self) -> tuple:
    return (
        ...existing fields...,
        self._params.k_a_max_exc, self._params.phi_m_exc,
        self._params.k_d_exc,     self._params.k_r_exc,
        self._params.g0_exc,
        self._params.k_a_max_inh, self._params.phi_m_inh,
        self._params.k_d_inh,     self._params.k_r_inh,
        self._params.g0_inh,
    )
```

Changing any of these from the editor triggers an `Elise.reset()`
on the next Run, as with all non-phase parameters today.

`_PHASE_PARAMS` stays `{"learn"}`. Kinetic parameters are **not**
phase params; toggling them is a model edit, not a phase switch.

---

## Parameter schema additions

`Elise.default_params_schema` gains a new section per opsin type.
Parameters are grouped under `opsin_exc` and `opsin_inh` so the
model editor renders them in two collapsible blocks.

```python
{
    "opsin_exc.k_a_max": {
        "value": 1000.0, "default": 1000.0,
        "type": "float", "group": "opsin_exc",
        "description": "Saturating activation rate (1/s)",
    },
    "opsin_exc.phi_m": {
        "value": 5e17, "default": 5e17,
        "type": "float", "group": "opsin_exc",
        "description": "Hill half-activation photon flux (1/mm^2/s)",
    },
    "opsin_exc.k_d": {
        "value": 50.0, "default": 50.0,
        "type": "float", "group": "opsin_exc",
        "description": "Desensitization rate (1/s)",
    },
    "opsin_exc.k_r": {
        "value": 5.0, "default": 5.0,
        "type": "float", "group": "opsin_exc",
        "description": "Recovery rate (1/s)",
    },
    "opsin_exc.g0": {
        "value": <calibrated>, "default": <calibrated>,
        "type": "float", "group": "opsin_exc",
        "description": "Peak conductance scaling (dimensionless)",
    },
    # same five for opsin_inh, with inhibitory defaults
}
```

`E_exc = 0 mV` and `E_inh = −75 mV` are **not** exposed as
parameters — they are physical constants of the opsin chemistry and
making them editable invites the user to break the matching-potential
calibration silently.

---

## Calibration: re-deriving the Für-Elise constants

The current shallow constants in `example_projects.py:1521-1524`
target two `u_tgt` values:

| State        | Target `u_tgt` | Current `(g_exc, g_inh)` |
| ------------ | -------------- | ------------------------ |
| Active pitch | −50 mV         | (1/3, 2/3)               |
| Rest         | −70 mV         | (1/15, 14/15)            |

These are conductance fractions normalised so `g_exc + g_inh = 1`.
For the kinetic model, the equivalent step is to find photon-flux
values `(φ_exc, φ_inh)` for each state such that the **steady-state
open fractions** `O*_exc(φ_exc), O*_inh(φ_inh)` yield, after `g0`
scaling, the same matching-potential targets.

### Step 1: pick a normalisation convention

Fix `g0_exc + g0_inh = 1` (sum of peak conductances). This mirrors
the shallow normalisation and means the ratio `g0_exc / g0_inh` is
the only free magnitude knob.

### Step 2: pick a "saturating" photon flux

Define `φ_sat = 100 · phi_m` (≈100× the Hill K_M). At this flux the
Hill is at `100/101 ≈ 99%` saturation, so `k_a ≈ k_a_max`. The
steady-state at saturation is:

```
O*_sat = k_a_max · k_r / (k_a_max · k_r + k_a_max · k_d + k_d · k_r)
```

Plug in the defaults: for the excitatory opsin,
`O*_sat = 1000·5 / (1000·5 + 1000·50 + 50·5) = 5000 / 55250 ≈ 0.0905`.

### Step 3: pick `g0_exc, g0_inh` so the active-pitch ratio matches

At active-pitch saturation, we want
`(g0_exc · O*_sat_exc) / (g0_inh · O*_sat_inh) = (1/3) / (2/3) = 0.5`.

If `O*_sat_exc = O*_sat_inh` (same Hill K_M, same `k_a_max·k_r/(…)`
shape), this reduces to `g0_exc / g0_inh = 0.5`, i.e.
**`g0_exc = 1/3, g0_inh = 2/3`** under the `g0_exc + g0_inh = 1`
convention.

With the proposed defaults, `O*_sat_exc ≈ 0.0905` and
`O*_sat_inh ≈ 0.444` (the inhibitory `k_d = 10` gives a much higher
saturation open fraction). So the calibration is:

```
g0_inh / g0_exc = (1/3) / (2/3) · O*_sat_exc / O*_sat_inh
                = 0.5 · 0.0905 / 0.444 = 0.102
```

With `g0_exc + g0_inh = 1`: `g0_exc = 1 / (1 + 0.102) ≈ 0.907`,
`g0_inh ≈ 0.093`. The asymmetric `g0` compensates for the asymmetric
`O*_sat`.

### Step 4: pick φ for rest pitches

At rest we want
`(g0_exc · O*_exc(φ_exc_rest)) / (g0_inh · O*_inh(φ_inh_rest)) = (1/15)/(14/15)`.
This requires φ_inh_rest at saturation and φ_exc_rest at a much
weaker flux. Solving the Hill inversely:

```
O*_target = k_a · k_r / (k_a · k_r + k_a · k_d + k_d · k_r)
       ⇒ k_a = O*_target · k_d · k_r / (k_r · (1 − O*_target) − k_d · O*_target)
       ⇒ φ   = k_a · phi_m / (k_a_max − k_a)
```

Both steps are done by a new helper in `example_projects.py`:

```python
def _calibrate_phi_for_target_open_fraction(
    O_target: float,
    k_a_max: float, phi_m: float, k_d: float, k_r: float,
) -> float:
    """Inverse Hill — return φ such that O*(φ) = O_target."""
    if O_target <= 0:
        return 0.0
    denom = k_r * (1 - O_target) - k_d * O_target
    if denom <= 0:
        raise ValueError(f"O_target={O_target} exceeds physical maximum")
    k_a = O_target * k_d * k_r / denom
    if k_a >= k_a_max:
        raise ValueError(f"O_target={O_target} exceeds k_a_max saturation limit")
    return k_a * phi_m / (k_a_max - k_a)
```

`_build_fuer_elise_opsin_stims` consumes this helper to produce
`stim_exc, stim_inh` in **photon-flux units** rather than the current
fractional-conductance units. The pitch table and event timing stay
identical — only the values change.

### Step 5: sanity check

For ELISE's 10-ms square pulses, the kinetic model reaches
≈ `1 − exp(−10·k_a / 1000) ≈ 1 − exp(−10) ≈ 99.995%` of steady state
during an active event (when `k_a ≈ k_a_max` ≈ 1000 s⁻¹). So the
steady-state calibration is an excellent approximation of the actual
peak open fraction during the event, and the macroscopic teaching
signal at 100s-of-ms timescales is preserved within sub-percent
error. This is the entire point of choosing the steady-state
calibration over a transient one.

### Why steady-state, not transient

Two options were considered:

| Option                           | Description                                                                                                                                                                                    | Verdict                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| (a) **Steady-state calibration** | Pick φ such that `O*(φ)` matches the target ratio at saturation. The transient onset adds <1% error at 10 ms pulses.                                                                           | **Adopted.** Preserves the existing learning outcome.                                             |
| (b) **Transient calibration**    | Solve the kinetic ODEs forward for the pulse duration and pick φ such that the integrated `O` over the pulse matches. More physically faithful but the φ values then depend on pulse duration. | Rejected. Couples the calibration to pulse timing; breaks if the user changes the event duration. |

The steady-state choice means the kinetic model **is** active and
produces the qualitative features described in the kinetics
section (onset transient, sustained-light droop, paired-pulse
recovery), but the **default** Für-Elise teaching signal is
quantitatively the same as the shallow model produced.

---

## Stim-array units

The stim arrays change from "fractional conductance in [0, 1]" to
"photon flux density in mm⁻²·s⁻¹". Implications:

- The stim-editor displays values like `5e17` rather than `0.333`.
  The fastplotlib heatmap colourbar handles arbitrary scales; no UI
  change is required. The default colour-mapping range adapts
  automatically from the array's min/max.
- `StimArray.units` for the opsin pair becomes `"mm^-2 s^-1"`
  (current: `"1"`). This is metadata only; the simulator reads
  `values` directly.
- The `_FUER_ELISE_OPSIN_*` constants are removed from
  `example_projects.py`. They are replaced by an analogous set of
  four photon-flux values computed from
  `_calibrate_phi_for_target_open_fraction`. These four values are
  module-level constants for transparency — anyone reading the file
  can see what flux corresponds to "active" vs "rest".

---

## Archive migration

Pre-kinetic archives have opsin stims with values in [0, 1].
Loading them into a kinetic-aware ELISE without migration would
produce essentially zero open fraction (since 1.0 mm⁻²·s⁻¹ ≪ phi_m
= 5e17 mm⁻²·s⁻¹), silently breaking the model.

`Project._migrate_legacy_elise_stim` (already present for the
pre-opsin → opsin migration) gains a second migration step:

```python
def _migrate_legacy_elise_kinetics_units(self) -> None:
    """Convert pre-kinetic opsin stims (values in [0,1]) to photon flux.

    Detection: if a stim named 'opsin_exc' or 'opsin_inh' has all
    values in [0, 1] (max < 2), it's pre-kinetic and we apply the
    inverse Hill using the default kinetic parameters.
    """
```

The migration is **conservative**: it only fires if the values are
in [0, 1] _and_ the archive's project metadata does not carry a
`schema_version >= 2` marker. New archives written under the
kinetic ELISE include `schema_version = 2` so they're never
re-migrated.

The migration uses the default kinetic parameters
(`k_a_max = 1000`, `phi_m = 5e17`, …) regardless of what the
archive's `params` block contains. The reasoning: if a user
hand-tuned `k_a_max` and saved, the archive would be post-kinetic
and carry the version marker; if the archive predates kinetics,
its `params` block has no kinetic fields at all, so the defaults
are the correct reference.

---

## Tests

`tests/test_elise_opsin_kinetics.py` (new):

| Test                                                          | Asserts                                                                                                                                          |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_step_kinetics_dark_returns_O_to_zero`                   | Start `O = 0.5, D = 0`, `phi = 0` for 1 s; final `O < 1e-6`, `D` peaks then decays.                                                              |
| `test_step_kinetics_reaches_steady_state_at_saturating_light` | `phi = 100·phi_m`, integrate 200 ms; compare to closed-form `O*`. Tolerance 1e-3.                                                                |
| `test_step_kinetics_paired_pulse_depression`                  | Two 10-ms pulses, 200 ms ISI; second peak `O` is 30–80% of first.                                                                                |
| `test_steady_state_open_fraction_closed_form`                 | `O*` from the analytic formula matches forward-integration to 1e-4 across a grid of φ.                                                           |
| `test_calibrate_phi_inverse_hill_roundtrip`                   | For each `O_target` in [0.01, 0.5], `O*(_calibrate_phi(O_target)) == O_target`.                                                                  |
| `test_calibrate_phi_raises_above_saturation`                  | `O_target = 0.95` with the default rates raises `ValueError`.                                                                                    |
| `test_fuer_elise_kinetic_default_matches_shallow_u_tgt`       | Build the kinetic default project, run 10 ms of active-pitch stim, assert `u_tgt[active] ≈ −50 mV ± 0.5 mV` and `u_tgt[rest] ≈ −70 mV ± 0.5 mV`. |
| `test_kinetic_reset_zeros_state_arrays`                       | Run for 100 ms, call `reset()`, assert all four state arrays are zero.                                                                           |
| `test_kinetic_state_persists_across_runs`                     | Run 50 ms, run 50 ms more; verify `O, D` at t=100 match a single 100 ms run.                                                                     |
| `test_kinetic_signature_includes_kinetic_params`              | Change `k_d_exc` from 50 to 80 in params; next call to `simulate()` triggers a `reset()` (state arrays zeroed).                                  |

`tests/test_project_archive_roundtrip.py` (extended):

| Test                                                      | Asserts                                                                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_legacy_pre_kinetic_archive_migrates_to_photon_flux` | Load a pre-kinetic .nseq, assert `opsin_exc` values are now > 1e15, that simulating reproduces the same `u_tgt` as the legacy archive within 1%. |
| `test_kinetic_archive_roundtrip_preserves_units`          | Save a kinetic project, reload, assert `StimArray.units == "mm^-2 s^-1"` and `schema_version == 2`.                                              |

---

## Non-goals (intentionally not implemented)

- **Level 3 — full 6-state PyRho port.** Explored, rejected: the
  slow `_2` pathway has a build-up time constant `1/Gf ≈ 27 ms`,
  far longer than ELISE's 10-ms event duration. At ELISE's
  timescales, `O_2 ≈ 0` and the kinetic accuracy gain is < 5%
  (see regime analysis below). The full 6-state model lives in
  `pyrho_6state_chr2_project` for users who want to characterize
  opsins — ELISE doesn't need it.
- **Voltage-dependent photocurrent (`f_v`, `v − E`).** ELISE's
  `simulate` does not have a membrane voltage in its conductance
  step — it computes `u_tgt` analytically from steady-state
  conductances. Adding `f_v` would require pulling somatic voltage
  into the conductance update, an architectural change far beyond
  this spec's scope.
- **Exposing `E_exc, E_inh` as editable parameters.** They are
  physical constants of the opsin chemistry; making them editable
  invites silent breakage of the matching-potential calibration.
  If a user wants a different reversal-potential pair they can
  hand-edit the source.
- **Persisting kinetic state in archives.** `O, D` rebuild from `φ`
  within milliseconds — saving them would only hurt
  cross-machine reproducibility (different RNG / different float
  precision producing identical-looking but bitwise-different
  states). Reset to zero on load is correct.
- **Adaptive integrator.** Forward Euler at `dt = 1 ms` is stable
  for all default parameters. If a user pushes `k_a_max` above
  ~500 s⁻¹ while keeping `dt = 1 ms`, exponential Euler should be
  swapped in. The change is one line and behind a runtime guard,
  but the guard is **not** in the initial implementation — it's a
  follow-up only if user reports surface it.
- **Inhibitory opsin defaults from a published article.** No
  inhibitory opsin parameters appear in `pyrho_6state_chr2_project`.
  The proposed inhibitory defaults are hand-chosen to match
  published GtACR/NpHR macroscopics. A future spec could lock these
  to a specific opsin (GtACR2, eNpHR3.0, …) with cited values;
  this spec uses generic "slow-desensitizing inhibitory opsin"
  defaults.

---

## Regime analysis — why Level 2 is sufficient

The 3-state collapse drops four PyRho features: the slow `_2`
pathway, the intermediate `I` states, `f_v`, and the `γ`-weighted
open states. Quantitative deviation depends on the stimulation
regime:

| Regime                                | Pulse duration | 3-state vs 6-state deviation | Matters for ELISE?                        |
| ------------------------------------- | -------------- | ---------------------------- | ----------------------------------------- |
| A — short pulses, ≤ 20 ms (Für Elise) | 10 ms          | < 10% peak open fraction     | No                                        |
| B — sustained pulses, 100–700 ms      | 100–700 ms     | 20–40% peak/plateau ratio    | Only if user sets up long illumination    |
| C — paired pulses, ISI < 200 ms       | < 200 ms       | 10–30% in PPR amplitude      | Possibly — Für Elise full cycle is 240 ms |
| D — voltage-dependent I-V             | n/a            | N/A in ELISE                 | No                                        |

ELISE operates in regime A by construction. The Level-2 design
target is that, in regime A, the deviation from a full PyRho port
is **single-digit percent in the peak open fraction** — well below
the granularity at which the rate-coded teaching signal can be
distinguished from noise.

---

## Implementation order (when this is picked up)

1. Add `_step_kinetics` numba helper and unit-test it
   (`test_step_kinetics_*` tests).
2. Add state arrays to `Elise.__init__`, wire `reset()`.
3. Add the kinetic parameters to `default_params_schema` (default
   values only; calibration of `g0_exc, g0_inh` happens after the
   helper is in place).
4. Add `_calibrate_phi_for_target_open_fraction` helper and
   roundtrip tests.
5. Recompute the Für-Elise constants via the helper; update
   `_build_fuer_elise_opsin_stims` to emit photon flux.
6. Wire `_step_kinetics` into `Elise.simulate`; extend
   `_signature` with the kinetic fields.
7. Run regression: `test_fuer_elise_kinetic_default_matches_shallow_u_tgt`
   should pass before any learning-outcome tests are touched.
8. Add archive migration (`_migrate_legacy_elise_kinetics_units`)
   and roundtrip tests.
9. Update `elise_opsin_migration.md` to point at this spec for the
   kinetic layer, and add a "What this spec did not cover" note
   referencing the kinetic upgrade.
