# ELISE — `clamp_visible` train/replay phase toggle

## Overview

The ELISE wrapper exposes a `clamp_visible` boolean parameter that lets the
user toggle between **training** (visible neurons are nudged toward the
stim) and **replay** (visible neurons evolve freely on the recurrent
input — the "ring out" phase from Kriener et al., 2024 §3.1) between
Runs, without losing the trained network state.

The upstream `Network.__call__` already distinguishes the two phases by
the `u_inp` argument: `u_inp=<vector>` adds a teaching current; `u_inp=None`
skips the entire teaching-current block in `total_diff_eq`. The previous
wrapper always sliced the stim and passed the vector, so a replay phase
was impossible from the GUI — a flat-E_l (-70 mV) stim isn't equivalent
to release because the conductance-based input current still actively
nudges the soma toward E_l.

The fix has two surfaces. The first is the obvious one: the
simulate-loop reads `clamp_visible` once per Run and passes `None` when
it's False. The second is the singleton-state issue: `NN.update_params`
auto-resets state on any param change, and `Elise.simulate`'s signature
cache also re-initialises the network on signature mismatch. Both
machineries had to learn to ignore a small set of **phase params** so
toggling between training and replay (or pausing plasticity with `learn`)
preserves the trained weights, voltages, and cumulative simulated time.

Applies to:

- `src/neuroseq/example_models/elise.py` — DEFAULT_PARAMS entry,
  `_PHASE_PARAMS` class attribute, `Elise.update_params` override,
  signature filter in `simulate`, conditional in the simulate loop body
- `tests/test_elise_clamp_visible.py` (new) — 5 wrapper-logic tests + 1
  mock-network test + 3 elise-gated integration tests

---

## The `clamp_visible` parameter

New entry in `DEFAULT_PARAMS` under the `Simulation` group, placed
immediately after `learn`:

```python
"clamp_visible": {
    "value": True,
    "default": True,
    "type": "bool",
    "group": "Simulation",
    "description": (
        "If True (training), drive visible neurons every step with the "
        "stim array. If False (replay), release the teaching nudging - "
        "let the network ring out from its current state. "
        "Toggling this between Runs does NOT reset the network."
    ),
},
```

Default `True` preserves the previous behaviour for existing projects
and `.nseq` archives — `NN.update_params` merges loaded params over the
default schema, so old archives lacking the key naturally pick up
`True` and behave identically.

---

## Phase params and state preservation

`Elise` declares a class-level `_PHASE_PARAMS` set containing the
parameters whose changes do **not** count as a trajectory change:

```python
class Elise(NN):
    _PHASE_PARAMS: frozenset[str] = frozenset({"clamp_visible", "learn"})
```

`learn` belongs in the set for symmetry — pausing plasticity between
Runs is the analogous "I want to observe without changing weights" toggle
and shouldn't reset state either.

### `Elise.update_params` override

Overrides `NN.update_params` to split the diff check by phase vs
trajectory:

```python
def update_params(self, params):
    merged = self._flat_defaults()
    if params:
        merged.update(params)
    old_traj = {k: v for k, v in self._params.items()
                if k not in self._PHASE_PARAMS}
    new_traj = {k: v for k, v in merged.items()
                if k not in self._PHASE_PARAMS}
    if old_traj != new_traj:
        self._params = merged
        self.reset()
    else:
        # Trajectory-affecting params unchanged: update phase params
        # in place, keep _network, _simulated_time_s, _signature.
        self._params = merged
```

If a Run changes a trajectory param alongside a phase param, the
trajectory diff wins and the network resets — covered by
`test_update_params_mixed_change_resets`.

### Signature cache in `Elise.simulate`

The same `_PHASE_PARAMS` filter is applied to the signature tuple used
as the network-cache key, so a pure phase change doesn't force
re-initialisation:

```python
signature = (
    num_vis,
    tuple(
        sorted(
            (k, v) for k, v in self._params.items()
            if k not in self._PHASE_PARAMS
        )
    ),
)
```

The two mechanisms (override + signature filter) belt-and-braces the
same invariant: changes restricted to phase params preserve
`self._network`, `self._signature`, and `self._simulated_time_s`.

---

## Simulate-loop conditional

`clamp_visible` is hoisted **above** the integration loop — read once per
`simulate()` call, since the phase is fixed for the duration of a single
Run:

```python
clamp_visible = _bool_param(self._params, "clamp_visible")
...
for step in range(n_steps):
    t_s = step * dt_ms / 1000.0
    t_abs_s = t0_s + t_s
    if clamp_visible:
        stim_idx = min(int(t_abs_s / stim_dt_s), stim_T - 1)
        u_inp_step = u_inp_full[stim_idx]
    else:
        u_inp_step = None
    self._network(u_inp=u_inp_step, learn=learn)
    ...
```

When `clamp_visible=False`, the stim array isn't read inside the loop at
all — but the up-front validation block (`u_inp_full.ndim != 2`,
non-empty time axis) still runs, so a malformed stim is still caught.

---

## How to use it from the GUI

The workflow that prompted this feature:

1. Open the ELISE example project. `clamp_visible = True` by default.
2. Click **Run** a few times to train. Each Run is one 1.92 s
   presentation of Für Elise; the singleton-state design accumulates
   weights and the absolute-time stim sampling keeps successive Runs
   continuing the same pattern.
3. Open the **model editor** → set `clamp_visible = False`. Click OK.
   The network state is **preserved** because `clamp_visible` is a
   phase param — neither `update_params` nor the signature cache wipes
   `_network`.
4. Click **Run** again — say for 2 s. The wrapper now calls
   `network(u_inp=None)` every step. Visible neurons drift on the
   somato-somatic recurrent input alone — the "ring out."
5. Flip `clamp_visible` back to True to resume training. The cycle is
   fully reversible.

---

## Tests

`tests/test_elise_clamp_visible.py` — 9 tests, all green:

| Test                                                                           | Asserts                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_clamp_visible_in_default_params`                                         | Schema entry has `value/default=True`, `type=bool`, `group=Simulation`.                                                                                                                                                                                            |
| `test_phase_params_includes_clamp_visible_and_learn`                           | Both names in `Elise._PHASE_PARAMS`.                                                                                                                                                                                                                               |
| `test_update_params_toggling_phase_does_not_reset`                             | Toggling `clamp_visible` keeps the sentinel `_network`, `_signature`, `_simulated_time_s`.                                                                                                                                                                         |
| `test_update_params_non_phase_change_resets`                                   | Changing `num_lat` calls `reset()`: `_network=None`, `_signature=None`, `_simulated_time_s=0.0`.                                                                                                                                                                   |
| `test_update_params_mixed_change_resets`                                       | A combined phase+trajectory change resets.                                                                                                                                                                                                                         |
| `test_simulate_clamp_visible_false_passes_none_and_preserves_network` (mocked) | Monkeypatches `_import_elise_types` with fakes. Verifies: first Run with `True` records non-None `u_inp` calls; second Run with `False` records `u_inp is None` for every step and reuses the SAME network instance; `simulated_time_seconds=0.4` after both Runs. |
| `test_clamp_visible_true_passes_stim_vector_to_network` (elise-gated)          | With a -50 mV stim and `clamp_visible=True`, at least one visible neuron's final `u` deviates from E_l by > 1 mV.                                                                                                                                                  |
| `test_clamp_visible_false_keeps_visible_near_rest` (elise-gated)               | With the same stim but `clamp_visible=False` and a fresh network, all visible `u_k` stay within 5 mV of E_l.                                                                                                                                                       |
| `test_clamp_visible_toggle_preserves_network_across_runs` (elise-gated)        | Train one Run with True, then release on the next with False. `id(sim._network)` unchanged across the boundary, cumulative time = 0.1 s.                                                                                                                           |

The mocked test is the strongest of the set — it directly verifies the
"pass `None` for every step in the False phase" behaviour without
requiring the elise package to be installed, and confirms network
identity is preserved across the toggle.

---

## Non-goals

- **Mid-run phase switching.** `clamp_visible` is read once at the top
  of `simulate()`. If a user wants to release halfway through a single
  Run, the right shape is a separate `release_after_seconds` scalar or a
  per-step mask channel; both are out of scope here.
- **A dedicated GUI button for replay.** The user toggles the param via
  the existing model editor; no new toolbar action.
- **On-disk simulator state.** The singleton lives in the running process
  only — closing the GUI still loses trained weights. Same constraint
  as before this feature.
