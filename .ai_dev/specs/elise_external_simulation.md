# ELISE Network — External-Simulation Example Project

## Overview

Integrate the ELISE rate-coded sequence-learning network (Kriener et al., 2024,
package `elise` at `../sequence-learning` via `[tool.uv.sources]`) into
neuroseq as an example project, reusing and extending the existing
"external simulation function" backbone
(`gui.py:_load_external_simulation_function`, `_do_external_simulation`).

ELISE is numpy/numba — not Brian2. Porting its full N×N matrix-vector dendritic
currents and per-neuron delays to Brian2 equations would lose Numba JIT and
duplicate the upstream model. Instead we wrap ELISE behind a small **`NN`
plugin contract** (`src/neuroseq/example_models/NN.py`) — a base class that
any non-Brian2 simulator can subclass to plug into neuroseq's Run button and
parameter editor.

The architecture is generic on purpose: a sibling plugin
(`example_models/sinus_offset.py`) ships in the same branch as a minimal demo

- template for new plugins. The integration described below is the ELISE
  implementation of that contract.

To make ELISE feel like a first-class example project (opens from
**Open → Example Project**, has editable parameters, runs from the main
**Run** button), the "external simulation function" state was migrated from
a GUI-window attribute onto `Project` itself, so it can be persisted in
`.nseq` archives and bundled with example projects.

Applies to:

- New: `src/neuroseq/example_models/NN.py` — the plugin contract base class
- New: `src/neuroseq/example_models/elise.py` — the ELISE plugin
- New: `src/neuroseq/example_models/__init__.py`
- `src/neuroseq/project.py` — new external-simulate fields, loader, save/load
- `src/neuroseq/io.py` — read/write extended `project_metadata.json`
- `src/neuroseq/gui.py` — read function from project; Run dispatch; param editor
- `src/neuroseq/model_editor.py` — `EditableExternalSimParamsTree`
- `src/neuroseq/example_projects.py` — `@register_example_project("ELISE …")`
- `pyproject.toml` — `elise` path dependency via `[tool.uv.sources]`
- `external_simulation_example/{simulate.py,run_brian2_externally.py}` —
  add `params=None` kwarg
- `tests/` — new unit + smoke tests

---

## The `NN` plugin contract — `example_models/NN.py`

Any non-Brian2 simulator that wants to plug into neuroseq's Run button and
parameter editor implements this small interface:

```python
class NN:
    def __init__(self, default_params_schema: Schema) -> None: ...
    def default_params_schema(self) -> Schema: ...        # for the GUI editor
    def flat_default_params(self) -> FlatParams: ...      # {name: value}
    def update_params(self, params) -> None: ...          # resets state on change
    @property
    def simulated_time_seconds(self) -> float: ...        # cumulative across runs
    def reset(self) -> None: ...                          # subclass override
    def simulate(self, stims_dict, simulation_time,
                 params=None) -> Experiment: ...          # subclass override
```

State semantics: a plugin instance retains backend state across
`simulate(...)` calls (so successive Run clicks **continue** the simulation
rather than restart it). `update_params` detects parameter changes via dict
equality and calls `reset()` when needed.

A plugin module is expected to expose either:

- A `get_simulator()` function returning a singleton instance (preferred), **or**
- An `NN` class symbol that `Project.load_NN()` will instantiate, **or**
- A module-level `simulate` function (stateless fallback for legacy plugins).

ELISE picks the first option; `sinus_offset.py` mirrors the same pattern.

---

## ELISE plugin — `example_models/elise.py`

| Name             | Type                                                       | Purpose                                    |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------ |
| `DEFAULT_PARAMS` | `dict[str, dict]`                                          | Editable parameter schema                  |
| `Elise(NN)`      | class                                                      | Stateful ELISE backend                     |
| `simulate`       | `(stims_dict, simulation_time, params=None) -> Experiment` | Module-level compat wrapper over singleton |
| `get_simulator`  | `() -> Elise`                                              | Returns the module singleton `_SIMULATOR`  |

`_SIMULATOR = Elise()` is created at module import time; `simulate(...)` and
the GUI's bound `external_simulate_function` both delegate to it. A fresh
`Elise()` is what tests use when they need an isolated instance.

### Stim contract

`stims_dict` must contain a `"u_inp"` entry shaped like every other
`stims_dict` value the GUI builds:

```python
stims_dict["u_inp"]["array"]   # ndarray[T, num_vis], membrane drive in mV
stims_dict["u_inp"]["dt"]      # float, seconds
```

`num_vis` is **derived** from the stim array's second dimension; it is not a
parameter. `num_lat` is a parameter. Per integration step, the stim is
sampled at the **absolute** simulation time `t_abs_s = t0_s + step * dt`
where `t0_s = self._simulated_time_s` — so continuing a 1 s run with another
1 s run reads stim samples from t=1 s onward, not from t=0 again.

### State carry-over and signature-based reset

```python
def simulate(self, stims_dict, simulation_time, params=None):
    ...
    self.update_params(params)
    signature = (num_vis, tuple(sorted(self._params.items())))
    if self._network is None or signature != self._signature:
        self._initialize_network(num_vis)
        self._signature = signature
        self._simulated_time_s = 0.0
    ...
    self._simulated_time_s += simulation_time
```

The `(num_vis, params)` signature is the cache key: if a Run uses the same
stim shape and same parameter dict, state continues; otherwise the network
is reinitialised. Editing any param in the GUI flushes the cache.

### DEFAULT_PARAMS schema

Each entry is `{name: {value, default, type, group, description}}` (no
min/max per request). Five groups, 30 entries total. Defaults track
`sequence-learning/scripts/config.toml`.

| Group        | Keys                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `Network`    | `num_lat`                                                                                                         |
| `Neuron`     | `C_v`, `C_u`, `E_l`, `E_exc`, `E_inh`, `g_l`, `g_den`, `g_exc_0`, `g_inh_0`, `a`, `b`, `lam`                      |
| `Weights`    | `weight_init_min`, `weight_init_max`, `d_den_min`, `d_den_max`, `d_som_min`, `d_som_max`, `d_int`, `p`, `q`, `p0` |
| `Simulation` | `dt_ms`, `eta_vis`, `eta_lat`, `learn`, `weight_snapshot_interval_ms`                                             |
| `RNG`        | `seed`                                                                                                            |

### Setup pattern

`Elise._initialize_network(num_vis)` mirrors
`sequence-learning/scripts/train.py:setup_network`:

1. Lazy import of `elise.{model, optimizer, rate_buffer, weights}` (gated
   behind `_import_elise_types`, with `np.seterr` save/restore so elise's
   module-level `np.seterr(all="raise")` doesn't leak into the rest of the
   process).
2. Spawn 4 child RNGs from `seed` via `bit_generator._seed_seq.spawn(4)`.
3. Build `SimpleNamespace`s for `network_params`, `neuron_params`,
   `weight_params` (ELISE accesses parameters dotted, not as dicts).
4. `DendriticWeights` and the **developed** `SomaticWeights` (the
   `RandomSomaticWeights` variant is not exposed; `weight_type="developed"`
   is hardcoded).
5. `Network(network_params, neuron_params, dendritic_weights, somatic_weights, Buffer)`.
6. `network.prepare_for_simulation(dt_ms, SimpleUpdater(eta_vis/dt_ms), SimpleUpdater(eta_lat/dt_ms))`
   — eta is divided by dt to match `simulator.py:33-34` upstream.

ELISE only ships `SimpleUpdater` (no Adam). Hardcoded.

### Output

```python
Experiment(
    name="ELISE simulation",
    duration=simulation_time,
    timepoints=[],
    data=[
        TraceData("somatic_potential_u",   shape=(N, n_steps), units="mV"),
        TraceData("dendritic_potential_v", shape=(N, n_steps), units="mV"),
        TraceData("firing_rate_r",         shape=(N, n_steps), units="1"),
        SynapseData("dendritic_weights",   values=(N, N, n_snapshots),
                    timestamps=(n_snapshots,)),
    ],
)
```

`n_steps = round(simulation_time * 1000 / dt_ms)`,
`n_snapshots = ceil(n_steps / round(weight_snapshot_interval_ms / dt_ms))`.
Traces are float32. The somatic weight matrix and delays are not exported
(fixed throughout the run). Weight-snapshot timestamps are **per-run local**
(0..simulation_time), so a chunk-based UI can offset them by
`self._simulated_time_s - simulation_time` to reconstruct an absolute axis.

---

## Project state migration

The pre-branch `external_simulation_function` lived on the main window
(`gui.py:74`) and was lost on project switch. It now lives on `Project` so
it persists in `.nseq` and can be bundled with example projects.

### New fields on `Project`

```python
self.external_simulate_path: str | None = None         # serialized
self.external_simulate_params: dict[str, dict] = {}    # serialized
self.external_simulate_function: Callable | None = None  # transient (bound method)
self.external_simulator: NN | None = None              # transient (full instance)
```

`external_simulator` is the full plugin instance (for code that needs
`reset()`, `simulated_time_seconds`, etc.); `external_simulate_function` is
its bound `.simulate` method (for the unchanged GUI call site).

### `Project.load_NN()`

```python
def load_NN(self) -> None:
    """Resolve `external_simulate_path` into a plugin instance + bound method."""
    path = self.external_simulate_path
    if path is None:
        self.external_simulator = None
        self.external_simulate_function = None
        return
    if path.startswith("neuroseq:"):
        module_name = path[len("neuroseq:") :]
        if module_name == "elise_simulate":  # legacy archive compat
            module_name = "elise"
        module = importlib.import_module(f"neuroseq.example_models.{module_name}")
        # 1) get_simulator() singleton; 2) NN class; 3) module-level simulate()
        if hasattr(module, "get_simulator"):
            self.external_simulator = module.get_simulator()
        elif hasattr(module, "NN"):
            self.external_simulator = module.NN()
        # ...
        self.external_simulate_function = (
            self.external_simulator.simulate
            if self.external_simulator is not None
            else module.simulate
        )
    else:
        self.external_simulator = None
        self.external_simulate_function = runpy.run_path(path)["simulate"]
```

A `load_external_simulator()` method exists as a back-compat alias that
delegates to `load_NN()`.

Called at the end of `__init__` if `external_simulate_path` is set, and from
`gui.py:_load_external_simulation_function` after writing the path. The
`elise_simulate` → `elise` rewrite inside `load_NN` keeps `.nseq` archives
saved before the module rename loadable.

### `project_metadata.json` extension

```json
{
  "dt": 0.001,
  "external_simulate_path": "neuroseq:elise",
  "external_simulate_params": { "num_lat": { "value": 50, ... }, ... }
}
```

Reading code in `Project.__init__` pulls both new keys with `.get(key, default)`
for backward compat with old archives.

`external_simulator` / `external_simulate_function` are **not** serialized —
they are reconstructed from the path on load.

---

## Run dispatch

`btn_continue_simulation_clicked` (the main "Run" action) is the dispatch
point:

```python
def btn_continue_simulation_clicked(self) -> None:
    simulation_time = getSimulationTime(self)
    if simulation_time is None:
        return
    if self.project.external_simulate_path is not None:
        self._do_external_simulation(simulation_time)
        return
    # ... existing b2_model path unchanged ...
```

The standalone **Run external stimulation function** action keeps working —
useful when a user manually loads a function on top of a Brian2 project.

### `_do_external_simulation`

- Reads `self.project.external_simulate_function` instead of
  `self.external_simulation_function`.
- Flattens params: `{k: v["value"] for k, v in self.project.external_simulate_params.items()}`.
- Calls `simulate(stims_dict, simulation_time, params=flat_params)`. For
  plugin-backed simulators this is the bound `Elise.simulate` method, so
  state persists across Run clicks.

### `_load_external_simulation_function`

```python
self.project.external_simulate_path = path
self.project.load_NN()
```

The window-level `self.external_simulation_function` attribute is removed.

### Backward compat for the legacy simulate signature

The two stock example simulators in `external_simulation_example/` accept
`(stims_dict, simulation_time)` only. To use the unified call site, both
gained a `params=None` kwarg (params is ignored by those legacy modules).

---

## Editable params in the model editor

`EditableExternalSimParamsTree` (in `model_editor.py`) subclasses
`EditableParameterTree` for IS-A only — it bypasses the b2_model-coupled
parent constructor by calling `ParameterTree.__init__` directly.

- Bound to `project.external_simulate_params` (mutated in place).
- Top-level groups come from each entry's `group` field; group order =
  insertion order in the dict.
- Per-entry pyqtgraph parameter type from `entry["type"]`:
  `"int"` → IntParameter, `"float"` → FloatParameter, `"bool"` → BoolParameter,
  `"str"` → StrParameter.
- Edits write back to `entry["value"]` via a closure-captured handler
  (`_connect_leaf` — the closure binding is mandatory: `sigValueChanged`
  emits `(parameter, new_value)` and a default-kwarg name binding gets
  silently overwritten by the second positional arg).
- No add/delete/rename: the schema is fixed.

### Dialog dispatch

`ModelEditorDialog.__init__` gains an optional `project: Project | None = None`
kwarg. When `project is not None and project.external_simulate_path is not None`,
the dialog routes through `_init_external_simulate(project, b2model)`:

- Deep-copies `project.external_simulate_params` onto
  `self.external_simulate_params` (non-destructive edit).
- Builds a single-tree vertical layout (group label, simulate-function-path
  caption, `EditableExternalSimParamsTree`, OK/Cancel).
- `accept()` short-circuits past the b2_model rebuild when
  `external_simulate_params is not None`.

Caller (`gui.py:btn_edit_clicked`) passes `project=self.project` and, on
accept, copies the edited dict back into the project. The b2 path is
untouched when `project=None` is passed (backward compat).

---

## Example project registration

`example_projects.py` adds `import copy`, the stim-pattern helper, and the
registered factory:

```python
def _build_elise_sequence_pattern(tfinal_ms, n_vis,
                                  slot_ms=50, rest_mv=-70.0, active_mv=-55.0):
    """[T, n_vis] one-hot rolling pulse train at 1 ms resolution."""
    ...

@register_example_project("ELISE Sequence Learning Network", section="Network")
def elise_project() -> Project:
    # Deferred so unrelated example projects don't pull in elise/numba.
    from neuroseq.example_models.elise import DEFAULT_PARAMS

    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_NN()

    n_vis, tfinal_ms = 4, 5000
    stim = _build_elise_sequence_pattern(tfinal_ms, n_vis)
    project.add_stim_array(StimArray(
        name="u_inp", stim_type=StimType.External,
        values=stim, units="mV", dt=0.001,
    ))
    return project
```

The pattern: each visible neuron is held at -55 mV for 50 ms in turn, with
the others at the default `E_l = -70 mV`; cycle period `n_vis * slot_ms` =
200 ms. At `n_vis=4` and `tfinal_ms=5000` the stim contains 25 cycles.
`num_lat` defaults to 50; the user can dial it down via the model editor
for smaller demos.

A sibling `sinus_offset_project()` is registered next to it under the same
section, using the same pattern with a different plugin module — concrete
evidence the abstraction generalises beyond ELISE.

The `elise` / `DEFAULT_PARAMS` import is deferred to the function body so
opening unrelated example projects does not pay the elise/numba/mlflow
import cost at GUI startup. `example_projects.py` is loaded eagerly by
`gui.py`.

---

## Dependency

`pyproject.toml` adds:

```toml
dependencies = [..., "elise"]

[tool.uv.sources]
elise = { path = "../sequence-learning", editable = true }
```

ELISE upstream pins `requires-python = ">=3.11"`; in dev environments running
Python 3.10 you can locally relax that pin in the path-deployed
`sequence-learning/pyproject.toml` (the imported modules — `model`,
`optimizer`, `weights`, `rate_buffer` — don't use 3.11-only syntax). CI on
macOS arm64 uses Python 3.10 + 3.12 and just consumes the upstream pin
unchanged.

> **Vendoring note.** An earlier branch (`implement_elise` → commit
> `b2f6659`, since reverted) explored vendoring the four upstream modules
> under `neuroseq.example_models._elise` to drop the path dep entirely.
> That was rolled back in favor of keeping the path dep, so this branch
> tracks upstream elise without copying. If/when elise is published to PyPI
> the `[tool.uv.sources]` block goes away and `"elise"` becomes a regular
> versioned dep.

---

## Implementation status

| Step | Scope                                                                           | Status |
| ---- | ------------------------------------------------------------------------------- | ------ |
| 1    | `elise.py` + `Elise(NN)` + `DEFAULT_PARAMS` + smoke tests + path dep            | done   |
| 2    | `NN` base class extracted; `sinus_offset` plugin as second implementation       | done   |
| 3    | `Project` fields, `load_NN`, save/load metadata, legacy sentinel rewrite        | done   |
| 4    | `gui.py`: Run dispatch, migrate readers/writers, simulate-signature kwarg       | done   |
| 5    | `EditableExternalSimParamsTree` + dialog dispatch                               | done   |
| 6    | `@register_example_project("ELISE …")` + sequence stim builder                  | done   |
| 7    | Docs: `developer/custom_simulators.md`, `user_guide/run_external_simulators.md` | done   |

---

## Tests

| Test                                                                                            | Asserts                                                                                                                          |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_elise_smoke.py::test_elise_runs`                                                    | Module-level `simulate()` with a 5-vis × 200 ms stim returns the four data objects with correct shapes; `r ∈ [0, 1]`             |
| `tests/test_elise_smoke.py::test_elise_simulator_continues_state_across_runs`                   | `Elise()` instance accumulates `simulated_time_seconds`; second run diverges from a fresh instance's first run                   |
| `tests/test_project_external_simulate.py::test_metadata_roundtrip`                              | `Project.save` → reload preserves `external_simulate_path` and `external_simulate_params`                                        |
| `tests/test_project_external_simulate.py::test_load_neuroseq_sentinel`                          | `load_NN()` resolves `"neuroseq:elise"` to a callable                                                                            |
| `tests/test_project_external_simulate.py::test_load_filesystem_path`                            | `load_NN()` resolves a `.py` file via `runpy.run_path`                                                                           |
| `tests/test_gui_external_simulate_dispatch.py::test_run_dispatches_to_external_when_path_set`   | When `project.external_simulate_path` is set, "Run simulation" routes to `_do_external_simulation`, not `project.run_simulation` |
| `tests/test_gui_external_simulate_dispatch.py::test_run_dispatches_to_b2_when_path_unset`       | Without external path, Run still hits the b2 path                                                                                |
| `tests/test_gui_external_simulate_dispatch.py::test_external_simulation_passes_params_kwarg`    | `_do_external_simulation` flattens `external_simulate_params` and passes it as `params=` kwarg                                   |
| `tests/test_external_sim_params_tree.py::test_tree_groups_and_types`                            | Tree builds typed leaves grouped by `group` field, in insertion order                                                            |
| `tests/test_external_sim_params_tree.py::test_tree_edit_writes_back_to_dict`                    | `leaf.setValue(...)` writes through to `params_dict[name]["value"]`                                                              |
| `tests/test_external_sim_params_tree.py::test_dialog_external_mode_when_path_set`               | Dialog deep-copies params; edits don't leak to project until accept                                                              |
| `tests/test_external_sim_params_tree.py::test_dialog_b2_mode_when_path_unset`                   | Dialog still builds the b2 5-tree layout when external path unset                                                                |
| `tests/test_external_sim_params_tree.py::test_dialog_b2_mode_when_project_omitted`              | Backward compat: `ModelEditorDialog(b2model)` (no `project=`) still works                                                        |
| `tests/test_external_sim_params_tree.py::test_dialog_accept_in_external_mode_does_not_touch_b2` | External-mode `accept()` short-circuits past `b2model.assemble_network()`                                                        |
| `tests/test_example_project_elise.py::test_elise_example_project_registered`                    | "ELISE Sequence Learning Network" appears under the "Network" section                                                            |
| `tests/test_example_project_elise.py::test_elise_example_project_builds`                        | Factory yields a project with `external_simulate_path == "neuroseq:elise"`, stim shape `(5000, 4)`, one-hot first pulse          |
| `tests/test_example_project_elise.py::test_elise_example_simulate_runs`                         | End-to-end call through the project (100 ms sim, num_lat=8) returns the four named data objects                                  |
| `tests/test_example_project_elise.py::test_sequence_pattern_cycles_correctly`                   | Pattern is periodic with period `n_vis * slot_ms`, slots non-overlapping, rest values match `rest_mv`                            |
| `tests/test_sinus_offset_smoke.py` / `test_example_project_sinus_offset.py`                     | Mirror coverage of the sibling plugin to verify the `NN` abstraction generalises                                                 |

CI green: 607 passed, 2 skipped, 77.99% total coverage / 88% diff-coverage,
all pre-commit hooks pass.
