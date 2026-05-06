# ELISE Network — External-Simulation Example Project

## Overview

Integrate the ELISE rate-coded sequence-learning network (Kriener et al., 2024,
package `elise` from `/mnt/live1/svintila/sequence-learning`) into neuroseq as
an example project, reusing the existing **external simulation function**
backbone (`gui.py:_load_external_simulation_function`,
`_do_external_simulation`).

ELISE is numpy/numba — not Brian2. Porting its full N×N matrix-vector dendritic
currents and per-neuron delays to Brian2 equations would lose Numba JIT and
duplicate the upstream model. Instead we wrap ELISE behind the
`simulate(stims_dict, simulation_time, params=None) -> Experiment` contract that
neuroseq already supports for external functions.

To make ELISE feel like a first-class example project (opens from
**Open → Example Project**, has editable parameters, runs from the main **Run**
button), the existing "external simulation function" state is migrated from a
GUI-window attribute onto the `Project` itself, so it can be persisted in
`.nseq` archives and bundled with example projects.

Applies to:

- New: `src/neuroseq/example_models/elise_simulate.py`
- New: `src/neuroseq/example_models/__init__.py`
- `src/neuroseq/project.py` — new external-simulate fields, loader, save/load
- `src/neuroseq/io.py` — read/write extended `project_metadata.json`
- `src/neuroseq/gui.py` — read function from project; Run dispatch; param editor
- `src/neuroseq/model_editor.py` — `EditableExternalSimParamsTree`
- `src/neuroseq/example_projects.py` — `@register_example_project("ELISE …")`
- `pyproject.toml` — `elise` path dependency
- `external_simulation_example/{simulate.py,run_brian2_externally.py}` —
  add `params=None` kwarg
- `tests/` — new unit + smoke tests

---

## ELISE wrapper — `elise_simulate.py`

Single file exposing two names:

| Name             | Type                                                       | Purpose                               |
| ---------------- | ---------------------------------------------------------- | ------------------------------------- |
| `DEFAULT_PARAMS` | `dict[str, dict]`                                          | Editable parameter schema (see below) |
| `simulate`       | `(stims_dict, simulation_time, params=None) -> Experiment` | The external simulate entry point     |

### Stim contract

`stims_dict` must contain a `"u_inp"` entry shaped like every other
`stims_dict` value the GUI builds (`gui.py:813`):

```python
stims_dict["u_inp"]["array"]   # ndarray[T, num_vis], membrane drive in mV
stims_dict["u_inp"]["dt"]      # float, seconds
```

`num_vis` is **derived** from the stim array's second dimension; it is not a
parameter. `num_lat` is a parameter. Per integration step, the stim is sampled
with nearest-index lookup (`min(int(t_s / stim_dt_s), T-1)`) — the last sample
holds for any time past the stim's end.

### DEFAULT_PARAMS schema

Each entry is `{name: {value, default, type, group, description}}` (no min/max
per request). Five groups, 30 entries total. Defaults track
`sequence-learning/scripts/config.toml`.

| Group        | Keys                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `Network`    | `num_lat`                                                                                                         |
| `Neuron`     | `C_v`, `C_u`, `E_l`, `E_exc`, `E_inh`, `g_l`, `g_den`, `g_exc_0`, `g_inh_0`, `a`, `b`, `lam`                      |
| `Weights`    | `weight_init_min`, `weight_init_max`, `d_den_min`, `d_den_max`, `d_som_min`, `d_som_max`, `d_int`, `p`, `q`, `p0` |
| `Simulation` | `dt_ms`, `eta_vis`, `eta_lat`, `learn`, `weight_snapshot_interval_ms`                                             |
| `RNG`        | `seed`                                                                                                            |

### Setup pattern

Mirrors `sequence-learning/scripts/train.py:setup_network`:

1. Spawn 4 child RNGs from `seed` via `bit_generator._seed_seq.spawn(4)`.
2. Build `SimpleNamespace`s for `network_params`, `neuron_params`, `weight_params`
   (ELISE accesses parameters dotted, not as dicts).
3. `DendriticWeights` and the **developed** `SomaticWeights` (the
   `RandomSomaticWeights` variant is not exposed; the schema hardcodes
   `weight_type="developed"`).
4. `Network(network_params, neuron_params, dendritic_weights, somatic_weights, Buffer)`.
5. `network.prepare_for_simulation(dt_ms, SimpleUpdater(eta_vis/dt_ms), SimpleUpdater(eta_lat/dt_ms))`
   — eta is divided by dt to match `simulator.py:33-34`.

ELISE only ships `SimpleUpdater` (no Adam). Hardcoded.

### Output

```python
Experiment(
    name="ELISE simulation",
    duration=simulation_time,
    timepoints=[],
    data=[
        TraceData("somatic_potential_u",  shape=(N, n_steps), units="mV"),
        TraceData("dendritic_potential_v", shape=(N, n_steps), units="mV"),
        TraceData("firing_rate_r",        shape=(N, n_steps), units="1"),
        SynapseData("dendritic_weights",  values=(N, N, n_snapshots),
                    timestamps=(n_snapshots,)),
    ],
)
```

`n_steps = round(simulation_time * 1000 / dt_ms)`,
`n_snapshots = ceil(n_steps / round(weight_snapshot_interval_ms / dt_ms))`.

Traces are float32; the somatic weights matrix and delays are not exported
(they are fixed throughout the run).

---

## Project state migration

The existing `external_simulation_function` lives on the main window
(`gui.py:74`) and is lost on project switch. Move it onto `Project` so it
persists in `.nseq` and can be bundled with example projects.

### New fields on `Project`

```python
# project.py — Project.__init__
self.external_simulate_path: str | None = None        # serialized
self.external_simulate_params: dict[str, dict] = {}   # serialized
self.external_simulate_function: Callable | None = None  # transient
```

### `Project.load_external_simulate()`

```python
def load_external_simulate(self) -> None:
    """Resolve `external_simulate_path` into `external_simulate_function`."""
    path = self.external_simulate_path
    if path is None:
        self.external_simulate_function = None
        return
    if path.startswith("neuroseq:"):
        # Built-in example, e.g. "neuroseq:elise_simulate"
        module_name = path.removeprefix("neuroseq:")
        module = importlib.import_module(f"neuroseq.example_models.{module_name}")
        self.external_simulate_function = module.simulate
    else:
        # User-loaded .py file
        self.external_simulate_function = runpy.run_path(path)["simulate"]
```

Called at the end of `__init__` if `external_simulate_path` is set, and from
`gui.py:_load_external_simulation_function` after writing the path.

### `project_metadata.json` extension

Currently written by `Project.save()` as `{"dt": self.dt}` (project.py:718).
Extend to:

```json
{
  "dt": 0.001,
  "external_simulate_path": "neuroseq:elise_simulate",
  "external_simulate_params": { "num_lat": { "value": 50, ... }, ... }
}
```

Reading code in `Project.__init__` (project.py:120-123) is extended to pull
both new keys with `.get(key, default)` for backward compat with old archives.

The `external_simulate_function` is **not** serialized — it is reconstructed
from the path on load.

---

## Run dispatch

`btn_continue_simulation_clicked` (gui.py:753, the main "Run" action wired at
gui.py:1596-1597) is the only entry point that needs to dispatch:

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

The standalone "**Run external stimulation function**" action
(gui.py:1797-1798) keeps working — useful when a user manually loads a
function on top of a Brian2 project.

### `_do_external_simulation` updates

`gui.py:803`:

- Reads `self.project.external_simulate_function` instead of
  `self.external_simulation_function`.
- Flattens params: `{k: v["value"] for k, v in self.project.external_simulate_params.items()}`.
- Calls `simulate(stims_dict, simulation_time, params=flat_params)`.

### `_load_external_simulation_function` updates

`gui.py:1055`:

```python
self.project.external_simulate_path = path
self.project.load_external_simulate()
```

The window-level `self.external_simulation_function` attribute is removed.
`btn_do_external_simulation_clicked` (gui.py:835) and the existing readers are
re-pointed at `self.project.external_simulate_function`.

### Backward compat for the simulate signature

The two existing example simulators accept `(stims_dict, simulation_time)`
only. To use the unified call site, both grow a `params=None` kwarg:

```python
def simulate(stims_dict, simulation_time, params=None):  # params ignored
    ...
```

Files: `external_simulation_example/simulate.py:7`,
`external_simulation_example/run_brian2_externally.py:9`.

---

## Editable params in the model editor

A new `EditableExternalSimParamsTree` class in `model_editor.py` subclassing
`EditableParameterTree` for IS-A only — it bypasses the b2_model-coupled
parent constructor by calling `ParameterTree.__init__` directly.

- Bound to `project.external_simulate_params` (mutated in place).
- Top-level groups come from each entry's `group` field (`Network`, `Neuron`,
  `Weights`, `Simulation`, `RNG`); group order = insertion order in the dict.
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

Caller (`gui.py:btn_edit_clicked`) passes `project=self.project` and, on accept,
copies the edited dict back: `self.project.external_simulate_params = mdiag.external_simulate_params`.

The b2 path is untouched when `project=None` is passed (backward compat).

---

## Example project registration

`example_projects.py` adds `import copy`, the helper, and the registered
factory at the bottom of the file:

```python
def _build_elise_sequence_pattern(tfinal_ms, n_vis,
                                  slot_ms=50, rest_mv=-70.0, active_mv=-55.0):
    """[T, n_vis] one-hot rolling pulse train at 1 ms resolution."""
    ...

@register_example_project("ELISE Sequence Learning Network", section="Network")
def elise_project() -> Project:
    # Deferred so unrelated example projects don't pull in elise/numba.
    from neuroseq.example_models.elise_simulate import DEFAULT_PARAMS

    project = Project()
    project.external_simulate_path = "neuroseq:elise_simulate"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_external_simulate()

    n_vis, tfinal_ms = 4, 5000
    stim = _build_elise_sequence_pattern(tfinal_ms, n_vis)
    project.add_stim_array(StimArray(
        name="u_inp", stim_type=StimType.External,
        values=stim, units="mV", dt=0.001,
    ))
    return project
```

The pattern: each visible neuron is held at -55 mV for 50 ms in turn, with the
others at the default `E_l = -70 mV`; cycle period `n_vis * slot_ms` = 200 ms.
At `n_vis=4` and `tfinal_ms=5000` the stim contains 25 cycles. `num_lat`
defaults to 50 (DEFAULT_PARAMS); the user can dial it down via the model
editor for smaller demos.

The `elise_simulate` import is deferred to the function body so opening
unrelated example projects does not pay the elise/numba/mlflow import cost
at GUI startup. `example_projects.py` is loaded eagerly by `gui.py`.

---

## Dependency

`pyproject.toml` adds:

```toml
dependencies = [..., "elise"]

[tool.uv.sources]
elise = { path = "/mnt/live1/svintila/sequence-learning", editable = true }
```

ELISE upstream pinned `requires-python = ">=3.11"`; relaxed to `>=3.10` in
the path-deployed `pyproject.toml` (the conda env this runs against is 3.10
and the imported modules — `model`, `optimizer`, `weights`, `rate_buffer` —
don't use 3.11-only syntax).

---

## Implementation status

| Step | Scope                                                                     | Status |
| ---- | ------------------------------------------------------------------------- | ------ |
| 1    | `elise_simulate.py` + `DEFAULT_PARAMS` + smoke test + path dependency     | done   |
| 2    | `Project` fields, `load_external_simulate`, save/load metadata            | done   |
| 3    | `gui.py`: Run dispatch, migrate readers/writers, simulate-signature kwarg | done   |
| 4    | `EditableExternalSimParamsTree` + dialog dispatch                         | done   |
| 5    | `@register_example_project("ELISE …")` + sequence stim builder            | done   |

---

## Tests

| Test                                                                                            | Asserts                                                                                                                          |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_elise_simulate_smoke.py::test_elise_simulate_runs` (existing)                       | Calling `simulate` with a 5-vis × 200 ms stim returns the four data objects with correct shapes; `r ∈ [0, 1]`                    |
| `tests/test_project_external_simulate.py::test_metadata_roundtrip`                              | `Project.save` → reload preserves `external_simulate_path` and `external_simulate_params`                                        |
| `tests/test_project_external_simulate.py::test_load_neuroseq_sentinel`                          | `load_external_simulate()` resolves `"neuroseq:elise_simulate"` to a callable                                                    |
| `tests/test_project_external_simulate.py::test_load_filesystem_path`                            | `load_external_simulate()` resolves a `.py` file via `runpy.run_path`                                                            |
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
| `tests/test_example_project_elise.py::test_elise_example_project_builds`                        | Factory yields a project with the expected sentinel path, stim shape `(5000, 4)`, and one-hot first pulse at neuron 0            |
| `tests/test_example_project_elise.py::test_elise_example_simulate_runs`                         | End-to-end call through the project (100 ms sim, num_lat=8) returns the four named data objects                                  |
| `tests/test_example_project_elise.py::test_sequence_pattern_cycles_correctly`                   | Pattern is periodic with period `n_vis * slot_ms`, slots non-overlapping, rest values match `rest_mv`                            |
