# Custom simulators

Any network (a simulator, a neuromorphic chip, a remote rig) can act as a
NeuroSeq backend: it receives the input arrays and returns an `Experiment`.
Write it as a standalone Python file so it needs no core changes. Using one from
the GUI is described in [Run external simulators](../networks/external_simulators.md).

## The `simulate` contract

```python
from neuroseq.data import Experiment


def simulate(stims_dict, simulation_time, params=None, progress=None) -> Experiment:
    ...
```

- `stims_dict`: `{name: {"array": ndarray (time × neurons), "dt": seconds}}`.
- `simulation_time`: seconds to simulate in this call.
- `params`: flat `{name: value}` of the project's `external_simulate_params`.
- `progress` (optional keyword): call `progress(fraction)` with a value in
  [0, 1] to drive the progress bar. It raises when the user cancels; let that
  propagate. Omit the keyword if you do not report progress.
- Return an `Experiment` with `TraceData`, `SpikeData` and/or `SynapseData`
  (`neuroseq.data`) for this chunk only. The GUI offers to import it as a new
  experiment or append it to an existing one.

`external_simulation_example/simulate.py` is a commented, file-based example.
`Project.load_NN()` (alias `load_external_simulator()`) loads either a file path
(via `runpy`, which must define `simulate`) or `neuroseq:<module>` for a module in
`src/neuroseq/example_models/`.

## Stateful backends with `NN`

For backends with parameters and state that persists between runs, subclass
`NN` from `neuroseq.example_models.NN`. It stores the editable parameter schema
and cumulative time; the subclass owns the runtime objects.

| Member                                                    | Role                                                   |
| --------------------------------------------------------- | ------------------------------------------------------ |
| `__init__(schema)`                                        | Copies the schema and initialises flat values and time |
| `default_params_schema()`                                 | Schema for the GUI parameter editor                    |
| `flat_default_params()`                                   | `{name: default}` from the schema                      |
| `update_params(params)`                                   | Merges overrides; calls `reset()` if anything changed  |
| `simulated_time_seconds`                                  | Time simulated so far (`self._simulated_time_s`)       |
| `reset()`                                                 | Subclass: clear runtime state                          |
| `simulate(stims_dict, simulation_time, params, progress)` | Subclass: run one chunk and return an `Experiment`     |
| `migrate_project(project)` (optional)                     | Upgrades legacy input formats after an archive loads   |

The defaults raise `NotImplementedError` (no `abc`). Keep schema keys and value
types simple: the parameter editor reads them. Stateful backends should continue
from the previous state and use `_simulated_time_s` to align time.

For `neuroseq:<module>`, NeuroSeq takes the simulator from `get_simulator()`,
else `NN()`, else the module-level `simulate`. The reference implementation is
`src/neuroseq/example_models/elise.py`.

```python
from neuroseq.data import Experiment
from neuroseq.example_models.NN import NN

DEFAULT_PARAMS = {
    "gain": {"value": 1.0, "default": 1.0, "type": "float",
             "group": "General", "description": "Input gain."},
}


class MyBackend(NN):
    def __init__(self) -> None:
        super().__init__(DEFAULT_PARAMS)
        self._runtime = None

    def reset(self) -> None:
        self._runtime = None
        self._simulated_time_s = 0.0

    def simulate(self, stims_dict, simulation_time, params=None, progress=None) -> Experiment:
        self.update_params(params)
        # build or reuse self._runtime, run, update self._simulated_time_s
        raise NotImplementedError


_SIMULATOR = MyBackend()


def get_simulator() -> MyBackend:
    return _SIMULATOR


def simulate(stims_dict, simulation_time, params=None, progress=None) -> Experiment:
    return _SIMULATOR.simulate(stims_dict, simulation_time, params=params, progress=progress)
```

To ship a backend with an example project, add a factory under
`src/neuroseq/example_projects/networks/` with `@register_example_project` and
import it in `example_projects/__init__.py` (see `elise_project.py`). Bundling a
backend this way is a core change; discuss it first (see [Contributing](contributing.md)).
