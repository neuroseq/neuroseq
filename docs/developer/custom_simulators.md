# Integration of Custom Simulators

This page describes the current structure used for NeuroSeq external simulators and how to implement a new backend.

## Overview

NeuroSeq has built-in Brian2 simulator but also supports arbitraty simulation backends. These backends are loaded through `Project.load_external_simulator()` and are expected to produce an `Experiment` object from stimulus arrays.

There are now two layers:

1. A shared base class in [src/neuroseq/example_models/simulator.py](../../src/neuroseq/example_models/simulator.py).
2. A backend implementation class such as [src/neuroseq/example_models/elise.py](../../src/neuroseq/example_models/elise.py).

The base class is intentionally lightweight. It does not use `abc`; instead, subclass methods are expected to override the base methods and the defaults raise `NotImplementedError`.

## Structure of the Base NN Class

The shared `NN` class in [src/neuroseq/example_models/simulator.py](../../src/neuroseq/example_models/simulator.py) provides common state and parameter handling.

It currently contains:

1. `__init__(default_params_schema)`
   Stores a deep copy of the editable parameter schema, initializes flat parameter values from the schema defaults, and initializes cumulative simulated time.
2. `simulated_time_seconds`
   Property exposing the amount of simulated time already accumulated by the backend instance.
3. `default_params_schema()`
   Returns a deep copy of the editable parameter schema for use in the GUI.
4. `flat_default_params()`
   Returns a flat `{name: default_value}` mapping derived from the schema.
5. `update_params(params)`
   Merges user-provided parameter overrides onto defaults and resets backend state if anything changed.
6. `_flat_defaults()`
   Internal helper that flattens the schema into default values.
7. `reset()`
   Subclass hook that must clear backend-specific runtime state.
8. `simulate(stims_dict, simulation_time, params=None)`
   Subclass hook that must run one simulation chunk and return an `Experiment`.

The important design point is that the base class owns parameter bookkeeping and cumulative time, while the subclass owns backend-specific runtime objects.

## Structure of a Backend Implementation

An implementation module should define a backend class derived from `NN`. The ELISE backend in [src/neuroseq/example_models/elise.py](../../src/neuroseq/example_models/elise.py) is the current reference implementation.

The backend class should typically provide:

1. `__init__()`
   Calls `super().__init__(DEFAULT_PARAMS)` and initializes backend-specific runtime fields such as cached networks, signatures, RNG state, or imported backend types.
2. `reset()`
   Clears runtime state that should be rebuilt when parameters change.
3. Backend-specific initialization helpers
   For example, ELISE uses `_import_elise_types()` and `_initialize_network(num_vis)`.
4. `simulate(...)`
   Validates inputs, applies parameter updates, initializes or reuses the backend state, advances the simulation, and returns an `Experiment` containing the generated data.

For stateful backends, `simulate(...)` should preserve runtime state between calls so repeated runs can continue from the previous state instead of starting over.

## Module-Level Entry Points

For integration with the current project and GUI code, the module should expose these entry points:

1. `simulate(stims_dict, simulation_time, params=None)`
   Compatibility wrapper used by the existing call sites.
2. `get_simulator()`
   Returns the singleton backend instance.
3. `NN`
   Constructor used by `Project.load_external_simulator()` when a module exposes a class-based backend entry point.
4. A singleton instance such as `_SIMULATOR = Elise()`.

This keeps the current GUI and project logic simple while still allowing a class-based implementation internally.

When restoring an archive, `Project` calls the backend instance's optional
`migrate_project(project)` method after loading the simulator and project data.
Backends can use this hook to upgrade their own legacy stimulus formats. The
ELISE implementation keeps these conversions, its example project factory
(`create_project()`), and teaching-pattern helpers in
`src/neuroseq/example_models/elise.py`. The example menu registers a small
wrapper in `example_projects/networks/elise_project.py` that delegates to this factory.

Example project factories live in `src/neuroseq/example_projects/`, with one
project per file under `neurons/` or `networks/`. The package's `__init__.py`
exports the factories and imports them in their existing menu order.
`_registry.py` provides the `register_example_project` decorator. New examples
should define a decorated factory in the appropriate folder, import it in
the package entry point, and add it to `__all__` so it is exported and appears
in the example-project menu.

## Expected Data Flow

The external simulator call path is:

1. The GUI collects stimulus arrays into `stims_dict`.
2. `Project.load_external_simulator()` loads either:
   a. a built-in NeuroSeq simulator module such as `neuroseq:elise`, or
   b. a Python file containing a `simulate(...)` entry point.
3. The backend `simulate(...)` method returns an `Experiment`.
4. The GUI offers the returned experiment for import or append into the project.

## Minimal Template

```python
from collections.abc import Mapping

from neuroseq.data import Experiment
from neuroseq.example_models.simulator import ParamValue
from neuroseq.example_models.simulator import NN

DEFAULT_PARAMS = {
    "example_param": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "General",
        "description": "Example parameter.",
    }
}


class MyBackend(NN):
    def __init__(self) -> None:
        super().__init__(DEFAULT_PARAMS)
        self._runtime = None

    def reset(self) -> None:
        self._runtime = None
        self._simulated_time_s = 0.0

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, ParamValue] | None = None,
    ) -> Experiment:
        self.update_params(params)
        # build or reuse runtime state
        # run the backend
        # return Experiment(...)
        raise NotImplementedError


def simulate(stims_dict, simulation_time, params=None):
    return _SIMULATOR.simulate(stims_dict, simulation_time, params=params)


def get_simulator() -> MyBackend:
    return _SIMULATOR


_SIMULATOR = MyBackend()
```

## Notes

1. The GUI parameter editor consumes the schema stored in `project.external_simulate_params`, so schema keys and value types should remain simple.
2. The `Experiment` returned by the backend should be self-contained for the run chunk that was just simulated.
3. If the backend is stateful, use `self._simulated_time_s` to align repeated runs with previously simulated time.
