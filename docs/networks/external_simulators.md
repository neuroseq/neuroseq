# Run External Simulators

An external simulator is a Python file exposing `simulate(...)`. It receives the
project's input arrays, runs its own network and returns an `Experiment`.

## Connect and run

1. **Experiment → Connect External Input Function** and choose the Python file.
   Use **Connect Brian2 External Input Function** instead if your backend builds
   its own Brian2 network: NeuroSeq then passes it the project's Brian2 units
   and [code-generation target](simulations.md#choosing-a-code-generation-backend).
2. Edit the input arrays.
3. **Experiment → Run External Input Function** (or **Simulate → Run
   Simulation**, which dispatches to the connected simulator), enter the
   duration.
4. Add the result as a new data group or append it to an existing one.

The connection is saved with the project. The bundled **ELISE Sequence Learning
Network** example project comes connected to
[`example_models/elise.py`](https://github.com/neuroseq/neuroseq/blob/main/src/neuroseq/example_models/elise.py).

## Data passed to the simulator

```python
{"stim_name": {"array": ndarray, "dt": seconds}}
```

The returned `Experiment` may hold trace, spike, synapse or other supported data.

## Errors

A file that fails to load clears the connection. A failing run shows
**External Simulation error**; typical causes are a missing entry point, a
stimulus name the simulator expects but the project lacks, or an exception in
the backend. To write a simulator, see
[Custom simulators](../developer/custom_simulators.md).
