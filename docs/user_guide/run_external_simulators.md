# Run External Simulators

NeuroSeq can run simulation backends that are outside the built-in Brian2 model. This is useful when a project should include models implemented with another frameworks or run external devices.

## What an External Simulator Does

An external simulator receives the current stimulus arrays from the project, runs its own backend, and returns an `Experiment`. NeuroSeq then lets you import that result as a new experiment or append it to an existing one.

## Connect an External Simulator

To load a custom simulator from a Python file:

1. Open the `Experiment` menu.
2. Select `Connect external stimulation function`.
3. Choose a Python file that exposes a compatible `simulate(...)` entry point.

NeuroSeq will load that file and store the connection in the current project.

## Run an External Simulator

Once a simulator is connected:

1. Prepare or edit the stimulus arrays in the Stimuli view.
2. Open the `Experiment` menu.
3. Select `Run external stimulation function`.
4. Enter the simulation time.
5. Choose whether to import the returned experiment as a new experiment or append it to an existing one.

If a project has an external simulator configured, the main `Run simulation` action can also dispatch to that simulator instead of Brian2.

## Built-In Example: ELISE

The bundled `ELISE Sequence Learning Network` example project already comes connected to the ELISE backend. In that project, the external simulator is loaded automatically and uses the class-based backend implementation in [src/neuroseq/example_models/elise.py](../../src/neuroseq/example_models/elise.py).

## What Data Is Passed to the Simulator

NeuroSeq passes stimulus arrays as a dictionary of the form:

```python
{
    "stim_name": {
        "array": ndarray,
        "dt": seconds,
    }
}
```

The backend returns an `Experiment` containing trace, spike, synapse, or other supported data objects.

## Errors and Troubleshooting

If loading fails, NeuroSeq shows an error dialog and clears the external simulator connection.

If execution fails, NeuroSeq shows an `External Simulation error` dialog. Common causes are:

1. The Python file does not expose the expected entry point.
2. The simulator expects a stimulus name that is not present in the project.
3. The simulator raises an exception while building or advancing its backend state.

If you are implementing the simulator yourself, see [developer/custom_simulators.md](../developer/custom_simulators.md).
