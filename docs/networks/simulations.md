# Simulations

NeuroSeq runs Brian2 spiking network simulations. The model, parameters, input
arrays and recorded monitors are saved together in the `.nseq` project.

| Action                                                        | What it does                                                                                                                                       |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Simulate → Run Simulation** (`Ctrl+R`)                      | Asks for a duration and continues from the last simulated time. Runs the [external simulator](external_simulators.md) instead if one is connected. |
| **Simulate → Store Simulation Checkpoint** (`Ctrl+Shift+S`)   | Saves the network state (variables, weights, clock) under a name; pick an existing name to overwrite it.                                           |
| **Simulate → Restore Simulation Checkpoint** (`Ctrl+Shift+R`) | Rolls the network back to a chosen checkpoint and truncates the monitors. Checkpoints are saved with the project.                                  |
| **File → Edit Model** (`Ctrl+E`)                              | Parameter tree for equations, parameters, synapses and monitors. Applies on the next run.                                                          |

Design the input first with the [static input editor](../input_design/static_input.md)
or a [dynamic input routine](../input_design/dynamic_input.md).

## Choosing a code generation backend

**Settings → Brian2 Simulation Backend**:

- `cython`: compiled, fastest; needs a C toolchain (see
  [installation](../getting_started/installation.md) for Windows).
- `numpy`: pure Python, always works, slower.

The backend cannot be changed while a simulation runs. For large synapse
matrices see **Settings → Synapse Matrix RAM Limit**
([GUI overview](../getting_started/gui_overview.md#synaptic-data-widget)).
