# Simulations

NeuroSeq runs Brian2 spiking network simulations. The model, parameters, stim
arrays, and recorded monitors all live inside the project and are persisted
together in the `.nseq` archive.

## Running a simulation

- **Simulate → Run simulation** (`Ctrl+R`) advances the network by the chosen
  duration. The run picks up from the last simulated time, so consecutive
  invocations extend the same trial.
- If the project has an external simulator connected (see
  [Run External Simulators](run_external_simulators.md)), the action dispatches
  to that backend instead of the built-in Brian2 model.

### Checkpointing

Long simulations can be checkpointed so you can experiment from a known state
without rerunning from scratch.

- **Simulate → Store simulation checkpoint** (`Ctrl+Shift+S`) saves the current
  Brian2 network state (neuron variables, synapse weights, clock).
- **Simulate → Restore simulation checkpoint** (`Ctrl+Shift+R`) rolls the
  network back to the most recently stored checkpoint. Trace, spike, and
  synapse monitors are truncated accordingly.

Checkpoints are part of the project file, so they survive save/load.

### Choosing a Brian2 backend

**Settings → Brian2 simulation backend** opens a small dialog to pick the
Brian2 code-generation target:

- `cython` — compiles C extensions on the fly. Fastest option for most
  networks; requires a working C toolchain (see the
  [installation page](../getting_started/installation.md) for the MSVC note
  on Windows).
- `numpy` — pure-Python fallback. Always works, slower. Tests use this.

## Designing stimulation

Prepare the input before running the model with the
[stimulation editor and static input guide](../Input%20Design%20and%20Readout%20Analysis/static_input_design.md).
For response-guided stimulation, see
[Design dynamic input](../Input%20Design%20and%20Readout%20Analysis/stdp_synaptic_tuning.md).

## Settings that affect simulations

- **Settings → Brian2 simulation backend** — described above.
- **Settings → Synapse matrix RAM limit** — see
  [GUI Overview → Synaptic data widget](../getting_started/gui_overview.md#synaptic-data-widget).

## Editing the model

**File → Edit model** (`Ctrl+E`) opens the parameter tree editor for the
underlying Brian2 model: neuron equations, parameter values, synapse rules,
monitors. Changes take effect on the next **Run simulation** and are saved
with the project.
