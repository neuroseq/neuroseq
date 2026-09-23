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

## Stimulation editor

The stim editor is the left-hand pane and operates in either _draw_ or _select_
mode, switched from the top of its toolbar. The general workflow is covered in
[GUI Overview → Stimulation editor](gui_overview.md#stimulation-editor); this
section documents the editor's auxiliary tools.

### Managing stim arrays

The toolbar exposes three array-level actions:

- **Create stim array** — opens a properties dialog: name, type
  (`light`, `current`, `voltage`, `opsin`, …), neuron count, physical units,
  and `dt`. The default `dt` is taken from
  **Settings → Stimulus automatic padding** (see below).
- **Edit stim array** — change any of the above on the currently selected
  array. Editing the neuron count or `dt` resizes/resamples the existing data.
- **Delete stim array** — drops the selected array from the project.

### Reshaping the time axis

- **Pad stim array** — extend the array's duration. You choose how much time
  to add and whether to pad at the start, end, or both. New samples are
  filled with zeros.
- **Trim stim array** — shrink the array's duration. Useful after a simulation
  to discard the unused tail.
- **Clear stim array** — zero every sample of the selected array without
  changing its shape.

### Selection-mode operations

When the editor is in select mode, mouse drags pick a rectangular region of
the stim array. The toolbar then exposes:

- **Select all** — extend the selection to the whole array.
- **Copy** / **Cut** — copy or move the selected region to the internal
  clipboard.
- **Paste** / **Paste inverted** — paste the clipboard at the current cursor
  position. _Inverted_ swaps zero and non-zero samples on the way in (useful
  for stamping refractory windows between pulse trains).
- **Save selection to file** / **Load selection from file** — read or write
  the selected region as a NumPy `.npy` blob. Loaded selections are inserted
  at the current cursor and respect the array's neuron axis.

Both modes share **Undo** / **Redo** at the top of the toolbar.

### Toggling the grid

**View → Toggle Grid** turns the time and neuron grid overlays on or off in
the stim editor. The grid spacing is fixed to the array's `dt` so it doubles
as a sample ruler.

## Pre-built stimulation patterns

For common stimulation motifs you do not need to draw manually. The toolbar
action **Design static input** opens a generator dialog that writes
patterns directly into the selected stim array:

| Pattern         | Description                                                                       |
| --------------- | --------------------------------------------------------------------------------- |
| Sequential      | One pulse per neuron, fired one after another with a configurable inter-pulse gap |
| Simultaneous    | All neurons (or one neuron) fire together, repeated N times                       |
| Für Elise       | A test motif used by the ELiSe sequence-learning examples                         |
| Constant        | Hold a single value across the requested window                                   |
| Linear gradient | Linearly ramp from an initial to a final magnitude across the pulse               |
| Ramp stimulus   | Repeated ramps with controllable rise/fall and inter-pulse interval               |

Common controls:

- **Scope** — apply to all neurons or to a specific neuron index.
- **Pulse duration / inter-pulse interval** — in milliseconds.
- **Magnitude** (or **Initial / Final magnitude** for the gradient patterns).
- **Repeats** — how many times the pattern is tiled along the time axis.
- **Start time** — where in the array the pattern begins.

The dialog stays open so you can iterate: change parameters, click **Apply**
to overwrite, or **Reset start time** to rewind the cursor.

## Closed-loop input design

**Design dynamic input** is a separate NeuroSeq feature for routines that
extend a stim array, run a simulation, analyze its traces, and decide what to
append next. Its first routine uses five-pulse EPSP measurements and STDP
pairings to move a postsynaptic response toward a percentage target. See the
dedicated [Design dynamic input guide](stdp_synaptic_tuning.md) for setup, controls, the
decision algorithm, live plots, and current limitations.

## Looping a stim array

**Loop stim array** (toolbar) repeats the entire current array end-to-end a
chosen number of times. The dialog lets you pick the number of repetitions;
the loop runs in place — the array is extended to accommodate the repeats.

## Settings that affect simulations

- **Settings → Stimulus automatic padding** — default duration (in seconds)
  used when a new stim array is created and when _Pad stim array_ prefills
  its dialog.
- **Settings → Brian2 simulation backend** — described above.
- **Settings → Synapse matrix RAM limit** — see
  [GUI Overview → Synaptic data widget](gui_overview.md#synaptic-data-widget).

## Editing the model

**File → Edit model** (`Ctrl+E`) opens the parameter tree editor for the
underlying Brian2 model: neuron equations, parameter values, synapse rules,
monitors. Changes take effect on the next **Run simulation** and are saved
with the project.
