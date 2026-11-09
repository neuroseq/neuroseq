# Stimulation editor and static input

Use stimulation arrays to define the input delivered to a simulation or
external network. This guide covers manual editing and predefined patterns.

## Stimulation editor

The stim editor is the left-hand pane and operates in either _draw_ or _select_
mode, switched from the top of its toolbar. The general workflow is covered in
[GUI Overview → Stimulation editor](../getting_started/gui_overview.md#stimulation-editor); this
section documents the editor's auxiliary tools.

### Managing stim arrays

The toolbar exposes three array-level actions:

- **Create stim array** — opens a properties dialog: name, type
  (`light`, `current`, `voltage`, `opsin`, …), neuron count, physical units,
  and `dt`. The default `dt` is taken from
  **Settings → Stimulus automatic padding** (see [below](#stimulus-settings)).
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

The same patterns, with the same names, are offered for **All neurons** and for
**Specific neurons**; Für Elise, which plays fixed neurons, is for all neurons
only.

| Pattern         | Description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| Sequential      | One neuron after another, each firing **Pulses per neuron** pulses before the next   |
| Simultaneous    | All neurons of the scope fire together, with the same magnitude every pulse          |
| Linear gradient | Like Simultaneous, but the magnitude steps linearly from the first to the last pulse |
| Ramp            | A continuous linear ramp of the magnitude, repeated                                  |
| Für Elise       | The opening notes of Für Elise on neurons 0-5, with **Pulses per note**              |

Common controls:

- **Scope** — apply to all neurons or to specific neuron indices.
- **Pulse duration / Inter-pulse interval** (or **Ramp duration / Interval
  between ramps**) — in milliseconds.
- **Magnitude**, or **First / Last pulse magnitude** for Linear gradient and
  **Start / End magnitude** for Ramp.
- **Sequence repetitions**, **Number of pulses**, **Number of ramps** or
  **Melody repetitions** — how often the pattern repeats, named for the pattern.
- **Start time** — where in the array the pattern begins.

The dialog stays open so you can iterate: change parameters, click **Apply**
to overwrite, or **Reset start time** to rewind the cursor.

## Closed-loop input design

**Design dynamic input** is a separate NeuroSeq feature for routines that
extend a stim array, run a simulation, analyze its traces, and decide what to
append next. Its first routine uses five-pulse EPSP measurements and STDP
pairings to move a postsynaptic response toward a percentage target. See the
dedicated [Design dynamic input guide](dynamic_input.md) for setup, controls, the
decision algorithm, live plots, and current limitations.

## Looping a stim array

**Loop stim array** (toolbar) repeats the entire current array end-to-end a
chosen number of times. The dialog lets you pick the number of repetitions;
the loop runs in place — the array is extended to accommodate the repeats.

## Stimulus settings

- **Settings → Stimulus automatic padding** — default duration (in seconds)
  used when a new stim array is created and when _Pad stim array_ prefills
  its dialog.
