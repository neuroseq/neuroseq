# Manual and static input

An input array holds one value per time step and neuron. It drives a
simulation and documents what was delivered to a real sample. Drawing basics
(draw/select mode, amplitude, profiles) are in the
[GUI Overview](../getting_started/gui_overview.md); this page covers the other
editor tools and the static input designer.

## Input arrays

Array actions are in the **Input array operations** button next to the
**Input data** selector (Data toolbar) and in the editor's right-click menu.

| Action                  | Effect                                                                                                                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Create input array      | Name, type (Excitatory, Inhibitory, External), number of neurons, units, timestep. Presets: SysCon2 (520.8125 µs), 1 ms, 0.1 ms, 50 µs, 30 Hz imaging.                                                                                                          |
| Edit input array        | Same fields plus duration. Changing neurons or timestep resizes or resamples the data.                                                                                                                                                                          |
| Delete / Clear          | Remove the array, or set all its values to zero (keeps its shape).                                                                                                                                                                                              |
| Pad input array         | Add time to the end of all input arrays: either **Seconds to add** (prefilled 5 s, then the last value) or **Pad to end time**, the time point (s) the arrays should end at; the difference is added. An end time before the current end is refused (use Trim). |
| Trim input array        | Remove time from the end of all input arrays.                                                                                                                                                                                                                   |
| Loop input array        | Repeat all input arrays end to end a chosen number of times. Clears the undo history.                                                                                                                                                                           |
| Trim to Mapped Neurons… | Keep only neurons with a mapped Cellpose cell, in every input array and experiment. Permanent.                                                                                                                                                                  |

**Settings → Input Automatic Padding** sets how many seconds of empty input are
added beyond the simulated time after each run (default 0).

## Select mode

Drag to select a neuron × time region, then:

| Action                          | Shortcut     | Effect                                                                                               |
| ------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| Select all                      | Ctrl+A       | Whole array.                                                                                         |
| Copy / Cut                      | Ctrl+C / X   | To the internal clipboard.                                                                           |
| Paste                           | Ctrl+V       | Click where the region should go.                                                                    |
| Paste inverted                  | Ctrl+Shift+V | Pastes `max − value`, so pulses become gaps and gaps become pulses.                                  |
| Save to file / Load from file   |              | The selection as a NumPy `.npy`. A loaded array is pasted like the clipboard and must fit the array. |
| Set amplitude to selected input |              | Sets all non-zero samples of the selection to an amplitude you enter.                                |
| Delete selected input region    | Delete       | Zeroes the selection.                                                                                |

Undo: Ctrl+Z, Redo: Ctrl+Shift+Z. **View → Toggle Grid** shows the time and
neuron grid.

## Scripting in the console

**View → Show Console** opens a Python console with `project`, `np`, `b2` and
`plt` defined. For example:

```python
project.stim_arrays["stimulus"].values[::10, ::2] = 1
```

## Design static input

**Design static input** (View toolbar or the editor's right-click menu) writes
a generated pattern into the selected input array.

| Pattern         | Description                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------ |
| Sequential      | One neuron after another, each firing **Pulses per neuron** pulses.                              |
| Simultaneous    | All neurons in scope fire together with the same magnitude.                                      |
| Constant        | Shown instead of Sequential and Simultaneous when the scope is a single neuron (see below).      |
| Linear gradient | Like Simultaneous, but magnitude (and optionally pulse duration) steps from first to last pulse. |
| Ramp            | A continuous linear ramp of the magnitude, repeated.                                             |
| Für Elise       | Opening notes of Für Elise on neurons 0-5 (**All neurons** scope only).                          |

- **Scope**: **All neurons**, or **Specific neurons** given as indices and
  ranges (`0, 2, 5-8`). Other neurons are left untouched. When the scope is a
  single neuron (one index, or **All neurons** of a one-neuron network),
  Sequential and Simultaneous would write the same pulses, so the two are
  replaced by one **Constant** pattern. The list updates while you type the
  indices; widening the scope again brings Sequential back if it was chosen.
- **Pulse duration** and **Inter-pulse interval** (ms), or **Ramp duration**
  and **Interval between ramps** for Ramp.
- **Magnitude**, **First/Last pulse magnitude** (Linear gradient) or
  **Start/End magnitude** (Ramp).
- **Pulse duration: Step from first to last pulse** (Linear gradient): the
  pulse duration also steps linearly, from **First pulse duration** to **Last
  pulse duration** (ms), each rounded to whole samples. The inter-pulse
  interval stays the same.
- Repeat count (named per pattern) and **Start time (ms)**.

**Apply** writes the pattern and moves **Start time** to its end, so repeated
clicks append patterns one after another. **Reset start time** returns to 0.
The window stays open until **Close**.

The designer keeps its values per input array: a newly created (or newly
selected) array starts from the default values, and an array you return to
shows the values last entered for it, including its start time.

For closed-loop input, see [Dynamic input](dynamic_input.md).
