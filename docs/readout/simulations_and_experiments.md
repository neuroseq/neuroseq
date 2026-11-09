# Connecting simulations and experiments

A NeuroSeq project holds both simulated and recorded data side by side.
Stim arrays drive simulations _and_ document what was applied to a real
sample, so the same draw-and-edit workflow works for both. This page
covers the actions that move data between projects and between
simulations and experiments.

## Project as a container

A project owns:

- one or more **stim arrays** (`Project.stim_arrays`),
- one Brian2 **model** (`Project.b2_model`),
- a set of **experiments** (`Project.experiments`), each grouping
  `TraceData`, `SpikeData`, and `SynapseData` from a single run or
  recording,
- optional **Cellpose segmentation** for hardware mapping.

Both the **Run simulation** action and the **Run external stimulation
function** action produce a new experiment that is appended to this
container.

## Importing from another project

**File → Import from project** merges data from an existing `.nseq` file
into the current project. This is the right action when you want to:

- pull in stim arrays designed in another project,
- attach experimental traces or spikes recorded under a different model,
- reuse a Cellpose segmentation across projects sharing the same
  microscope sample.

After picking the file, NeuroSeq shows the items it can pull in (stim
arrays, experiments, segmentation) so you can opt in selectively. The
current project's own data is left untouched.

## Manage Data

**File → Manage Data** opens the data manager dialog, which lists project
data grouped by simulation run or recording. Each group corresponds to an
`Experiment` in `Project.experiments`. From here you can:

- inspect each group's name, duration, and number of datasets,
- inspect which data objects (`TraceData`, `SpikeData`, `SynapseData`)
  belong to each group and their dimensions,
- delete selected groups and all the datasets they contain.

## Comparing simulated and recorded data

The widgets (trace, spike, synapse, custom analysis) all draw from the
project as a whole, so simulated and imported data can be displayed
together:

1. Run a simulation, or load a previously simulated experiment.
2. Import experimental traces via
   [Importing imaging data](imaging_data.md) or
   [Importing Igor2 waves](igor2_waves.md).
3. In the **Neuron data** dropdown, switch between the simulated and the
   imported `TraceData` — they share the same time axis whenever both
   were aligned to the same stim array.
4. Use the **All data** ("Data") widget to display two `TraceData`
   objects in the same view for direct comparison.

The [Analysis Tools and Custom Plugins](analysis.md) page
covers the right-click analyses that operate on a selected region of any
widget — these work identically on simulated and recorded data, which is
the point of keeping both in one container.
