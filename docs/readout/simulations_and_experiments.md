# Simulations vs. experiments

A project holds input arrays, one Brian2 model, and data groups (experiments)
from simulation runs and recordings, plus an optional Cellpose segmentation.
The same input array drives a simulation and documents what was delivered to a
sample, so simulated and recorded responses can be compared directly.

**Simulate → Run Simulation** and **Experiment → Run External Input Function**
each add a new data group.

## Import from another project

**File → Import from Project** loads parts of another `.nseq` file. NeuroSeq
offers to save the current project first, then asks what to take:

| Option            | Effect                                                  |
| ----------------- | ------------------------------------------------------- |
| Input Arrays      | Replaces the current input arrays.                      |
| Model             | Replaces the model.                                     |
| Simulation data   | The stored simulation state (only together with Model). |
| Experimental Data | Replaces the current data groups.                       |

Chosen parts replace, not merge with, the current ones, and the undo history
is cleared. The Cellpose segmentation is not imported.

## Compare simulated and recorded data

1. Run a simulation.
2. Import recordings: [imaging data](imaging_data.md) or
   [Igor2 waves](igor2_waves.md). Map recorded cells to the model's neuron IDs
   with the Cellpose [ID mapping](imaging_data.md#neuron-id-mapping).
3. Show one dataset in **Neuron data** (Traces panel) and the other in **All
   data** (Data panel, **View → Show Data**). Both share the time axis of the
   input arrays.
4. Select a region and run the same [analysis](analysis.md) on each.

Use **File → Manage Data** to inspect or delete data groups (see
[Experiments](experiments.md)).
