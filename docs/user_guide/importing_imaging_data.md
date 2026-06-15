# Importing imaging data

NeuroSeq accepts imaging data from two distinct sources:

- **Cellpose** — spatial cell segmentation (which pixels belong to which
  neuron). This is what NeuroSeq needs to map drawn stimuli back onto the
  physical sample for holographic export.
- **Suite2p** — fluorescence traces extracted from a calcium imaging
  recording. These appear as `TraceData` in the project alongside any
  simulated traces.

Patch-clamp electrophysiology imports are also covered here for completeness.

## Cellpose segmentation

Open one of:

- **File → Import data → Import cell segmentation** — full importer dialog.
- **File → Import cell spatial footprints** — shortcut that opens the same
  dialog directly from the top-level File menu.

### Dialog sections

- **File Selection** — point at the Cellpose output (a NumPy `.npy` array of
  per-pixel segmentation IDs, or the `_seg.npy` file produced by the Cellpose
  GUI).
- **Data Information** — read-only summary: image dimensions, number of
  detected cells, value range.
- **ID Mapping Options** — choose how Cellpose IDs are mapped onto the
  project's neuron indices:
  - **By ID** — use the segmentation IDs as-is.
  - **By order** — renumber sequentially in the order Cellpose found cells.
  - **Custom shape mapping** — assign each cell to a target neuron index via
    a `SquareAOI` or `CircleAOI` overlay role. Use this when the project's
    neuron layout (e.g. a grid of square AOIs) does not match Cellpose's
    discovery order.
- **Areas of Interest** — optional roles drawn on top of the segmentation
  (square/circle AOIs). Cells that fall inside an AOI inherit its role.

After import, the cell footprints are available in the cell overlay widget
and are bundled into the project file. They are required for the
[SysCon2 holographic export](multiphoton_holographic_stimulation.md).

### Cell overlay widget

The cell overlay widget visualizes the imported segmentation on top of the
sample image. You can:

- Pan and zoom with the mouse wheel and middle-button drag, matching the
  other widgets.
- See cells that have been mapped to project neurons highlighted relative
  to unmapped ones.
- Drag the corner/edge handles of square and circle AOI overlays to refine
  their position after import — the cell-to-neuron assignment recomputes
  whenever an AOI moves.

### Generating test data

If you do not have a real recording yet, NeuroSeq can synthesize a Cellpose
test image with a clean grid of cells:

```bash
neuroseq --generate-cellpose-test-data ./cellpose_test_grid.tif \
  --cellpose-grid-rows 10 \
  --cellpose-grid-cols 10 \
  --cellpose-radius 11 \
  --cellpose-gap 30
```

See [Multiphoton Holographic Stimulation → Generating test data](multiphoton_holographic_stimulation.md#generating-test-data-cli)
for the full set of options.

## Suite2p calcium imaging

**File → Import data → Import trace → Import calcium imaging** opens the
Suite2p importer.

- **File Selection** — point at a Suite2p output folder (the directory that
  contains `F.npy`, `Fneu.npy`, `iscell.npy`, …). Individual file paths can
  also be supplied directly via the **Individual File Selection** group when
  the folder layout is non-standard.
- **Data Preview** — shows a few representative traces so you can confirm
  the right cells are being imported before committing.
- **Import Options** — neuropil subtraction, dF/F normalization, decimation,
  and which Suite2p classification mask (`iscell`) to apply.

Imported traces become a new `TraceData` entry under the project's experiments
and appear in the **Neuron data** dropdown.

## Patch-clamp electrophysiology

**File → Import data → Import trace → Import patch clamp** loads voltage
traces recorded with Igor Pro / SutterPatch:

- **Select `.npy` patchclamp file** — the converted patch-clamp recording.
- **Data Preview** — shows the first few sweeps so you can confirm the data.
- **Import Options** — which sweeps to pull in, sample-rate handling, and
  channel selection.
- **Peak Detection Parameters** — optional thresholding to derive a
  `SpikeData` alongside the trace.

For raw Igor Pro waves see
[Importing Igor2 Waves](importing_igor2_waves.md). For stimulus-only patch
clamp recordings use **Import data → Import stimulus → Import patch clamp**
on the same menu.
