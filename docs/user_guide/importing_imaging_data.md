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
- **Select file for segmentation…** — segments a mean-fluorescence TIFF or
  PNG with the chosen Cellpose model and loads the result. The files are saved
  beside the image with the model in their name, e.g. `fov_cpsam_seg.npy` and
  `fov_cpsam_cp_masks.png`, so segmenting with another model keeps the
  earlier results, and an `fov_seg.npy` made in the Cellpose GUI is not
  overwritten. With **Adjust brightness first** ticked, the image is shown
  with the Brightness/Contrast controls of the Spatial Footprints panel before
  Cellpose runs, and Cellpose segments the adjusted image; **Reset** keeps the
  original. The window used is stored in the `_seg.npy`.
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
  (square/circle AOIs). Cells that fall inside an AOI inherit its role. The
  **Linear** strategy shows the same shapes, which can be moved and resized
  there too, but they do not change its mapping; they are kept with the
  imported segmentation.

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

### Showing one channel of a movie

Movies recorded with several channels store them interleaved frame by frame,
so playing every frame makes the Imaging tab flicker between channels. Set
**Channels** in the Imaging tab's bottom bar to the number of channels and pick
the one to watch under **Show**; **All frames** plays every frame in file
order. The frame rate entered on import is the rate of each channel, so one
channel plays in step with the time slider. Imports through the TIFF pipeline
start with its channel count and functional channel. The choice is saved per
imaging data entry. ScanImage recordings use the channel selector in the
**Image acquisition** dialog instead.

### Saving imaging data

**Save imaging data…** in the Imaging tab first asks which frames to save.
Frames are counted from 1, as in the time slider tooltip and in Fiji; **Current
frame** fills in the frame shown at the slider position, so a short period found
by scrubbing can be saved directly. Choose what to save:

- **Original frames (TIFF substack)** — the recorded pixel values of the shown
  channel, as one stack that Fiji opens with its frame interval.
- **With stimulation marks (TIFF or GIF)** — the frames as displayed, with the
  stimulated cells marked by a red "x" sized to the image. All saved frames
  share one grey scale, taken from robust percentiles of the saved frames, so
  the background does not flicker from frame to frame.

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

## GPU acceleration (Cellpose & Suite2p)

Both Cellpose segmentation and the Suite2p extraction/deconvolution step run on
**PyTorch** and are dramatically faster on a GPU. NeuroSeq uses the GPU
automatically when one is available — **CUDA** on NVIDIA cards or **MPS** on
Apple Silicon — and otherwise falls back to the CPU (which can be unusably slow
for segmentation).

Two things are required: a GPU that these tools support, and a GPU build of
PyTorch. **Cellpose can use only NVIDIA (CUDA) and Apple (MPS) GPUs — Intel
integrated graphics (Iris Xe / UHD) are not supported.**

**Check what you have** (in the NeuroSeq environment):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A version ending in `+cpu` (e.g. `2.13.0+cpu`), or `False`, means a CPU-only
build is installed — the Cellpose "use GPU" option will be greyed out.

**Enable the GPU** (on a machine with an NVIDIA or Apple-Silicon GPU) by
installing a GPU build of PyTorch into the NeuroSeq environment:

```bash
uv pip install --torch-backend=auto torch
```

`uv` auto-detects the machine's CUDA driver and installs the matching wheel.
On **Windows this step is required**, because PyPI's default `torch` wheel is
CPU-only there (on Linux the default PyPI wheel is already CUDA-enabled). A
later `uv sync` can revert to the CPU build; re-run the command, or set
`UV_TORCH_BACKEND=auto` in the environment. Verify with the check command above
(expect e.g. `2.x.x+cu128 True`), then relaunch NeuroSeq.

If NeuroSeq is installed into an environment `uv` doesn't manage (e.g. a conda
environment), install a CUDA build directly with pip instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Pick the `cuXXX` index matching your NVIDIA driver if `cu128` doesn't fit
(check the [PyTorch install page](https://pytorch.org/get-started/locally/)).

**No supported GPU** (e.g. an Intel-only laptop)? GPU segmentation is not
possible on that machine. Instead, run Cellpose on one machine that _does_ have
an NVIDIA/Apple GPU and import its `_seg.npy` output (the file-selection path
above needs only the segmentation, not a GPU), and/or use a lighter Cellpose
model or downsample the image before segmenting.

When a GPU is unavailable, NeuroSeq shows a one-time hint after launching
Cellpose and logs the same information before running the Suite2p pipeline.

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
