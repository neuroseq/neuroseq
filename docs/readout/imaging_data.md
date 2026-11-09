# Imaging data

Calcium imaging enters a project in two parts:

- **Cell segmentation** (Cellpose): which pixels belong to which cell, and
  which cell is which project neuron. Required for
  [holographic stimulation](../networks/holographic_stimulation.md).
- **Traces**: fluorescence per cell, extracted from raw TIFFs or imported from
  Suite2p output, added as trace (and spike) data.

Patch-clamp recordings are covered in [Igor2 waves](igor2_waves.md).

## Cell segmentation

Open **File → Cell segmentation**, or drop a TIFF or PNG on the Cell
segmentation tab (this segments it right away).

**Segment an image**: choose a **Model** (`cpsam`, `cpsam_v2`, `cpdino`,
`cpdino-vitb`; `cpsam_v2` is more sensitive on low-SNR fields) and click
**Select file for segmentation…** with a mean-fluorescence TIFF or a PNG.
Results are saved next to the image with the model in the name
(`fov_cpsam_seg.npy`, `fov_cpsam_cp_masks.png`), so other models' results and
a Cellpose GUI `fov_seg.npy` are kept. **Adjust brightness first** lets you set
brightness/contrast before segmenting; Cellpose then segments the adjusted
image.

**Or load existing files**: an image (PNG/TIF) and a segmentation (`.npy`,
including Cellpose `_seg.npy`, or `.csv`). **Open Cellpose** and **Open Fiji**
launch the external tools.

### Neuron ID mapping

**Mapping strategy** decides which Cellpose cell is which input neuron:

| Strategy                      | Mapping                                                                                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear                        | Input neuron 0 is the cell with Cellpose ID equal to the offset, then upward; lower IDs wrap to the end.                                                                              |
| Custom                        | Type pairs `input:cellpose` (e.g. `1:12,2:3`), or tick **Assign by clicking cells** and click cells in order (unclicked cells get the remaining IDs in Cellpose order).               |
| Custom with areas of interest | A square and/or circle, each **preferential** or **exclusion**. Cells in preferential areas get the lowest IDs, cells in exclusion areas the highest, each tier in Cellpose-ID order. |

Area shapes can be set in pixels or percent, and dragged and resized in the
viewer; the mapping updates as they move. Under **Linear** the shapes are shown
and kept but do not change the mapping.

The segmentation is saved in the project. **Delete Cellpose Data** removes it.

## Raw calcium imaging TIFFs

**File → Import data → Output → Calcium Imaging (raw .tif)** extracts traces
from one or more 2-photon TIFF stacks in the background.

1. **Add TIFF…** the stack(s).
2. Pick **Cellpose masks**: those loaded in the project (with mapping), a
   `_seg.npy` file, or Suite2p auto-detection (set **Cell diameter**).
3. Set the parameters and click **Run pipeline**.

Raw stacks over ~1 GB crash Cellpose. Segment a projection instead: **Open
Fiji**, **Image → Stacks → Z Project…** (Max for sparse, Mean for dense
activity), save it, segment it with **Open Cellpose**, then load the
`_seg.npy`. The pipeline still runs on the full stack.

| Parameter            | Default | Notes                           |
| -------------------- | ------- | ------------------------------- |
| Pipeline             | Suite2p | Or **Simply average from ROI**. |
| Frame rate (fs)      | 30 Hz   | Per channel.                    |
| # channels           | 2       | 1 when ROI average is chosen.   |
| Functional channel   | 1       |                                 |
| Indicator tau        | 0.5 s   | jGCaMP8m.                       |
| Neuropil coefficient | 0.7     |                                 |
| Spike threshold      | 0.02    | Suite2p.                        |

**Simply average from ROI** streams the stack, so long recordings (15 min at
512×512, about 14 GB) need not fit in memory. Options:

- **Neuropil correction**: subtracts coefficient × an annulus of the 350
  nearest pixels more than 2 px from any cell.
- **Compute ΔF/F and deconvolve spikes**: ΔF/F against Suite2p's sliding
  maximin baseline (30 s) of the raw fluorescence; OASIS spikes counted above
  **Spike threshold (ΔF/F)** (default 0.1).
- **Input artefact suppression**: replaces frames lit by an input array's
  stimulation with the mean of the neighbouring clean frames. **Imaging
  started at** is the first frame's time on that array's axis (0 if both
  started together, positive if imaging started later).

## Suite2p output

**File → Import data → Output → Calcium Imaging (Suite2p outputs)**: choose
the output folder (with `F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`), or pick
files under **Individual File Selection** (`.npy` or `.csv`).

- Traces are `F − 0.7·Fneu` (when `Fneu` is present), normalised to 0-1 per
  neuron.
- Spikes come from `spks.npy`, kept above **Spike threshold** (default 0.02).
- **Max neurons to import** defaults to 2; **Sample rate** to 30 Hz.
- Rows are mapped to input neurons through the Cellpose mapping when one is
  loaded.

## Imaging tab

- **Channels / Show**: for interleaved multi-channel movies, set the channel
  count and pick the channel to play; **All frames** plays every frame. The
  TIFF pipeline presets both. ScanImage recordings use **Functional channel**
  in the ScanImage tab instead.
- **Moved files**: projects store only the movie path. If it is missing, click
  **Locate file…**; other parts of a split movie are found in the same folder.
  Save the project to keep the new path.
- **Save imaging data…**: choose frames (counted from 1; **Current frame** and
  **All frames** buttons), then **Original frames (TIFF substack)** or **With
  stimulation marks (TIFF or GIF)**, which marks stimulated cells with a red
  "x" on a fixed grey scale.

## GPU acceleration

Cellpose and Suite2p run on PyTorch and are far faster on an NVIDIA (CUDA) or
Apple Silicon (MPS) GPU. Intel integrated graphics are not supported. Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A `+cpu` version or `False` means a CPU-only build (the Cellpose GPU option is
greyed out). Install a GPU build:

```bash
uv pip install --torch-backend=auto torch
```

This is required on Windows, where the default wheel is CPU-only. A later
`uv sync` can revert it; set `UV_TORCH_BACKEND=auto` to avoid that. In a conda
environment use
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
(pick the `cuXXX` matching your driver). Without a supported GPU, segment on
another machine and import the `_seg.npy`.
