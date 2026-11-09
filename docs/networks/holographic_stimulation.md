# Multiphoton Holographic Stimulation (SysCon2)

NeuroSeq exports input arrays as SysCon2 `.seq` sequences for the HOLO4D
two-photon holographic stimulation rig and can import, upload and run them
through SysCon2. SysCon2 must run on the same Windows PC (local named pipe).

## Requirements

| Item              | Requirement                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| Input array `dt`  | `0.0005208125` s (520.8125 µs, one SysCon2 tick). Presets in **Create/Edit input array**. |
| Cell segmentation | Cellpose masks loaded (**File → Cell segmentation**), rows mapped to ROIs                 |
| Power calibration | A SysCon `.lut` per laser wavelength, if power control is used                            |

## Drawing input

Each active sample is a laser TTL pulse at that tick; NeuroSeq derives the SLM
holograms from the pulses.

- Pulses can be one tick (520.8 µs) long. With spiral scanning, use at least
  16.7 ms (one spiral).
- The SLM needs one blank 16.666 ms window (32 ticks) before the first pulse to
  load the hologram.
- A new hologram starts whenever the set of active neurons changes between
  consecutive non-empty windows. Empty gaps keep the current hologram loaded.
- Each hologram needs one blank load window before its first pulse. With
  **Exact stimulus timing** (default) its shape starts one window before that
  pulse and pulses are delayed only where the previous hologram leaves no room
  (the first stimulus is always delayed by one window). Unticked, shapes snap to
  the 16.7 ms window grid, which can delay stimuli by up to one window.
- TTL pulses keep their exact tick. Re-importing an export at the SysCon2 `dt`
  restores each pulse at the same sample.

## Sanity check

Click **Check Input Sanity** in the export dialog. It detects and offers to fix,
directly in the input array (undoable):

- wrong `dt` (**Clear Array** or **Resample (Nearest-Neighbor)**),
- first input less than 16.66 ms from the start (**Translate All Inputs Right**),
- SLM load-time gaps between neuron groups (**Translate Inputs Right**),
- SLM off/on overlaps (**Space Input Right**),
- neurons stimulated together with unequal durations (**Equalize (to lowest
  value)**), and overlapping pulse groups (align to the shortest),

then proposes any remaining timing or amplitude changes for confirmation.

## Export

Open the export dialog with **File → Export → Optogenetic Input → SysCon2**
(writes files only) or **Export with settings…** in the SysCon2 panel (also
imports into SysCon2 when connected).

| Setting                                  | Default                                                      |
| ---------------------------------------- | ------------------------------------------------------------ |
| Laser timeline                           | 940 nm digital (also 1030, 1100 nm)                          |
| Target power                             | 1 mW; `0` disables power control (intensity 0)               |
| Force binary export intensity            | off (amplitudes kept, clipped to 0-1)                        |
| Spiralling (TTL 1 tick before SLM ready) | off                                                          |
| Exact stimulus timing                    | on                                                           |
| Use 10 input spots per ROI               | on (placement: farthest apart, or equal spacing from border) |
| Output folder                            | last used, first `Documents\NeuroSeq\SysCon exports`         |

Then click **Generate Scripts**. Files written:

| File                       | Content                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| `<name>.seq`               | Shape and TTLPulse objects                                                |
| `<name>_original_stim.npz` | The exact input array; restored on re-import. Keep it next to the `.seq`. |
| `<name>_overview.png`      | Timeline overview                                                         |

**Power LUT.** **Load LUT…** loads a SysCon `.lut` for the selected wavelength
(one slot per wavelength). The requested mW is converted to the laser's
intensity percentage and written as a fraction (0-1): with a linear 85 mW
maximum, 5 mW is written as `0.058824`. Each object's description records
`neuron=… power_mW=… target_mW=…`. LUTs, export settings and the export folder
are saved in the `.nseq`; save the project after loading a LUT. Use the rig's
measured LUT, not a linear example.

## SysCon2 panel

**View → Show Stimulation setup**, tab **SysCon2 (HOLO4D)**:

- **Connection**: **Connect** / **Disconnect**.
- **Sequence Manager**: **Export with settings…**; **Export current array to
  stimulation setup** re-exports the shown array with the last settings and its
  own **Target power** box (these scripts are temporary and deleted when
  NeuroSeq closes). **Refresh**, **Select**, **Delete** the sequences in
  SysCon2, or **Browse…** and **Import** a local `.seq` into the **Target
  directory** (default `neuroseq\<date>`).
- **Preview**: collapsed by default (**Show preview**).
- **Holo4D**: select the imported sequence, set **Runs** (0 = endless, asks for
  confirmation), **Start condition** and **Trigger behavior** under _Advanced
  options_, then **Upload** (waits for Idle) and **Run**. The panel shows
  Busy/Idle and elapsed time, not percent complete. A triggered run waits until
  Busy is seen. **Stop** also cancels a queued Run.
- **Laser**: read-only. **Get Status** discovers the lightsources (preferring
  940 nm). Set power, shutter and emission in SysCon2.

## Dynamic input routines on the rig

Choose the **SysCon2 (calcium imaging)** target in the rheobase or STDP window
dialog; **Use SysCon2 dt** fixes the array's timestep.

**Hardware settings (SysCon2 / ScanImage)** holds target power, laser, spiral
scanning and 10 spots per cell (shared with the SysCon2 panel) and the recording
options:

- **Run blocks automatically and record them with ScanImage** (default on):
  each block is uploaded, run and recorded, and ΔF/F and spikes are extracted;
  see [ScanImage acquisition](scanimage_acquisition.md#dynamic-input-routines).
  Rheobase counts a response when ΔF/F rises by the **Response threshold**
  (default 20 %) in the post-pulse window, or when OASIS finds a spike of at
  least the **Spike threshold**, depending on **Calcium readout**.
- Unticked: each block is only imported into SysCon2. Run, record and import
  it yourself, then click **Calcium data imported**. Suite2p's normalised traces
  are refused for ΔF/F rise.

Safety limits: a block is refused without a target power above 0 mW and a LUT
for the chosen laser, with a target above the LUT maximum, with amplitudes
outside (0, 1], or with a stimulated neuron lacking a Cellpose ROI. Export
warnings are asked once per routine.

Rheobase defaults on this target: 16.7 ms pulse (one spiral), up to 100 ms, 1 s
recovery, 300 ms post-pulse window, **Fixed duration, varying amplitude**, 3
repeats per probe. Each target keeps its own values. With deconvolved spikes, a
spike in the frame after the window still counts.

If the input array already holds stimuli, **Start** offers **Clear and start
over**, **Continue after last stimulus**, or **Start in new arrays** (new empty
array and data group named after the run, e.g.
`Exc_stimulation rheobase_2026-10-22_143012`).

Each run saves its blocks in its own folder in the export folder, e.g.
`SysCon exports\rheobase_2026-10-22_143012\rheobase_2026-10-22_143012_001.seq`
(with PNG and sidecar), imported into SysCon2 under the same name.

## Importing a `.seq` file

**File → Import data → Input → Optogenetics** rebuilds a sequence in a new or
existing input array at 520.8125 µs.

- If the `_original_stim.npz` sidecar is next to the file and matches it, the
  exact original array is restored. Otherwise timings are the hardware-adjusted
  ones (SLM load delays).
- Targets map to consecutive rows from **Target neuron offset**, in order of
  first occurrence, not to project neuron IDs or Cellpose cells.
- **Replace mapped target channels** clears only the receiving rows; unticked,
  values merge by maximum.
- Amplitudes are raw file intensities (1 for zero-intensity shapes), not optical
  power. One sequence pass is imported (`runs` ignored).
- Rejected: files with no stimulation, arrays over 512 MiB, objects with several
  timing entries or lightsources.
- A differing `dt` of an existing target array is resampled after confirmation.

## Generating test data (CLI)

```bash
# 512 × 512 image with a 10 × 10 grid of cells
neuroseq --generate-cellpose-test-data ./cellpose_512.tif \
 --cellpose-grid-rows 10 --cellpose-grid-cols 10 \
 --cellpose-radius 11 --cellpose-gap 30

# 1300 × 1300 with auto-fitted radius and gap
neuroseq --generate-cellpose-test-data ./cellpose_1300.tif \
 --cellpose-grid-rows 10 --cellpose-grid-cols 10 \
 --cellpose-square-size 1300
```
