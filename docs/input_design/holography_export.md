# Holography export

An input array becomes a `.seq` sequence for the HOLO4D two-photon holographic
stimulation rig: one laser TTL pulse per active sample and one SLM hologram per
group of co-stimulated neurons. To upload and run sequences on the rig, see
[Holographic stimulation](../networks/holographic_stimulation.md).

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
If shifted pulses no longer fit, the array is extended (with a warning)
instead of cutting them off.

## Export settings

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

## Re-importing a sequence

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
