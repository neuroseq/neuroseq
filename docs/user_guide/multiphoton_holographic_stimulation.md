# Multiphoton Holographic Stimulation (SysCon2 Export)

NeuroSeq can export stimulation patterns as SysCon2 `.seq` files for use with
Holo4D multiphoton holographic stimulation hardware.

---

## Overview

The stim array you draw in NeuroSeq represents **laser TTL pulses** directly.
Each sample corresponds to one SysCon2 hardware tick
(`dt = 0.0005208125 s = 520.8125 µs`). NeuroSeq infers the SLM shape objects
needed for each neuron from the drawn TTL pulse positions — you do not need to
manually define hologram boundaries.

---

## Required setup

| Parameter             | Required value                                        |
| --------------------- | ----------------------------------------------------- |
| `stim_array.dt`       | `0.0005208125` s (one SysCon2 tick)                   |
| Cellpose segmentation | loaded and mapped to stimulus channels                |
| SysCon2 template      | a previously imported `.seq` file in the same session |

The dialog will reject (or offer to fix) stim arrays whose `dt` does not match
the SysCon2 tick quantum.

---

## Drawing TTL pulses

Draw active samples in the stimulation editor exactly where you want the laser
to fire. A few rules apply:

- **No minimum duration**: pulses can be as short as one tick (520.8 µs).
- **dt must be the SysCon2 tick**: use the _Fix dt_ button in the sanity-check
  dialog if you need to convert an existing array.
- **Blank lead-in required**: the SLM hardware needs one full 16.666 ms window
  (32 ticks) of blank time before the first laser pulse to load the hologram.
  The sanity fixer inserts this automatically if it is missing, but you can also
  add it yourself with _Shift right for initial SLM load_.

---

## How shape objects are derived

The export pipeline maps your raw TTL pulses onto SLM shape objects as follows:

1. **Window coverage** — each raw pulse is mapped to the 32-tick window(s) it
   falls in. This determines which SLM pattern windows must be loaded.
2. **Hologram grouping** — active windows are grouped into holograms. A new
   hologram starts when the active neuron set changes relative to the previous
   non-empty window: either a neuron disappears or a new neuron appears.
   Gaps (fully empty windows) do **not** break the hologram — the pattern stays
   loaded across the gap.
3. **Pre-load blank window** — a blank 16.666 ms window is placed immediately
   before the first active window of each hologram. The SLM uses this time to
   load the pattern before any laser pulse fires. If the blank is missing the
   sanitizer inserts one by shifting the tail right.
4. **Shape object bounds** — for each neuron within a hologram:
   - shape starts at `(first_active_window - 1) × 32` ticks (the blank load window),
   - shape ends at `last_active_window_excl × 32` ticks (minimum span: load + 1 active window).

---

## TTL pulse timing

Laser TTL pulses are emitted at **exact tick positions**, not at window
boundaries. If a pulse was drawn at tick 100, the exported `.seq` file
contains a `TTLPulse` starting at tick 100 (= 52.08 ms).

If the sanitizer inserted a blank window before the first active window (because
the first pulse was too early), all raw pulse positions are shifted by 32 ticks
per inserted blank. The shift preserves the sub-window offset of each drawn
pulse relative to its containing window.

**Roundtrip accuracy** — when a `.seq` file exported in this mode is imported
back into NeuroSeq with `dt = 0.0005208125 s`, each pulse reconstructs at the
same sample index it was exported from (`sample_idx == tick_idx`).

---

## Hologram boundary rule

| Situation                                                            | Effect                                    |
| -------------------------------------------------------------------- | ----------------------------------------- |
| Same neuron set, contiguous windows                                  | continue same hologram                    |
| Same neuron set, gap between windows                                 | continue same hologram (SLM stays loaded) |
| A neuron that was active in the **last** non-empty window disappears | start new hologram                        |
| A neuron that was **never** in the current hologram appears          | start new hologram                        |

A new hologram is triggered whenever the active neuron set changes — either a neuron
leaves or a new one joins relative to the last non-empty window. Gaps (fully empty
windows) are ignored when checking for disappearances; only transitions between
consecutive non-empty windows matter.

---

## Sanity checks

Run **Check Input Sanity** in the stimulation generator dialog before exporting.
The checker validates:

- `dt` is the SysCon2 tick quantum (offers _Fix dt_ button if not),
- first active sample is at least 32 ticks from the start (SLM lead-in),
- no SLM gap violations between successive neuron groups,
- no off→on SLM overlap violations,
- no simultaneous neurons with mismatched durations (optional; use
  _Equalize durations_),
- all active neurons have a valid Cellpose mapping.

Feedback is shown in a scrollable dialog. If fixes were applied you can write
the corrected values back to the stim array via _Apply fixes_.

---

## Exporting

1. Open **File → Export data → Export light stimulus** (or the toolbar button) and choose `SysCon2` in the pre-dialog.
2. Select the stim array and Cellpose segmentation.
3. Choose laser wavelength (`940 nm`, `1030 nm`, or `1100 nm` digital; default `940 nm`).
4. Optionally set a **Target power** and load a power LUT (see below).
5. Click **Check Input Sanity** and resolve any errors.
6. Enter an output script name and click **Export**.

### Target power and power LUT

The dialog has a **Target power** (mW) spinner next to a **Load LUT…**
button:

- Setting **Target power** to `0` disables power control — the exported
  `Intensity_0` fields are written as zero, which matches the historical
  default behaviour.
- Loading a SysCon `.lut` file maps the requested mW to the laser's
  intensity percentage at the currently selected wavelength. The exporter divides
  that percentage by 100 for the `.seq` intensity fields (0–1). Each
  wavelength has its own LUT slot.
- LUTs are bundled into the `.nseq` archive when you save the project, so
  the export stays reproducible across machines. Loading a LUT marks the project
  as modified; save it to retain the calibration.

The **status label** next to the button shows which LUT (if any) is
currently loaded for the active wavelength. Switching the wavelength
combobox swaps to that wavelength's LUT slot.

The exporter writes:

| File                       | Content                                             |
| -------------------------- | --------------------------------------------------- |
| `<name>.seq`               | SysCon2 sequence — Shape objects + TTLPulse objects |
| `<name>_original_stim.npz` | the exported input array, restored on re-import     |
| `<name>_overview.png`      | timeline overview image                             |

---

## Generating test data (CLI)

```bash
# 512 × 512 image with 10 × 10 grid
neuroseq --generate-cellpose-test-data ./cellpose_512.tif \
 --cellpose-grid-rows 10 \
 --cellpose-grid-cols 10 \
 --cellpose-radius 11 \
 --cellpose-gap 30

# 1300 × 1300 with auto-fitted radius/gap
neuroseq --generate-cellpose-test-data ./cellpose_1300.tif \
 --cellpose-grid-rows 10 \
 --cellpose-grid-cols 10 \
 --cellpose-square-size 1300
```

## Importing a `.seq` file

Use **File → Import stimulus → Import light stimulus → SysCon2** to reconstruct a
sequence in an existing input array. Browsing and cancelling do not change the
project. Import applies the file's header as the session's export template only
after conversion succeeds. If the project sample interval changes to 520.8125 µs,
existing stimulus and trace data are remapped after confirmation.

Targets map to consecutive channels starting at **Target neuron offset**, in their
first-occurrence order for Holo4D sequences. They are not matched to project neuron
IDs or Cellpose identities. Repeated point Shapes at identical coordinates and
with the same lightsource share a channel; nearby legacy Entities retain the
1 µm grouping rule. Multiple spots in one ROI may therefore import as separate
channels. The preview shows the same grouping used for conversion.

**Replace mapped target channels** clears only channels receiving imported data;
other channels are preserved. Disable it to merge using the maximum value at
each sample. Dropped channels, clipped duration, and capped legacy repeats require
confirmation before the project changes. Files that reconstruct no stimulation
are rejected, as are arrays exceeding the 512 MiB import allocation limit.

Holo4D timing is reconstructed from each target's laser TTL pulses and SLM
availability windows. The **Legacy repeat tick** setting does not change Holo4D's
520.8125 µs TTL quantum. One sequence pass is imported; the header's `runs` count
is not expanded. Imported amplitudes are raw file intensity values, with a nominal
1 used for zero-intensity shapes, rather than recovered optical power or the
original drawing amplitudes. Spiralling latency and hardware behavior cannot be
inferred from the file, so this is not a measurement of delivered stimulation.
Objects with multiple timing entries or multiple lightsources are rejected rather
than partially imported.

Every export also writes `<name>_original_stim.npz` next to the `.seq`: the exact
input array, including silent neurons. The `.seq` holds hardware-adjusted timings
(SLM load delays), so importing it alone reproduces those adjusted timings. When
the sidecar sits next to the file and still matches it, the import restores the
original array with its exact timings and neuron rows. Keep the two files together.

## Setup panel and hardware execution

Export defaults are **940 nm**, **Force binary export intensity off**,
**Spiralling off**, **Exact stimulus timing on**, and **Use 10 input spots per ROI on**.
With exact timing, each hologram shape starts one 16.7 ms SLM load window before
its first pulse instead of on the window grid, and stimuli are delayed only where
the previous hologram leaves no room for that load (the first stimulus is always
delayed by one window). Untick it to snap shapes to the window grid, which can
delay stimuli by up to a window per hologram. Previously remembered
settings still apply to quick exports. With a linear 85 mW maximum calibration,
5 mW is exported as approximately `0.058824`, not `5.8824`, in both Shape and
laser TTL intensity fields. Use the rig's measured wavelength-specific LUT;
the linear example is not a replacement calibration.

Main-menu exports populate the setup panel's generated preview without importing
into hardware. Exports from the setup panel import when connected. Routine blocks
also update the preview after import acknowledgement. Preview images resize with
the panel and remain distinct from hardware status.

Laser controls are currently read-only: **Get Status** discovers lightsources on
its first use, preferring 940 nm, and retrieves status. The single selector displays
friendly names while retaining the original SCI identifiers internally. Set power,
shutter, and emission in SysCon2 itself.

Select the imported sequence, choose Runs and trigger settings, and click
**Upload**. After confirmation of an endless upload, NeuroSeq uploads and waits for
hardware Idle. Then click **Run** and confirm to start the sequence. The panel shows
phases, elapsed time, and hardware Busy/Idle state; SCI provides no
percentage-complete measurement. A triggered run remains waiting until Busy is
observed. A short automatic run can finish between polls; that case is reported
without claiming measured completion. **Stop** also cancels any queued Run.

The preview is collapsed by default; click **Show preview** to expand it.

The dynamic-routine dialogs offer **Use SysCon2 dt** and prompt for resampling when
starting with the wrong timestep. **Edit Input Array** also provides exact presets
for SysCon2 (520.8125 µs), 1 ms, 0.1 ms, 50 µs, and 30 Hz imaging.
