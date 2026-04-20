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
3. Choose laser wavelength (default 1140 nm).
4. Click **Check Input Sanity** and resolve any errors.
5. Enter an output script name and click **Export**.

The exporter writes:

| File                      | Content                                             |
| ------------------------- | --------------------------------------------------- |
| `<name>.seq`              | SysCon2 sequence — Shape objects + TTLPulse objects |
| `<name>_export_debug.csv` | per-window/segment/hologram timing debug table      |
| `<name>_overview.png`     | timeline overview image                             |

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
