# Dynamic input

A dynamic input routine runs an experiment step by step: append one input
block to the selected input array, deliver it, read the response, choose the
next block. Test a routine on a Brian2 simulation, then switch its
**Stimulation target** to run the same protocol on hardware.

## Starting a routine

1. Select the input array to extend.
2. Click **Design dynamic input** (View toolbar or the editor's right-click
   menu), pick a routine and click **Continue**.
3. Set the parameters and click **Start** (**Start routine** for STDP tuning).

If the array already holds input, choose **Clear and start over** or
**Continue after last stimulus** (hardware rheobase also offers **Start in new
arrays**). Blocks are appended after the simulated time, never into the past,
and stay in the array afterwards (undo, edit, save as usual).

One routine runs at a time: **Start** in another routine is refused until the
running one stops. While it runs, its array stays selected and cannot be drawn
on, cleared or given a pattern, and undo/redo are off.

**Pause** and **Stop** act before the next block (a running Brian2 run is not
cancelled). **Clear input array** zeroes the array; **Reset input array**
trims it back to the simulated time. The log records every probe and decision;
live plots open in a separate window.

On a fresh model the routine first runs a short simulation to create trace
monitors. If no `/ v` trace appears, add a `StateMonitor` for `v` in
**File → Edit Model**.

## Stimulation targets

| Target                       | Routines              | Amplitude means                            |
| ---------------------------- | --------------------- | ------------------------------------------ |
| Brian2 simulation            | all                   | Input array units                          |
| SysCon2 (calcium imaging)    | rheobase, STDP window | Fraction (0-1) of the export target power  |
| BrainScaleS-2 (neuromorphic) | rheobase              | Fraction (0-1) of the tab's `i_offset` max |

For SysCon2, **Hardware settings** sets power, laser, spots, recording tail
(3 s), artefact removal and trace extraction (Suite2p or ROI average). With
**Run blocks automatically and record them with ScanImage** (default) each
block is run, recorded and extracted for you. Unticked, you run, record and
import each block, then click **Calcium data imported** (rheobase) or **Block
done / data imported** (STDP window). **Use SysCon2 dt** sets the array
timestep to the SysCon2 grid. Details:
[Holographic stimulation](../networks/holographic_stimulation.md),
[ScanImage acquisition](../networks/imaging_acquisition.md),
[BrainScaleS-2](../networks/neuromorphic_hardware.md).

## Minimal input for spiking (rheobase)

Finds the smallest input that makes one neuron spike, by bisection rather than
a full grid scan. Each probe is a recovery period, a pulse, and a post-pulse
window in which spikes are counted. The search tests the maximum amplitude (no
spike: stop, raise it), then the minimum (spike: threshold is at or below it),
then halves the bracket until it is below the tolerance or **Maximum
iterations** (25) is reached.

**Sweeping strategy**:

- **Amplitude and duration**: rheobase at the longest duration, then the
  chronaxie (duration at twice the rheobase) by bisecting duration.
- **Fixed duration, varying amplitude**: threshold at one duration.
- **Fixed amplitude, varying duration**: shortest spiking duration.

| Parameter                        | Brian2          | SysCon2            |
| -------------------------------- | --------------- | ------------------ |
| Amplitude range / tolerance      | 1-100 / 0.05    | 0.1-1.0 / 0.05     |
| Fixed pulse amplitude            | 10              | 0.5                |
| Pulse duration range / tolerance | 10-50 ms / 1 ms | 16.7-100 ms / 1 ms |
| Recovery time                    | 200 ms          | 1000 ms            |
| Post-pulse window                | 20 ms           | 300 ms             |
| Repeats per probe                | 1               | 3                  |

With several repeats a probe counts as a response when most trials respond.
Brian2 counts crossings of **Spike threshold** (-20 mV). SysCon2 uses the
**Calcium readout**: a ΔF/F rise of at least **Response threshold** (20 %), or
OASIS-deconvolved spikes above **Spike threshold** (0.1 ΔF/F). BrainScaleS-2
starts with the hardware amplitudes and repeats and counts the chip neuron's
spikes; connect from the dialog or the BrainScaleS-2 tab.

## STDP timing window (LTP/LTD vs Δt)

Sweeps Δt = t_post − t_pre and measures the synaptic change at each point. Use
a suprathreshold amplitude (the rheobase routine can find one).

- **Readout**: **EPSP amplitude change (%)** (five presynaptic probe pulses
  after each point) or **Synaptic weight change (%)** (reads the synapse
  weight directly; use it on recurrent networks).
- **Sweep mode**: **Reset to baseline each Δt** restores a Brian2 checkpoint
  before every point. **Continuous sweep** keeps one synapse and visits Δt by
  increasing magnitude, alternating sign. SysCon2 is always continuous, with a
  calcium readout.

Defaults: amplitude 100 (0.8 on SysCon2), pulse 5 ms, Δt -100 to 100 ms in 21
points (Δt = 0 excluded), 1 pairing per point, recovery 150 ms, post-pulse
window 50 ms.

## STDP synaptic tuning

Drives one synapse's EPSP to a target relative to its initial size (Brian2
only). Try it in **File → Open example project → Network → Two Hodgkin-Huxley
Neurons with STDP** with array `Exc_stimulation`, trace `Simulation :: P / v`,
presynaptic 0, postsynaptic 1.

1. Five presynaptic probe pulses, 100 ms apart; their mean EPSP is the 100 %
   baseline.
2. Outside the **Tolerance window** (1 %) of **Target relative EPSP amplitude**
   (120 %), append one pairing: pre before post below target, post before pre
   above.
3. Probe again; repeat until within tolerance or after **Maximum plasticity
   induction iterations** (20). **Run until the last iteration regardless of
   convergence** keeps pairing.

**Pairing strategy**: **Fixed pre-post pairing time** (30 ms) uses the same
interval each time. **Adaptive (single parameter, dt increment)** starts at
30 ms and shrinks the interval by **dt increment** (1 ms) when the remaining
error exceeds the last EPSP change, otherwise grows it (between one sample and
1000 ms).

Pulses default to 100 (array units) for 5 ms. EPSPs are measured with the
**EPSP amplitudes** plugin; the routine stops if the trace lacks the
postsynaptic neuron or an EPSP is missed. It assumes excitatory EPSPs and
pre-before-post potentiation, and timings snap to the array timestep.
