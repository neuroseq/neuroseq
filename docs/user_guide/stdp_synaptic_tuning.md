# STDP synaptic tuning

**Design dynamic input** is a closed-loop stimulation feature in NeuroSeq. Unlike
**Design static input**, which writes a complete pattern before a simulation,
Design dynamic input repeatedly performs three operations:

1. append stimulation to the selected stim array,
2. continue the Brian2 simulation, and
3. analyze the resulting voltage trace to decide what to append next.

This guide covers **STDP synaptic tuning**, a routine that targets a
postsynaptic EPSP amplitude by applying spike-timing-dependent plasticity
(STDP) pairings. The decision policy and protocol described below target this
specific use case.

## Opening Design dynamic input

Select the stim array that should receive the generated protocol, then click
**Design dynamic input** in the stimulation-editor toolbar. A prompt asks which
design routine to run; choose **STDP synaptic tuning** to open the designer.
The selected array must have columns for both the pre- and postsynaptic
neuron IDs.

For a ready-to-run model, open:

**File → Open example project → Network → Two Hodgkin-Huxley Neurons with STDP**

In that example, use:

- stim array: `Exc_stimulation`,
- trace: `Simulation :: P / v`,
- presynaptic neuron ID: `0`, and
- postsynaptic neuron ID: `1`.

STDP synaptic tuning appends data; it does not clear an existing stim array. New
blocks are placed after both the current array end and the current simulation
time, so generated stimulation is never inserted into the already simulated
past.

## Closed-loop routine

When **Start routine** is pressed, STDP synaptic tuning performs the following
sequence:

1. Append five probe pulses (default 1 ms each) for the presynaptic neuron.
   Probe pulses are spaced 100 ms apart.
2. Continue the simulation through the appended block.
3. Detect and average the five EPSP amplitudes in the postsynaptic voltage
   trace. This first average is the baseline and is defined as **100%**.
4. Compare the current percentage with **Target relative EPSP amplitude**.
5. Append and simulate one plasticity-induction pairing. The selected pairing
   strategy determines the pulse order and timing.

6. Append and simulate another five-pulse presynaptic probe block.
7. Measure the new five-EPSP average and repeat from step 4 until the target is
   reached or the maximum number of induction pairings is exhausted.

The target is achieved when:

```text
abs(current EPSP % - target EPSP %) < tolerance %
```

## Controls

| Control                                                                  | Meaning                                                                        |                     Default |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------------------------: |
| **Trace to analyze**                                                     | Simulation voltage trace containing the postsynaptic neuron                    |       First available trace |
| **Presynaptic neuron ID**                                                | Stim-array column receiving probe pulses and the presynaptic side of a pairing |                         `0` |
| **Postsynaptic neuron ID**                                               | Trace index used for EPSP measurement and the postsynaptic side of a pairing   |                         `1` |
| **Pulse amplitude**                                                      | Value written into the selected stim array during each pulse                   |        `1.0` in array units |
| **Pulse duration (ms)**                                                  | Width of every probe and pairing pulse                                         |                      `1 ms` |
| **Pairing strategy**                                                     | Fixed timing with automatic direction, or adaptive signed timing               | Fixed pre-post pairing time |
| **Pre-post pairing time**                                                | Fixed-strategy interval; pulse order is selected automatically                 |                     `30 ms` |
| **Initial guess for pre-post pairing time**                              | Adaptive-strategy starting magnitude                                           |                     `30 ms` |
| **dt increment**                                                         | Adaptive-strategy change applied to the signed pairing time each iteration     |                      `1 ms` |
| **Target relative EPSP amplitude**                                       | Desired EPSP relative to the first five-pulse baseline                         |                      `120%` |
| **Tolerance window (deadband)**                                          | Maximum absolute distance from the target that counts as success               |                        `1%` |
| **Maximum plasticity induction iterations**                              | Safety limit for closed-loop iterations                                        |                        `20` |
| **Run until the last iteration regardless of convergence to the target** | Ignore early convergence and continue until the maximum pairing count          |                    Disabled |
| **Follow new input on each iteration**                                   | Move the synchronized timeline camera to each appended block                   |                     Enabled |
| **Visible history**                                                      | Number of generated-block lengths kept visible throughout execution            |                         `3` |

Set **Visible history** to `0` to show only the newly appended block. Disable
**Follow new input on each iteration** to retain manual control of the shared pan and zoom state.

Normally, entering the tolerance stops the routine immediately. Enable **Run
until the last iteration regardless of convergence to the target** to keep
applying induction pairings until **Maximum plasticity induction iterations** is reached.
Convergence is still measured and recorded in the log while this override is
active.

### Pairing strategies

**Fixed pre-post pairing time** keeps the configured interval magnitude
constant. Below the EPSP target it uses pre-before-post timing; above the target
it uses post-before-pre timing.

**Adaptive (single parameter, dt increment)** starts with the initial pairing
time magnitude. After each new EPSP measurement, it compares the remaining
target error with the change since the preceding measurement. If
`abs(EPSP new - target) > abs(EPSP new - EPSP old)`, the magnitude is reduced by
**dt increment** (not below zero); otherwise it is increased by **dt
increment**. The current target side determines the sign: positive/pre-before-
post below target and negative/post-before-pre above target.

## EPSP measurement

STDP synaptic tuning reuses NeuroSeq's existing **EPSP amplitudes** analysis plugin.
The plugin:

- converts supported voltage units to millivolts,
- detects peaks using an adaptive prominence threshold,
- measures amplitude relative to a local pre-peak baseline, and
- rejects action-potential-sized peaks.

STDP synaptic tuning associates detections with the five known probe times and uses
their mean as the current EPSP amplitude. If the selected trace does not record
the postsynaptic neuron, or if an EPSP cannot be detected after every probe,
the routine stops and reports the reason in its log.

## Creating simulation trace monitors

A freshly loaded model may not expose its simulation trace names until Brian2
has initialized the network. If no trace is available when the routine starts,
STDP synaptic tuning opens **Create stimulation trace monitor** and offers to run a
short initialization simulation.

- default duration: `0.001 s`,
- precision: three decimal places,
- minimum duration: `0.001 s`.

After the initialization run, STDP synaptic tuning refreshes the trace list and
starts the EPSP routine automatically. If no trace becomes available, add a
Brian2 `StateMonitor` for the membrane-voltage variable in **File → Edit
model**, then try again.

## Live plots

The dialog contains three synchronized Matplotlib plots with enough spacing to
keep their labels and tick marks separate:

- **EPSP amplitude (%)** shows every five-pulse measurement relative to the
  initial baseline. The target appears as a dashed horizontal line after the
  baseline has been measured. The shaded region is the tolerance deadband.
- **Δ target (%)** shows the signed difference between the current EPSP and the
  target. Zero means that the measured value equals the target; the shaded band
  shows the same tolerance around zero.
- **Pairing time (ms)** shows the signed pre-post pairing time used for each
  induction. Positive points are pre-before-post and negative points are
  post-before-pre.

The text below the plots reports the signed difference, absolute difference,
and active tolerance. Plot colors follow NeuroSeq's light or dark theme.

## Stopping and reviewing the result

**Stop** prevents the routine from appending another block after the current
simulation finishes. It does not cancel a Brian2 task already in progress; use
the simulation progress dialog's cancellation control when the active run must
be interrupted.

All generated blocks remain in the selected stim array and can be inspected,
edited, undone, saved with the project, or used for subsequent simulations.
The routine log records baseline measurements, relative changes, induction
direction, termination, and errors.

## Current limitations

- The routine currently supports the built-in Brian2 simulation path.
- The decision policy assumes excitatory EPSPs and an STDP model in which
  pre-before-post potentiates and post-before-pre depresses.
- Probe timing is fixed to five pulses (of configurable duration) at 100 ms
  intervals.
- Only one selected stim array is modified, although NeuroSeq pads other arrays
  as needed to maintain a compatible project timeline.
- EPSP detection requires a voltage `TraceData` object and a detectable response
  after each probe pulse.
