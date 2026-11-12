# Neuromorphic hardware

The **BrainScaleS-2** tab of the **Stimulation setup** pane (**View → Show
Stimulation setup**) delivers the current input array to the BrainScaleS-2
(BSS-2) accelerated neuromorphic chip and adds the response as a new data
group. The chip cannot inject arbitrary currents, so input is either a constant
current per segment or spikes from a virtual source population.

Verified on `hxcube7fpga0chip57_1` via `quiggeldy.bioai.eu:11750` (October
2026): in a 10-neuron staircase each neuron fired only during its own pulse.

## How it works

The BSS-2 software (`pynn_brainscales`) exists only in the EBRAINS container.
**Connect** starts `resources/bss2/neuroseq_bss2_helper.py` inside it with
`singularity exec`, and NeuroSeq talks to it over JSON lines. The helper reaches
the chip through _quiggeldy_ (the BSS-2 job server). One connection is shared by
all project tabs; each tab receives only its own results.

## Setup

1. **Container image.** Lab machines: `/containers/stable/latest` (default)
   with **Container app** `dls`. Elsewhere pull the EBRAINS image and pick the
   `.sif` with **Browse…** (leave the app empty):

   ```bash
   mkdir -p tmp
   SINGULARITY_TMPDIR=$PWD/tmp singularity pull \
     docker://docker-registry.ebrains.eu/esd/master/image:lab_latest.oci
   ```

2. **Setup.** `quiggeldy.bioai.eu` ports **11717 (stable)**, **11750
   (experimental, default)** and **11780 (special)**. Only allow-listed IPs can
   connect; ask the BSS-2 team to add yours. With `lab_latest` only
   experimental worked (October 2026); the others fail with `Could not submit
request`. **Refresh** also downloads the EBRAINS setup list. Host, port and
   expected **Identifier** stay editable.
3. **Token (optional).** Not needed from an allow-listed IP. Otherwise paste
   `clb_oauth.get_token()` output, or set `QUIGGELDY_TOKEN`. The token stays in
   memory only, is passed through the environment, and is masked in the log.
4. **Connect.** **Connected chip** shows the identifier (with a warning if it
   differs from the expected one).

Settings except the token are remembered.

### Start-up time

- Without `squashfuse`, each start unpacks the `.sif` (about 1.5 min). Build a
  sandbox once and select that folder instead (starts in about 1 s):

  ```bash
  singularity build --sandbox image_lab_latest_sandbox image_lab_latest.oci.sif
  ```

- The first activation of the container's spack environment takes about 50 s;
  it is then cached per image (`~/.cache/neuroseq/bss2` on Linux).

## Input modes

**Neurons** (_auto_: one per input row) and **Max neurons** (512, one chip).
Row `i` drives hardware neuron `i`; extra rows are dropped and missing ones get
no input, with a warning. A single-row array drives every neuron.

**Constant current (input array)** (default). The array is split into segments
where any row changes value. Each segment is a separate hardware run starting
from rest (chained runs did not work on the chip), with

`constant_current_i_offset = round(clip(value / Stim value at max, 0, 1) × Max i_offset)`

and the current enabled where the value is above 0. **Max i_offset**: 500
(range 0-1022). **Stim value at max**: array maximum. Each segment costs about
0.5 s of hardware time; more than 1000 segments trigger a warning.

**Spike input.** A `SpikeSourceArray` (one source per neuron) with one-to-one
excitatory synapses of **Weight** 63 (0-63). **Spike source**:

- **From input array (threshold)**: sample the array every **Event interval**
  (20 ms biological) and spike where it exceeds **Threshold** (0).
- **From project spike data**: a spike dataset from a data group; neuron `i`
  drives hardware neuron `i`.

## Time base

BSS-2 runs about 1000x faster than biology. **Speed-up** (default 1000):

- hardware ms = biological s × 1000 / speed-up (1 s of input is a 1 ms run),
- biological s = hardware ms / 1000 × speed-up.

## Delivery and readback

**Deliver current input array** shows a summary and asks for confirmation.
Neurons are `HXNeuron`s with the chip's nightly calibration. The result is a
data group `BSS-2 <stim name> <date time>` (undoable) with:

- **BSS-2 spikes** of all driven neurons, and
- **BSS-2 membrane voltage (neuron _n_)** for the neuron in **Record v of
  neuron** (one MADC, so one neuron per run), resampled to a regular step, raw
  units.

**Stop** kills the helper; a run already on the chip is discarded and you must
reconnect. A running delivery keeps the project tab from closing.

## Rheobase routine

In **Design dynamic input → rheobase**, choose **BrainScaleS-2
(neuromorphic)**. **Connect** in the dialog uses this tab's settings.

- Each probe is delivered as constant current with the tab's settings. Probe
  amplitudes are fractions (0-1) of **Max i_offset**.
- The probed neuron's membrane voltage is recorded. All probes go into one data
  group, `BSS-2 rheobase <date time>`, aligned with the input.
- Every probe starts from rest; spikes after the current ends are not recorded.
- The chip is noisy near threshold: **Repeats per probe** (3 on hardware
  targets) counts a response when most runs spike.
- If the maximum cannot make a short pulse fire, that duration is marked (red
  triangle); the chronaxie search continues only if the maximum is at least
  twice the rheobase. Raise **Max i_offset** (up to 1022) for more current.
- **Stop** ends the search without killing the shared helper.

## Debug log

The **Debug log** lists every step (launch command, environment, helper
versions, and with **Show protocol traffic** every message). The token never
appears. **Copy log** copies it with version, host and settings for bug
reports; **Clear** empties it.

## Chip notes (October 2026)

- `pynn.run()` durations are hardware ms; runs need `pynn.reset()` in between.
- Constant current requires `HXNeuron` (not `CalibHXNeuronCoba`) on the nightly
  calibration.
- Chained `APPEND` runs executed only the first part, hence one run per segment.
- The membrane is sampled about every 32 ns hardware time (32 µs biological at
  1000x).
- Neurons differ: at `i_offset` 500 they fired 2-9 spikes per 100 ms pulse.
