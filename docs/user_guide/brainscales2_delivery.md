# Deliver stimuli to BrainScaleS-2

The **Stimulation setup** pane (View → Show Stimulation setup) has one tab per
stimulus-delivery target. Next to **SysCon2 (HOLO4D)**, which drives the
holographic stimulation rig, the **BrainScaleS-2** tab delivers the current
stim array to the BrainScaleS-2 (BSS-2) accelerated neuromorphic chip and adds
the chip's response to the project as a new data group.

!!! warning "Not yet verified on hardware"

    The BSS-2 path has been tested only against a simulated helper. The
    container calls and the time conversion listed under
    [Assumptions](#assumptions-to-check-on-the-real-system) still have to be
    confirmed on a real chip.

## How it works

NeuroSeq does not import the BSS-2 software stack (`pynn_brainscales`,
`dlens_vx_v3`); it only exists inside the EBRAINS container image. On
**Connect**, NeuroSeq starts a small helper script inside the container:

```bash
singularity exec --bind <neuroseq>/resources/bss2 [--app <app>] <image> \
  bash -c '{ export SYSTEMNAME="ebrainslab";
             source /esd/vendor/spack/share/spack/setup-env.sh;
             spack env activate esd; } 1>&2 && exec python -u neuroseq_bss2_helper.py'
```

NeuroSeq and the helper exchange JSON lines over stdin/stdout. The helper
opens a connection to the chip through _quiggeldy_ (the BSS-2 job server)
and runs the network with `pynn_brainscales`. The connection is shared by all
open project tabs, but each tab only receives the results of its own runs.

## Setup

1. **Get the container image.** On lab machines the image is
   `/containers/stable/latest` (use the app name `dls`). Elsewhere, download the
   EBRAINS image, for example:

   ```bash
   mkdir -p tmp
   SINGULARITY_TMPDIR=$PWD/tmp singularity pull \
     docker://docker-registry.ebrains.eu/esd/master/image:lab_latest.oci
   ```

   and choose the resulting `.sif` file with **Browse…** (leave **Container
   app** empty for such an image unless it defines apps).

2. **Choose a setup.** **Refresh** downloads the quiggeldy setup list
   (`https://quiggeldy.bioai.eu/nmpi/quiggeldy_setups.csv`, columns
   `Host,Port,Identifier`). The default setup is the one the EBRAINS examples
   pick: `sha256(user name) mod number of setups`, where the user name is
   `$USER`. Host, port and expected identifier stay editable, for example to
   use another quiggeldy host than the one listed.

3. **Enter your EBRAINS token.** Paste the output of `clb_oauth.get_token()`
   (EBRAINS Collaboratory) into **Token**. If the field is empty, an existing
   `QUIGGELDY_TOKEN` environment variable is used. The token is kept in memory
   for this session only: it is passed to the container through the helper's
   environment (`QUIGGELDY_TOKEN` and `SINGULARITYENV_QUIGGELDY_TOKEN`), never
   on the command line, and it is not saved in the settings or written to the
   log. Anything that looks like the token in helper output is masked.

4. **Connect.** The first activation of the container's software environment
   takes around 90 seconds; the status line reports each step. Once quiggeldy
   answers, **Connected chip** shows the chip identifier. If it differs from the
   selected setup's identifier, the status says so.

The tab settings (image, app, setup, input mode and its parameters, recorded
neuron, speed-up) are saved for next time; the token is not.

## Input modes

**Neurons** sets how many hardware neurons are driven; _auto_ uses one neuron
per stim channel. Stim channel `i` drives hardware neuron `i`. Extra channels
are dropped and missing ones receive no input, both with a warning. A
single-channel stim array drives every neuron. **Max neurons** (default 512, one
chip) caps the count.

### Constant current (stim array) — default

The stim array is split into segments wherever any channel's value changes;
runs of identical samples form one segment. For each segment and neuron the
helper sets

- `constant_current_enable = value > 0`, and
- `constant_current_i_offset = round(clip(value / value_at_max, 0, 1) * max_i_offset)`

and then runs the segment with `pynn.run(duration, RunCommand.APPEND)`; the
last segment is run with `RunCommand.EXECUTE`. **Max i_offset** is in hardware
units (0–1022); **Stim value at max** defaults to the array maximum. Segments
that become identical after rounding are merged. Very many segments (more than 1000) are allowed but slow to prepare, so the confirmation warns about them.

### Spike input

Spike trains are fed through a `SpikeSourceArray` population, one source per
hardware neuron, with one-to-one excitatory projections of the chosen
**Weight** (0–63). The spike source is either

- **From stim array**: the stim array is sampled every **Event interval**
  (biological ms); a spike is emitted wherever a channel exceeds
  **Threshold**, as in the PyNN example model, or
- **From project spike data**: a spike dataset from a data group of the
  project; neuron `i`'s spikes drive hardware neuron `i`.

## Time base (speed-up)

BSS-2 runs about 1000 times faster than biological neurons, and
`pynn_brainscales` run durations and recorded times are in hardware
milliseconds. NeuroSeq works in biological seconds and converts with the
**Speed-up** factor (default 1000):

- hardware ms = biological s × 1000 / speed-up — 1 s of stimulus is a 1 ms run;
- biological s = hardware ms / 1000 × speed-up — a spike at 0.5 ms on the chip is
  stored at 0.5 s.

## Delivery and readback

**Deliver current stim array** shows a summary (neurons, segments or spikes,
biological and hardware duration, recorded neuron) and asks for confirmation.
While the run executes, the status line shows the helper's progress. The
neurons are `HXNeuron`s configured from the chip's nightly calibration (as in
the BrainScaleS-2 demos), which the helper loads once per connection with
`pynn.helper.chip_from_nightly()`, falling back to the EBRAINS calibration
download for the connected chip. A running BSS-2 run keeps the project tab
from closing.

When the run finishes, a data group named `BSS-2 <stim name> <date time>` is
added (undoable with Edit → Undo) containing

- **BSS-2 spikes**: spikes of all driven neurons (hardware neuron indices,
  biological time), and
- **BSS-2 membrane voltage (neuron _n_)**: the membrane voltage of the neuron
  chosen under **Record v of neuron**. BSS-2 has a single MADC, so only one
  neuron's voltage is recorded per run. Its irregular samples are resampled to
  a regular time step; the values are raw MADC readings (`a.u.`).

**Stop** terminates the helper. A run already sent to the chip cannot be
recalled; its results are discarded and you have to reconnect.

## Debug log

The **Debug log** box at the bottom of the tab records every step with a
timestamp: the exact `singularity exec …` launch command and the variables
NeuroSeq passes into the container, the helper process start and exit code,
everything the container prints while its software environment activates,
the helper's Python/package versions and quiggeldy settings, and (with
**Show protocol traffic** on) every message exchanged with the helper, with
arrays summarised by type and shape.

The EBRAINS token never appears in the log: NeuroSeq and the helper only
report whether it was given and how many characters it has.

To report a connection problem, click **Copy log**. It copies the log
together with the NeuroSeq version, host, Singularity version, connection
state and the tab's settings, ready to paste into a message. The text in the
log can also be selected and copied directly. **Clear** empties the log.

## Verified against the EBRAINS container

Checked in `image_lab_latest.oci.sif` (Python 3.11, PyNN 0.12.3, neo 0.13.3):

- `pynn.run(runtime, command)` with `RunCommand.APPEND`/`EXECUTE`; durations
  are hardware milliseconds, as in the AdEx tutorial (0.1 + 0.6 + 0.3 ms).
- `constant_current_enable`, `constant_current_i_offset` (and
  `constant_current_type`) are `HXNeuron` parameters; `CalibHXNeuronCoba` does
  not have them, so delivery uses `HXNeuron` on top of the nightly calibration
  (`pynn.setup(initial_config=calib)`).
- `HXNeuron` records `spikes`, `v`, `exc_synin`, `inh_synin`, `adaptation`;
  `SpikeSourceArray`, `OneToOneConnector` and `StaticSynapse` exist.
- `hxcomm.ManagedConnection().get_unique_identifier()` returns a list; the
  first entry is the chip identifier.
- The helper starts through NeuroSeq's launch command and answers `ping`.

### Start-up time

- Without `squashfuse` on the host, Singularity unpacks the whole `.sif` on
  every start (about 1.5 minutes). Build an unpacked copy once and choose that
  folder as the container image instead (the container then starts in about
  1 s):

  ```bash
  singularity build --sandbox image_lab_latest_sandbox image_lab_latest.oci.sif
  ```

- Activating the container's spack environment takes about 50 s. NeuroSeq
  caches the activation (`spack env activate --sh`) per image in the user
  cache folder (`~/.cache/neuroseq/bss2` on Linux), so only the first
  connection after installing or updating an image pays this; later
  connections start in seconds.

## Still to check on the hardware

- Per-neuron lists in `Population.set(...)` apply from the next `APPEND`
  segment on.
- Recording `spikes` on the whole population and `v` on a one-neuron view in
  the same run; spikes are read from
  `get_data("spikes").segments[-1].spiketrains`, the membrane from
  `get_data("v").segments[-1].irregularlysampledsignals[0]` (falling back to
  `analogsignals`).
- The calibrated `HXNeuron` defaults respond to `constant_current_i_offset` in
  the 0–1022 range as expected (the demos use 0–1000).
