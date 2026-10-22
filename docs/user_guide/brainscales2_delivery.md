# Deliver stimuli to BrainScaleS-2

The **Stimulation setup** pane (View → Show Stimulation setup) has one tab per
stimulus-delivery target. Next to **SysCon2 (HOLO4D)**, which drives the
holographic stimulation rig, the **BrainScaleS-2** tab delivers the current
stim array to the BrainScaleS-2 (BSS-2) accelerated neuromorphic chip and adds
the chip's response to the project as a new data group.

!!! note "Verified on hardware"

    Connection and constant-current delivery work on the real system (chip
    `hxcube7fpga0chip57_1` through `quiggeldy.bioai.eu:11750`, October 2026):
    a 10-neuron staircase made every neuron fire only during its own pulse,
    with the membrane of the recorded neuron read back over the whole run.
    See [Results on the chip](#results-on-the-chip).

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

2. **Choose a setup.** The picker always lists the quiggeldy instances on
   `quiggeldy.bioai.eu`, which run different software versions:
   **11717 (stable)**, **11750 (experimental)** and **11780 (special)**. They
   are reachable only from allow-listed addresses; ask the BSS-2 team to add
   your computer's IP. With the EBRAINS `lab_latest` image only the
   **experimental** instance accepted the connection (October 2026); it is
   the default. The others fail with `Could not submit request`. **Refresh**
   keeps the selected instance when it is still listed. **Refresh** additionally downloads
   the EBRAINS setup list
   (`https://quiggeldy.bioai.eu/nmpi/quiggeldy_setups.csv`, columns
   `Host,Port,Identifier`); its default entry is the one the EBRAINS examples
   pick: `sha256(user name) mod number of setups`, where the user name is
   `$USER`. Host, port and expected identifier stay editable.

3. **Token (optional).** From an allow-listed IP, the quiggeldy instances
   accept the connection with your user name alone; leave **Token** empty.
   For token-based access (e.g. through EBRAINS), paste the output of
   `clb_oauth.get_token()` (EBRAINS Lab) into **Token**. If the field is
   empty, an existing `QUIGGELDY_TOKEN` environment variable is used. The
   token is kept in memory
   for this session only: it is passed to the container through the helper's
   environment (`QUIGGELDY_TOKEN`, also with the `SINGULARITYENV_` and `APPTAINERENV_` prefixes), never
   on the command line, and it is not saved in the settings or written to the
   log. Anything that looks like the token in helper output is masked.

4. **Connect.** The first activation of the container's software environment
   takes around 50 seconds; later connections start in under a second (see
   [Start-up time](#start-up-time)). The status line reports each step. Once quiggeldy
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

and runs the segment as its own hardware run (`pynn.reset()` before every
segment but the first, then `pynn.run(duration)`), reading spikes and membrane
voltage right after it and shifting them to the segment's start time. Each
segment therefore starts from the neurons' resting state. (Chaining segments
with `RunCommand.APPEND`, as in the AdEx tutorial, executed and recorded only
the first chained part on the chip.) **Max i_offset** is in hardware
units (0–1022); **Stim value at max** defaults to the array maximum. Segments
that become identical after rounding are merged. Every segment costs about 0.5 s of hardware time (20 segments ≈ 12 s); very many segments (more than 1000) are allowed but slow, so the confirmation warns about them.

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

## Rheobase routine

The **Design dynamic input → rheobase** routine can run its probes on the chip:
choose the target **BrainScaleS-2 (neuromorphic)** (connect in this tab first).

- Each probe block (recovery, pulse, post-pulse window) is delivered as
  constant current, like **Deliver** does, with the tab's settings: container,
  setup, speed-up and **i_offset max**.
- Probe amplitudes are fractions (0-1) of **i_offset max**, so a probe of 0.5
  drives half of it, whatever the other probes are. Stim neuron _i_ is
  hardware neuron _i_, and the probed neuron's membrane voltage is recorded.
- Every probe's readback is added to the project as its own data group,
  `BSS-2 rheobase <timestamp>`; the routine counts the probed neuron's spikes
  in the pulse plus the post-pulse window and continues its search.
- Because each constant-current segment is its own hardware run starting from
  rest (see _Constant current_ above), every probe starts from rest too; spikes
  after the current switches off are not recorded.
- **Stop** ends the search and ignores a probe still running; it does not kill
  the shared helper.

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

- `pynn.run(runtime)` durations are hardware milliseconds; consecutive runs
  need `pynn.reset()` in between.
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

## Results on the chip

Measured on `hxcube7fpga0chip57_1` (quiggeldy.bioai.eu:11750, October 2026):

- **Chained APPEND runs do not work with this software version**: only the
  first chained part was executed and recorded (membrane data stopped after
  it; later pulses produced no spikes), whether only `constant_current_enable`
  or also `constant_current_i_offset` changed between parts. Delivery
  therefore runs one hardware run per segment.
- **One run per segment works**: each neuron fires only while its current is
  on, and `constant_current_i_offset` can change from segment to segment
  (one neuron: 200 → 0 spikes, 500 → 5, 1000 → 15 spikes per 0.1 ms).
- Recording `spikes` of the population and `v` of one neuron in the same run
  works; the membrane is sampled every ≈32 ns of hardware time
  (≈32 µs biological at speed-up 1000).
- Neurons differ: at the same `i_offset` they fire at different rates
  (2–9 spikes per 100 ms pulse at `i_offset` 500 in the staircase test), as
  expected for analog circuits.
