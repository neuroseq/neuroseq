# Holographic stimulation

NeuroSeq uploads and runs `.seq` sequences on the HOLO4D two-photon holographic
stimulation rig through SysCon2, which must run on the same Windows PC (local
named pipe). Sequences come from [Holography export](../input_design/holography_export.md).

## Stimulation setup panel

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
  see [ScanImage acquisition](imaging_acquisition.md#dynamic-input-routines).
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

If the input array already holds input, **Start** offers **Clear and start
over**, **Continue after last input**, or **Start in new arrays** (new empty
array and data group named after the run, e.g.
`Exc_stimulation rheobase_2026-10-22_143012`).

Each run saves its blocks in its own folder in the export folder, e.g.
`SysCon exports\rheobase_2026-10-22_143012\rheobase_2026-10-22_143012_001.seq`
(with PNG and sidecar), imported into SysCon2 under the same name.

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
