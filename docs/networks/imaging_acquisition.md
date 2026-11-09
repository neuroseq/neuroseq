# Calcium imaging acquisition

The **ScanImage (imaging)** tab of the **Stimulation setup** pane (toolbar
button **Image acquisition**, or **View → Show Stimulation setup**) controls a
running ScanImage MATLAB session: Focus, record or arm a recording, abort, and
grab a field of view for Cellpose. Completed recordings open in the Imaging tab.
There is no live image in NeuroSeq (streaming frames slowed ScanImage down);
watch ScanImage's own display.

Scope: **ScanImage Premium 2023.1.0**, single-plane raster imaging. Volumes,
FastZ, multi-ROI and arbitrary-line scans are not supported (a configured
Z-stack is fine with stacks disabled). Validate on the acquisition PC before
experiments; automated tests do not prove hardware compatibility or timing.

## Setup

1. Start ScanImage with the rig configuration and, in its MATLAB window, run:

   ```matlab
   matlab.engine.shareEngine('NeuroSeqScanImage')
   ```

2. Create a separate Python interpreter supported by that MATLAB release (e.g.
   3.10 for R2023a, 3.10/3.11 for R2023b; see the
   [MathWorks table](https://www.mathworks.com/support/requirements/python-compatibility.html))
   with the **matching** MATLAB Engine version, `numpy` and `tifffile`. Do not
   install it into NeuroSeq's own environment (Python 3.12+).
3. In the tab, choose that Python executable, enter the session name and click
   **Connect**. NeuroSeq only attaches to an existing session; it never starts
   MATLAB. The Python path is remembered; the recording folder is remembered per
   project.
4. Configure scanners, PMTs, channels and trigger wiring in ScanImage. Set
   **Functional channel** to the calcium indicator channel (shown in the Imaging
   tab and measured by routines; it must be saved). It resets to 1 each session.

NeuroSeq adds its `.m` files to the MATLAB path and registers its callbacks next
to existing user functions (ScanImage must be idle). Disconnecting removes them;
MATLAB keeps running.

## Recording

| Control                                   | Default / behaviour                                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Recording folder**                      | Each recording is one `<date>_<number>.tif`; **Next file** shows the name. Never overwrites. |
| **Experiment name**                       | Optional name of the recording's data group (else the file name)                             |
| **Frames**                                | 300 (routines set their own)                                                                 |
| **Save channels**                         | Channel 1; tick several to save them together                                                |
| **Wait for external acquisition trigger** | Off: software start. On: waits for ScanImage's configured trigger input.                     |
| **First-frame timeout**                   | 60 s, then abort                                                                             |

- **Focus** starts unsaved continuous imaging. Abort it before recording.
- **Record / Arm** starts or arms the recording. "Waiting for first frame" does
  not prove the hardware is ready for a TTL.
- **Abort acquisition** stops it.

For each recording NeuroSeq temporarily sets ScanImage's logging folder, file
stem and counter, frames per file (unlimited), logging averaging and
acquisitions per loop (1), saved channels and trigger, uses Grab (not Loop), and
restores your values once ScanImage is idle after completion, abort or a failed
start. A setting you change in ScanImage meanwhile is kept.

A recording completes when ScanImage reports it and the files are stable. It
must contain all requested frames; an "aborted" recording that has them all
counts as complete, other aborted ones are attached as aborted. The recording
becomes the project's selected imaging data; the selector next to **Image
acquisition** lists all imaging data (**Import Imaging Data**, **Remove imaging
data** act on the list).

Files beside each TIFF, written before the hardware command:

- `<date>_<number>.json`: request, stimulus (name, `dt`, shape, SHA-256),
  routine block (`.seq` path, SysCon2 entry) and export settings. It records the
  request, not proof of completion.
- `<date>_<number>_stimulus.npy`: the stimulus samples.

The `.nseq` stores only metadata and a frame index; keep the TIFFs at their
paths. **Save imaging data…** exports a rendered copy, not the original. GIF
keeps recorded frame intervals; other movie formats use a constant rate.

### Field of view for cell segmentation

**Grab FOV for segmentation** records 300 frames (software start, no
stimulation), saves the mean of the **Functional channel** as
`<date>_<number>_mean.tif`, and opens Cellpose on it, so masks share the
recordings' pixel grid. **Auto brightness** (default on, remembered) stretches
contrast to 0.35 % saturated pixels, like Fiji's _Enhance Contrast_; the same
adjustment is shown in the segmentation dialog and Spatial Footprints.

## Stimulation and timing

For manual runs, export, select and upload in the SysCon2 tab and arm imaging
here with the external trigger. Manual recordings are marked as unverified
synchronization, so planned stimulation markers are not drawn.

## Dynamic input routines

With the SysCon2 target and **Run blocks automatically and record them with
ScanImage**, rheobase and STDP window blocks run without manual steps:

1. Export the block, import, select and upload it in SysCon2 (1 run).
2. Arm ScanImage to wait for the external trigger, for the block plus
   **Record after block** (default 3 s).
3. Run the sequence (start condition _auto_); its TTL must trigger ScanImage.
4. Extract ΔF/F per Cellpose ROI (F0 = mean before the block's first stimulus)
   and deconvolve it with OASIS (jGCaMP8m, 0.5 s decay; events of at least
   **Spike threshold** are spikes). Options:
   - **Remove stimulation artefact** (default on): frames lit by the laser are
     replaced by the nearest clean frames.
   - **Trace extraction**: **Suite2p (motion-corrected, neuropil-corrected)**
     (default; rigid and non-rigid registration to the image the masks were
     drawn on, F − 0.7 × neuropil, ΔF/F relative to raw F) or **ROI average
     (fast)**. **Warn if Suite2p takes over** (120 s) only logs a warning.
5. Place the traces and spikes in the run's data group (e.g.
   `rheobase_2026-10-27_153313`) at the block's time.

STDP pairing blocks are not recorded. Progress appears in the toolbar progress
bar and the routine's log.

Requirements and limits:

- SysCon2 and ScanImage connected, ScanImage frame rate reported, Cellpose masks
  of the same FOV on the same pixel grid.
- A recording not finished 15 s after block plus tail is aborted and the routine
  stops (usually a missing trigger).
- Check the **Functional channel** before each run.
- Verify the trigger wiring (SysCon2 `TargetConnectorPort`) and the delay from
  sequence start to first frame on the rig.

## Interruptions

The helper is terminated if it does not answer within 30 s; MATLAB and the
scanner keep running and the acquisition state is unknown, so check ScanImage
before reconnecting. Closing NeuroSeq does not stop a running acquisition: use
**Abort acquisition** first.

## Rig acceptance checklist

Verify on the acquisition PC:

- ScanImage settings are restored after completion, abort, failed start and
  disconnect, and a following manual Grab uses them.
- File names continue across restarts; frame counts and pixel values are
  correct, including two-channel recordings.
- Playback orientation matches a known asymmetric target and the Cellpose masks.
- External trigger, first-frame timeout and abort work; a routine block's TTL
  starts the recording and the response window lines up with the pulse.
- Frame rate and dropped frames are the same with NeuroSeq connected (it polls
  status 5 times a second; each reply reports `elapsed_ms`).
- Recordings survive save/reopen and attach to their own project after a
  project switch.

MATLAB adapter tests without a microscope (separate session, no `hSI`, from the
repository root): `assertSuccess(runtests('tests/matlab'))`.
