# ScanImage acquisition on the same PC

The **ScanImage (imaging)** tab of the **Stimulation setup** pane connects to a
running ScanImage MATLAB session; the **Image acquisition** toolbar button shows
it. It starts Focus, starts or externally arms a recording, aborts an
acquisition, and opens completed TIFF files in the Imaging tab. ScanImage owns
the scanner, its own live display and the original recording.

NeuroSeq does not stream live images. Fetching frames through MATLAB ran on the
thread that also runs ScanImage and slowed acquisition down (issues #628, #651),
so the tab has controls only; watch the image in ScanImage's display windows.
Placing acquisition next to stimulation also lets the dynamic input routines
drive both (see [below](#dynamic-input-routines)).

This integration targets **ScanImage Premium 2023.1.0**, single-plane raster
imaging. Its MATLAB adapter requires validation on the actual acquisition PC;
automated Python tests do not establish hardware compatibility or timing.
Volumes, FastZ, multi-ROI and arbitrary-line acquisitions are outside this
initial implementation. A configured Z-stack is fine as long as stacks are
disabled. For its own recordings NeuroSeq sets logging averaging and
acquisitions per loop to 1, so every raw frame is saved in one acquisition, and
restores your values when the recording ends, is aborted or fails to start.
NeuroSeq uses Grab, not Loop, so the loop repeat period is not used.

## Setup

1. Start ScanImage normally and load the microscope's configuration. In that
   MATLAB window run:

   ```matlab
   matlab.engine.shareEngine('NeuroSeqScanImage')
   ```

2. Prepare a separate Python installation compatible with that MATLAB release.
   NeuroSeq itself requires Python 3.12+, so do not install an incompatible
   engine into NeuroSeq's environment. For example, MATLAB R2023a supports
   Python 3.10; R2023b also supports Python 3.11. Consult the
   [MathWorks compatibility table](https://www.mathworks.com/support/requirements/python-compatibility.html).
   Install MATLAB Engine **matching the installed MATLAB release**, plus
   `numpy` and `tifffile`, in the helper interpreter. A current unpinned engine
   package is not necessarily compatible with older MATLAB.

3. Click **Image acquisition** in the toolbar, select that Python executable and enter
   the shared session name. Click **Connect**. The helper connects only to an
   existing session with that name; it never starts another MATLAB instance.
   Required ScanImage properties and user-function registration are probed.
   NeuroSeq remembers the Python executable as the default for next time; the
   recording folder is remembered per project (and the last one used becomes
   the default for new projects).

4. Configure scanners, PMTs, saved/displayed channels and trigger wiring in
   ScanImage. Set **Functional channel** to the channel of the calcium
   indicator: it is the channel shown for recordings in the Imaging tab.

The bundled `.m` files are added to the MATLAB path for the session. The adapter
registers acquisition lifecycle callbacks alongside existing user functions;
it does not replace the user's callback table. Reconnecting reconciles the
adapter's entries without duplicating them, including after clearing MATLAB
functions. New callback registration requires ScanImage to be idle. A normal
idle disconnect removes its callbacks. It does not shut down MATLAB.

## Focus and recording

**Focus** starts ScanImage's continuous unsaved imaging, shown in ScanImage.
Abort Focus before starting a recording.

Choose a recording folder and frame count; the experiment name is optional.
**Save channels** lists ScanImage's channels and starts with the ones ScanImage
saves; tick several to record them together in one acquisition. ScanImage's own
selection is restored after the recording, like the other settings NeuroSeq
changes.

Every recording is **one TIFF directly in the recording folder**, named after the
date and a recording number that continues from the files already there, e.g.
`2026-10-22_00003.tif`; **Next file** shows the name the next recording gets.
NeuroSeq sets ScanImage's file stem to the date, its file counter to the
recording number, and frames per file to unlimited, and restores them
afterwards. A recording never overwrites an existing file. The experiment name,
which keeps its capitalisation, names the recording's data group in the project;
without one, the file name is used.

**Record / Arm** enables ScanImage TIFF logging. When
**Wait for external acquisition trigger** is selected, ScanImage waits for
its configured trigger input. Otherwise it starts through software.
The first-frame timeout requests an abort if no acquisition start is observed.
"Waiting for first frame" is not proof that hardware is ready to accept a TTL.

NeuroSeq reports state from ScanImage, waits for completed/aborted acquisition
events and readable, stable files, then indexes the saved channels and frame
timestamps. A successful complete acquisition must contain the requested frame
count. Should ScanImage still split a recording into several files, they are grouped as
one recording. Aborted recordings with readable complete
channel groups can also be attached, labeled as aborted.

After completion, abort, or a failed start, the adapter restores the previous
logging folder, filename stem/counter, frames per file, logging enable,
external-trigger enable, and frames-per-slice settings once ScanImage is idle. If the terminal callback
precedes the idle state, the next status poll retries restoration; idle
Disconnect also restores pending settings, as do the next Preview and Record if
ScanImage became idle without any terminal callback. Verify restoration on your rig,
including after abort, before starting another manual Grab. Acquisition metadata
retains the settings used during the recording.

A completed acquisition is added to the project that requested it and becomes
its selected imaging data. Imported imaging data and earlier recordings stay in
the list.
The tab's **Functional channel** picks the channel shown for recorded
playback. It is kept when the recording saved it; otherwise it changes to the
first saved channel and displays a message.

The imaging data selector next to **Image acquisition** lists all imaging data in
the project: imported TIFF/GIF/movie files and finished recordings. **Import
Imaging Data** adds to this list, and **Remove imaging data** removes only the
selected entry.
Files remain external to `.nseq`; moving them without restoring their paths
makes playback unavailable. Projects save the run metadata and index, not the
raw movie. Frame indexes use compact columns and file indices and are stored
once per run; the selected movie references that run. Full frame descriptions
remain in the original TIFFs and can be reread for future alignment analysis.
Save the project to retain its acquisition history.

A small, human-readable `<date>_<number>.json` beside the TIFF stores the
request, the stimulus it relates to (name, sample interval, shape and SHA-256 of
its samples, not the samples themselves), the block of a dynamic input routine
(with the path of its `.seq` file and its SysCon2 sequence-manager entry), and
the SysCon export settings, written before the hardware command. It aids
recovery if NeuroSeq exits before the project is saved, and is removed again if
ScanImage refuses to start. Its requested state is not evidence that the
acquisition completed.

The existing **Save imaging data…** action exports a rendered derivative. It does not
replace ScanImage's original TIFF recording. Recorded ScanImage playback uses channel-specific frame
timestamps; brightness conversion affects only the display/derivative. GIF
exports use the recorded inter-frame intervals, subject to GIF timing precision.
Other movie formats use a constant frame rate and do not preserve irregular
acquisition timing; retain the TIFF timestamps for analysis.

## Stimulation and timing

For manual recordings, use the SysCon2 tab to export/select/upload stimulation.
Imaging can be armed here and triggered through the configured physical input.
Manual recordings are marked as having unverified synchronization: software
command arrival times are not measured stimulation times, so planned
stimulation crosses are suppressed for them.
The 2023 [auxiliary trigger feature](https://archive.scanimage.org/SI2023/Concepts/Triggers/Auxiliary%2BTrigger.html)
can record stimulation markers on supported resonant systems. Their frame
descriptions remain in the original TIFFs for a future calibrated alignment
step.

## Dynamic input routines

With the SysCon2 target and **Run blocks automatically and record them with
ScanImage** ticked in their hardware settings, the rheobase and STDP window
routines run each block without manual steps:

1. Export the block, import it into SysCon2, select it and upload it (1 run).
2. Arm a ScanImage recording that **waits for the external trigger**, with
   enough frames for the block plus the **Record after block** time at
   ScanImage's frame rate.
3. Run the sequence (start condition _auto_); its TTL output must trigger
   ScanImage, so the recording starts with the block.
4. When the recording is complete, extract the ΔF/F trace of every cell with the
   project's Cellpose masks (frames lit by the stimulation laser are
   interpolated; F0 is the mean before the block's first stimulus), deconvolve
   it into spikes with OASIS (as Suite2p does, with jGCaMP8m's 0.5 s decay; an
   event of at least the **Spike threshold**, in ΔF/F, is a spike), add both to
   the recording's data group, and measure the response.

STDP pairing blocks need no data, so they are run without recording. The
routine needs SysCon2 and ScanImage connected, ScanImage's frame rate reported,
and a Cellpose segmentation of the same field of view whose masks share the
TIFFs' pixel grid. These recordings are marked `stimulation_trigger`
synchronization. Verify the trigger wiring (the SysCon2 configuration's
`TargetConnectorPort`) and the delay between sequence start and the first frame
on the rig before relying on the measured windows.

## Interruptions and rig acceptance

An unresponsive helper is terminated after 30 seconds; MATLAB and the scanner
are not killed. Acquisition state is then unknown. Inspect ScanImage before
reconnecting. Closing NeuroSeq stops its helper transport; it does not promise
to stop an independently running microscope acquisition. Use **Abort acquisition**
and verify idle before closing when you intend to stop imaging.

Before experimental use, verify on the acquisition PC:

- Previous ScanImage logging/trigger/frame-count settings are restored after
  completion, abort, failed start, and idle disconnect; a subsequent manual Grab
  uses the original settings.
- The probe succeeds on the exact ScanImage/MATLAB build and reports settings.
- Recorded playback orientation matches a known asymmetric target and the
  Cellpose masks of the same field of view.
- Recordings are written as `<date>_<number>.tif` in the chosen folder, the
  number continues across restarts, and ScanImage's own stem, counter and
  frames-per-file settings are restored afterwards.
- A short software-triggered recording has the requested frames and original
  pixel values, including a two-channel recording and a file rollover.
- External triggering, first-frame timeout and abort behave as expected.
- An automatic routine block arms ScanImage, the sequence's TTL starts the
  recording, and the response window lines up with the delivered pulse.
- ScanImage runs as fast with NeuroSeq connected as without: compare its frame
  rate and dropped frames during Focus and a recording. Each status reply
  reports `elapsed_ms`, the time the call spent in ScanImage's MATLAB thread;
  NeuroSeq polls status five times a second. (The removed live view
  base64-encoded every frame in MATLAB, about 160 ms per 512×512 frame on
  that same thread, which is what slowed ScanImage down.)
- Completed and interrupted recordings survive project save/reopen, and a
  recording finishing after a project switch attaches to its originating project.
- Hardware stimulation markers and image timestamps agree before claiming any
  synchronization accuracy.

## Future remote transport

`ScanImageConnection` exchanges request IDs and JSON operations with a separate
helper process. Commands, lifecycle status and recording indexes are independent
of Qt rendering. A future remote transport can implement the
same operations on the acquisition PC, with authentication, bounded messages,
connection-loss reconciliation, and explicit recording transfer/path mapping.
MATLAB Engine itself is used only locally. The current implementation opens no
network server and relies on local filesystem access for recording verification.

Do not assume the built-in ScanImage TCP server speaks this protocol: its
[2023 documentation](<https://archive.scanimage.org/SI2023/Premium%2BFeatures/ScanImage%2BRemote%2BControl%2B(TCP_IP).html>)
describes a MATLAB client. Native [ZeroMQ streaming](https://docs.scanimage.org/Premium%2BFeatures/ZeroMQ.html)
requires ScanImage Premium 2025 or later and is not used here.

## Adapter regression tests without a microscope

In a separate MATLAB session with no `hSI` variable, from the repository root:

```matlab
results = runtests('tests/matlab');
assertSuccess(results);
```

These tests use simulated ScanImage components to exercise settings restoration,
callback registration, single-acquisition validation, and file naming.
They do not establish compatibility with the installed ScanImage version or
hardware timing.
