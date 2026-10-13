# ScanImage acquisition on the same PC

The calcium-video widget can connect to a running ScanImage MATLAB session,
show live images, start or externally arm a recording, abort an acquisition,
and open its completed TIFF files. ScanImage owns the scanner and original
recording. NeuroSeq's preview is sampled at up to 5 updates/second and is not
the recording stream.

This integration targets **ScanImage Premium 2023.1.0**, single-plane raster
imaging. Its MATLAB adapter requires validation on the actual acquisition PC;
automated Python tests do not establish hardware compatibility or timing.
Volumes, FastZ, multi-ROI and arbitrary-line acquisitions are outside this
initial implementation. Logging averaging must be 1 to retain raw frames.

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

3. In the imaging panel's video tab, select that Python executable and enter
   the shared session name. Click **Connect**. The helper connects only to an
   existing session with that name; it never starts another MATLAB instance.
   Required ScanImage properties and user-function registration are probed.

4. Configure scanners, PMTs, saved/displayed channels and trigger wiring in
   ScanImage. Choose a displayed channel in NeuroSeq for live viewing.

The bundled `.m` files are added to the MATLAB path for the session. The adapter
registers acquisition lifecycle callbacks alongside existing user functions;
it does not replace the user's callback table. A normal idle disconnect removes
its callbacks. It does not shut down MATLAB.

## Preview and recording

**Preview (Focus)** starts continuous unsaved imaging. **Live view** chooses
whether incoming preview frames are shown; disabling it does not stop imaging.
The playback slider has no effect on the live image. Abort Focus before
starting a recording.

Choose a recording folder, experiment name and frame count. **Record / Arm**
creates a unique run directory and enables ScanImage TIFF logging. When
**Wait for external acquisition trigger** is selected, ScanImage waits for
its configured trigger input. Otherwise it starts through software.
The first-frame timeout requests an abort if no acquisition start is observed.
"Waiting for first frame" is not proof that hardware is ready to accept a TTL.

NeuroSeq reports state from ScanImage, waits for completed/aborted acquisition
events and readable, stable files, then indexes the saved channels and frame
timestamps. A successful complete acquisition must contain the requested frame
count. Split TIFF files are grouped as one recording. Aborted recordings with
readable complete channel groups can also be attached, labeled as aborted.

The **Recorded acquisitions** selector opens earlier recordings in the project.
Files remain external to `.nseq`; moving them without restoring their paths
makes playback unavailable. Projects save the run metadata and index, not the
raw movie. Save the project to retain its acquisition history.

A `<run-id>.neuroseq.json` sidecar beside the run directory stores the original
request, stimulus snapshot, and available SysCon export settings before the
hardware command. It aids recovery if NeuroSeq exits before the project is
saved. Its requested state is not evidence that the acquisition completed.

The existing **Save video…** action exports a rendered derivative. It does not
replace ScanImage's original TIFF recording. Live preview is not exportable
through this action. Recorded ScanImage playback uses channel-specific frame
timestamps; brightness conversion affects only the display/derivative.

## Stimulation and timing

Use the existing SysCon widget to export/select/upload stimulation. Imaging can
be armed from the video widget and triggered through the configured physical
input. This version does **not** automatically run SysCon or assert that its
selected sequence matches the editor snapshot. The snapshot records the
planned input at the time imaging was requested.

Recordings are marked as having unverified synchronization. Planned
stimulation crosses are suppressed for these recordings and for live images:
software command arrival times are not measured stimulation times.
The 2023 [auxiliary trigger feature](https://archive.scanimage.org/SI2023/Concepts/Triggers/Auxiliary%2BTrigger.html)
can record stimulation markers on supported resonant systems. Their frame
descriptions are preserved by the recording index for a future calibrated
alignment step. Verify the rig's trigger topology before automating joint runs.

Movie completion does not automatically resume experimental routines that need
calcium traces or inferred activity. That requires a separate analysis handoff.

## Interruptions and rig acceptance

An unresponsive helper is terminated after 30 seconds; MATLAB and the scanner
are not killed. Acquisition state is then unknown. Inspect ScanImage before
reconnecting. Closing NeuroSeq stops its helper transport; it does not promise
to stop an independently running microscope acquisition. Use **Abort acquisition**
and verify idle before closing when you intend to stop imaging.

Before experimental use, verify on the acquisition PC:

- The probe succeeds on the exact ScanImage/MATLAB build and reports settings.
- Preview orientation matches a known asymmetric image and channel selection.
- A short software-triggered recording has the requested frames and original
  pixel values, including a two-channel recording and a file rollover.
- External triggering, first-frame timeout and abort behave as expected.
- The UI remains responsive and ScanImage logging does not lose frames under
  normal and deliberately slowed preview consumption.
- Completed and interrupted recordings survive project save/reopen, and a
  recording finishing after a project switch attaches to its originating project.
- Hardware stimulation markers and image timestamps agree before claiming any
  synchronization accuracy.

## Future remote transport

`ScanImageConnection` exchanges request IDs and JSON operations with a separate
helper process. Commands, lifecycle status, preview frames and recording indexes
are independent of Qt rendering. A future remote transport can implement the
same operations on the acquisition PC, with authentication, bounded messages,
connection-loss reconciliation, and explicit recording transfer/path mapping.
MATLAB Engine itself is used only locally. The current implementation opens no
network server and relies on local filesystem access for recording verification.

Do not assume the built-in ScanImage TCP server speaks this protocol: its
[2023 documentation](<https://archive.scanimage.org/SI2023/Premium%2BFeatures/ScanImage%2BRemote%2BControl%2B(TCP_IP).html>)
describes a MATLAB client. Native [ZeroMQ streaming](https://docs.scanimage.org/Premium%2BFeatures/ZeroMQ.html)
requires ScanImage Premium 2025 or later and is not used here.
