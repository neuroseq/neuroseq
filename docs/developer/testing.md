# Testing

A proper development of the code implies the use of tests with the `pytest` framework. Tests are located in the `tests/` folder.

## To run all tests

```bash
pytest
```

## To run a specific subset of tests

```bash
pytest tests/test_gui.py
```

## SysCon2 control

The protocol, connection manager, panel, and routine tests use fake transports and
recordings. On Windows, `tests/test_syscon2_client.py` also opens a real local named
pipe to check timeouts when a server never replies or drains a write. These tests
do not require SysCon2 or a connected stimulation device.

```bash
uv run pytest tests/test_syscon2_client.py tests/test_syscon2_connection_manager.py tests/test_syscon2_panel.py tests/test_rheobase_dialog.py tests/test_stdp_window_dialog.py
```

Real-rig checks are still needed for command execution and acquisition alignment.
The SCI v1.2 manual confirms the `Sequence Manager` instance name for the Holo4D
profile and that Runs = 0 means endless repetition, but does not specify the wire
width of the Runs integer. The client currently uses a 32-bit DWord.

Pipe commands have a 30-second deadline, configurable through
`NamedPipeTransport(timeout=...)`. Disconnect and shutdown cancel pending I/O.
A timeout closes the connection and requires an explicit reconnect; it does not
confirm that a queued Stop command reached the device.

### Real-rig checklist

This updates the [export/control test](https://github.com/ssciwr/neuroseq/pull/573#issuecomment-6046355724)
and [rheobase test](https://github.com/ssciwr/neuroseq/pull/573#issuecomment-6046356049)
in the PR. Those comments describe the overall workflow correctly, but the
following details matter when testing the current implementation.

1. Run NeuroSeq and SysCon2 on the same Windows PC, with the Holo4D profile and a
   verified Cellpose-to-stim-ID mapping. Connect opens the pipe; it does not prove
   the laser is connected or ready. Click **Get Status** to discover lightsources and retrieve status; the selector
   prefers 940 nm. Check the mode and hardware state in SysCon2. Direct laser
   mutation controls are hidden.
2. Prepare a short pattern and use **Export with settings…**. Select the intended input
   array, wavelength, target power, LUT, and output directory. The export target
   power is in mW. LUT percentages are divided by 100 in `.seq` files; verify
   this scaling on the rig. Live status uses the unit configured in SysCon2,
   which SCI does not identify.
3. Run **Check Input Sanity** and resolve every reported violation. The exact
   required timestep is **520.8125 µs** (`0.0005208125` seconds), not exactly
   520.8 µs. Let the dialog resample it. Around 20 ms is a starting pulse duration,
   but it does not by itself guarantee valid SLM timing; inspect the generated
   overview and debug CSV after the sanity fixes.
4. Generate the script. Filenames use the dialog's **Script name**, whose initial
   value follows `export_seq_<date>_<n>`, rather than the array name.
   The PNG is a generated preview. The panel shows whether import is pending,
   confirmed, failed, or not attempted (offline). Red target crosses appear only
   after acknowledgement of that preview's import and exclude skipped neurons.
   They do not show remote selection, upload, or live illumination. Confirm the
   sequence also appears in SysCon2 itself before proceeding.
5. Select the imported sequence, set **Runs = 1**, choose the start condition and
   trigger edge, then click **Upload**, wait for "Press Run", then click **Run** and confirm. Verify the
   progress label moves through uploading, hardware readiness, starting, and running. A triggered
   run must wait for its trigger. Stop during upload must not send Run afterward.
   SCI reports phases rather than percentage completion. Validate actual timing,
   pattern, and delivered power using the rig's controlled bench-test procedure.
   Configure emission, shutter, and laser intensity in SysCon2 itself.
6. Edit the array and test **Export current array to stimulation setup**. It reuses the last
   dialog export's settings but writes `export_seq_<date>_<n>.seq` and its diagnostic
   files to a temporary directory, not the dialog's chosen output folder. It
   opens the full export dialog if settings are unavailable. As above, verify
   the import, then select and use Upload then Run for the changed sequence.
7. For rheobase, select the SysCon2 target, disable binary intensity export, use
   amplitudes within 0–1, and choose
   the response window from the actual acquisition frame rate. At 30 Hz,
   100–150 ms spans several frames; neither 30 Hz nor a particular useful response
   window is guaranteed by the SCI protocol. For raw TIFF import, use the Suite2p
   processing option to obtain deconvolved spike data. Verify the sample rate,
   spike threshold, and neuron mapping in the import dialog.
8. Start the routine. For **every** block, refresh the sequence list, select the
   newly imported block, set **Runs = 1**, then **Upload**, **Run** and acquire. The host only
   imports; it does not select, upload, or run. Arm a triggered run before firing
   the acquisition trigger, and verify its alignment with the recording. Manual
   clicks alone do not establish a shared time origin.
9. Import each probe into a **new experiment with a unique name**, starting at
   time zero for that block. Avoid appending it after a prior recording: the
   routine measures times relative to the current block, without subtracting an
   import offset. A repeated experiment name can make the import dialog select
   **Apply to existing experiment** automatically; check this choice explicitly.
10. Return to the routine and click **Calcium data imported**. Fresh-data selection
    is resolved when that button is clicked, so a new recording need not already
    be selected immediately after import. Existing recordings are rejected,
    including on the first probe. Importing data must preserve the open routine,
    selected stimulus, and export settings; this now has a GUI regression test.
11. Check pause, stale-data rejection, and stopping. **Routine Stop** ends the
    software search and discards its pending result; it does not send a hardware
    Stop command or retract an already queued/imported block. Use **Holo4D Stop**
    to request a hardware stop, and verify it on the rig. If communication stalls,
    a queued Stop cannot be assumed delivered; Disconnect cancels communication,
    not stimulation.

Record the script/debug CSV, selected sequence, Runs, trigger settings, wavelength,
power settings, actual frame rate, import threshold, and per-probe response.
Confirm the Runs wire representation on the real system; the Sequence Manager
instance name is documented in the manual and is no longer an unresolved naming
assumption. The same per-block import/time-origin rules apply to the STDP target;
pairing blocks require acknowledgement after execution, but no new recording.

## Linux teardown crash with Conda

If tests report success and then segfault during process exit, inspect the native
backtrace. On the development host, this occurred in Conda's `libstdc++` exit
handlers after graphics libraries unloaded. Running with the system C++ runtime
resolved the crash without changing application code:

```bash
conda run --no-capture-output -n neuroseq env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 QT_QPA_PLATFORM=offscreen QT_API=pyqt6 PYTEST_QT_API=pyqt6 .venv/bin/python -m pytest
```

The library path is specific to Debian/Ubuntu x86-64; check the system library
location before using this workaround elsewhere.

## Pre-commit

pre-commit run --all-files

Export validation is shared by the dialog, quick export, and routine host. Timing
fixes must first be reviewed and applied with **Check Input Sanity**. Other
warnings (including skipped neurons, missing power calibration, and power clamping) require confirmation
before a dialog or quick export can proceed; unattended routine exports reject
these warnings. Existing dialog export files require overwrite confirmation.
An import uses a private snapshot, so regenerating the same filename cannot
change an already queued request. Disconnecting or changing projects clears
preview import confirmation and target markers. Importing an arbitrary file
clears the generated preview association because its targets are unknown.
