# Threaded External Simulation

## Summary

External simulation dispatch in NeuroSeq now runs off the Qt main thread.
The GUI performs the usual pre-flight bookkeeping on the active project,
then hands the actual simulator call to a dedicated `QThread` worker.

This replaces the earlier partial fix where the main thread stayed inside
the simulator and manually pumped Qt events from the progress callback.

## Components

### `ExternalSimWorker`

`src/neuroseq/gui.py` defines `ExternalSimWorker` as a module-level
`QtCore.QObject`.

Responsibilities:

- receive the resolved external simulator function and its immutable
  dispatch inputs
- run the simulator on a dedicated `QThread`
- poll a shared `threading.Event` from the progress callback to support
  cooperative cancellation
- communicate back to the GUI exclusively through Qt signals

Signals:

- `progress(float)`: reports completion fractions in `[0, 1]`
- `finished(object)`: emits the resulting `Experiment`
- `cancelled()`: emitted when the cancel event is observed and the worker
  raises `SimulationCancelledError`
- `failed(str)`: emits a stringified exception message

The worker has no QObject parent so it can be moved cleanly to a worker
thread.

## Dispatch Flow

`Gui._do_external_simulation()` now performs only dispatch-time work on the
main thread:

1. reject dispatch when no external simulator is loaded
2. reject re-entry if another external simulation is already in flight
3. compute the cumulative target duration and tile stimulation arrays
4. rebuild the stim-editor view of the project and extract `stims_dict`
5. flatten external simulation parameters
6. detect whether the simulator accepts a `progress=` keyword argument
7. capture the current project as the dispatch target
8. flip the shared progress dialog to `NonModal` (it is `WindowModal` by
   default for the other long-running paths that block the main thread)
   and show it with a temporary Cancel button
9. connect `progress_dialog.canceled` to a fresh `threading.Event`
10. create and start a fresh `QThread` + `ExternalSimWorker`

The method returns immediately after `thread.start()`.

## Progress and Cancellation

The worker's local progress callback:

- checks the cancel event before emitting progress
- raises `SimulationCancelledError` when cancellation is requested
- emits `progress(float)` otherwise

The main-thread slot `_on_external_sim_progress()` only clamps the fraction
and updates the shared progress dialog. No manual `processEvents()` call is
used anymore because the main thread is no longer blocked inside the
simulator.

## UI Gating

`Gui._set_simulation_actions_enabled(enabled)` disables both simulation
entry points while a worker is active:

- `continue_simulation_action`
- `_do_external_simulation_action`

This prevents accidental re-entry from menus or toolbars while the worker
thread is still running.

## Terminal Handling

All terminal paths call `_teardown_external_sim()` first. That helper:

- restores the override cursor
- restores the progress dialog's modality to `WindowModal` for the
  next synchronous user of the shared dialog
- removes the temporary cancel button from the shared progress dialog
- disconnects the dialog's `canceled` signal from the active
  `threading.Event`
- clears `_external_sim_thread`, `_external_sim_worker`,
  `_external_sim_cancel`
- re-enables the simulation actions

After teardown:

- `_on_external_sim_finished()` resets the progress dialog and routes the
  resulting `Experiment`
- `_on_external_sim_cancelled()` resets the dialog and shows an
  informational cancellation message
- `_on_external_sim_failed()` resets the dialog and shows an error message

## Dispatch-Project Capture

The worker completion path must not blindly append to `self.project`.
Users can swap projects while a long external simulation is still running.

`_do_external_simulation()` therefore captures the active project at
dispatch time and stores it on the `Gui` instance for the finished slot.

Completion semantics:

- if the current project is still the dispatch-time project, NeuroSeq uses
  the normal `_append_experimental_data()` GUI flow
- if the user has switched projects, the finished experiment is appended
  directly to the original project, and the GUI shows a warning dialog

This prevents long-running external simulations from silently landing in the
wrong project.

## Shutdown Behaviour

`Gui.closeEvent()` now drains an in-flight external simulation before the
rest of the widget teardown continues.

Behaviour:

- set the active cancel event if a worker thread is still running
- wait up to 2 seconds for the worker thread to stop cooperatively
- if it does not stop in time, request `thread.quit()` and wait again

This ensures the worker does not outlive the GUI during shutdown.

## Compatibility

External simulator compatibility remains unchanged:

- legacy simulators that only accept
  `simulate(stims_dict, simulation_time, params=None)` still work
- newer simulators that advertise `progress=` receive the progress callback
- cancellation remains cooperative and depends on progress polling

## Validation

Automated coverage includes:

- worker-level signal and thread-semantics tests
- GUI-level dispatch, cancel, idempotency, project-swap, and close-event
  tests
- existing external-simulation regression tests updated to wait for worker
  completion

Manual rollout on real hardware is still required for responsiveness and
shutdown verification during long ELiSe runs.

## Known Limitations

These are intentional and acceptable, but should be remembered when
extending the threaded path:

- **Single-flight.** Only one external simulation runs at a time. Re-entrant
  dispatch is a no-op (logged) by design. Users who want to compare multiple
  parameter settings must script their runs sequentially.

- **Pure Python + NumPy in the inner loop.** The ELiSe simulator does not
  use Numba JIT. CPU utilisation by the worker is bounded by how often the
  inner loop releases the GIL via NumPy C calls. A future Numba pass on
  `elise.Network.__call__` would multiply throughput 10–100× without
  changing the threading design.

- **Stim arrays pass by reference.** `_do_external_simulation()` builds
  `stims_dict` from live `StimArray.values` references, not copies. If the
  user edits a stim through the stim editor while a sim is running, the
  worker thread will observe the mutation mid-run. The actions-disable
  gate only locks the _Run_ buttons; the stim editor remains live. The
  consequence is a corrupted simulation (not a crash) and the workflow is
  unusual. Hardening — e.g. `v.stim_array.values.copy()` at dispatch — is
  deferred until a user reports the issue.

- **Simulator state on the project.** The worker mutates
  `project.external_simulator.simulated_time_seconds` (and any other
  internal state on the simulator instance) while running. No main-thread
  code currently reads those fields during a run. If such a read is ever
  added, locking will be required.
