# Threaded Brian2 Simulation

## Summary

Native Brian2 simulation runs no longer execute on the Qt main thread.
When a project does not use `external_simulate_path`, the Run action
dispatches `Project.run_simulation()` through the existing
`BackgroundIOWorker` infrastructure.

This keeps the GUI event loop responsive while Brian2 compiles and runs
the model.

## Dispatch Path

`Gui.btn_continue_simulation_clicked()` now branches like this:

- external simulator configured: use the existing threaded external
  simulation path
- no external simulator configured: call
  `Gui._do_brian2_simulation_threaded()`

The old inline `try/except` around `project.run_simulation(...)` is gone.

## Worker Reuse

Brian2 does not introduce a new worker class.

`Gui._do_brian2_simulation_threaded()` builds a closure compatible with
`BackgroundIOWorker`:

- capture the current `project`
- capture the dispatch-time `stim_padding_secs`
- adapt Brian2's 4-argument `report` callback to the worker's
  `report(percent: int)` callback

The closure returns `None`. Completion is handled on the main thread by
the `on_success` callback passed into `_dispatch_io_task()`.

## Progress Dialog Behaviour

The shared `progress_dialog` is shown non-modally through
`_dispatch_io_task()`.

Label behaviour:

- initial label: `Compiling Brian2 model...`
- first Brian2 progress callback: queued label update to
  `Simulating Brian2...`

The compile phase can legitimately sit at `0%` for a while because
Brian2 does not emit progress during cython compilation. No synthetic
compile progress is injected.

## Cross-Thread UI Safety

Brian2 invokes its `report` callback on the worker thread.

The percent update is safe because `BackgroundIOWorker` emits it through
its Qt signal and the GUI receives it via `_on_io_progress()`.

The label flip is also main-thread-safe:

- the worker callback uses `QtCore.QMetaObject.invokeMethod(...)`
- the call is queued onto the progress dialog's thread
- the widget itself is never touched directly from the worker thread

## UI Gating

Brian2 simulation dispatch reuses `_set_io_actions_enabled()`.

While a simulation is running:

- the File menu is disabled
- the File toolbar is disabled
- the simulation actions are disabled

This prevents save/load/import and additional simulation dispatches from
racing a worker that is mutating `project.b2_model`.

## Success and Failure

On success, the main-thread completion callback logs completion and calls:

- `Gui._update_data(keep_undo_history=True)`

On failure, the generic IO failure path is reused, but
`_dispatch_io_task()` now accepts per-dispatch failure metadata so the
Brian2 dialog is simulation-specific:

- title: `Simulation error`
- body prefix: `Could not run simulation:`

Save/load/import continue to use the existing file-operation wording.

## Re-entrancy

Brian2 dispatch uses the existing `_io_worker is not None` guard.

If the user tries to start another long-running operation while a Brian2
simulation is already in flight, the GUI shows a `Busy` information
dialog and ignores the second request.

## Shutdown

`Gui.closeEvent()` already drains `_io_thread`, and Brian2 now uses that
same path.

Close-time behaviour:

- if the worker thread is still running, the GUI waits for it
- if it does not finish within the existing timeout, the thread is
  force-quit

This is a shutdown safety measure, not a user-facing cancellation API.

## Non-Goals

- no Cancel button for Brian2 runs
- no new Brian2-specific worker lifecycle
- no change to external simulation threading
- no synthetic progress during compile

## Validation

Coverage for the shipped path includes:

- worker-thread dispatch for Brian2 GUI runs
- progress-dialog label transition
- `_update_data(..., keep_undo_history=True)` on success
- simulation-specific failure dialog wording
- re-entrancy blocking
- close-event drain of an in-flight Brian2 worker

Manual hardware validation is still required for real responsiveness
during longer native Brian2 runs.
