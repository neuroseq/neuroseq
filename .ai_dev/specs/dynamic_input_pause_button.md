# Pause/Resume for closed-loop routine dialogs

`InputDesignerDialog`, `RheobaseDialog`, and `StdpWindowDialog` (all under
"Design dynamic input") each have a **Pause**/**Resume** toggle button next
to Stop, letting the user hold a running routine between iterations without
losing the iteration currently in flight.

## Behavior

- Clicking **Pause** while a routine is running:
  - Any iteration already in progress (a Brian2 `Network.run()` call and its
    resulting measurement/plot update) completes normally.
  - The _next_ simulation request is captured instead of being sent to the
    host, and the button label switches to **Resume**.
- Clicking **Resume** immediately sends the captured request and the
  closed loop continues exactly as if it had never paused.
- **Stop** and closing the dialog always fully reset pause state (see
  below), regardless of whether the routine was paused.
- Pause is soft: it never cancels an in-flight Brian2 run. The existing
  "Cancel" button on the Brian2 progress dialog remains the only way to
  abort a run already in progress; the two features don't interact.

## Implementation

All three dialogs share the same closed-loop shape: every iteration, in
every phase (including each dialog's one-off "run 0.001s to create trace
monitors" warm-up step in `start()`), asks the host to run a simulation
through exactly one shared method, `_request_simulation(seconds)`. Pause is
implemented by intercepting only that one choke point - no dialog-specific
phase logic needed changes.

Per dialog (`input_designer_dialog.py`, `rheobase_dialog.py`,
`stdp_window_dialog.py`):

- `__init__` adds `self._paused = False` and
  `self._pending_simulation_seconds: float | None = None`.
- `_request_simulation(seconds)` checks `self._paused` first: if paused, it
  stores `seconds` in `self._pending_simulation_seconds`, logs a message,
  and returns without emitting `simulation_requested`.
- A new `self.btn_pause` (`QDialogButtonBox`, placed between Stop and Clear
  input array) is enabled whenever Stop is enabled (i.e. while `_running`),
  and disabled/reset by `_finish()`.
- `toggle_pause()` flips `self._paused`, updates the button's text
  ("Pause"/"Resume"), and on transition back to unpaused, if a request was
  captured, sends it (`self._awaiting_simulation = True;
self.simulation_requested.emit(seconds)`).
- `_finish()` resets `_paused` and `_pending_simulation_seconds` alongside
  the existing `_running`/`_awaiting_simulation` resets, so Stop, natural
  completion, and closing the dialog while running all leave pause state
  clean.

## Known accepted edge case

If the user clicks Pause at the exact moment a routine was about to call
`_finish(...)` instead of requesting another simulation (e.g. it just
converged or hit its iteration cap), Pause is a no-op for that run - there
is nothing left to defer. No special-casing was added for this; it's
harmless.

## Tests

`tests/test_input_designer_dialog.py`, `tests/test_rheobase_dialog.py`,
`tests/test_stdp_window_dialog.py` each add one test
(`test_*_pause_defers_next_simulation_and_resets_on_finish`) covering:
pausing defers `_request_simulation`'s emission (`_pending_simulation_seconds`
set, no `simulation_requested` signal fired), resuming immediately emits
the deferred request, and `_finish()` resets `_paused`,
`_pending_simulation_seconds`, and the pause button's enabled state/text.
