# Shared closed-loop backend for the three dynamic-input routines

`RheobaseDialog`, `StdpSynapticTuningDialog`, and `StdpWindowDialog` each
drive a closed-loop protocol (append a stim block, run Brian2 forward,
react to the result, repeat) through a Qt-free `RoutineBackend` subclass
that owns its routine's full decision graph, plus two small shared,
Qt-free modules for block-appending and measurement.

## Behavior

No user-visible change. This is an internal architecture change: the same
routines behave identically, with one real bug fix along the way (see
"Gap-timing fix" below).

## Architecture

### `src/neuroseq/stimulation/dynamic_input_routine.py`

- `SimulationHost` (Protocol): what a backend needs from its dialog -
  `request_simulation(seconds)` and `append_block(block) -> int`.
- `DynamicInputDialog` (Protocol): the contract
  `StimEditor._dynamic_input_dialogs()` assumes of every cached routine
  dialog (`simulation_finished`/`simulation_failed`/`refresh`/`apply_theme`).
- `RoutineBackend`: Qt-free run/pause/finish plumbing shared by all three
  routines - the `running`/`awaiting_simulation`/`paused`/
  `pending_simulation_seconds`/`phase` flags, `begin()`, `toggle_pause()`,
  `request_simulation()`, `simulation_finished()`/`simulation_failed()`,
  `finish()`, and callback hooks (`on_log`, `on_finished`, `on_advance`,
  `on_measurement`) a dialog wires to its own Qt-touching methods. Each
  routine subclasses this and adds its own domain state, phase graph, and
  `_advance()`.
- `backend_property(name)`: a small property factory dialogs use to expose
  backend-owned state through their existing private-attribute names (e.g.
  `RheobaseDialog._low = backend_property("low")`), so most pre-existing
  test assertions on those names kept working unchanged through the
  migration.

### `src/neuroseq/stimulation/dynamic_input_blocks.py`

- `append_block(editor, stim, block) -> int`: trims `stim` back to the
  current simulation head (`editor.reset_stim_array()`) before writing
  `block` immediately after it, then returns the sample index written at.
  Does not touch the live view/camera at all - each dialog's own
  "follow new input" behavior (`chk_follow_input`/`focus_time_range`) stays
  entirely dialog-side, called right after this function returns.
- `run_to_sample(project, stim, stop_sample) -> float`: seconds to
  simulate forward to reach `stop_sample`.

### `src/neuroseq/stimulation/dynamic_input_measurement.py`

- `EPSPMeasurement`, `NoEpspDetected` (a `ValueError` subclass).
- `measure_spike_count(...)`, `measure_epsp(...)`: shared trace-windowing
  and detection logic used by all three routines' measurement steps.

### Per-routine backends

Each dialog builds a frozen `<Routine>RunConfig` dataclass once from its
widget values (in its thin `start()`/`_begin_*` method) and hands it to
its backend's `begin_*(config, ...)`. From then on the backend reads
`self.config`/`self._cfg()` instead of touching any widget, and reaches
the dialog only through the `SimulationHost` protocol (`append_block`/
`request_simulation`) and the `on_log`/`on_finished`/`on_advance`/
`on_measurement` (and, for `StdpWindowBackend`, `on_restore_baseline`)
callback hooks.

- **`RheobaseBackend`** (`src/neuroseq/rheobase_dialog.py`): owns the
  bisection search / duration sweep. `begin_search(config)`, `probe()`,
  `record_probe_result(spike_count)` is the entry point the dialog calls
  after measuring a probe's spike count.
- **`StdpSynapticTuningBackend`** (`src/neuroseq/stdp_synaptic_tuning_dialog.py`):
  owns the EPSP-targeting protocol (adaptive/fixed pairing strategy).
  `begin_protocol(config)`, `probe()`, `pairing(potentiate)`,
  `record_measurement(amplitude, units)` is the entry point after each
  probe measurement.
- **`StdpWindowBackend`** (`src/neuroseq/stdp_window_dialog.py`): owns the
  Δt sweep (both EPSP and synaptic-weight readout, both reset-to-baseline
  and continuous sweep modes). `begin_sweep(config, delta_ts,
weight_baseline=...)`, `probe()`, `pairing(delta_t_ms)`,
  `record_baseline_probe(amplitude, units)`, `handle_probe_measurement(
measurement)`, `record_weight_point(weight)` are the entry points the
  dialog calls after each measurement (trace-fetch and synapse-weight
  reads stay dialog-side; reset-mode's checkpoint restore is a distinct
  dialog-side pre-step reached via the `on_restore_baseline` hook).

Each dialog keeps only what genuinely needs Qt: widget construction, the
one-off "initialize_trace"/"initialize_weight" phase pre-check (needs
`refresh()`/trace-name lookups), trace/synapse-weight reads, plot/heatmap
rendering, and its own `SimulationHost.append_block` implementation (which
wraps the shared `dynamic_input_blocks.append_block` plus the unchanged
follow-camera/heatmap-update code).

## Gap-timing fix

`dynamic_input_blocks.append_block()` calling `editor.reset_stim_array()`
before computing where to write the next block fixes a real bug: the
project-wide "Input Automatic Padding" setting (`Settings.stim_padding_secs`,
File > "Input Automatic Padding") could previously zero-pad a stim array
beyond a block's intended end during a simulation run, and the next probe/
pairing block would then be appended after that stale padding too -
silently inflating the configured inter-block gap. `reset_stim_array()`
only ever removes unsimulated trailing padding (never anything already
delivered to the simulation) and is a cheap no-op when there's nothing to
trim, so calling it before every block is safe and free in the common case.

## Tests

- `tests/test_dynamic_input_routine.py`, `tests/test_dynamic_input_blocks.py`,
  `tests/test_dynamic_input_measurement.py`: Qt-free tests of the three
  shared modules, including a regression test
  (`test_append_block_absorbs_padding_beyond_simulation_head`) that proves
  the gap-timing fix.
- `tests/test_rheobase_backend.py`, `tests/test_dynamic_input_stdp_backends.py`:
  Qt-free tests that drive each routine's full closed loop (via
  `begin_search`/`begin_protocol`/`begin_sweep` and a `while backend.running:
...` loop against a scripted measurement sequence and a fake
  `SimulationHost`) through to real convergence/completion - no
  QApplication involved.
- The existing per-dialog Qt test suites (`tests/test_rheobase_dialog.py`,
  `tests/test_stdp_synaptic_tuning_dialog.py`, `tests/test_stdp_window_dialog.py`,
  `tests/test_rheobase_integration.py`, `tests/test_stdp_window_integration.py`)
  needed no behavioral changes - only mechanical adjustments where they
  asserted on renamed/relocated private attributes, covered by the
  `backend_property` compatibility shim.

## Known follow-on fixed alongside this

`tests/test_stdp_window_integration.py`'s lightweight fake editor had a
no-op `reset_stim_array()` that predated this gap-timing fix landing;
fixed to actually trim (mirroring `StimEditor.reset_stim_array()`), since
the gap-fix now depends on it.
