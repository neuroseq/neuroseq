# NeuroSeq: Detailed Functional Walkthrough (Code-Scanned)

This document explains how NeuroSeq works by scanning the repository code, especially:

- `src/neuroseq/**/*.py`
- `external_simulation_example/*.py`
- `tests/test_*.py`

Scan date: 2026-03-25 (module paths and menus refreshed 2026-10-19).

## 1) What NeuroSeq is

NeuroSeq is a Qt desktop GUI around a Brian2 simulation model plus data import/export and visualization tooling. It lets you:

- define/edit neuron-network models,
- define/edit stimulation arrays over time,
- run and continue simulations,
- inspect traces/spikes/synapse matrices,
- import experimental data (raw TIFF, Suite2p, Cellpose, patch-clamp),
- import SysCon2 `.seq` stimulation timelines into existing stim arrays,
- save/load full projects,
- run external simulation functions,
- generate synthetic datasets.

Runtime entrypoint:

- CLI script `neuroseq` -> `neuroseq.__main__:main`.

## 2) End-to-end runtime flow

### 2.1 Startup

`__main__.py`:

- Parses CLI options (`--log-level`, `--log-file`, optional filename) plus Cellpose test-data generation flags.
- In normal mode, initializes Brian2 logging preferences, builds `QApplication`, and opens `Gui(filename=...)`.
- In generation mode (`--generate-cellpose-test-data`), writes synthetic Cellpose TIFF + segmentation `.npy` and exits without launching Qt.

`gui.py` constructor:

- Creates `GuiSettings`, status bar with CPU/GPU/RAM monitor (`ResourceMonitorLabel`), and the project tab bar.
- Each project tab owns a complete `ProjectWorkspace` (`project_workspace.py`); opening a project or example opens a new tab, leaving other tabs untouched.
- Each workspace builds its central multi-pane UI:
  - stimulation editor,
  - trace view,
  - spike view,
  - generic data view,
  - synapse heatmap,
  - custom analysis pane,
  - embedded Python console,
  - time slider.
- Each workspace creates its menus/toolbars and all action handlers.
- Loads default project if no file was passed.

#### 2.1.1 Default startup state (no file passed)

When NeuroSeq starts without a project filename, the first tab loads `default_project()` ("Getting Started Project").

What is shown by default:

- Main window opens maximized.
- Visible panes: Stimuli, Traces, Spikes, Heatmap, Custom analysis, Console.
- Hidden pane: generic `Data` pane (off by default).
- Stim editor is populated immediately with the default stimulation arrays (`stimulus` and `stimulus_inh`), with the first stim selected.
- Trace/Spike/Synapse comboboxes are populated from simulation monitors (`Simulation :: P / v`, `Simulation :: P`, `Simulation :: Ce / we`), but plots are initially effectively empty until simulation data is produced.

Default theoretical model:

- A conductance-based Hodgkin-Huxley style network model is loaded by default.
- Neuron group `P` has 40 neurons (32 excitatory `Pe`, 8 inhibitory `Pi`) with HH gating dynamics (`m`, `n`, `h`) and excitatory/inhibitory conductance terms.
- Synapses include recurrent excitatory `Ce` (with STDP via `Apre/Apost` and `we`) plus `Cei` and `Cie` connections between excitatory and inhibitory subgroups.

### 2.2 Project state in GUI

Each `ProjectWorkspace` (one per tab) keeps one active `Project` object. Only one simulation runs at a time across all tabs, because Brian2's seed and code-generation settings are process-wide.

- `update_project()` refreshes all widgets, comboboxes, and console namespace.
- `update_limits()` pushes global time/neuron limits into the shared pan/zoom controller and time slider.

### 2.3 Simulation

`ProjectWorkspace.btn_continue_simulation_clicked()`:

- asks for simulation duration,
- calls `Project.run_simulation(...)`.

`Project.run_simulation()`:

- pre-pads stim arrays to ensure simulation horizon exists,
- delegates to `B2Model.run_simulation(...)`.

`B2Model.run_simulation()`:

- rebuilds network if model definition changed (`must_be_rebuilt`),
- creates namespace from global parameters + stim arrays converted to Brian2 `TimedArray`,
- runs `network.run(...)` with progress callback,
- checkpoints initial state automatically.

### 2.4 Display

After simulation/import changes, GUI updates comboboxes and visual widgets:

- `TraceWidget` for `TraceData`,
- `SpikeWidget` for `SpikeData`,
- `HeatmapWidget` for `SynapseData`,
- `DataWidget` as dynamic wrapper that switches among the above.

All time-driven widgets are synchronized by `TimeSliderWidget`.

## 3) Core data model

`data.py` defines the canonical data structures:

- `TraceData(name, traces[neurons,time], indices, units, dt)`
- `SpikeData(name, spike_indices, spike_times)`
- `SynapseData(name, values[src,dst,time], timestamps)`
- `Experiment(name, duration, timepoints, data[])`

`stimulation/stim.py` defines stimulation abstractions:

- `StimType`: Excitation/Inhibition/External
- `StimArray(name, stim_type, values[time,neuron], units, dt)`
- draw helpers for editor: `DrawStimType`, `DrawStim`

## 4) Brian2 model layer (`b2_model.py`)

`B2Model` is the central simulation definition engine.

### 4.1 Main design

It keeps an intermediate dictionary representation:

- `parameters`
- `neurongroups`
- `synapses`
- `monitors`
- `initials`

and exposes many `define_*` methods to mutate this representation.

Any mutation that affects runtime marks `must_be_rebuilt=True` (via `_trigger_rebuild` decorator or explicit handling).

### 4.2 Assembly and reconstruction

- `assemble_network()` builds a fresh Brian2 `Network` from dictionaries.
- Reconstruction helpers (`reconstruct_neurongroups`, `reconstruct_synapses`, etc.) rebuild from serialized state.
- `from_dict(...)` reconstructs full model + stored state/checkpoints.

### 4.3 Simulation outputs

After a run, model exposes:

- `get_simulation_trace_names()/data()`
- `get_simulation_spike_names()/data()`
- `get_simulation_synapse_names()/data()`

Synapse monitor data is transformed into full `n x n x t` matrices via `utils.construct_synapse_matrix(...)` with RAM-aware downsampling.

### 4.4 State/checkpoint/save bridge

- `store(name)` / `restore(name)` for checkpoints.
- `to_dict()` serializes model definition + current state metadata.
- `save_to_h5(path)` exports monitor-recorded data to HDF5.

## 5) Project layer (`project.py`)

`Project` packages everything user-visible:

- one `B2Model`,
- many experiments,
- many stim arrays,
- optional `CellposeImportResult`.

Responsibilities:

- load/save/import `.nseq` archives,
- compute global limits (`max_time_secs`, `max_neurons`),
- add/rename/delete/update stim arrays,
- append or create experiments,
- run simulation and keep limits consistent.

Important append logic:

- `issues_with_appending_experimental_data(...)` validates compatibility.
- `append_experimental_data(...)` merges by name/type and time offset.

## 6) Persistence and serialization (`io.py`)

NeuroSeq project files are zip archives (`.nseq`) with:

- JSON records for model/stims/experimental data/cellpose metadata,
- `.npy` blobs for numerical arrays,
- optional pickle for Brian2 stored states.

`io.py` provides:

- custom JSON encoder/object-hook factories handling `TraceData`, `SpikeData`, `SynapseData`, `StimArray`, `Experiment`, Brian2 `Quantity`, and `CellposeImportResult`,
- `append_to_zip(...)`, `load_from_zip(...)`,
- `as_timed_array(stim_array)` for Brian2 simulation namespace.

## 7) GUI and visualization internals

### 7.1 Main window and actions (`gui.py`, `project_workspace.py`)

`Gui` is the application shell holding the project tabs; each tab's `ProjectWorkspace` wires all actions:

- File: new/open/open example/import from project/save, `Import data → Input` (Optogenetics, Patch Clamp) and `Import data → Output` (Calcium Imaging raw `.tif` / Suite2p outputs, Patch Clamp), `Export → Patch Clamp Input / Optogenetic Input`, edit model, synthetic data, `Manage Data`.
- Simulate: run/continue/store checkpoint/restore checkpoint.
- Playback: speed, play/pause, skip, stop.
- View: show/hide panes, selection text toggles.
- Settings: Brian2 backend, input padding, LLM assistant, synapse/trace RAM limits, theme.
- Experiment: stimulation script generator + external simulation integration.

### 7.2 Shared plotting base (`visualization/fpl_widget.py`)

`FPLWidget` wraps fastplotlib figure/subplot creation and common cleanup.

### 7.3 Timeseries base (`visualization/timeseries_widget.py`)

`TimeseriesWidget` adds:

- pointer interactions (select/draw),
- ROI selection overlays,
- hover text,
- timeline marker,
- gridline layer,
- context menu for custom analysis on selected slices.

### 7.4 Specific viewers

`visualization/trace_widget.py`:

- stacked line plot per neuron,
- normalization dialog (percent, center median, clip).

`visualization/spike_widget.py`:

- scatter plot spike raster (time vs neuron index).

`visualization/heatmap_widget.py`:

- time-indexed synapse matrix image,
- optional subtraction against reference timestamp,
- LUT/histogram panel.

`visualization/data_widget.py`:

- stacked wrapper that selects trace/spike/heatmap viewer by incoming `Data` type.

### 7.5 Camera and timeline controls

`visualization_support/constrained_pan_zoom_controller.py`:

- clamps zoom/pan to valid data extents,
- keeps optional x/y scrollbars synchronized,
- updates gridline density visibility by zoom.

`visualization/time_slider_widget.py`:

- slider + timer-based playback loop,
- adjustable relative speed,
- broadcasts time updates to connected widgets.

`visualization_support/gridlines.py` / `visualization_support/region_of_interest.py` (time marker and hover text are drawn inside `visualization/timeseries_widget.py`):

- visual helper graphics for grid, current time marker, hover label, and selection rectangle.

### 7.6 Custom analysis pane (`visualization/custom_analysis_widget.py`)

Offers quick histogram and 2D histogram analyses over selected neuron/time slices for trace or spike data.

### 7.7 Interactive console (`console_widget.py`)

Embedded Python REPL with namespace:

- `np`, `b2`, `project`.

After each command, emits update signal and refreshes project limits, enabling live model/stim edits.

### 7.8 Resource monitor (`ResourceMonitorLabel` in `gui.py`)

Periodic process stats in status bar:

- CPU, RAM,
- optional GPU/VRAM via NVML.

## 8) Stimulus editing subsystem

### 8.1 Stim editor (`stimulation/stim_editor.py`)

Core capabilities:

- draw/erase stimuli directly on 2D stim arrays,
- select/copy/cut/paste/paste-inverted regions,
- save/load selected regions to/from file,
- create/edit/delete stim arrays,
- pad/trim all stim arrays,
- configurable draw intensity profile:
  - constant,
  - linear,
  - logarithmic,
  - random.

### 8.2 Undo/redo (`stimulation/stim_array_modifier.py`)

`StimArrayModifier` stores reversible slice actions as commands in the project's shared `ProjectHistory` (`project_history.py`), and updates plot texture incrementally. The same session history also covers data-group deletion (File → Manage Data) and imports.

### 8.3 Stim visuals (`visualization/stim_widget.py`)

Heatmap-like display of stim values with colormap depending on `StimType`.
Supports background overlay of non-selected stim arrays.

Color/intensity interpretation:

- Color hue indicates stimulus type:
  - excitation: magenta,
  - inhibition: blue,
  - external: diverging purple/green.
- Brightness/opacity indicates amplitude magnitude relative to the selected array's current scale.
- For excitation/inhibition arrays, `0` is transparent and larger positive values become more saturated/opaque.
- For external arrays, display range is symmetric (`-vmax` to `+vmax`): negative values map toward purple, positive values toward green, and values near `0` are transparent.
- Ramps appear as gradients because amplitude changes over time; constant-amplitude blocks appear as uniform color.
- Non-selected stim arrays (when "Show all stim arrays" is enabled) are drawn as semi-transparent overlays.

### 8.4 Stim dialogs

`stimulation/stim_dialog.py`:

- edit metadata (name/type/neurons/units).

`stimulation/stim_intensity_dialog.py`:

- configure draw profile values with validation rules.

### 8.5 Stimulation script generator (`export/stimulation_generator_dialog.py`)

Provides export dialog for hardware script generation.
Current implementation status:

#### Scope and outputs

- Opened from `File → Export → Optogenetic Input` (choose `SysCon2`; writes files only) or from the Stimulation setup panel's `Export with settings…` button (also imports the generated sequence into a connected SysCon2).
- Hardware type list currently only `HOLO4D`.
- Output extension currently always `.seq`.
- Exports are generated from selected stim array + imported Cellpose segmentation using `export/exporter_syscon2.py`.
- Generator writes three files per run:
  - `<script_name>.seq`,
  - `<script_name>_export_debug.csv`,
  - `<script_name>_overview.png`.

#### Direct-tick export model (current)

The stim array represents laser **TTL pulses** directly. Each sample is one SysCon2 tick (`dt = 0.0005208125 s = 520.8125 µs`), so `sample_idx == tick_idx` exactly. SLM shape objects are _inferred_ from the pulse activity and are not drawn by the user.

- Required `stim_array.dt == 0.0005208125` (exact constant `_SYSCON2_TTL_PULSE_QUANTUM_S`); export and sanity check both enforce this.
- Raw active segments (`stim_value > threshold`) represent the desired TTL pulses. No minimum duration is imposed.
- Intensity is either fixed binary (`1.0`, default) or preserved/clipped to `[0,1]`.

**Shape object inference (SLM hardware requirement)**

- For shape derivation, raw segments are quantized to 32-tick (16.666 ms) window boundaries:
  - start snaps to the left window boundary,
  - duration is rounded up to whole windows.
- Overlapping window-level segments are normalized in connected overlap components (all starts align left, durations trimmed to shortest).
- Hologram grouping is inferred from the normalized window-level activity:
  - a new hologram is started only when a neuron appears that has **never** been part of the current hologram's neuron set (`current_neurons` is monotonically growing),
  - gaps (empty windows) between pulse groups of the same neuron set do **not** break the hologram — the SLM shape object spans across the gap.
- Pre-load blank enforcement:
  - each hologram must have a blank 16.666 ms window immediately before its first active window (SLM load-in requirement),
  - if missing, sanitizer inserts a blank window by shifting all subsequent content right by 32 ticks,
  - raw segment tick positions are adjusted by the same shift so TTL pulses remain inside the shape's usable window after insertion,
  - this also applies to the first hologram; tail windows may be truncated if shifting exceeds the array length.

**TTL pulse emission**

- Laser TTL pulse intervals are derived from the **adjusted raw segment positions** (raw positions after accounting for any preblank insertions), not from window boundaries.
- This preserves sub-window timing: a pulse drawn at tick 100 is emitted at exactly tick 100 (or tick 100 + shift if a preblank window was inserted before it).
- Multiple TTL pulses from different raw segments can occur within the same hologram.

**Shape object emission**

- Sanitized segments are mapped to Cellpose centroids; active segments without valid mapping are skipped and reported.
- For each neuron within a hologram:
  - `shape_start_win = hologram.start_win - 1` (clamped at `0`) — the blank load window,
  - `shape_end_win` covers the last active window for that neuron (minimum `shape_start_win + 2`),
  - shape tick range: `[shape_start_win * 32, shape_end_win * 32)`.
- Multiple TTL pulses can occur within one hologram's shape span.

#### Timeline layout and emitted object order

- Lane scheduling uses interval coloring with lane count `N = max` simultaneously active shape windows.
- Fixed timeline layout: `0..N-1` = shape/neuron lanes, `N+0..N+2` = laser timelines (`940`, `1030`, `1140` nm), `N+3` = `TTL Out 1`, `N+4` = `TTL Out 2`.
- Exactly one laser timeline is active per export (selector in dialog; default `1140nm`).
- Laser/light-source mapping is fixed:
  - `TimelineInfo0_LightsourceID_0=dummy_0_0` -> `940nm`,
  - `TimelineInfo0_LightsourceID_0=dummy_1_0` -> `1030nm`,
  - `TimelineInfo0_LightsourceID_0=dummy_2_0` -> `1140nm`.
- Deterministic block order:
  - all shape blocks first (`start_tick, lane, neuron_index` order),
  - then active-laser `TTLPulse` blocks from merged pulse intervals,
  - then `TTL Out 1` mirrored pulse blocks for the same intervals,
  - `TTL Out 2` is intentionally left empty.
- Timing encoding rules: `TTLPulse` duration = `(repeats + 1) * 520.8125us`, `Shape` duration = `(repeats + 2) * 16.666ms`.
- Generated sequence text is parser-validated immediately (`parse_syscon2_seq_text`) and summarized via `SysconExportReport`.

#### Debug CSV (current schema usage)

- Exporter emits a structured debug CSV with fixed columns and row types covering:
  - mapping windows (`window_map`),
  - raw/sanitized stim segments (`stim_raw`, `stim_sanitized`),
  - sanity fixes (`fix`),
  - inferred holograms and members (`hologram`, `hologram_member`),
  - emitted SysCon objects (`sys_shape`, `sys_pulse_laser`, `sys_ttl_out1`),
  - warnings (for example unmapped active neurons).
- Rows include both NeuroSeq and SysCon timing representations (ms/us, window indices, tick indices), IDs/parent IDs, timeline/lane, and fix metadata.

#### Timeline overview PNG

- Exporter writes `<script_name>_overview.png` with timeline rows, object spans, SLM lead-in overlay, and explicit timeline labels (`Neuron lane`, `Laser <wavelength>`, `TTL Out`/port when inferable).
- For long timelines, renderer adapts figure width, time units, and minimum visible pulse width to improve readability.

#### Sanity check workflow in generator dialog

- `Check Input Sanity` runs `check_syscon2_input_sanity`, which validates `dt`, checks for SLM gap/overlap/preblank violations, and performs a dry-run export.
- Feedback is displayed in a scrollable dialog so long fix lists do not overflow the screen.
- If the stim array has the wrong `dt`, a **Fix dt** button sets `stim_array.dt = 0.0005208125` and re-runs the check automatically.
- If sanitizer changed values (preblank insertions, overlap normalization), the dialog offers to apply the corrected values back into the selected stim array.
- Sanity warnings include applied fix descriptions, truncation count, binary/clipping warnings, and unmapped active neurons.

#### Header/template lifecycle

- Header/template fields come from HOLO4D config text (lines before first object block) and are normalized to SysCon2 v3 required defaults.
- When a SysCon2 file is imported, its header block becomes the default export template for later script generation in the same project/session.

## 9) Data import pipelines

### 9.1 Suite2p import

`importers/suite2p_import_dialog.py` + `importers/importer_suite2p.py` (File → Import data → Output → Calcium Imaging (Suite2p outputs)):

- reads `F/Fneu/spks/stat` from `.npy` or CSV,
- supports folder auto-detection and per-file selection,
- computes contrasted traces (`F - 0.7*Fneu`),
- per-neuron normalization for traces/spikes,
- optional spike extraction via thresholding,
- returns `TraceData` and optional `SpikeData`.

### 9.2 Cellpose import

`importers/cellpose_import_dialog.py` + `importers/importer_cellpose.py` + `visualization/cell_overlay_widget.py`:

- loads base image (PNG/TIFF stack mean-projection supported),
- loads segmentation (CSV, cellpose dict NPY, or suite2p stat+iscell),
- shifts to 0-based internal IDs with background `-1`,
- generates colored overlay and legend,
- supports mapping strategies between stimulus IDs and cellpose IDs:
  - linear offset-based,
  - custom pairs.
- preview overlay can show segmentation colors, cellpose IDs, and mapped stimulus IDs.

The result is stored as `CellposeImportResult` in project state.

### 9.2.1 Cellpose synthetic test-data generator (CLI)

`__main__.py` + `cellpose_test_data.py`:

- `neuroseq --generate-cellpose-test-data <output.tif>` creates synthetic Cellpose-compatible files and exits.
- Default geometry is a `10 x 10` grid of circular targets with radius `10 px` and `10 px` gap.
- Companion segmentation output is saved as `<stem>_segmentation.npy` by default, or explicitly via `--cellpose-segmentation-output`.
- Generator options include `--cellpose-grid-rows`, `--cellpose-grid-cols`, `--cellpose-radius`, `--cellpose-gap`, and optional `--cellpose-square-size`.
- When `--cellpose-square-size` is set (for example `1300`), output is exactly `1300x1300`.
- Default square behavior is auto-fit: rows/cols stay fixed while radius/gap are adjusted to fill the square canvas.
- New fixed-geometry square behavior: add `--cellpose-preserve-radius-gap` to keep the exact `--cellpose-radius` and `--cellpose-gap`; the grid is centered in the square and generation fails if it does not fit.
- New fixed-radius auto-gap behavior: add `--cellpose-preserve-radius-auto-gap` to keep `--cellpose-radius` fixed while auto-fitting only the gap from `rows/cols/radius/square-size`.
- `--cellpose-auto-fit-radius-gap` can be passed explicitly to force the default auto-fit behavior.
- Generated segmentation uses Cellpose-style integer labeling (`0` background, targets labeled `1..N`).

Copy-paste examples:

```bash
# 1) Default test data (10x10, radius=10, gap=10)
python -m neuroseq \
  --generate-cellpose-test-data example_data/cellpose_default.tif
```

```bash
# 2) Custom rows/cols/radius/gap and explicit segmentation output path
python -m neuroseq \
  --generate-cellpose-test-data example_data/cellpose_custom.tif \
  --cellpose-segmentation-output example_data/cellpose_custom_segmentation.npy \
  --cellpose-grid-rows 8 \
  --cellpose-grid-cols 12 \
  --cellpose-radius 7 \
  --cellpose-gap 5
```

```bash
# 3) Fixed square output (1300x1300) with default auto-fit radius/gap
python -m neuroseq \
  --generate-cellpose-test-data example_data/cellpose_1300_autofit.tif \
  --cellpose-grid-rows 10 \
  --cellpose-grid-cols 10 \
  --cellpose-radius 10 \
  --cellpose-gap 10 \
  --cellpose-square-size 1300
```

```bash
# 4) Fixed square output (1300x1300) with fixed radius/gap (no auto-fit)
python -m neuroseq \
  --generate-cellpose-test-data example_data/cellpose_1300_fixed.tif \
  --cellpose-grid-rows 10 \
  --cellpose-grid-cols 10 \
  --cellpose-radius 10 \
  --cellpose-gap 10 \
  --cellpose-square-size 1300 \
  --cellpose-preserve-radius-gap
```

```bash
# 5) Fixed square output with fixed radius and auto-fitted gap
python -m neuroseq \
  --generate-cellpose-test-data example_data/cellpose_1300_fixed_radius_auto_gap.tif \
  --cellpose-grid-rows 8 \
  --cellpose-grid-cols 12 \
  --cellpose-radius 10 \
  --cellpose-square-size 1300 \
  --cellpose-preserve-radius-auto-gap
```

### 9.3 Patch-clamp output import

`importers/patch_clamp_data_output_import_dialog.py` + `importers/importer_patch_clamp_data.py` (File → Import data → Output → Patch Clamp):

- loads recorded traces from Igor2 `.pxp` / `.ibw` or `.csv` / `.txt` (one trace per column, shared source sampling rate),
- creates new trace data or imports into existing trace data (extend / replace options); a different sample interval triggers the same Δt choice as input import (section 9.6),
- runs peak-based spike detection with tunable parameters.

### 9.4 Import mode dialog

`add_experiment_data_dialog.py` governs whether new imported data:

- creates a new experiment, or
- appends into existing experiment at selected timepoint,
  with validation/error messaging.

### 9.5 SysCon2 `.seq` import (stimulation data)

`importers/syscon2_import_dialog.py` + `importers/importer_syscon2.py`:

- GUI entrypoint is `File → Import data → Input → Optogenetics`; the dialog can create a new input array or import into an existing one.
- Parser accepts `.seq` files whose first non-empty line is either `nohash` or a 16-byte hash string (`AA-BB-...` format).
- Header is parsed as `key=value` lines before the first object block; required keys are `version`, `scanmode`, `runs`, `invertedportcnt`, `AutoTriggerRulesEnabled`, `AutoTriggerRulesCount`, `timelinezoomfactor`, `timelineviewposx`, `timelineviewposy` (`DeviceSequenceID` is accepted when present and defaulted for compatibility when missing).
- Object blocks must be `[index] ... [/index]`; malformed/mismatched block markers raise an import error.
- Each object extracts timing from `TimelineInfo0_*`: `starttime` (us, float), `repeats`, `Stepsize`, and `Intensity_0`; values are clamped/sanitized (`start>=0`, `repeats>=0`, `step>=1`).
- Eligibility uses timed `Shape`/`Entity` objects only (non-shape timed objects are no longer an import option in the dialog).
- If `Intensity_0` is missing, default intensity is `1.0` for `Shape`/`Entity` and `0.0` for other object types.
- Circular-entity grouping for the generic timed-object path uses geometry type `2` entities and center distance (`<=1.0um`) based on `CenterX/CenterY`.
- Import path selection is pulse-first:
  - if any laser `TTLPulse` objects exist (`LightsourceID` present), importer uses Holo4D pulse-based reconstruction,
  - otherwise importer uses generic timed object expansion from `Shape`/`Entity`.
- In pulse-based reconstruction:
  - active laser timeline is selected as the timeline with the most laser pulses (tie -> lowest index),
  - `TTLPulse` duration uses `duration_us = (repeats + 1) * 520.8125`,
  - pulses from that active laser timeline are the true stimulation events,
  - `Shape`/`Entity` objects define placement windows, with unusable lead period `16.666ms` (`usable_start = object_start + 16.666ms`),
  - shape duration uses SysCon quantum (`(repeats + 2) * 16.666ms`),
  - if multiple point objects share a lane, a later object clips/replaces earlier placement on that lane,
  - TTL output pulses (`LightsourceCount=0`) are ignored for reconstruction.
- In pulse-based reconstruction (`dt = 0.0005208125 s`):
  - pulse start/duration are converted by `round(t_us / dt_us)` giving `sample_idx == tick_idx` exactly,
  - sub-window pulse timing is preserved end-to-end: a pulse at tick 100 imports at sample 100.
- In generic timed-object fallback mode (no laser pulses), start indices and durations are converted by `round(t / dt)` using object timing metadata.
- Mapping order is deterministic: grouped timed objects are assigned to neuron channels in file order.
- If repeated/overlapping events hit the same channel/time window, amplitude merge uses `max(existing, incoming)`.
- Safety cap limits each object to `500000` events during expansion; excess repeats are counted as truncated repeats in the result report.
- Import target controls: `neuron_offset` chooses starting destination neuron column; channels beyond available target neurons are dropped and reported.
- Time-axis handling: `extend_to_fit=True` grows the target stim array to imported duration; otherwise overflow is truncated and reported.
- Value-write mode: `replace_existing=True` replaces destination values; otherwise imported values are merged with existing data by pointwise `max`.
- Preview panel reports parsed header fields, object counts, timed-object counts, estimated grouped channels, and center-distance merged counts before import.
- Completion dialog reports eligible objects, grouped channels, mapped/unmapped channels, generated timepoints, merged circular entities, and any truncation counters.

### 9.6 Patch-clamp input import: Igor2 `.ibw` / `.pxp` and `.csv` / `.txt` (stimulation data)

`importers/patch_clamp_data_input_import_dialog.py` + `importers/importer_patch_clamp_data.py`:

- GUI entrypoint is `File → Import data → Input → Patch Clamp` (no source-software pre-dialog).
- Import is available regardless of whether the project already has stim arrays:
  - if no stim arrays exist, only `Create new stim array` is available,
  - if stim arrays already exist, user can either create a new stim array or import into an existing one.
- Supported file types are:
  - Igor Binary Wave `.ibw`,
  - Igor Packed Experiment `.pxp`,
  - simple Igor-style `.csv` / `.txt`.
- Current scope is one selected 1D wave per import action (packed experiments stay open so several waves can be imported in turn):
  - `.ibw` and `.csv` provide a single wave directly,
  - `.pxp` files may contain many waves, but the dialog imports one selected wave at a time,
  - multidimensional Igor waves are rejected or skipped,
  - the importer currently expects one 1D command/stimulation trace per import operation.

#### Supported file semantics

- Igor-style `.csv` import expects:
  - first non-empty row = wave name,
  - every following non-empty row = exactly one numeric sample value.
- Multi-column sample rows in `.csv` are rejected as invalid input.
- `.ibw` loading uses the optional Python dependency `igor2`.
- `.pxp` loading also uses the optional Python dependency `igor2`.
- For `.ibw`, importer extracts when available:
  - wave name,
  - raw sample values,
  - source sample interval (`dt`) from Igor header scaling fields,
  - data units,
  - x-axis / dimension units,
  - optional wave note text for preview display.
- For `.pxp`, importer walks the packed experiment, extracts all importable 1D waves, and presents them in a wave-picker list using slash-joined folder paths such as `root/stims/HodgkinHuxley`.
- `.pxp` loading is tolerant to per-record unpack failures (e.g. upstream igor2 0.5.x raises `TypeError: buffer is too small for requested array` on some dependency-formula / multidimensional waves). The loader iterates the packed record stream itself and wraps each `record_type(header, data, ...)` call in try/except:
  - records that fail to unpack are logged at WARNING and recorded as `SkippedPxpRecord(record_index, record_type, reason)`,
  - the remaining records are still passed through `igor2.packed._build_filesystem(...)` and parsed as usual,
  - `load_igor_pxp_file()` returns an `IgorPxpLoadReport(waves, skipped)` dataclass,
  - when `skipped` is non-empty the dialog shows a `QMessageBox.warning` ("Some waves were skipped") after populating the wave picker, and writes the full per-record breakdown to the application log.

#### Packed experiment selection flow

- When a `.pxp` file is selected:
  - the dialog scans the packed experiment for importable 1D waves,
  - a `Wave Selection` panel becomes visible,
  - the list contains the wave path keys derived from the packed-experiment folder hierarchy.
- If exactly one wave is found:
  - it is auto-selected immediately,
  - preview, suggested stim name, and suggested units populate automatically.
- If multiple waves are found:
  - no wave is selected initially,
  - preview stays empty,
  - import remains disabled until the user chooses one wave from the list.
- Loading a non-`.pxp` file hides the wave picker again and returns to the original single-wave flow.

#### Preview and source sampling metadata

- After file selection, the dialog shows a preview containing:
  - wave name,
  - source format,
  - source sample count,
  - detected source `dt` if present in file metadata,
  - the source `dt` currently selected for import,
  - wave units,
  - axis units,
  - value range (`min..max`),
  - optional Igor note text.
- Source sample interval is always user-editable in the dialog.
- If the file contains no source `dt` metadata:
  - the dialog defaults to `0.001 s`,
  - preview explicitly warns that the file did not store the sample interval,
  - import still proceeds using the currently selected source `dt`.

#### Timeline conversion model

- There is no project-wide time grid: every stim array and dataset keeps its own `dt`.
- Samples are imported at the source sample interval, so a new stim array is created on the source/file `dt`.
- When importing into an existing stim array whose `dt` differs, a dialog offers:
  - `Resample wave to target Δt` (default),
  - `Remap target to wave Δt` (the whole target array is remapped; may shift existing timing by up to one sample and can make the target very large),
  - `Cancel`.
- If source `dt` already equals the target `dt`, sample values are copied directly.
- When the wave is resampled onto a different target `dt`, the importer uses zero-order hold:
  - target timestamps are generated on the selected target grid,
  - each target sample takes the most recent source sample value,
  - this is intended for step-like command waveforms,
  - fine structure below the chosen target `dt` is intentionally collapsed in the NeuroSeq representation.

#### Import target modes

The dialog offers two mutually exclusive modes.

##### A) Create new stim array

- A new `StimArray` is created from the imported wave.
- Suggested stim name defaults to the Igor wave name.
- Suggested units default to the imported wave units when available.
- Default stim type is `External`.
- Imported values are stored as a single-neuron stim matrix:
  - shape = `(timepoints, 1)`.
- Stored stim metadata:
  - `name` = dialog name field,
  - `stim_type` = dialog type selection,
  - `units` = dialog units field,
  - `dt` = selected source `dt`.
- Duplicate stim names are rejected before import.

##### B) Import into existing stim array

- User selects:
  - target stim array,
  - target neuron column,
  - whether to extend target duration,
  - whether to replace or add to existing values.
- Import writes the 1D Igor wave into exactly one neuron column of the chosen target stim array.
- If the selected target stim array uses a different `dt` than the wave, the resample/remap/cancel choice above applies; the import is recorded as one undoable edit.
- Time-axis behavior:
  - if `extend_to_fit=True`, target stim array grows to fit the imported duration,
  - otherwise overflow samples are truncated.
- Value-write behavior:
  - if `replace_existing=True`, destination neuron column is cleared first and then overwritten with imported values,
  - if `replace_existing=False`, imported values are added to existing destination values sample-by-sample.
- Other neuron columns are preserved.

#### Resulting project/GUI updates

- New-stim mode calls `Project.add_stim_array(...)`.
- Existing-stim mode mutates the selected `StimArray.values` in place.
- After successful import, GUI refreshes project state and sets the stim combobox to the imported/updated stim array.
- Completion dialog reports:
  - source sample count,
  - imported sample count,
  - target stim name,
  - target neuron index for existing-stim mode,
  - whether resampling occurred,
  - source and target `dt`,
  - any truncation when target extension was disabled.

#### Current limitations

- No multi-wave batch import yet.
- No automatic reconstruction of stimulation patterns from Sutter routine/paradigm metadata yet.
- Because zero-order hold is used, the importer is best aligned with command/stimulation waveforms that are piecewise constant or approximately step-like.

## 10) Synthetic data generation (`synthetic_data_dialog.py`)

Two synthetic workflows:

- Patch-clamp style traces:
  - select source traces,
  - subset neurons by index expression,
  - add Gaussian noise,
  - export (`.npy`/`.csv`) or add as new experiment.
- Imaging style traces from spikes:
  - converts spikes to fluorescence-like traces,
  - applies decay dynamics, downsampling, lognormal variability, Poisson noise,
  - export or add to project.

## 11) Model editor subsystem (`model_editor.py`)

Provides a rich parameter-tree dialog for editing B2Model internals:

- neurongroups/subgroups,
- synapses,
- monitors,
- parameters,
- initial conditions,
- random seed.

It clones the model via `B2Model.from_dict(...)`, edits clone state, and on accept rebuilds if needed.

## 12) Example projects (`example_projects/`)

The GUI menu `File -> Open example project` shows only functions registered with `@register_example_project(...)`.

Currently, 7 projects are selectable:

### 12.1 `Getting Started Project`

Purpose:

- Small HH-style E/I network with explicit excitatory + inhibitory stimulation and plastic recurrent E-to-E synapses.

Model:

- 40 neurons total (`P`), split into `Pe` (32 excitatory) and `Pi` (8 inhibitory).
- Hodgkin-Huxley membrane dynamics (`m`, `n`, `h`) with synaptic conductances `ge`/`gi` and optogenetic-like conductance inputs (`geopto`, `giopto`).
- Threshold `v > -20 mV`, refractory `3 ms`, solver `exponential_euler`.

Synapses:

- `Ce`: `Pe -> Pe`, condition `i != j`, STDP-like variables `Apre`/`Apost`, weight `we`, clipping to `[-gmax, gmax]`.
- `Cie`: `Pi -> Pe`, fixed inhibitory weight equation (`wi = 67 nS`), pre-event `gi += wi`.
- `Cei`: `Pe -> Pi`, fixed excitatory weight equation (`we = 6 nS`), pre-event `ge += we`.

Initialization and monitors:

- Randomized initial `v`, `ge`, `gi`.
- Monitors for trace (`P / v`), spikes (`P`), and synapse state (`Ce / we`).

Stimuli:

- `stimulus` (Excitation, units `siemens`, `dt=1 ms`): sparse pulses on excitatory neurons around 120 ms, 320 ms, and 520 ms.
- `stimulus_inh` (Inhibition, units `siemens`, `dt=1 ms`): short inhibitory pulse around 800 ms on excitatory indices.

### 12.2 `non_linear`

Purpose:

- Minimal nonlinear synapse demo (no stimulation arrays) to inspect synaptic state dynamics.

Model:

- Input group `neuron_input` with 2 neurons, ramping `v` (`dv/dt = 1/(10 ms)`), threshold/reset spiking.
- Target group `neurons` with 1 neuron, state `g`, dynamics `dv/dt = (g-v)/(10 ms)`.

Synapse:

- `S` from input to target with nonlinear state equations:
  - `dg_syn/dt = -a*g_syn + b*x*(1-g_syn)`
  - `dx/dt = -c*x`
  - `g_post = g_syn (summed)`
  - `on_pre: x += w`

Initialization and monitors:

- Two synaptic weights initialized to `[1.0, 10.0]`.
- Input neuron initial `v = [0.0, 0.5]` (phase offset).
- Monitors `g` on synapse (`M`) and target neuron (`Mn`).

### 12.3 `Hodgkin-Huxley Neuron`

Purpose:

- Single-neuron HH electrophysiology example showing response to external current plus excitatory/inhibitory optogenetic channels.

Model:

- 1 neuron (`neuron`) with HH gating (`m`, `n`, `h`), external current term `I_external(t,i)`, and optogenetic-like conductances from `Exc_stimulation`/`Inh_stimulation`.
- Threshold `v > -20 mV`, refractory `2 ms`, solver `exponential_euler`, neuron `dt = 0.1 ms`.

Initialization and monitors:

- Randomized initial membrane potential around `El`.
- State monitor for `v` and spike monitor.

Stimuli (`tfinal = 3100 ms`, all `dt = 1 ms`):

- `I_external` (`pA`): ramp segments plus constant steps (e.g., 200/400/600 pA windows).
- `Exc_stimulation` (`1`): ramp and step blocks between ~1100 ms and 1999 ms.
- `Inh_stimulation` (`1`): ramp and step blocks between ~2100 ms and 2999 ms.

What the stim-panel color patterns mean for this project:

- On load, the selected stim is typically `I_external` (first stim array), shown in the `External` colormap (diverging: negative=purple, zero=transparent/white, positive=green). Since `I_external` is non-negative here, its active windows appear in green shades.
- By default, `Show all stim arrays` is enabled, so non-selected stim arrays are overlaid as semi-transparent background layers:
  - excitation arrays are magenta,
  - inhibition arrays are blue.
- Therefore, for this project you usually see:
  - green pattern blocks around ~100–999 ms (`I_external`),
  - translucent magenta patterns around ~1100–1999 ms (`Exc_stimulation`),
  - translucent blue patterns around ~2100–2999 ms (`Inh_stimulation`).
- Ramps are rendered as color gradients; constant plateaus are rendered as flat-color rectangles.

### 12.4 `Leaky-Integrate-and-Fire Neuron`

Purpose:

- Single-neuron LIF counterpart to the HH single-neuron example, with the same three stimulation channels.

Model:

- 1 neuron with:
  - `dv/dt = -(v-v0)/tau + Rm*I_stim/tau`
  - `I_stim = I_ext + I_opto_ext + I_opto_inh`
  - `I_ext = I_external(t,i)`
  - optogenetic current terms driven by `Exc_stimulation`/`Inh_stimulation`.
- Threshold `v > -10 mV`, reset `v=-65 mV`, refractory `2 ms`, `dt=0.1 ms`.

Initialization and monitors:

- Initial `v = 5 mV`.
- Trace monitor (`v`) and spike monitor.

Stimuli (`tfinal = 3100 ms`, `dt=1 ms`):

- Same temporal pattern style as the single HH project:
  - `I_external` (`pA`) ramp/step protocol.
  - `Exc_stimulation` (`mwatt / um2`) ramp/step blocks.
  - `Inh_stimulation` (`mwatt / um2`) ramp/step blocks.

### 12.5 `Network of Hodgkin-Huxley Neurons`

Purpose:

- Large-scale recurrent E/I HH network example for network-level activity and synapse monitoring.

Model:

- 4000 neurons total (`P`), split into `Pe` (3200 excitatory) and `Pi` (800 inhibitory).
- HH dynamics with external current term and optogenetic conductance channels.
- Threshold `v > -20 mV`, refractory `3 ms`, solver `exponential_euler`.

Synapses:

- `Ce`: `Pe -> P`, connection probability `0.02`, variable `we`.
- `Ci`: `Pi -> P`, connection probability `0.02`, variable `wi`.
- Pre rules:
  - `Ce`: `ge += clip(we, 0*nS, 18*nS)`
  - `Ci`: `gi += clip(wi, 0*nS, 210*nS)`

Initialization and monitors:

- Randomized initial `v`, `ge`, `gi`.
- Randomized synaptic weights:
  - `Ce.we = exp(0.25*randn())*6*nS`
  - `Ci.wi = exp(0.25*randn())*67*nS`
- Monitors for `P / v`, spikes, and `Ce / we`.

Stimuli (`tfinal = 2000 ms`, `dt=1 ms`):

- `I_external` (`pA`) initialized to zeros.
- `Exc_stimulation` and `Inh_stimulation` (`mwatt / um2`) contain patterned pulse trains over subsets of neurons (early block-group pulses and later sequential pulses).

### 12.6 `Network of Hodgkin-Huxley Neurons with STDP`

Purpose:

- Medium-size HH E/I network focused on recurrent excitatory plasticity (STDP) with synapse-state monitoring.

Model:

- 40 neurons total (`P`), split into `Pe` (32) and `Pi` (8).
- HH dynamics with external current and optogenetic conductances.
- Threshold `v > -20 mV`, refractory `3 ms`, solver `exponential_euler`.

Synapses:

- Plastic excitatory synapse `Ce` (`Pe -> P`):
  - probability `0.02`, condition `i != j`
  - state: `we`, `Apre`, `Apost`
  - on-pre: increase `ge`, update `Apre`, update/clip `we`
  - on-post: update `Apost`, update/clip `we`
  - parameters: `taupre`, `taupost`, `dApre`, `dApost`, `gmax`.
- Inhibitory synapse `Ci` (`Pi -> P`), probability `0.02`, pre rule `gi += clip(wi, 0*nS, 210*nS)`.

Initialization and monitors:

- Randomized initial `v`, `ge`, `gi`.
- Randomized `Ce.we` and `Ci.wi`.
- Monitors for `P / v`, spikes, and `Ce / we`.

Stimuli (`tfinal = 2000 ms`, `dt=1 ms`):

- `I_external` (`pA`) is zero baseline.
- `Exc_stimulation` and `Inh_stimulation` (`mwatt / um2`) use the same pulse-pattern style as the non-STDP network example.

### 12.7 `ChR2 in HEK293`

Purpose:

- Photocurrent-focused PyRho 6-state ChR2 model in a voltage-clamp-like setup across multiple clamp voltages.

Model:

- 10 neurons (`G0`), with `dt = 0.1 ms`, `tfinal = 1000 ms`.
- Membrane equation in clamp form: `dv/dt = (E_vc-v)/tau_m`.
- Full PyRho 6-state channel equations (`C_1`, `C_2`, `I_1`, `I_2`, `O_1`, `O_2`) with light-dependent transitions and photocurrent `I`.
- Threshold `v > v_t0` with `v_t0 = 10000 mV` (effectively suppresses standard spiking behavior for this protocol), reset `v=E_m`, refractory `t_ref`.

Initialization and monitors:

- Per-neuron clamp potentials and initial voltages from `-100 mV` to `80 mV` (step `20 mV`).
- Initial channel state mainly in `C_1`.
- State monitors for:
  - voltage (`v`)
  - photocurrent (`I`)
  - all channel states (`C_1`, `C_2`, `O_1`, `O_2`, `I_1`, `I_2`).

Stimulus:

- Single stim array `stim` (Excitation, units `1`, `dt=1 ms`) with square pulse amplitude `0.00442` from 100 ms to 700 ms.

Notes:

- Additional project-generating functions in `example_projects/` are present but not registered, so they do not appear in the GUI example menu.

## 13) External simulation plugin interface

`ProjectWorkspace` (`project_workspace.py`) can load external Python function `simulate(...)` via `runpy.run_path`.

Expected signature:

- `simulate(stims_dict, simulation_time) -> Experiment`

where `stims_dict[name]` provides `{"array": 2D stim values, "dt": seconds}`.

Examples in:

- `external_simulation_example/simulate.py`
- `external_simulation_example/run_brian2_externally.py`

Returned `Experiment` is appended/imported through the same experiment import flow.

## 14) Settings and app-level options

`gui_settings.py` stores persistent settings in user config (`platformdirs`):

- Brian2 backend (`numpy`/`cython`),
- synapse heatmap RAM cap,
- automatic stim padding seconds.
- BrainScaleS-2 tab settings (container image/app, quiggeldy setup, input mode, speed-up, recorded neuron; never the EBRAINS token).

`save_load_options.py` provides dialog/options for selective save/import of:

- stim arrays,
- model,
- simulation data,
- experimental data.

## 15) Utility layer highlights (`utils.py`)

Important utilities:

- Brian2 logging preference setup,
- wait-cursor decorator,
- QAction helper creation,
- combobox width auto-adjust,
- nested dict update,
- safe concat,
- append time-aligned data into existing traces/spikes/synapse series,
- remove selected neurons from trace/spike data,
- construct synapse matrices with downsampling,
- colormap selection by stim type,
- simulation time input dialog.

## 16) Module-by-module quick index

The following table maps each `src/neuroseq` module or package to primary functionality.

| Module                                               | Primary responsibility                                                                                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                                        | package version export                                                                                                                                                    |
| `__main__.py`                                        | CLI entrypoint, Cellpose test-data generation, and Qt app bootstrap                                                                                                       |
| `add_experiment_data_dialog.py`                      | append-vs-new experiment import chooser                                                                                                                                   |
| `analysis_plugins.py`                                | analysis plugin discovery (built-in and per-user plugin folders)                                                                                                          |
| `aoi_shapes.py`                                      | geometry of AOI shapes                                                                                                                                                    |
| `b2_model.py`                                        | Brian2 model definition/assembly/simulation/serialization                                                                                                                 |
| `b2_units.py`                                        | Brian2 quantity parsing/display helpers                                                                                                                                   |
| `background_jobs.py`                                 | workers and lifecycle for main-window background jobs                                                                                                                     |
| `cellpose_annotation.py`                             | programmatic Cellpose annotation for mean-fluorescence images                                                                                                             |
| `cellpose_test_data.py`                              | synthetic Cellpose grid TIFF + segmentation generator                                                                                                                     |
| `console_art.py`                                     | console wordmark and imagery                                                                                                                                              |
| `console_widget.py`                                  | embedded Python REPL bound to project                                                                                                                                     |
| `data.py`                                            | core dataclasses (`TraceData`, `SpikeData`, `SynapseData`, `Experiment`)                                                                                                  |
| `data_manager_dialog.py`                             | File → Manage Data: inspect and delete data groups (undoable)                                                                                                             |
| `external_tools.py`                                  | launch Fiji/ImageJ and Cellpose from the GUI                                                                                                                              |
| `fiji.py`                                            | locate and launch Fiji/ImageJ                                                                                                                                             |
| `gpu.py`                                             | PyTorch GPU-availability helpers for imaging imports                                                                                                                      |
| `gui.py`                                             | application shell: project tabs, resource monitor                                                                                                                         |
| `gui_settings.py`                                    | persistent app settings                                                                                                                                                   |
| `icon_manager.py`                                    | icon loading (`IconManager`)                                                                                                                                              |
| `io.py`                                              | zip/json/npy persistence and typed reconstruction                                                                                                                         |
| `main_scroll.py`                                     | whole-window scrolling with nested-scroll priority                                                                                                                        |
| `model_descriptions.py`                              | one-line descriptions for model fields and parameters                                                                                                                     |
| `model_editor.py`                                    | parameter-tree model editing dialog                                                                                                                                       |
| `project.py`                                         | aggregate project state and orchestration                                                                                                                                 |
| `project_history.py`                                 | session-local undo/redo shared by a project's data editors                                                                                                                |
| `project_workspace.py`                               | per-tab workspace: panes, menus/toolbars, action wiring                                                                                                                   |
| `qt_helpers.py`                                      | shared generic Qt widgets and window helpers                                                                                                                              |
| `save_imaging_dialog.py`                             | options for saving imaging data shown in the Imaging tab                                                                                                                  |
| `save_load_options.py`                               | save/import include-options dialog                                                                                                                                        |
| `spike_detection.py`                                 | threshold and peak-based spike detection                                                                                                                                  |
| `startup_dialog.py`                                  | launch dialog: example, open, recent, or blank                                                                                                                            |
| `synthetic_data_dialog.py`                           | synthetic patch/imaging data generation                                                                                                                                   |
| `theme.py`                                           | light/dark/retro themes                                                                                                                                                   |
| `utils.py`                                           | cross-cutting helpers                                                                                                                                                     |
| `workspace_acquisition_controller.py`                | workspace-local ScanImage acquisition and live-image lifecycle                                                                                                            |
| `workspace_devices.py`                               | shared device connections with per-workspace reply routing                                                                                                                |
| `device_control/`                                    | ScanImage connection/controls; SysCon2 client, protocol, connection manager, panel, stimulation host; BrainScaleS-2 protocol, connection, tab; Stimulation setup tab host |
| `elise_support/`                                     | ELiSe protocol dialog, workflows, replay metrics                                                                                                                          |
| `example_projects/`                                  | built-in example project factories (`neurons/`, `networks/`, `_registry.py`)                                                                                              |
| `export/exporter_igor2.py`                           | Igor2 `.ibw`/`.csv`/`.pxp` stimulus export                                                                                                                                |
| `export/exporter_syscon2.py`                         | SysCon2 `.seq` export generation from stim arrays + cellpose segmentation                                                                                                 |
| `export/igor2_pxp_stim_export_dialog.py`             | Igor2 `.pxp` (all arrays) export dialog                                                                                                                                   |
| `export/igor2_single_wave_export_dialog.py`          | Igor2 single-wave export dialog                                                                                                                                           |
| `export/power_lut.py`                                | laser power look-up tables                                                                                                                                                |
| `export/stimulation_generator_dialog.py`             | hardware script export dialog                                                                                                                                             |
| `export/stimulus_export_dialog.py`                   | File → Export entry point (format chooser)                                                                                                                                |
| `export/syscon2_export_service.py`                   | Qt-free SysCon2 `.seq` export path shared by dialog, panel and host                                                                                                       |
| `importers/cellpose_import_dialog.py`                | GUI flow for cellpose image+seg import and mapping                                                                                                                        |
| `importers/importer_cellpose.py`                     | cellpose loading, normalization, overlays, mapping payload                                                                                                                |
| `importers/importer_patch_clamp_data.py`             | Igor2/CSV patch-clamp wave loading and conversion                                                                                                                         |
| `importers/importer_scanimage.py`                    | index ScanImage TIFF recordings without loading movies                                                                                                                    |
| `importers/importer_suite2p.py`                      | suite2p contrast/normalize/convert pipeline                                                                                                                               |
| `importers/importer_syscon2.py`                      | SysCon2 `.seq` parser, center-distance grouping, and stim-array conversion                                                                                                |
| `importers/importer_tif.py`                          | Suite2p + Cellpose pipeline for raw 2-photon TIFF stacks                                                                                                                  |
| `importers/patch_clamp_data_input_import_dialog.py`  | patch-clamp input (stimulus) import dialog                                                                                                                                |
| `importers/patch_clamp_data_output_import_dialog.py` | patch-clamp output (recording) import dialog with spike detection                                                                                                         |
| `importers/suite2p_import_dialog.py`                 | Suite2p/CSV import dialog                                                                                                                                                 |
| `importers/syscon2_import_dialog.py`                 | GUI preview/options flow for SysCon2 `.seq` stim import                                                                                                                   |
| `importers/tif_import_dialog.py`                     | raw TIFF import dialog (runs the pipeline as a background job)                                                                                                            |
| `llm/`                                               | chat assistant: backends (Anthropic, OpenAI, Ollama), chat widget, settings, read access, plugin-write confirmation                                                       |
| `stimulation/stim.py`                                | stimulation dataclasses/enums                                                                                                                                             |
| `stimulation/stim_array_modifier.py`                 | undoable stim edit commands                                                                                                                                               |
| `stimulation/stim_dialog.py`                         | stim metadata editor dialog                                                                                                                                               |
| `stimulation/stim_editor.py`                         | interactive stim editing workstation                                                                                                                                      |
| `stimulation/stim_intensity_dialog.py`               | draw-intensity profile dialog                                                                                                                                             |
| `stimulation/stim_loop_dialog.py`                    | loop stim array dialog                                                                                                                                                    |
| `stimulation/stim_timing.py`                         | stim sample-interval helpers (resampling, remapping)                                                                                                                      |
| `stimulation/static_input/`                          | Design static input: pattern dialog, patterns, controller                                                                                                                 |
| `stimulation/dynamic_input/`                         | Design dynamic input: routines (STDP window, STDP tuning, rheobase), controller, measurement                                                                              |
| `visualization/`                                     | plot widgets: trace, spike, heatmap, data, stim, custom analysis, spatial footprint, cell overlay, calcium video, time slider, timeseries/fpl bases                       |
| `visualization_support/`                             | shared pan/zoom controller, gridlines, ROI, view state, Cellpose ids, neuron subset and brightness dialogs                                                                |
