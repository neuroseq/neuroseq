# NeuroSeq: Detailed Functional Walkthrough (Code-Scanned)

This document explains how NeuroSeq works by scanning the repository code, especially:

- `src/neuroseq/*.py`
- `external_simulation_example/*.py`
- `tests/test_*.py`

Scan date: 2026-03-03.

## 1) What NeuroSeq is

NeuroSeq is a Qt desktop GUI around a Brian2 simulation model plus data import/export and visualization tooling. It lets you:

- define/edit neuron-network models,
- define/edit stimulation arrays over time,
- run and continue simulations,
- inspect traces/spikes/synapse matrices,
- import experimental data (Suite2p, Cellpose, patch-clamp),
- save/load full projects,
- run external simulation functions,
- generate synthetic datasets.

Runtime entrypoint:

- CLI script `neuroseq` -> `neuroseq.__main__:main`.

## 2) End-to-end runtime flow

### 2.1 Startup

`__main__.py`:

- Parses CLI options (`--log-level`, `--log-file`, optional filename).
- Initializes Brian2 logging preferences.
- Builds `QApplication` and opens `Gui(filename=...)`.

`gui.py` constructor:

- Creates `GuiSettings`, `Project`, status bar with CPU/GPU/RAM monitor.
- Builds central multi-pane UI:
  - stimulation editor,
  - trace view,
  - spike view,
  - generic data view,
  - synapse heatmap,
  - custom analysis pane,
  - embedded Python console,
  - time slider.
- Creates menus/toolbars and all action handlers.
- Loads default project if no file was passed.

### 2.2 Project state in GUI

`Gui` keeps one active `Project` object.

- `update_project()` refreshes all widgets, comboboxes, and console namespace.
- `update_limits()` pushes global time/neuron limits into the shared pan/zoom controller and time slider.

### 2.3 Simulation

`Gui.btn_continue_simulation_clicked()`:

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

`stim.py` defines stimulation abstractions:

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

### 7.1 Main window and actions (`gui.py`)

`Gui` wires all actions:

- File: new/open/import/save/export/edit model + data imports.
- Simulate: run/continue/store checkpoint/restore checkpoint.
- Playback: speed, play/pause, skip, stop.
- View: show/hide panes, selection text toggles.
- Settings: Brian2 backend, stim padding, synapse RAM limit.
- Experiment: stimulation script generator + external simulation integration.

### 7.2 Shared plotting base (`fpl_widget.py`)

`FPLWidget` wraps fastplotlib figure/subplot creation and common cleanup.

### 7.3 Timeseries base (`timeseries_widget.py`)

`TimeseriesWidget` adds:

- pointer interactions (select/draw),
- ROI selection overlays,
- hover text,
- timeline marker,
- gridline layer,
- context menu for custom analysis on selected slices.

### 7.4 Specific viewers

`trace_widget.py`:

- stacked line plot per neuron,
- normalization dialog (percent, center median, clip).

`spike_widget.py`:

- scatter plot spike raster (time vs neuron index).

`heatmap_widget.py`:

- time-indexed synapse matrix image,
- optional subtraction against reference timestamp,
- LUT/histogram panel.

`data_widget.py`:

- stacked wrapper that selects trace/spike/heatmap viewer by incoming `Data` type.

### 7.5 Camera and timeline controls

`constrained_pan_zoom_controller.py`:

- clamps zoom/pan to valid data extents,
- keeps optional x/y scrollbars synchronized,
- updates gridline density visibility by zoom.

`time_slider_widget.py`:

- slider + timer-based playback loop,
- adjustable relative speed,
- broadcasts time updates to connected widgets.

`gridlines.py` / `timeline.py` / `text.py` / `region_of_interest.py`:

- visual helper graphics for grid, current time marker, hover label, and selection rectangle.

### 7.6 Custom analysis pane (`custom_analysis_widget.py`)

Offers quick histogram and 2D histogram analyses over selected neuron/time slices for trace or spike data.

### 7.7 Interactive console (`console_widget.py`)

Embedded Python REPL with namespace:

- `np`, `b2`, `project`.

After each command, emits update signal and refreshes project limits, enabling live model/stim edits.

### 7.8 Resource monitor (`resource_monitor_label.py`)

Periodic process stats in status bar:

- CPU, RAM,
- optional GPU/VRAM via NVML.

## 8) Stimulus editing subsystem

### 8.1 Stim editor (`stim_editor.py`)

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

### 8.2 Undo/redo (`stim_array_modifier.py`)

`StimArrayModifier` stores reversible slice actions with local undo/redo stack, and updates plot texture incrementally.

### 8.3 Stim visuals (`stim_widget.py`)

Heatmap-like display of stim values with colormap depending on `StimType`.
Supports background overlay of non-selected stim arrays.

### 8.4 Stim dialogs

`stim_dialog.py`:

- edit metadata (name/type/neurons/units).

`stim_intensity_dialog.py`:

- configure draw profile values with validation rules.

### 8.5 Stimulation script generator (`stimulation_generator_dialog.py`)

Provides export dialog for hardware script generation.
Current implementation status:

- hardware type list currently only `HOLO4D`,
- output extension currently always `.seq`,
- exported content currently is the editable HOLO4D config text block,
- additional appending of cellpose/stim data exists as commented code (inactive).

## 9) Data import pipelines

### 9.1 Suite2p import

`import_dialog.py` + `importer_suite2p.py`:

- reads `F/Fneu/spks/stat` from `.npy` or CSV,
- supports folder auto-detection and per-file selection,
- computes contrasted traces (`F - 0.7*Fneu`),
- per-neuron normalization for traces/spikes,
- optional spike extraction via thresholding,
- returns `TraceData` and optional `SpikeData`.

### 9.2 Cellpose import

`cellpose_import_dialog.py` + `importer_cellpose.py` + `cell_overlay_widget.py`:

- loads base image (PNG/TIFF stack mean-projection supported),
- loads segmentation (CSV, cellpose dict NPY, or suite2p stat+iscell),
- shifts to 0-based internal IDs with background `-1`,
- generates colored overlay and legend,
- supports mapping strategies between stimulus IDs and cellpose IDs:
  - linear offset-based,
  - custom pairs.
- preview overlay can show segmentation colors, cellpose IDs, and mapped stimulus IDs.

The result is stored as `CellposeImportResult` in project state.

### 9.3 Patch-clamp import

`patchclamp_import_dialog.py` + `importer_patchclamp.py`:

- loads 1D or 2D patch-clamp traces from `.npy`,
- mapping strategy for target stimulus/neuron IDs (linear/custom),
- integrates with existing cellpose mapping,
- runs peak-based spike detection with tunable parameters,
- returns `TraceData` (`Patch Clamp Traces`) plus per-neuron `SpikeData` (`Patch Clamp Spikes`).

### 9.4 Import mode dialog

`import_experiment_dialog.py` governs whether new imported data:

- creates a new experiment, or
- appends into existing experiment at selected timepoint,
  with validation/error messaging.

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

## 12) Example projects (`example_projects.py`)

Registered GUI examples include:

- `Getting Started Project`: HH-like network with synapses and stimuli.
- `non_linear`: compact nonlinear synapse dynamics demonstration.
- `Hodgkin-Huxley Neuron`: single-neuron electrophysiology/opto-style stim arrays.
- `Leaky-Integrate-and-Fire Neuron`: single LIF with multiple stim channels.
- `Network of Hodgkin-Huxley Neurons`: larger E/I network with synapses.
- `Network of Hodgkin-Huxley Neurons with STDP`: HH network + plasticity rules.
- `ChR2 in HEK293`: PyRho 6-state ChR2 optogenetic model.

These functions create fully configured `Project` instances by scripting `B2Model.define_*` plus stim arrays.

## 13) External simulation plugin interface

`gui.py` can load external Python function `simulate(...)` via `runpy.run_path`.

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

The following table maps each `src/neuroseq` file to primary functionality.

| Module                               | Primary responsibility                                                   |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `__init__.py`                        | package version export                                                   |
| `__main__.py`                        | CLI entrypoint and Qt app bootstrap                                      |
| `b2_model.py`                        | Brian2 model definition/assembly/simulation/serialization                |
| `b2_units.py`                        | Brian2 quantity parsing/display helpers                                  |
| `cell_overlay_widget.py`             | cellpose overlay visualization widget                                    |
| `cellpose_import_dialog.py`          | GUI flow for cellpose image+seg import and mapping                       |
| `console_widget.py`                  | embedded Python REPL bound to project                                    |
| `constrained_pan_zoom_controller.py` | bounded camera + scrollbar sync                                          |
| `custom_analysis_widget.py`          | histogram/hist2d analysis views                                          |
| `data.py`                            | core dataclasses (`TraceData`, `SpikeData`, `SynapseData`, `Experiment`) |
| `data_widget.py`                     | polymorphic trace/spike/heatmap container                                |
| `example_projects.py`                | built-in preconfigured projects                                          |
| `fpl_widget.py`                      | base fastplotlib Qt wrapper                                              |
| `gridlines.py`                       | layered grid and stripes graphics                                        |
| `gui.py`                             | main window/controller and action wiring                                 |
| `gui_settings.py`                    | persistent app settings                                                  |
| `heatmap_widget.py`                  | synapse matrix viewer over time                                          |
| `import_dialog.py`                   | Suite2p/CSV import dialog                                                |
| `import_experiment_dialog.py`        | append-vs-new experiment import chooser                                  |
| `importer_cellpose.py`               | cellpose loading, normalization, overlays, mapping payload               |
| `importer_patchclamp.py`             | patch-clamp conversion and spike extraction pipeline                     |
| `importer_suite2p.py`                | suite2p contrast/normalize/convert pipeline                              |
| `io.py`                              | zip/json/npy persistence and typed reconstruction                        |
| `model_editor.py`                    | parameter-tree model editing dialog                                      |
| `patchclamp_import_dialog.py`        | patch-clamp import GUI and options                                       |
| `pretty_print_enum.py`               | enum string formatting helpers                                           |
| `project.py`                         | aggregate project state and orchestration                                |
| `region_of_interest.py`              | ROI graphics + text                                                      |
| `resource_monitor_label.py`          | CPU/RAM/GPU status label                                                 |
| `save_load_options.py`               | save/import include-options dialog                                       |
| `spike_detection.py`                 | threshold and peak-based spike detection                                 |
| `spike_widget.py`                    | spike raster plot widget                                                 |
| `stim.py`                            | stimulation dataclasses/enums                                            |
| `stim_array_modifier.py`             | undo/redo action engine for stim edits                                   |
| `stim_dialog.py`                     | stim metadata editor dialog                                              |
| `stim_editor.py`                     | interactive stim editing workstation                                     |
| `stim_intensity_dialog.py`           | draw-intensity profile dialog                                            |
| `stim_widget.py`                     | stim image plot and overlays                                             |
| `stimulation_generator_dialog.py`    | hardware script export dialog                                            |
| `synthetic_data_dialog.py`           | synthetic patch/imaging data generation                                  |
| `text.py`                            | hover text graphic                                                       |
| `time_slider_widget.py`              | playback/seek timeline controller                                        |
| `timeline.py`                        | current-time vertical marker graphic                                     |
| `timeseries_widget.py`               | shared timeseries interaction base class                                 |
| `trace_widget.py`                    | trace line-stack viewer                                                  |
| `utils.py`                           | cross-cutting helpers                                                    |

## 17) Test suite coverage map (high-level)

The `tests/` folder validates major subsystems:

- Model core: `test_b2model.py`, `test_b2_units.py`, `test_main.py`.
- GUI integration/actions: `test_gui.py`, `test_gui_settings.py`.
- Visualization widgets: trace/spike/heatmap/time-slider/gridlines/data-widget tests.
- Editing tools: stim editor/dialog/intensity/modifier tests.
- Importers/dialogs: suite2p, cellpose, patch-clamp, experiment import tests.
- Persistence and project logic: `test_io.py`, `test_project.py`, `test_save_load_options.py`.
- Utilities: `test_utils.py`.

This means the documented functionality above is not only implemented but broadly unit/integration tested across core areas.

## 18) Practical caveats observed in code

- `stimulation_generator_dialog.py` currently exports mostly static/config text (`.seq`) despite taking stim/cellpose inputs.
- `StimWidget` comments note `dt_ms != 1` caveat for shared camera assumptions with image-based x-spacing.
- `utils.get_stim_type(...)` infers type from stim name as fallback heuristic.
- Some experimental/support paths are explicitly marked as limited or less rigorously tested in comments (e.g., suite2p stat-to-seg conversion notes).
