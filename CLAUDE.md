# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# NeuroSeq — Claude Code Guide

## Project Overview

NeuroSeq is a Qt-based desktop GUI application for computational neuroscience. It combines Brian2 spiking neural network simulation with interactive stimulus design, multi-source experimental data import, and hardware export for holographic stimulation (HOLO4D `.seq` files).

**Entry point**: `neuroseq` CLI → `src/neuroseq/__main__.py`

## Experimental Context (Wiegert lab, Mannheim)

Reference: Baio, C. (2024–2025) "Multiphoton Holographic Optogenetics in Hippocampal Slice Culture to Manipulate and Measure Neural Activity at Single-Cell and Network Level". Master's thesis, Università degli Studi di Trieste / Universität Heidelberg (Prof. Cingolani, co-supervised by Prof. Wiegert and Dr. Formozov). Stored at `/mnt/live3/svintila/random/papers/Baio et al. - Multiphoton Holographic Optogenetics in Hippocampal Slice Culture to Manipulate and Measure Neural A.pdf`.

NeuroSeq supports a high-throughput all-optical pipeline that combines two-photon calcium imaging with two-photon holographic optogenetic stimulation on rat hippocampal organotypic slice cultures (hOSCs), targeting the CA3 module to study recurrent network connectivity.

### Biological prep

- **Tissue**: P5 rat pup hippocampal organotypic slice cultures, 300 µm cut on McIlwain Tissue Chopper, kept on Millicell-CM membranes in slice culture medium (MEM + 20% horse serum + insulin + glutamine + ascorbic acid).
- **Maturation**: imaging done at DIV 17 (~100 µm thick at that point).
- **Viral transduction at DIV 3** (always 5× dilution):
  - **Calcium indicator**: `rAAV9-CaMKIIa-jGCaMP8m` — **jGCaMP8m** is the chosen indicator; picked for the best speed/sensitivity tradeoff (faster decay than jGCaMP7f / GCaMP6s, still large ΔF/F per spike).
  - **Excitatory opsin**: `rAAV9-Syn-ChrimsonR-tdTomato` (red-light cation channel, tdTomato co-expression for identification).
  - **Bidirectional opsin** (optional): `rAAV2-9-CaMKIIa_somBiPOLES-mCerulean` (ChrimsonR–GtACR2 tandem for depolarisation/hyperpolarisation in one construct, tagged with mCerulean).

### Imaging acquisition (drives `importers/importer_tif.py` defaults)

- **Microscope**: RAPP HOLO-4D (Movable Objective Microscope + Ti:Sapphire laser + SLM holographic stimulation path).
- **2-photon imaging laser**: tunable 660–1320 nm + fixed 1040 nm. **GCaMP excitation at 940 nm.**
- **Detection**: 4 GaAsP PMTs (2 green, 2 red), plus Dodt contrast.
- **Spontaneous Ca activity recording**: **27 000 frames over 15 minutes at 30 Hz** (this is where `DEFAULT_FS = 30.0` and `tau = 0.5 s` in `src/neuroseq/importers/importer_tif.py` come from), 512×512 pixels, saved as multi-frame TIFF.
- **Pre-record acquisitions**: an anatomical Dodt snap (2048×2048) and a 41-slice 1 µm Z-stack for spatial registration.

### Holographic stimulation (drives `.seq` export and `stimulation/stim_editor.py` semantics)

- **2-photon stimulation lasers**: Carbide low-rep-rate / high-peak-pulse-power, two paths — fixed 1040 nm (up to ~40 W) and a tunable >/< 1000 nm path (~60 mW per output).
- **ChrimsonR stimulation**: 1140 nm, 11 pulses × 16.6 ms at 10 Hz, power swept 23 → 60 mW per spot. Spiral illumination (≥16.6 ms minimum due to spiral period, 10 µm spiral diameter typical).
- **BiPOLES reference (Vierock 2021)**: 1100 nm, 15 ms pulses at 10 Hz; AP probability curve saturates around 0.3 mW/µm².
- **Heating budget**: holographic illumination can raise local temperature by 1–5 °C; protocols are bounded by power × number of simultaneous spots.
- **Stimulation coordination software**: RAPP **SysCon** (driver for `.seq` exports — see `export/exporter_syscon2.py`, `importers/importer_syscon2.py`); imaging via _ScanImage_; laser control via _Shortcut Chameleon_, _Carbide_, _Topas4_; power via _ThorLabs_.

### Analysis pipeline (mirrors what `importer_tif.py` automates inside neuroseq)

The thesis processes raw TIFFs externally before this PR existed; that workflow is what `importer_tif.run_suite2p_pipeline` now collapses into the GUI:

1. **Cellpose** is run on the mean (or Z-projected) image to produce a mask of pyramidal somas.
2. A **suite2p-based custom Python script** (modelled on W. Yang 2021 and A. Formozov 2022) takes those masks, extracts per-neuron F/Fneu traces, does neuropil subtraction, baseline correction, and OASIS deconvolution.
3. **Fiji/ImageJ** is used for response heatmaps (per-stim ΔF over a 30-frame pre-stim baseline) and for projecting large TIFFs before Cellpose — the same role as the "Run Fiji" / Z-projection workflow documented in the TIFF import dialog.

### Phenotypes seen across slices (useful sanity checks when running pipelines)

- ~1/3 of slices are **silent**, ~1/3 show **highly synchronous network bursts**, ~1/3 are **sporadic** with isolated cellular / sub-population transients.
- ~**40% of opsin-expressing cells** respond to single-cell holographic stimulation (vs. ~1/5 of opsin-negative controls — small sample, not definitive).
- Single-cell stimulation rarely activates downstream network; reciprocal-connection probability in CA3 is low (≈1% in rats, up to ~10% short-range in mice).

## Development Setup

```bash
# Install and run (uv recommended)
uv run neuroseq

# Run tests (serial full suite ~6 min; -n auto parallelises it to ~40 s)
uv run pytest -n auto

# Run specific test file
uv run pytest tests/test_gui.py

# Serve docs locally
uv run mkdocs serve   # → http://127.0.0.1:8000/neuroseq
```

Pre-commit hooks use **ruff** (lint + format), **mypy**, **prettier**, and **nbstripout**. Run `pre-commit install` to set them up.

## Architecture

```
Project (project.py)          — top-level state container
├── B2Model (b2_model.py)     — Brian2 simulation engine (neurons, synapses, monitors, checkpoints)
├── StimArray (stimulation/stim.py) — stimulation sequences
├── Experiment (data.py)      — imported experimental data (traces, spikes, synapses)
└── Cellpose info             — cell segmentation for hardware mapping

gui.py                        — application shell; hosts one ProjectWorkspace per project tab
project_workspace.py          — per-tab workspace: menus/toolbars, action wiring, pane management
background_jobs.py            — worker threads and lifecycle for long-running main-window jobs
project_history.py            — Qt-free session undo/redo shared by a project's data editors
qt_helpers.py                 — shared generic Qt widgets / window helpers
io.py                         — .nseq archive persistence (ZIP + JSON + .npy blobs)
```

Packages under `src/neuroseq/`:

- `importers/` — importers and import dialogs (TIFF, Suite2p, Cellpose, patch clamp, SysCon2, ScanImage)
- `export/` — Igor2 and SysCon2 exporters, export dialogs, power LUTs
- `device_control/` — ScanImage, SysCon2 and BrainScaleS-2 connections plus the "Stimulation setup" pane (`stimulation_setup_widget.py`, one tab per target: the SysCon2/HOLO4D panel, the BrainScaleS-2 tab `bss2_panel.py`, and the ScanImage acquisition controls `scanimage_controls.py`, owned by `workspace_acquisition_controller.py`). Closed-loop hardware delivery: `syscon2_stimulation_host.py` runs a dynamic input routine's blocks through SysCon2 and, via the `BlockAcquisition` protocol the acquisition controller implements, records each one with ScanImage and extracts ΔF/F (`importers/importer_scanimage_traces.py`). BSS-2: `bss2_protocol.py` (Qt-free stim → segments/spikes, request/readback, launch command), `bss2_connection.py` (QProcess transport to `resources/bss2/neuroseq_bss2_helper.py`, which runs inside the EBRAINS container and must import only stdlib/numpy/the BSS-2 stack, never `neuroseq`)
- `stimulation/` — `StimArray`, stim editor and dialogs; `static_input/` (drawn patterns) and `dynamic_input/` (closed-loop routines: STDP, rheobase)
- `visualization/`, `visualization_support/` — plot widgets and their shared helpers (see below)
- `llm/` — chat assistant backends, settings, and plugin-write dialog
- `elise_support/` — ELiSe protocol dialog, workflows, replay metrics
- `example_projects/` — example project factories (see table below)

### How files are grouped (follow this when adding or moving code)

The code is grouped **by feature/workflow, not by kind of file**: an importer, its
dialog and its helpers sit together in one package rather than in global
`dialogs/` or `widgets/` folders. When adding code:

- **Pick the package by workflow.** Data coming into a project → `importers/`
  (`importer_<source>.py` holds the Qt-free parsing, `<source>_import_dialog.py`
  the GUI). Data leaving a project → `export/` (`exporter_<target>.py` + dialogs).
  Talking to lab hardware/software (ScanImage, SysCon2/HOLO4D, BrainScaleS-2) →
  `device_control/`.
  Designing stimuli → `stimulation/`: drawn/static patterns in
  `stimulation/static_input/`, closed-loop routines (rheobase, STDP) in
  `stimulation/dynamic_input/`. Plotting → `visualization/`. Model-specific support
  (ELiSe) → its own package (`elise_support/`), not core modules.
- **Separate logic from widgets.** Keep Qt-free logic in its own module so it can
  be tested without a GUI. Coordination between widgets/dialogs and the project goes
  in a controller (`static_input_controller.py`, `dynamic_input_controller.py`,
  `workspace_acquisition_controller.py`), not in the widget and not in
  `project_workspace.py`. `project_workspace.py` should only build menus/panes and
  delegate; `gui.py` is only the tab shell.
- **Layering.** `visualization_support/` and `stimulation/` must not import
  `neuroseq.visualization` at runtime (only inside `if TYPE_CHECKING:`);
  `tests/test_import_layering.py` enforces this. The single allowed exception
  is `stimulation/stim_editor.py`, which embeds `StimWidget`.
- **Shared infrastructure, not copies.** Generic Qt helpers go in `qt_helpers.py`.
  Every finite background task runs through `background_jobs.BackgroundJob`, never
  a hand-built `QThread`. Every change to project data goes through `Project`
  methods (`replace_stim_values`, `add_experiment`, `edit_experiments`, …) so the
  shared undo history (`project_history.py`) is recorded or cleared; never assign
  `stim_array.values`/`.dt` or `project.experiments[...]` from a dialog.
- **Plugin folder is not a code location.** `resources/analysis_plugins/` holds
  only thin, auto-discovered analysis plugins; core code must never import from it.
  LLM-authored plugins are written (after user confirmation) to the per-user folder
  from `analysis_plugins.user_plugins_directory()`.
- **Example projects** are one factory per file under `example_projects/neurons/`
  or `example_projects/networks/`, registered with `@register_example_project`, and
  must be imported in `example_projects/__init__.py` (inside the `isort: off`
  block — import order is menu order).
- **Moves are clean breaks.** Use absolute `neuroseq.<package>.<module>` imports,
  and don't leave re-export shims at old paths. Update imports, test monkeypatch
  string targets, docs, and this file in the same commit. Tests stay flat in
  `tests/` as `test_<module>.py`.

### Key source files

| File                                     | Purpose                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `b2_model.py`                            | Brian2 network construction, parameter management, state checkpointing (~2700 lines) |
| `gui.py`                                 | Main window shell and project tabs (~350 lines)                                      |
| `project_workspace.py`                   | Per-tab workspace, menus, all action wiring (~3900 lines)                            |
| `stimulation/stim_editor.py`             | Interactive 2D stim editing, undo/redo (~1600 lines)                                 |
| `export/exporter_syscon2.py`             | Hardware script generation for HOLO4D (~4000 lines)                                  |
| `export/stimulation_generator_dialog.py` | Export config GUI and sanity checks (~1300 lines)                                    |
| `model_editor.py`                        | Parameter-tree GUI for model editing (~1200 lines)                                   |
| `data.py`                                | Core dataclasses: `TraceData`, `SpikeData`, `SynapseData`, `Experiment`              |
| `stimulation/stim.py`                    | `StimArray`, `StimType` enum, draw profile types                                     |
| `utils.py`                               | Cross-cutting helpers: Brian2 logging, decorators, synapse matrix construction       |
| `example_projects/`                      | Individual project factories grouped into `neurons/` and `networks/`                 |
| `conftest.py` (tests/)                   | Comprehensive pytest fixtures for Qt, Brian2, projects (~900 lines)                  |

### Visualization widgets

Visualization widgets live in `src/neuroseq/visualization/`. Interactive data views use fastplotlib for GPU-accelerated rendering; custom analysis uses Matplotlib.

Shared plot navigation, view state, selection, and brightness controls live in `src/neuroseq/visualization_support/` (which must not import `visualization/`; the playback time slider lives in `visualization/`).

- `trace_widget.py` — stacked neuron voltage traces
- `spike_widget.py` — spike raster scatter plot
- `heatmap_widget.py` — synapse weight matrix
- `stim_widget.py` — stimulus array heatmap
- `custom_analysis_widget.py` — histogram analysis on selections
- `timeseries_widget.py` — base class for shared pan/zoom/ROI interaction

### Import pipelines

All importers live in `src/neuroseq/importers/` (`cell_overlay_widget.py` is in `visualization/`):

- Raw 2-photon TIFF (Cellpose + suite2p pipeline): `importer_tif.py` + `tif_import_dialog.py`
- Suite2p fluorescence imaging: `importer_suite2p.py` + `suite2p_import_dialog.py`
- Cellpose segmentation: `importer_cellpose.py` + `cellpose_import_dialog.py` + `cell_overlay_widget.py`
- Patch-clamp input/output (Igor2 `.ibw`/`.pxp`, `.csv`/`.txt`): `importer_patch_clamp_data.py` + `patch_clamp_data_input_import_dialog.py` / `patch_clamp_data_output_import_dialog.py`
- SysCon2 holographic sequences: `importer_syscon2.py` + `syscon2_import_dialog.py`
- ScanImage TIFF recordings (indexing without loading movies): `importer_scanimage.py`

## Code Style & Tooling

- **Linter/formatter**: `ruff` — line length limit ignored (`E501` ignored), isort with `force-single-line = true`
- **Type checking**: `mypy` with `--ignore-missing-imports`, stubs for numpy and Qt bindings via qtpy
- **Qt abstraction**: use `qtpy` imports (not `PyQt6`/`PySide6` directly) for portability
- **Test coverage**: minimum 75% (both full suite and diff coverage via `diff-cover`)
- **Test markers**: `@pytest.mark.screenshot` for screenshot-based tests

## Testing

Tests are in `tests/`. The `conftest.py` provides fixtures for:

- `qtbot` (pytest-qt)
- Brian2 model instances at various configurations
- Full project instances with sample data
- Synthetic trace/spike/synapse data

Run with `uv run pytest`. Coverage reports use `pytest-cov`; memory profiling available via `pytest-memray` (Linux/macOS only).

### Screenshot tests

Screenshot tests are skipped by default. To update reference screenshots instead of running all other tests:

```bash
uv run pytest --update-screenshot
```

### Qt testing patterns

Two `autouse` fixtures apply to every test automatically:

- **`suppress_message_boxes`** — patches `QMessageBox.information/warning/critical` to return `Ok` silently.
- **`patch_b2w_exec_show`** — replaces `.exec()` with `.show()` on `QDialog` and `QMessageBox` to prevent event-loop blocking.

Use the opt-in **`qmessagebox_calls`** fixture to capture dialog calls and assert on them (supports both static `QMessageBox` methods and `StimulationGeneratorDialog._show_scrollable_message`).

### Brian2 in tests

`conftest.py` sets `b2.prefs.codegen.target = "numpy"` globally for all tests, disabling C++ code generation for speed and reproducibility.

## File Format

Projects are saved as `.nseq` archives: ZIP files containing a JSON metadata file and `.npy` blobs for numerical arrays. See `io.py` for serialization logic.

## Important Notes

- `pyqtgraph!=0.14.0` is excluded due to known bugs
- `pyparsing!=3.2.4` excluded due to SysCon2 parsing regression
- GPU monitoring uses `nvidia-ml-py` (optional, gracefully absent on non-NVIDIA systems)
- The `external_simulation_example/` directory is excluded from ruff and mypy
- Example models in `src/neuroseq/example_models/` are excluded from pre-commit hooks
- The **ELISE** example model (`src/neuroseq/example_models/elise.py`) imports a separate
  private `elise` package (`elise.model`/`optimizer`/`rate_buffer`/`weights`). It is not a
  declared dependency, so running the ELISE example raises a clear `ImportError` until it is
  installed into the active venv. The source lives at
  `/mnt/live1/svintila/sequence-learning` (package `elise` v0.3.x, `src/` layout). Install it
  editable without deps — the imported modules need only `numba`+`numpy`, both already present:

  ```bash
  VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps -e /mnt/live1/svintila/sequence-learning
  ```

  The `VIRTUAL_ENV` prefix is required: a bare `uv pip` resolves to the conda env at
  `~/conda-ws/envs/neuroseq`, not the project `.venv` that `uv run` uses, so it reports a
  successful install while `uv run python -c "import elise"` keeps failing. `uv run` does
  not prune the editable install once it is in the right venv. If you later hit
  `ModuleNotFoundError` for `mlflow`/`networkx`/`tomli_w`/`coolname`/`tqdm` (pulled in by
  `elise.simulator`/`main`/`data`), do a full `uv pip install -e` of that path instead (same
  `VIRTUAL_ENV` prefix).

- When `elise` is missing, ~27 ELISE tests skip rather than fail, so `uv run pytest` still
  reports a green suite. If a full-suite run shows ~29 skipped instead of ~2, that install has
  gone missing — reinstall it before comparing pass counts against a baseline.
