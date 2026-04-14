# NeuroSeq — Claude Code Guide

## Project Overview

NeuroSeq is a Qt-based desktop GUI application for computational neuroscience. It combines Brian2 spiking neural network simulation with interactive stimulus design, multi-source experimental data import, and hardware export for holographic stimulation (HOLO4D `.seq` files).

**Entry point**: `neuroseq` CLI → `src/neuroseq/__main__.py`

## Development Setup

```bash
# Install and run (uv recommended)
uv run neuroseq

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_gui.py

# Serve docs locally
uv run mkdocs serve   # → http://127.0.0.1:8000/neuroseq
```

Pre-commit hooks use **ruff** (lint + format), **mypy**, **prettier**, and **nbstripout**. Run `prek install` to set them up.

## Architecture

```
Project (project.py)          — top-level state container
├── B2Model (b2_model.py)     — Brian2 simulation engine (neurons, synapses, monitors, checkpoints)
├── StimArray (stim.py)       — stimulation sequences
├── Experiment (data.py)      — imported experimental data (traces, spikes, synapses)
└── Cellpose info             — cell segmentation for hardware mapping

gui.py                        — main window, menu/toolbar wiring, pane management
io.py                         — .nseq archive persistence (ZIP + JSON + .npy blobs)
```

### Key source files

| File                              | Purpose                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| `b2_model.py`                     | Brian2 network construction, parameter management, state checkpointing (~2100 lines) |
| `gui.py`                          | Main window, all action wiring (~1300 lines)                                         |
| `stim_editor.py`                  | Interactive 2D stim editing, undo/redo (~1200 lines)                                 |
| `exporter_syscon2.py`             | Hardware script generation for HOLO4D (~3200 lines)                                  |
| `stimulation_generator_dialog.py` | Export config GUI and sanity checks (~900 lines)                                     |
| `model_editor.py`                 | Parameter-tree GUI for model editing (~1000 lines)                                   |
| `data.py`                         | Core dataclasses: `TraceData`, `SpikeData`, `SynapseData`, `Experiment`              |
| `stim.py`                         | `StimArray`, `StimType` enum, draw profile types                                     |
| `utils.py`                        | Cross-cutting helpers: Brian2 logging, decorators, synapse matrix construction       |
| `example_projects.py`             | 7 built-in preconfigured projects (~1300 lines)                                      |
| `conftest.py` (tests/)            | Comprehensive pytest fixtures for Qt, Brian2, projects (~700 lines)                  |

### Visualization widgets (all use fastplotlib for GPU-accelerated rendering)

- `trace_widget.py` — stacked neuron voltage traces
- `spike_widget.py` — spike raster scatter plot
- `heatmap_widget.py` — synapse weight matrix
- `stim_widget.py` — stimulus array heatmap
- `custom_analysis_widget.py` — histogram analysis on selections
- `timeseries_widget.py` — base class for shared pan/zoom/ROI interaction

### Import pipelines

- Suite2p fluorescence imaging: `importer_suite2p.py` + `import_dialog.py`
- Cellpose segmentation: `importer_cellpose.py` + `cellpose_import_dialog.py` + `cell_overlay_widget.py`
- Patch-clamp electrophysiology: `importer_patchclamp.py` + `patchclamp_import_dialog.py`
- SysCon2 holographic sequences: `importer_syscon2.py` + `syscon2_import_dialog.py`
- Igor2 binary waveforms: `importer_igor2.py` + `igor_import_dialog.py`

## Code Style & Tooling

- **Linter/formatter**: `ruff` — line length limit ignored (`E501` ignored), isort with `force-single-line = true`
- **Type checking**: `mypy` with `--ignore-missing-imports`, stubs for numpy and PySide6
- **Qt abstraction**: use `qtpy` imports (not `PyQt5`/`PySide6` directly) for portability
- **Test coverage**: minimum 75% (both full suite and diff coverage via `diff-cover`)
- **Test markers**: `@pytest.mark.screenshot` for screenshot-based tests

## Testing

Tests are in `tests/`. The `conftest.py` provides fixtures for:

- `qtbot` (pytest-qt)
- Brian2 model instances at various configurations
- Full project instances with sample data
- Synthetic trace/spike/synapse data

Run with `uv run pytest`. Coverage reports use `pytest-cov`; memory profiling available via `pytest-memray` (Linux/macOS only).

## File Format

Projects are saved as `.nseq` archives: ZIP files containing a JSON metadata file and `.npy` blobs for numerical arrays. See `io.py` for serialization logic.

## Important Notes

- `pyside6!=6.9.1` and `pyqtgraph!=0.14.0` are pinned exclusions due to known bugs
- `pyparsing!=3.2.4` excluded due to SysCon2 parsing regression
- GPU monitoring uses `nvidia-ml-py` (optional, gracefully absent on non-NVIDIA systems)
- The `external_simulation_example/` directory is excluded from ruff and mypy
- Example models in `src/neuroseq/example_models/` are excluded from pre-commit hooks
