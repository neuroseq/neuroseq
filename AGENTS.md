# Repository Guidelines

## Project Structure & Module Organization

`src/neuroseq/` contains the installable package and most application code. GUI entry lives in `__main__.py`; visualization widgets and their plotting base classes live in `visualization/`, including `stim_widget.py`. Stimulation editing logic remains in `stimulation/`, import code in `importers/`, and export code in `export/`. Other shared modules, widgets, and dialogs remain directly under `src/neuroseq/`, using names like `*_widget.py` and `*_dialog.py`. `tests/` mirrors that surface area with `test_*.py` files, shared fixtures in `tests/conftest.py`, and sample inputs in `tests/sampledata/`. `docs/` holds MkDocs content, with developer notes under `docs/developer/`, user docs under `docs/user_guide/`, and images in `docs/images/`. `external_simulation_example/`, `example_data/`, and `notebooks/` are supporting material, not core package code.

`visualization_support/` contains shared plot navigation, gridlines, selection regions, playback controls, trace/spatial view state, neuron selection, and brightness/contrast controls.

`example_projects/` contains one project factory per file, grouped under `neurons/` and `networks/`. Its package entry point exports the existing factory names and registers projects in menu order; `_registry.py` contains the registration decorator.

## NeuroSeq Functionality Overview

NeuroSeq is a Qt desktop application for designing neuronal stimulation sequences, running Brian2-based network simulations, importing experimental data, and comparing simulated and recorded activity in one workspace. The main window is assembled in `src/neuroseq/gui.py`; shared project state lives in `src/neuroseq/project.py`; the simulation/model layer lives in `src/neuroseq/b2_model.py`; canonical data structures live in `src/neuroseq/data.py`; and project serialization lives in `src/neuroseq/io.py`.

Key user-visible workflows:

- Stimulation editing: users work primarily in the stimulation editor (`stim_editor.py`, `stim_widget.py`, `stim.py`). NeuroSeq supports multiple stim arrays per project, draw/erase/select interactions, amplitude and duration controls, padding/trimming, array metadata editing, and showing multiple arrays together.
- Simulations and model editing: NeuroSeq can run and continue Brian2 simulations from the GUI, keep checkpoints, and load a default Hodgkin-Huxley-style example project. Model definitions include neuron groups, synapses, monitors, parameters, and initial values; editing paths are centered around `b2_model.py`, `model_editor.py`, and `example_projects/`.
- Data visualization and playback: trace, spike-raster, and synapse-heatmap views are first-class widgets (`trace_widget.py`, `spike_widget.py`, `heatmap_widget.py`) synchronized by the time slider (`time_slider_widget.py`) and shared pan/zoom controls. `data_widget.py` provides an extra generic viewer for comparing datasets.
- Analysis and scripting: selected regions can be sent to the custom analysis pane/window (`custom_analysis_widget.py`). The embedded Python console (`console_widget.py`) exposes project/model/stim objects for inspection and scripted stimulus manipulation, but direct low-level Brian2 object mutation can desynchronize GUI state.
- Experimental data import/export: supported flows include Suite2p, Cellpose, patch-clamp, Igor2, and SysCon2 import paths (`importer_*.py`, `*_import_dialog.py`) plus exporter flows such as Igor2 patch-clamp CSV and SysCon2 sequence export (`exporter_*.py`). There is also Cellpose synthetic test-data generation exposed from `__main__.py`.
- Experiments and persistence: `.nseq` project archives save/load model definitions, stimulus arrays, experiments, imported data, and stored simulation state. NeuroSeq can also import selected content from another project and manage project data via dialogs such as `data_manager_dialog.py` and `import_experiment_dialog.py`.

When working on a feature or bug, identify which of these surfaces is affected first: stimulation editing, simulation/model assembly, import/export, visualization, analysis, or project persistence. Tests usually mirror those surfaces directly, for example `tests/test_stim_editor.py`, `tests/test_b2model.py`, `tests/test_importer_*.py`, `tests/test_exporter_*.py`, `tests/test_io.py`, and `tests/test_gui.py`.

## Build, Test, and Development Commands

Use the Conda environment named `neuroseq` for all commands in this repository so the expected toolchain is available, including `rg`. Prefer `conda run -n neuroseq <command>` for one-off commands if shell activation is not already in place.

Use `uv` for the fastest local workflow:

- `uv run neuroseq` launches the GUI from the current checkout.
- `uv run pytest -sv` runs the full test suite.
- `uv run pytest tests/test_gui.py -sv` runs one test module.
- `uv run pytest --cov --cov-report=xml` generates coverage data used in CI.
- `uv run diff-cover coverage.xml -c pyproject.toml` checks the 75% diff-coverage gate.
- `prek run -a` or `pre-commit run -a` runs lint, format, mypy, notebook cleanup, and workflow checks.
- `uv run mkdocs serve` serves the docs locally.

Without `uv`, install in editable mode with `pip install -e . --group dev`.

## Coding Style & Naming Conventions

Target Python 3.10+ and use 4-space indentation. Keep names in `snake_case` for modules, functions, and tests. Follow the existing flat-module conventions instead of adding deep package nesting unless a new subsystem clearly needs it. Ruff handles formatting and import sorting; pre-commit also runs mypy, so add type hints where they improve public APIs and test fixtures.

## Testing Guidelines

Tests use `pytest` and `pytest-qt`. Add or update tests with every behavior change, especially for GUI flows, import/export logic, and data transformations. Keep test files named `test_<feature>.py`. If you need CI parity for Qt, run with `QT_API=pyqt6` and `PYTEST_QT_API=pyqt6`. Coverage must stay at or above the configured 75% threshold.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit subjects such as `fix CI issues` or scoped summaries like `fix tests: use correct SysCon2 dt`. Keep the first line concise and descriptive. Pull requests should explain the user-visible change, link the relevant issue or discussion, list the commands you ran, and include screenshots for UI changes. Do not commit generated files such as `.coverage`, `coverage.xml`, cache directories, or `.venv/`.
