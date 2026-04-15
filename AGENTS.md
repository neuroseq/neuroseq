# Repository Guidelines

## Project Structure & Module Organization

`src/neuroseq/` contains the installable package and most application code. The layout is intentionally flat: GUI entry lives in `__main__.py`, widgets and dialogs use names like `*_widget.py` and `*_dialog.py`, and import/export code uses `importer_*.py` and `exporter_*.py`. `tests/` mirrors that surface area with `test_*.py` files, shared fixtures in `tests/conftest.py`, and sample inputs in `tests/sampledata/`. `docs/` holds MkDocs content, with developer notes under `docs/developer/`, user docs under `docs/user_guide/`, and images in `docs/images/`. `external_simulation_example/`, `examplary_data/`, and `notebooks/` are supporting material, not core package code.

## Build, Test, and Development Commands

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

Tests use `pytest` and `pytest-qt`. Add or update tests with every behavior change, especially for GUI flows, import/export logic, and data transformations. Keep test files named `test_<feature>.py`. If you need CI parity for Qt, run with `QT_API=pyside6` and `PYTEST_QT_API=pyside6`. Coverage must stay at or above the configured 75% threshold.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit subjects such as `fix CI issues` or scoped summaries like `fix tests: use correct SysCon2 dt`. Keep the first line concise and descriptive. Pull requests should explain the user-visible change, link the relevant issue or discussion, list the commands you ran, and include screenshots for UI changes. Do not commit generated files such as `.coverage`, `coverage.xml`, cache directories, or `.venv/`.
