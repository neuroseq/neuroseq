# Documentation

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## To run docs locally

For the uv workflow, install the development dependencies:

```bash
uv sync --group dev
```

Start the local documentation server from the repository root:

```bash
uv run mkdocs serve
```

Then open the local URL shown in the terminal (`http://127.0.0.1:8000`).

If NeuroSeq is installed directly into Conda, activate that environment,
install the documentation packages with `python -m pip install mkdocs-material`,
and run `python -m mkdocs serve` instead. See
[environment selection](../getting_started/installation.md#using-uv-from-a-conda-shell).

## Optional: build static docs

To generate static files for deployment:

```bash
uv run mkdocs build
```

The generated site is written to the `site/` folder.
