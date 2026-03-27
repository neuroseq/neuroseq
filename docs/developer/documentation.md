# Documentation

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## To run docs locally

Install the required packages in conda environment:

```bash
pip install mkdocs mkdocs-material
```

Start the local documentation server from the repository root:

```bash
mkdocs serve
```

Then open the local URL shown in the terminal (`http://127.0.0.1:8000`).

## Optional: build static docs

To generate static files for deployment:

```bash
mkdocs build
```

The generated site is written to the `site/` folder.
