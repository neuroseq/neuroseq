# Developer Guide on installation and use

Choose either uv or a direct Conda installation. uv can manage Python and the
project `.venv` independently, so it does not require Conda. The two workflows
are alternatives; you do not need to set up both.

## Developer Installation with uv

First ensure you have [installed uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Then you can install [prek](https://prek.j178.dev/) (a faster implementation of [pre-commit](https://pre-commit.com/)) using uv:

```bash
uv tool install prek
```

Then clone the repository and set up the pre-commit hooks for this repository:

```bash
git clone git@github.com:neuroseq/neuroseq.git
cd neuroseq
uv sync --group dev
prek install
```

The documentation is in the docs folder as Markdown and rendered using [Zensical](https://zensical.org/docs/).
To serve the documentation locally run:

```bash
uv run zensical serve
```

Open the URL shown in the terminal to see the generated documentation.
See [Documentation](documentation.md) for build checks and deployment details.

`uv sync` creates a project environment in `.venv` and installs dependencies
using `pyproject.toml` and `uv.lock`. `uv run` checks that environment before
launching a command, so no manual activation of `.venv` is needed. See
[uv's project workflow](https://docs.astral.sh/uv/guides/projects/).

After pulling repository updates, run `uv sync` again. To deliberately resolve
newer dependency versions, use `uv lock --upgrade` followed by `uv sync` and
test the result; this changes `uv.lock`.

## Developer Installation without uv

Alternatively, if you aren't using uv, then create a new conda environment or virtual environment and activate it.
Then do an editable install of the package (which means that any changes to the source code will be reflected in the installed package without needing to reinstall),
and install the pre-commit hooks:

```bash
git clone git@github.com:neuroseq/neuroseq.git
cd neuroseq
conda create -n neuroseq python=3.12
conda activate neuroseq
python -m pip install --upgrade pip
python -m pip install -e . --group dev
python -m pip install prek
prek install
```

The `--group dev` option requires a recent pip. This installs directly into
Conda and does not reproduce `uv.lock` or uv's package-index configuration.
If you install uv inside Conda and then use `uv sync` / `uv run`, the project
uses its separate `.venv` by default. See
[Using uv from a Conda shell](#using-uv-from-a-conda-shell).

Pip does not use `uv.lock` or `[tool.uv.sources]`, so this installation can have
different package versions and PyTorch builds from the uv workflow.

## Using uv from a Conda shell

Combining uv and Conda is optional, not a third required setup step. You may
install uv in an activated Conda environment:

```bash
conda activate neuroseq
python -m pip install uv
uv sync
uv run neuroseq
```

With the default project configuration, `uv run` uses the repository's `.venv`,
even when Conda is active. Conda supplies the shell/tool installation; the
application dependencies remain in the separate uv environment. In particular,
`conda run -n neuroseq python` and `uv run python` can load different packages.
Updating one does not repair or update the other.

For a fully independent uv setup, install standalone uv using the link above
and run it from a shell without Conda activated. If your current uv executable
is installed inside Conda, install standalone uv before removing that Conda
environment. An existing project `.venv` may also use a Conda-provided Python;
when retiring that Python installation, recreate `.venv` with a uv-managed
Python interpreter.

To identify the environment and Brian2/NumPy versions actually used, run:

```bash
uv run python -c "import sys, brian2, numpy; print(sys.executable); print(brian2.__version__, numpy.__version__)"
```

For a direct Conda installation, run the same check with
`conda run -n neuroseq python -c "import sys, brian2, numpy; print(sys.executable); print(brian2.__version__, numpy.__version__)"`.
In the application's embedded console, `import sys; print(sys.executable)`
identifies the running interpreter. Point your IDE at that same interpreter.

## Developer Use

The GUI is launched with:

```bash
uv run neuroseq
```

(if you are not using uv, just run `neuroseq`)

To run with profiling:

```bash
uv run neuroseq --profile --profile-output latest.prof
```

and to see the results:

```bash
 uv run snakeviz latest.prof
```

To run the test suite:

```bash
uv run pytest
```

For the direct Conda installation, use `python -m pytest` in the activated
environment, or `conda run -n neuroseq python -m pytest`. Run application and
tests with the same interpreter when investigating dependency failures.
