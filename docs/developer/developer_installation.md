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
git clone git@github.com:ssciwr/neuroseq.git
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

## Developer Installation without uv

Alternatively, if you aren't using uv, then create a new conda environment or virtual environment and activate it.
Then do an editable install of the package (which means that any changes to the source code will be reflected in the installed package without needing to reinstall),
and install the pre-commit hooks:

```bash
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
[installation and environment selection](../getting_started/installation.md#using-uv-from-a-conda-shell).

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
