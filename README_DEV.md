# Developer Guide

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
prek install
```

The documentation is in the docs folder as markdown and rendered using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/getting-started/).
To serve the documentation locally run:

```bash
uv run mkdocs serve
```

Then go to http://127.0.0.1:8000/neuroseq to see the generated documentation.

## Developer Installation without uv

Alternatively, if you aren't using uv, then create a new conda environment or virtual environment and activate it.
Then do an editable install of the package (which means that any changes to the source code will be reflected in the installed package without needing to reinstall),
and install the pre-commit hooks:

```bash
pip install -e . --group dev
pip install prek
prek install
```

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

## Connection to Holo4D profile

C:\ProgramData\ROE\SysCon 2\2.7.2\profiles\Holo4D\Sequence Manager\Experiments\NeuroSeq_tests
