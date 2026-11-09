# Developer installation

Use either uv (recommended) or Conda. You need only one of them.

## With uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
and [prek](https://prek.j178.dev/) (a fast [pre-commit](https://pre-commit.com/)), then:

```bash
uv tool install prek
git clone git@github.com:neuroseq/neuroseq.git
cd neuroseq
uv sync --group dev
prek install
```

`uv sync` creates `.venv` from `pyproject.toml` and `uv.lock`; `uv run` uses it
without activation. Run `uv sync` again after pulling. To upgrade dependencies,
run `uv lock --upgrade`, then `uv sync`, and test (this changes `uv.lock`).

## With Conda

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

`--group` needs a recent pip. Pip ignores `uv.lock` and `[tool.uv.sources]`, so
package versions and PyTorch builds can differ from the uv setup. Drop `uv run`
from the commands in these docs (for example `python -m pytest`).

!!! warning "uv inside Conda"

    If uv is installed in an active Conda environment, `uv run` still uses the
    repository's `.venv`, not Conda's Python. `conda run -n neuroseq python` and
    `uv run python` can then load different packages.

Check which interpreter and versions you are using (point your IDE at the same one):

```bash
uv run python -c "import sys, brian2, numpy; print(sys.executable, brian2.__version__, numpy.__version__)"
```

In the embedded console, `import sys; print(sys.executable)` shows the running interpreter.

## Everyday commands

| Task            | Command                                                                                 |
| --------------- | --------------------------------------------------------------------------------------- |
| Launch the GUI  | `uv run neuroseq`                                                                       |
| Debug logging   | `uv run neuroseq --log-level DEBUG`                                                     |
| Run tests       | `uv run pytest` (see [Testing](testing.md))                                             |
| Run all hooks   | `prek run -a`                                                                           |
| Serve docs      | `uv run zensical serve` (see [Documentation](documentation.md))                         |
| Profile the GUI | `uv run python -m cProfile -o latest.prof -m neuroseq`, then `uvx snakeviz latest.prof` |
