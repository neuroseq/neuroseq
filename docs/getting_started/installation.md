# Installation

NeuroSeq requires Python 3.12 or newer. Choose one of the following workflows
and use the same environment for launching, testing, and troubleshooting.

| Option         | Environment                    | Requires Conda? | Launch command              |
| -------------- | ------------------------------ | --------------- | --------------------------- |
| uv             | Project `.venv`, managed by uv | No              | `uv run neuroseq`           |
| Conda with pip | Named Conda environment        | Yes             | `neuroseq` after activation |

## Option 1: Repository installation with uv

uv works independently of Conda and can download a suitable Python interpreter
when needed. You do not need to create or activate a Conda environment for
this option.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone git@github.com:ssciwr/neuroseq.git
cd neuroseq
uv sync
uv run neuroseq
```

GitHub SSH access is required for this clone URL. `uv sync` creates a project
environment in `.venv` and installs dependencies using `pyproject.toml` and
`uv.lock`. `uv run` checks that environment before launching a command.
No manual activation of `.venv` is needed. See
[uv's project workflow](https://docs.astral.sh/uv/guides/projects/).

After pulling repository updates, run `uv sync` again. To deliberately resolve
newer dependency versions, use `uv lock --upgrade` followed by `uv sync` and
test the result; this changes `uv.lock`.

## Option 2: Conda

Conda is an alternative to the project `.venv`. Install
[Anaconda](https://www.anaconda.com/) if needed.

Once Anaconda installed, create and activate a new environment in a command line:

```
conda create -n neuroseq python=3.12
conda activate neuroseq
```

Alternatively, the new enviroment can be created within Anaconda Navigator. For this, open Anaconda Navigator and go to "Enviroments". Click "Create" button, write "neuroseq" as a name and select Python 3.12.

### Install NeuroSeq into Conda

NeuroSeq requires Python 3.12 or newer.

If you have ssh authentication to GitHub set up you can install directly from this repository with pip:

```bash
pip install git+ssh://git@github.com/ssciwr/neuroseq
```

Alternatively you can clone the repository and then install it:

```bash
git clone git@github.com:ssciwr/neuroseq.git
cd neuroseq
pip install .
```

Run `neuroseq` from the activated Conda environment. Prefer
`python -m pip install .` to explicitly install into its Python. To update
the installation after pulling changes, run `python -m pip install --upgrade .`.
Use `python -m pip check` to check dependency metadata.

Pip does not use `uv.lock` or `[tool.uv.sources]`, so this installation can have
different package versions and PyTorch builds from the uv workflow. For GPU
setup, see [GPU troubleshooting](../common_issues.md#imaging-segmentation).

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

## Windows desktop shortcut

On Windows, NeuroSeq can instead be installed with a desktop shortcut that
updates it from GitHub before each start. See
[Windows desktop shortcut](windows_shortcut.md).

## How to install Cython on Windows

To speed up simulations with cython on Windows one will need an installation of the Microsoft Visual C++ 14.0 or higher.

See this [stackoverflow post](https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8) for further instructions on installation:

Visual Studio 2022 is available [here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes#17.10.5)
