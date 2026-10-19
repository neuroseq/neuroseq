# NeuroSeq - Neuronal Sequencer

[![CI](https://github.com/ssciwr/neuroseq/actions/workflows/ci.yml/badge.svg)](https://github.com/ssciwr/neuroseq/actions/workflows/ci.yml)

Private repository containing the neuroseq prototype GUI.

![screenshot](docs/images/screenshot.png)

## Installation

NeuroSeq requires Python 3.12 or newer. Choose one of two setup options:

- **uv:** manages Python and the project's `.venv`. Conda is not required.
- **Conda:** provides a named Python environment; install NeuroSeq into it with pip.

### Option 1: uv

For a repository checkout, use
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the
project environment:

```bash
git clone git@github.com:ssciwr/neuroseq.git
cd neuroseq
uv sync
uv run neuroseq
```

`uv sync` installs the project dependencies into `.venv`; `uv run` launches
commands in that environment. Use `uv run pytest` for tests in the same
environment. See the [installation guide](docs/getting_started/installation.md)
for details.

### Option 2: Conda

Conda is an alternative when you want to install NeuroSeq directly into a
named environment. Anaconda can be downloaded from the
[official webpage](https://www.anaconda.com/).

Once Anaconda installed, create and activate a new environment in a command line:

```
conda create -n neuroseq python=3.12
conda activate neuroseq
```

Alternatively, the new enviroment can be created within Anaconda Navigator. For this, open Anaconda Navigator and go to "Enviroments". Click "Create" button, write "neuroseq" as a name and select Python 3.12.

#### Install NeuroSeq into Conda

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

Use `python -m pip` in place of `pip` to explicitly target the active Python.
Installing with pip in Conda does not use `uv.lock` or uv's configured package
indexes; see the [installation guide](docs/getting_started/installation.md)
for GPU package setup.

You can also install uv inside Conda with `python -m pip install uv` and use
the uv workflow above. In that case, the project still runs in `.venv` by
default, separately from Conda. Updating packages in Conda does not update
the uv environment, and vice versa.

Combining the tools is optional. With standalone uv installed, you do not need
to create or activate a Conda environment. uv can download a suitable Python
when needed. If uv is currently installed inside Conda, install standalone uv
before removing that Conda environment.

### How to install Cython on Windows

To speed up brian2 simulations with Cython on Windows one will need an installation of the Microsoft Visual C++ version 14.0 or higher.

See this [stackoverflow post](https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8) for further instructions on installation. Visual Studio 2022 is available [here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes#17.10.5)

## Use

After installing, you should be able to run the following command to launch the GUI:

```bash
neuroseq
```

For the uv workflow, run `uv run neuroseq` from the repository instead.

## Documentation

Documentation for the project is availible [here](docs/index.md).

## Contributing and providing feed-back

The NeuroSeq is an open-source project. Everyone can suggest and implement improvements as well as improve overall reliability and stability of its operation.

To report a bug or suggest a new feature functionality please open a new issue. To implement a new feature, please fork the repository, modify the code accordingly and open a discussion with core developers via submitting a pull request.
