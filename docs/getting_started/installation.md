# Installation

NeuroSeq requires Python 3.12 or newer. Install the latest release from
[PyPI](https://pypi.org/project/neuroseq/) with pip into its own environment,
so that its dependencies do not interfere with other Python software.

## Create an environment

With Python's built-in `venv`:

```bash
python -m venv neuroseq-env
source neuroseq-env/bin/activate     # Linux / macOS
neuroseq-env\Scripts\activate        # Windows
```

Or with [Conda](https://www.anaconda.com/):

```bash
conda create -n neuroseq python=3.12
conda activate neuroseq
```

The Conda environment can also be created in Anaconda Navigator: go to
"Environments", click "Create", enter "neuroseq" as the name and select
Python 3.12.

## Install NeuroSeq

In the activated environment, run:

```bash
python -m pip install neuroseq
```

Then launch the GUI with:

```bash
neuroseq
```

The first launch can take longer while dependencies prepare local caches.
A loading window appears during initialization, and the plotting workspace loads
after you choose a project on the welcome screen. Reuse the same Python
environment to retain its caches for later launches.

To upgrade to the latest release, run `python -m pip install --upgrade neuroseq`.

For GPU-accelerated cell segmentation, see
[GPU troubleshooting](../common_issues.md#imaging-segmentation).

To work on NeuroSeq itself from a repository checkout (with uv or Conda), see the
[developer installation guide](../developer/developer_installation.md).

For Windows desktop shortcuts, see
[Windows desktop shortcut](../developer/miscellaneous.md#windows-desktop-shortcut).

## How to install Cython on Windows

To speed up simulations with cython on Windows one will need an installation of the Microsoft Visual C++ 14.0 or higher.

See this [stackoverflow post](https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8) for further instructions on installation:

Visual Studio 2022 is available [here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes#17.10.5)
