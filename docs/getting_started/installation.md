# Installation

NeuroSeq needs Python 3.12 or newer. Install it from
[PyPI](https://pypi.org/project/neuroseq/) into its own environment.

```bash
# venv
python -m venv neuroseq-env
source neuroseq-env/bin/activate     # Linux / macOS
neuroseq-env\Scripts\activate        # Windows

# or Conda
conda create -n neuroseq python=3.12
conda activate neuroseq

python -m pip install neuroseq       # upgrade: add --upgrade
neuroseq                             # launch the GUI
```

The first launch is slower while dependencies build their caches. Reuse the same
environment to keep them.

Further setup:

- GPU for Cellpose and Suite2p: see [Troubleshooting](../common_issues.md#cellpose-use-gpu-is-greyed-out).
- Working from a repository checkout: [developer installation](../developer/developer_installation.md).
- Windows desktop shortcut: [Windows shortcut](../developer/windows_shortcut.md).
- Optional `cython` Brian2 backend (**Settings → Brian2 Simulation Backend**, default `numpy`):
  on Windows this needs Microsoft Visual C++ 14.0 or newer, e.g. from
  [Visual Studio 2022](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes).
