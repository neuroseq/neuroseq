# Installation

## Conda

It is recommended to use Anaconda environment (package manager) for the installation. It can be downloaded from the [official webpage](https://www.anaconda.com/).

Once Anaconda installed, create and activate a new environment in a command line:

```
conda create -n neuroseq python=3.11
conda activate neuroseq
```

Alternatively, the new enviroment can be created within Anaconda Navigator. For this, open Anaconda Navigator and go to "Enviroments". Click "Create" button, write "neuroseq" as a name and select Python 3.11.

## Installation of the neuroseq

NeuroSeq requires Python 3.11 or newer.

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

## How to install Cython on Windows

To speed up simulations with cython on Windows one will need an installation of the Microsoft Visual C++ 14.0 or higher.

See this [stackoverflow post](https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8) for further instructions on installation:

Visual Studio 2022 is available [here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes#17.10.5)
