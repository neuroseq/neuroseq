# NeuroSeq - Neuronal Sequencer

[![CI](https://github.com/ssciwr/neuroseq/actions/workflows/ci.yml/badge.svg)](https://github.com/ssciwr/neuroseq/actions/workflows/ci.yml)

Private repository containing the neuroseq prototype GUI.

![screenshot](screenshot.png)

## Installation

### Conda

It is recommended to use Anaconda environment (package manager) for the installation. It can be downloaded from the official webpage https://www.anaconda.com/.

Once Anaconda installed, create and activate a new environment in a command line:

```
conda create -n neuroseq python=3.10
conda activate neuroseq
```

Alternatively, the new enviroment can be created within Anaconda Navigator. For this, open Anaconda Navigator and go to "Enviroments". Click "Create" button, write "neuroseq" as a name and select Python 3.10.

### Installation of the neuroseq

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

### How to install Cython on Windows

To speed up brian2 simulations with Cython on Windows one will need an installation of the Microsoft Visual C++ version 14.0 or higher.

See further instructions on installation here: https://stackoverflow.com/questions/60322655/how-to-use-cython-on-windows-10-with-python-3-8

Visual Studio 2022 is available here: [https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes#17.10.5](https://visualstudio.microsoft.com/)

## Use

After installing, you should be able to run the following command to launch the GUI:

```bash
neuroseq
```

To generate synthetic Cellpose test data (TIFF image + matching `.npy` segmentation):

```bash
neuroseq --generate-cellpose-test-data ./cellpose_test_grid.tif
```

## Documentation

- Getting started
- Control of simulations
- Examples
- Control of experiments
- Import of experimental data

## GUI Console

The GUI console is a text box at the bottom of the GUI that can be used to run Python code.
The code is executed in the same Python environment as the GUI, with the following variables/objects made available (run dir() command to see the list):

- `np`: numpy
- `b2`: brian2
- `stims`: a dict of stimuli, each of which is a 2d numpy array
- `model`: a class that defines the entire model

The console also has a history feature, which can be accessed by pressing the up and down arrow keys.

### Stimuli editing

The console allows users to modify the numpy arrays of stimuli using numpy and Python code.
Any changes you make to the stimuli will be checked for validity, and if valid then they will be updated in the model and the changes will be displayed in the GUI, e.g.

```python
>>> stims['stimulus'] = np.random.randint(2,size=(500, 24))
```

Example for generating a simple sequential stimulation pattern:

```python
>>> for i in range(100):
...     project.stim_arrays['stimulus'].values[0+i*10:10+i*10,i] = 1
```

If the changes are invalid, then an error message will be displayed in the console, and the model will not be modified, e.g.

```python
>>> stims['stimulus'] = np.random.randint(2,size=(5, 5, 5))
    Invalid stims: expected dict[str, 1d or 2d numpy array], found dict with 3d numpy array for key 'stimulus'
```

### Model editing

It is also possible to modify other aspects of the model using the console, by interacting with the `project.b2_model` object.
Try `help(project.b2_model)` to see what methods and attributes are available.

### Brian2 objects

The objects from the underlying brian2 model can also be accessed. For example, you can get a list of monitors with

```python
>>> project.b2_model.monitors.keys()
dict_keys(['Traces_v', 'spikemon', 'synmon'])
```
And you can then get the underlying brian2 monitor object with e.g.

```python
>>> monitor = model.get_monitor('Traces_v')['object']
```

Note that these objects should not be directly modified using the console, as it will cause the GUI and `model` to become out of sync with the underlying brian2 model.
