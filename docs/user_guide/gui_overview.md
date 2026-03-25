# GUI Overview

## General Layout

![GUI layout](../images/layout.png)

The main layout of NeuroSeq consist of several widgets: the main widget for modification (drawing) of stimulation patterns (1), the widgets for visulisation of neuronal activity in form of traces (2) or spikes (3), monitor of synaptic connectivity (4). It also include a pad to quickly show some relevant analysis plots (5) and a GUI console (6).

The navigation in widgets (1-3) can be done with scrolling sliders or by holding a middle mouse button and moving the field of view of a given widget. By default, the field of new in all widgets is connected.

Zooming in and zooming out is done by corresponding "+" and "-" buttons or by holding "ctrl" or "shift" buttons and rotating mouse wheel for x- and y-axis, respectively.

Each widget can be shown or hidden in menu "View" by using the corresponding toolbar:

![Show/hide widgets](../images/toggle_widgets.png)

In addition to elements 1-6, the toolbar contains a universal widget that can dupicate widgets (2) and (3), useful for subsequent visulisation and data analysis. The button with capital "T" toggles and untoggles text annotation next to a mouse cursor.

The project may contrain not only one, but several stimulation arrays. The user can select which one to show in the widget one, using a drop down manu in the "Stim data:" section of the folloing toolbar:

![Data selection](../images/data_selection.png).

Similarly, the trace, spike and synapce data may contain a list of different data entries (generated in computational, biologial or neuromorphic experiments) with corresponding drop down menues. "All data" corresponds to a universal widget where any data in a form of traces or spikes can be independently selected. 

![Data selection](../images/data_selection.png).

 ## Stimulation editor

The stimulatation pattern can be edited using the toolbar located at the left side of a main window. The editing can be operated in draw and select modes. In the drawing mode, the user can switch between "draw stimulus" and "erise stimulus" regimes and choose the amplitude and duration of the a drawn stimuli. The amplitude of a stimulus can be constant, gradient and log-gradient and random.

![Stimulus intensity](../images/amplitude_selection.png).

Mouse can be used for drawing stimuli: left click is drawing and right click is erising. In erise mode, the functionality of the left and right buttons is inverted.
A longer stimulus can be drawing by holding a mouse button.

The intensity of stimulation is coded with transparency and normalized on the maximum value in an array. 

If "select mode" is active, the drawing options are not availible anymore, insteed other options are availible for selecting, copying, cutting, saving selected region and so on. The user can "do" and "undo" modifications in the editor using the corresponding buttons.

The toolbar also contains a pad with for creating/deleting new stim arrays, modifying the properties of the existing stimulation array (name, type, number of neurons and physical units) and padding/trimming the duration of an existing array.

The option "show all stim arrays" gives a possibility to show all stim arrays at the same time in one widget. In this case, the selected array remains the same, while other arrays will be shown in background with additional transparency applied.  

## Neuron data and spike data widgets 

Once the simulation is run or trace data is exported, it became availible in drop down menu "Neuron data". Natigation is done as described above. Normalization of traces can be set with buttons next it with a possibility to normalize traces on a median value and clip/upclip them to permit to restrict traces to the provided space for each neuron or provide a possibility to occupy an unlimited space within the widget. Spike data widget is analogous to the neuron data widget but constructed to represent binary spike events.

## Synaptic data widget

The synaptic data widget contains the information about the states of synaptic connections in a given time point. Using the replay mechanism described below one can access the modifications of synapic changes.

## Universal widget

During analysis, it may be desirable to show an additional copy of the widget, for example, to compare two different periods or neuronal populations within the same widget and at the same time. For this a universal "all data" widget was constructed.

## Analysis widget

The user can select a given region of the "neuron data" widget and access an analysis menu by right click of a mouse, and choosing a desired analysis plot/routine.
It can shown within "analysis widget" or opened in a new window.

## GUI Console

The GUI console is a text box at the bottom of the GUI that can be used to run a custom Python code. 

The code is executed in the same Python environment as the GUI, with the following variables/objects made available (run dir() command to see the list):

- `np`: numpy
- `b2`: brian2
- `stims`: a dict of stimuli, each of which is a 2d numpy array
- `model`: a class that defines the entire model

The console also has a history feature, which can be accessed by pressing the up and down arrow keys.

### Stimuli Editing

One of the purposes of the console is to allow users to modify the numpy arrays of stimuli using numpy and Python code.
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

It is also possible to modify other aspects of the model using the console, by interacting with the `model` object.
Try `help(model)` to see what methods and attributes are available.

### brian2 objects

The objects from the underlying brian2 model can also be accessed. For example, you can get a list of monitors with

```python
>>> model.monitors.keys()
dict_keys(['Traces_v', 'spikemon', 'synmon'])
```

And you can then get the underlying brian2 monitor object with e.g.

```python
>>> monitor = model.get_monitor('Traces_v')['object']
```

Note that these objects should not be directly modified using the console, as it will cause the GUI and `model` to become out of sync with the underlying brian2 model.

