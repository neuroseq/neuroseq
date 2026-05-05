# Analysis Tools and Custom Plugins

NeuroSeq provides built-in custom analysis tools to support major types of analysis of neuronal data. It also supports user-defined analysis plugins.

## Use Built-In Analysis Tools

1. Generate trace or spike data in a simulation or real experiments.
2. In corresponding widget, drag to select a neuron/time region.
3. Right-click the selected area.
4. Choose one of the built-in analysis actions:
5. Optionally pick the `(new window)` variant to open results in a separate window.

## Use Installed Analysis Plugins

NeuroSeq discovers plugins from this folder:

`src/neuroseq/resources/analysis_plugins`

Any `*.py` file in that folder is treated as a plugin module (except files starting with `_`).

Plugins can be added and tested during the operation of NeuroSeq: no restarts are necessary.

After placing or editing a plugin file, open the right-click analysis menu again: you should see the plugin of interest in the menu.

## Create Your Own Plugin

Users can create a new plugins in `src/neuroseq/resources/analysis_plugins`, for example:

`src/neuroseq/resources/analysis_plugins/my_analysis.py`

Each plugin should define:

- `PLUGIN_ID`: a short stable internal identifier used in error logging
- `PLUGIN_NAME`: a label that will be shown in the context menu
- `PLUGIN_ICON`: optional QtAwesome icon name (for example `mdi6.chart-line`)
- `render_analysis(widget, data, neuron_slice, time_slice)`: plotting function

For more technical implementation details, see the [Developer Guide: Analysis plugins](../developer/analysis_plugins.md).
