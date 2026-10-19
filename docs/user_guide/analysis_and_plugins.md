# Analysis Tools and Custom Plugins

NeuroSeq provides built-in custom analysis tools to support major types of analysis of neuronal data. It also supports user-defined analysis plugins.

## Use Built-In Analysis Tools

1. Generate trace or spike data in a simulation or real experiments.
2. In corresponding widget, drag to select a neuron/time region.
3. Right-click the selected area.
4. Choose one of the built-in analysis actions:
5. Optionally pick the `(new window)` variant to open results in a separate window.

## The Custom Analysis widget

When an analysis is run _without_ the `(new window)` variant, its output
lands in the **Custom analysis** dock widget. Toggle it from **View → Show
Custom analysis** in the menu or via the toolbar.

The widget reuses the standard pan/zoom interaction of the other timeseries
panels, so you can drag-select inside the analysis output itself and run a
follow-up analysis on it.

## ELiSe replay analysis

Projects loaded from the **ELiSe Sequence Learning Network** example get an
extra one-click action on the stim editor toolbar:
**Analyse ELiSe replay (target MSE)**.

It compares the simulated `firing_rate_r` trace against the binary teaching
pattern derived from the project's `opsin_exc` stim, computing the
paper-style replay MSE and correlation. The button is enabled only when the
project is the ELiSe one _and_ a simulation has produced the required
trace. Results are appended as a new analysis output and a metrics `.csv`
file is written alongside.

## Use Installed Analysis Plugins

NeuroSeq discovers plugins from this folder:

`src/neuroseq/resources/analysis_plugins`

Any `*.py` file in that folder is treated as a plugin module (except files starting with `_`).

NeuroSeq also discovers plugins from a per-user folder outside the package
(`<user data dir>/neuroseq/analysis_plugins`, e.g.
`~/.local/share/neuroseq/analysis_plugins` on Linux, or the folder named by the
`NEUROSEQ_USER_PLUGIN_DIR` environment variable). Plugins written by the chat
assistant go there. A user plugin whose filename or `PLUGIN_ID` matches a
built-in plugin is skipped with a warning.

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
