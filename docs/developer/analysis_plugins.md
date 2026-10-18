# Analysis Plugins (Developer)

This page describes how the custom analysis plugin system works internally and how to implement new analysis plugins.

## Architecture Overview

Plugin support is split across three core modules:

- `src/neuroseq/analysis_plugins.py`
  - Discovers plugin modules from `src/neuroseq/resources/analysis_plugins`
  - Loads each plugin file dynamically
  - Builds `AnalysisPlugin` objects (`id`, `label`, `icon`, `render`)
- `src/neuroseq/visualization/timeseries_widget.py`
  - Rebuilds right-click analysis menu on demand
  - Adds built-in analysis actions and plugin actions
  - Emits `custom_analysis_requested` with either a built-in `CustomAnalysisType` or an `AnalysisPlugin`
- `src/neuroseq/visualization/custom_analysis_widget.py`
  - Executes built-in analysis logic
  - Executes plugin `render` callback for plugin analyses
  - Catches plugin exceptions, logs the plugin id, and shows a failure message in the plot

## Plugin Discovery

A plugin is a Python file in: `src/neuroseq/resources/analysis_plugins`. It must have `*.py` extention (files starting with `_` will be ignored) and define callable `render_analysis`. Each pluging has ID, NAME and ICON. The default value for ID and NAME is a filename without extention but user can also define them as `PLUGIN_ID`, `PLUGIN_NAME`, and `PLUGIN_ICON` (e.g. `mdi6.puzzle`). `PLUGIN_ID` should be stable over time. Use a machine-oriented id (for example, `mean_activity`) and keep display text in `PLUGIN_NAME`. At the present momement dropdown menu does not distinguish the type of the data that is being analysed and shows all improrted plugins regardless of datatype.

See [`src/neuroseq/resources/analysis_plugins/mean_activity.py`](../../src/neuroseq/resources/analysis_plugins/mean_activity.py) for a working example plugin.

## Advanced Interactive Example

The ELiSe plugins keep their supporting code in the same directory:
`_replay_metrics.py` provides replay metrics and CSV export,
`_elise_workflows.py` connects project data to the analysis windows, and
`_elise_protocol_dialog.py` provides protocol settings. Their leading
underscores exclude them from plugin discovery.

For an advanced plugin example with UI controls, see [`src/neuroseq/resources/analysis_plugins/advanced_mean_activity.py`](../../src/neuroseq/resources/analysis_plugins/advanced_mean_activity.py).

This plugin extends the basic mean-activity analysis by adding an interactive control bar in the same analysis window.

important: for the analysis widgets opened in the same window as it is the case for "show custom analysis" pad it is important to prevent duplicating of controls when the same plugin is run repeatedly in one open analysis widget.
