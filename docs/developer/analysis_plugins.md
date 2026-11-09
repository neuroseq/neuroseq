# Analysis plugins

An analysis plugin adds an entry to the right-click analysis menu of the trace
and spike views without changing core code. Using plugins is described in
[Analysis](../readout/analysis.md).

## Writing a plugin

A plugin is one `.py` file (names starting with `_` are ignored) that defines
`render_analysis`:

```python
from neuroseq.data import TraceData

PLUGIN_ID = "mean_activity"  # stable, machine-oriented; default: file name
PLUGIN_NAME = "Mean activity"  # menu label; default: file name
PLUGIN_ICON = "mdi6.chart-timeline-variant"  # qtawesome name; default: mdi6.puzzle


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    ax = widget.subplot  # Matplotlib axes
    if isinstance(data, TraceData):
        ax.plot(widget.traces.mean(axis=0))  # selection already sliced
```

`widget` is the `CustomAnalysisWidget`; useful attributes are `subplot`,
`traces` or `spike_times_ms` (the selected region), `dt_ms`, `neuron_slice`
and `time_slice`. The menu lists every plugin for every data type, so check
`data` (`TraceData`, `SpikeData`, `SynapseData`) and show a message for
unsupported types. Exceptions are caught, logged with the plugin id and shown
in the plot.

Examples in `src/neuroseq/resources/analysis_plugins/`: `mean_activity.py`
(basic) and `advanced_mean_activity.py` (adds a control bar to the analysis
window; guard against adding duplicate controls when the plugin runs again in
the same window).

## Where plugins are found

| Folder                                     | Use                                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `src/neuroseq/resources/analysis_plugins/` | Built-in plugins                                                                               |
| `user_plugins_directory()`                 | Per-user plugins, e.g. written by the chat assistant; override with `NEUROSEQ_USER_PLUGIN_DIR` |

User plugins never shadow a built-in file name or `PLUGIN_ID`. The folders are
rescanned each time the menu opens.

## Internals

- `analysis_plugins.py`: `discover_analysis_plugins()` loads files into
  `AnalysisPlugin(id, label, icon, render)` objects.
- `visualization/timeseries_widget.py`: builds the menu and emits
  `custom_analysis_requested` with a `CustomAnalysisType` or an `AnalysisPlugin`.
- `visualization/custom_analysis_widget.py`: runs built-in analyses and plugin
  `render` callbacks.

Keep plugins thin. Supporting code belongs in a regular package (for ELiSe,
`src/neuroseq/elise_support/`); core code must never import from the plugin
folder, which is excluded from coverage.
