# Analysis and plugins

Analyses run on a selection in any trace, spike or data widget, so they work
the same on simulated and recorded data.

## Run an analysis

1. Drag to select a neuron × time region.
2. Right-click it and choose an analysis.

Built-in: **Histogram**, **Histogram 2D**, **Spike count per neuron**,
**Pearson correlation**. Installed plugins are listed below them.

**Open analysis in a new window** (checkbox at the top of the menu, on by
default) opens each result in its own window. Unticked, results go to the
**Custom Analysis** panel (**View → Show Custom Analysis**), where you can
select again and run a follow-up analysis.

## Bundled plugins

| Plugin                                   | Purpose                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------- |
| Mean activity / Advanced: Mean activity  | Mean trace (or spike activity) of the selection; the advanced one has controls. |
| EPSP amplitudes                          | Detects EPSPs and plots their amplitudes (also used by the STDP routines).      |
| Estimate membrane parameters             | Passive membrane parameters from voltage and injected current.                  |
| Replay MSE (cycle vs reference)          | MSE and correlation of replay cycles against the first selected cycle.          |
| ELiSe target MSE / ELiSe learning curves | ELiSe model metrics.                                                            |

For the **ELISE Sequence Learning Network** example, the stimulation editor
toolbar also shows **Analyse ELiSe Replay (target MSE)** (replay MSE and
correlation of `firing_rate_r` against the teaching pattern from `opsin_exc`,
saved as `.csv`) and **Run ELiSe Protocol (learning curves)**. They are enabled
once their inputs exist.

## Plugin folders

Plugins are discovered from:

- `src/neuroseq/resources/analysis_plugins/` (bundled), and
- a per-user folder: `<user data dir>/neuroseq/analysis_plugins` (e.g.
  `~/.local/share/neuroseq/analysis_plugins` on Linux), or the folder in
  `NEUROSEQ_USER_PLUGIN_DIR`. The
  [chat assistant](../getting_started/chat_assistant.md) writes plugins here
  after you confirm them.

Every `*.py` not starting with `_` is a plugin. A user plugin whose file name
or `PLUGIN_ID` matches a bundled one is skipped with a warning. New or edited
plugins appear the next time the menu opens, without a restart.

## Write a plugin

A plugin module defines:

- `PLUGIN_ID`: stable internal identifier (used in logs),
- `PLUGIN_NAME`: menu label,
- `PLUGIN_ICON`: optional QtAwesome icon (e.g. `mdi6.chart-line`),
- `render_analysis(widget, data, neuron_slice, time_slice)`: draws the result.

Details: [Developer Guide: Analysis plugins](../developer/analysis_plugins.md).
