# Input Design and Readout Analysis

This section covers the shared workflow for designing stimulation and analyzing
network responses, across simulations, neuromorphic hardware, and biological
experiments. For model execution and device connections, see
[Neural Networks](../networks/overview.md).

## Input design

- [Stimulation editor and static input](static_input.md): create and edit
  stimulation arrays, generate patterns, and repeat protocols.
- [Design dynamic input](dynamic_input.md): use response measurements to
  guide subsequent stimulation with STDP synaptic tuning.
- [Export patch-clamp stimuli](patch_clamp_export.md): prepare protocols
  for external acquisition software.

## Readout analysis

- [Experiments](../readout/experiments.md): organize recordings and generate synthetic data.
- [Import Igor2 waves](../readout/igor2_waves.md) and
  [imaging data](../readout/imaging_data.md): bring external recordings into a project.
- [Analysis tools and plugins](../readout/analysis.md): measure selected data
  and run built-in or custom analyses.
- [Connecting simulations and experiments](../readout/simulations_and_experiments.md):
  manage project data and compare simulated and recorded responses.
