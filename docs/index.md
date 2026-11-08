# NeuroSeq

Welcome to the official documentation for NeuroSeq, a first implementation of the neuronal sequencer.

NeuroSeq is an open-source framework for designing and running experiments with biological neuronal circuits, neuromorphic hardware, and computational neural network models. It brings stimulation design, simulation, data acquisition, visualization, and analysis into a shared graphical workspace.

A neuronal sequencer organizes precisely timed patterns of inputs to neural networks and helps analyze their responses. With NeuroSeq, users can create stimulation sequences, run simulations and experimental hardware. It supports predefined stimulation protocols and interactive closed-loop experiments, where network responses guide subsequent stimulation.

The source code is available in the [official NeuroSeq repository](https://github.com/neuroseq/neuroseq).

![NeuroSeq graphical user interface](https://raw.githubusercontent.com/neuroseq/neuroseq/main/docs/images/screenshot.png)

### Getting started

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [In-app LLM assistant (experimental)](getting_started/chat_assistant.md)

### User guide

- [GUI overview](user_guide/gui_overview.md)
- [Simulations](user_guide/simulations.md)
- [Design dynamic input](user_guide/stdp_synaptic_tuning.md)
- [Run external simulators](user_guide/run_external_simulators.md)
- [Experiments](user_guide/experiments.md)
- [Analysis tools and custom plugins](user_guide/analysis_and_plugins.md)
- [Connecting simulations and experiments](user_guide/simulations_and_experiments.md)
- [Importing Igor2 Waves](user_guide/importing_igor2_waves.md)
- [Importing Imaging Data](user_guide/importing_imaging_data.md)
- [Export stimuli for patch clamp recordings](user_guide/export_patch_clamp_stimuli.md)
- [Multiphoton Holographic Stimulation](user_guide/multiphoton_holographic_stimulation.md)

### Examples

- [Examples](examples.md)

### Developer guide

- [Installation and use](developer/developer_installation.md)
- [Documentation](developer/documentation.md)
- [Implementation of custom simulators](developer/custom_simulators.md)
- [Analysis plugins](developer/analysis_plugins.md)
- [Making a release](developer/releases.md)
- [Miscellaneous](developer/miscellaneous.md)
- [Testing](developer/testing.md)
- [Contributing](developer/contributing.md)

### Troubleshooting

- [Common issues](common_issues.md)

### License

NeuroSeq is released under the [MIT License](https://github.com/neuroseq/neuroseq/blob/main/LICENSE).
Copyright © 2024-2026 Andrey Formozov.
See [AUTHORS](https://github.com/neuroseq/neuroseq/blob/main/AUTHORS.md) for the list of contributors.
