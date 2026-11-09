# NeuroSeq

Welcome to the official documentation for NeuroSeq, a first implementation of the neuronal sequencer.

NeuroSeq is an open-source framework for designing and running experiments with biological neuronal circuits, neuromorphic hardware, and computational neural network models. It brings stimulation design, simulation, data acquisition, visualization, and analysis into a shared graphical workspace.

A neuronal sequencer organizes precisely timed patterns of inputs to neural networks and helps analyze their responses. With NeuroSeq, users can create stimulation sequences, run simulations and experimental hardware. It supports predefined stimulation protocols and interactive closed-loop experiments, where network responses guide subsequent stimulation.

The source code is available in the [official NeuroSeq repository](https://github.com/neuroseq/neuroseq).

![NeuroSeq graphical user interface](https://raw.githubusercontent.com/neuroseq/neuroseq/main/docs/images/screenshot.png)

### Getting started

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [GUI overview](getting_started/gui_overview.md)
- [In-app LLM assistant (experimental)](getting_started/chat_assistant.md)

### Neural Networks

- [Overview](networks/overview.md)
- [Simulations](networks/simulations.md)
- [Run external simulators](networks/external_simulators.md)
- [ScanImage acquisition](networks/scanimage_acquisition.md)
- [Multiphoton holographic stimulation](networks/holographic_stimulation.md)
- [Deliver stimuli to BrainScaleS-2](networks/brainscales2.md)

### Input Design and Readout Analysis

- [Overview](input_design/overview.md)
- [Stimulation editor and static input](input_design/static_input.md)
- [Design dynamic input](input_design/dynamic_input.md)
- [Export stimuli for patch clamp recordings](input_design/patch_clamp_export.md)
- [Experiments](readout/experiments.md)
- [Analysis tools and custom plugins](readout/analysis.md)
- [Connecting simulations and experiments](readout/simulations_and_experiments.md)
- [Importing Igor2 Waves](readout/igor2_waves.md)
- [Importing Imaging Data](readout/imaging_data.md)

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
