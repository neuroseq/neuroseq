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

- [Overview](Neural%20Networks/overview.md)
- [Simulations](Neural%20Networks/simulations.md)
- [Run external simulators](Neural%20Networks/run_external_simulators.md)
- [ScanImage acquisition](Neural%20Networks/scanimage_acquisition.md)
- [Multiphoton holographic stimulation](Neural%20Networks/multiphoton_holographic_stimulation.md)
- [Deliver stimuli to BrainScaleS-2](Neural%20Networks/brainscales2_delivery.md)

### Input Design and Readout Analysis

- [Overview](Input%20Design%20and%20Readout%20Analysis/overview.md)
- [Stimulation editor and static input](Input%20Design%20and%20Readout%20Analysis/static_input_design.md)
- [Design dynamic input](Input%20Design%20and%20Readout%20Analysis/stdp_synaptic_tuning.md)
- [Export stimuli for patch clamp recordings](Input%20Design%20and%20Readout%20Analysis/export_patch_clamp_stimuli.md)
- [Experiments](Input%20Design%20and%20Readout%20Analysis/experiments.md)
- [Analysis tools and custom plugins](Input%20Design%20and%20Readout%20Analysis/analysis_and_plugins.md)
- [Connecting simulations and experiments](Input%20Design%20and%20Readout%20Analysis/simulations_and_experiments.md)
- [Importing Igor2 Waves](Input%20Design%20and%20Readout%20Analysis/importing_igor2_waves.md)
- [Importing Imaging Data](Input%20Design%20and%20Readout%20Analysis/importing_imaging_data.md)

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
