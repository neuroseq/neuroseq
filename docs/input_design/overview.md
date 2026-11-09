# Input Design and Readout

In NeuroSeq, input is the central element. An input array streams information
into a network, sets its dynamical state, and probes it (rheobase,
connectivity, plasticity). The same array and the same analysis run on digital
(Brian2), biological (holographic optogenetics with calcium imaging, patch
clamp) and neuromorphic (BrainScaleS-2) networks. For running models and
connecting devices, see [Neural Networks](../networks/overview.md).

## Input design

Three levels, from manual to closed-loop:

| Level                                                        | What it does                                                                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| [Manual input designer](static_input.md)                     | Draw, copy, paste and load input in the stimulation editor, or script it in the embedded console.                                    |
| [Static input designer](static_input.md#design-static-input) | Ready-made routines that generate whole patterns (sequential, simultaneous, gradient, ramp).                                         |
| [Dynamic input designer](dynamic_input.md)                   | Step-wise experiments: deliver input, read the response, choose the next input (rheobase search, STDP window, STDP synaptic tuning). |

Dynamic routines can be tested on a simulation first and then routed unchanged
to the microscope (SysCon2) or to BrainScaleS-2.

To run an input array on an external rig, see
[Export patch-clamp stimuli](patch_clamp_export.md) and
[Holographic stimulation](../networks/holographic_stimulation.md).

## Readout

- [Experiments](../readout/experiments.md): manage data groups and generate
  synthetic data.
- [Imaging data](../readout/imaging_data.md): Cellpose segmentation, raw TIFF
  and Suite2p traces, neuron ID mapping.
- [Igor2 waves](../readout/igor2_waves.md): patch-clamp input and output.
- [Analysis and plugins](../readout/analysis.md): built-in analyses and custom
  plugins on any selection.
- [Simulations vs. experiments](../readout/simulations_and_experiments.md):
  combine projects and compare simulated with recorded data.
