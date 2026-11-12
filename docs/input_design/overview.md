# Input design overview

In NeuroSeq, input is the central element. An input array streams information
into a network, sets its dynamical state, and probes it (rheobase,
connectivity, plasticity). The same array and the same analysis run on digital
(Brian2), biological (holographic optogenetics with calcium imaging, patch
clamp) and neuromorphic (BrainScaleS-2) networks. For running models and
connecting devices, see [Neural Networks](../networks/overview.md).

Three levels, from manual to closed-loop:

| Level                                                        | What it does                                                                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| [Manual input designer](static_input.md)                     | Draw, copy, paste and load input in the input editor, or script it in the embedded console.                                          |
| [Static input designer](static_input.md#design-static-input) | Ready-made routines that generate whole patterns (sequential, simultaneous, gradient, ramp).                                         |
| [Dynamic input designer](dynamic_input.md)                   | Step-wise experiments: deliver input, read the response, choose the next input (rheobase search, STDP window, STDP synaptic tuning). |

Dynamic routines can be tested on a simulation first and then routed unchanged
to the microscope (SysCon2) or to BrainScaleS-2.

To run an input array on an external rig, export it with
[Holography export](holography_export.md) or
[Patch clamp export](patch_clamp_export.md). Responses come back through
[Readout](../readout/overview.md).
