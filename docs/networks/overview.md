# Neural Networks overview

NeuroSeq is a neuronal sequencer: it controls the input to a neural network and
reads out its output, whether the network is digital, biological or
neuromorphic. For any network, input is split into **controlled** (what
NeuroSeq delivers) and **uncontrolled** (everything else), and output into
**observed** (what is recorded) and **unobserved**. The same input protocol and
the same analysis run on every substrate.

Input arrays have one row per neuron. On biological networks, neuron ID matching
links each row to a real cell through a Cellpose segmentation
(**File → Cell segmentation**, _ID Mapping Options_).

## Supported substrates

| Substrate    | Network                                               | Input                                                                                           | Readout                                                                                           |
| ------------ | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Digital      | [Brian2](simulations.md)                              | Input arrays drive the built-in model                                                           | Simulated traces, spikes, synapses                                                                |
| Digital      | [External simulators](external_simulators.md)         | Input arrays passed to your Python `simulate(...)`                                              | The `Experiment` your backend returns                                                             |
| Biological   | [Holographic stimulation](holographic_stimulation.md) | Two-photon holographic optogenetics on HOLO4D: export `.seq`, upload and run through SysCon2    | [ScanImage](imaging_acquisition.md) calcium imaging: ΔF/F and deconvolved spikes per Cellpose ROI |
| Biological   | Patch clamp                                           | [Export protocols](../input_design/patch_clamp_export.md) for the acquisition software          | [Import Igor2 waves](../readout/patch_clamp_data.md) or CSV                                       |
| Neuromorphic | [BrainScaleS-2](neuromorphic_hardware.md)             | Constant current per neuron, or spikes from a virtual source population; chip runs 1000x faster | Spikes of all driven neurons and the membrane voltage of one neuron                               |

## Workflow

1. **Design input** with the [static input editor](../input_design/static_input.md)
   or a [dynamic input routine](../input_design/dynamic_input.md) (rheobase, STDP).
2. **Deliver** it: run a simulation, export a file, or send it to a connected
   device from the **Stimulation setup** pane (**View → Show Stimulation setup**).
3. **Read out**: the result is added to the project as a data group of traces,
   spikes or imaging data.
4. **Analyse and adapt**. Dynamic input routines close this loop automatically on
   Brian2 and SysCon2 + ScanImage (rheobase also on BrainScaleS-2).

Integrations use file export/import, a direct API, or both (for example, SysCon2
export plus automatic upload, run and ScanImage recording). New backends can be
added through the [custom simulator interface](../developer/custom_simulators.md).
