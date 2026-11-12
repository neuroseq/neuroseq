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

## Terminology

- **Import**: loading files of any kind into a project. **File → Import data**
  has an **Input** submenu (e.g. SysCon2 `.seq`, patch-clamp command waveforms)
  and an **Output** submenu (calcium imaging, patch-clamp recordings).
- **Input**: the arrays that drive the network, one row per neuron (input
  arrays, shown in the **Input** panel and chosen with the **Input data**
  selector). They are designed with static or dynamic input design and
  imported or exported as files.
- **Output**: what is read out from the network: traces, spikes and synapses,
  simulated or recorded (selected with **Neuron data**, **Spike data**,
  **Synapse data**).
- **Stimulation**: the physical delivery of input on a rig, e.g. holographic
  optogenetic stimulation through SysCon2, configured in the **Stimulation
  setup** pane. A stimulation artefact is laser light picked up by the imaging.

## Supported substrates

These are the integrations in use today. They are starting points for new
substrates, third-party software and APIs rather than a fixed list.

| Substrate    | Network                                               | Input                                                                                                                                                    | Readout                                                                                                                                    |
| ------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Digital      | [Brian2](simulations.md)                              | Input arrays drive the built-in model                                                                                                                    | Simulated traces, spikes, synapses                                                                                                         |
| Digital      | [External simulators](external_simulators.md)         | Input arrays passed to your Python `simulate(...)`                                                                                                       | The `Experiment` your backend returns                                                                                                      |
| Biological   | [Holographic stimulation](holographic_stimulation.md) | Two-photon holographic optogenetics on HOLO4D: export `.seq`, upload and run through SysCon2                                                             | [ScanImage](imaging_acquisition.md) calcium imaging, started or armed through its MATLAB API: ΔF/F and deconvolved spikes per Cellpose ROI |
| Biological   | Patch clamp                                           | [Export protocols](../input_design/patch_clamp_export.md) for the acquisition software                                                                   | [Import patch-clamp data](../readout/patch_clamp_data.md)                                                                                  |
| Neuromorphic | [BrainScaleS-2](neuromorphic_hardware.md)             | PyNN runs on the remote chip through a container helper: constant current per neuron, or spikes from a virtual source population; chip runs 1000x faster | Spikes of all driven neurons and the membrane voltage of one neuron                                                                        |

## Workflow

1. **Design input** with the [static input editor](../input_design/static_input.md)
   or a [dynamic input routine](../input_design/dynamic_input.md) (rheobase, STDP).
2. **Deliver** it: run a simulation, export a file, or send it to a connected
   device from the **Stimulation setup** pane (**View → Show Stimulation setup**).
3. **Read out**: the result is added to the project as a data group of traces,
   spikes or imaging data.
4. **Analyse and adapt**. Dynamic input routines close this loop automatically on
   Brian2 and SysCon2 + ScanImage (rheobase also on BrainScaleS-2).

## Interaction mechanisms

How NeuroSeq talks to a network depends on what its software and hardware offer:

- **File export and import**: export a protocol, run it in the other
  application, then import the readout. Existing protocols can be imported where
  the format is supported.
- **Direct API**: configure and run a simulator or device through its API and
  get the response back.
- **Combined**: generate protocol files, upload and run them through a control
  API, and import the recordings automatically, as with SysCon2 holographic
  stimulation plus ScanImage acquisition.

A file-based connection is useful on its own or as one step of an automated
experiment. New backends can be added through the
[custom simulator interface](../developer/custom_simulators.md).
