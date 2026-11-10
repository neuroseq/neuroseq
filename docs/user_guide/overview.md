# Neural Networks overview

NeuroSeq provides a shared workspace for designing stimulation protocols,
running them on neural networks, and inspecting the resulting activity. The
networks can be computational models, neuromorphic hardware, or biological
neuronal circuits. The implementation is evolving, with existing integrations
providing the basis for additional substrates, third-party software, and APIs.

## Overall structure

A project brings together stimulation arrays, network or device settings, and
experiments containing the recorded responses. The workflow has four parts:

1. **Design input.** Create stimulation arrays with the static input editor or
   generate successive blocks with a dynamic input routine.
2. **Deliver stimulation.** Run a computational model, export a protocol for
   external software, or send commands to a connected device.
3. **Collect readout.** Import recordings or receive results through an API,
   then add them to the project as traces, spikes, or imaging data.
4. **Inspect and adapt.** Compare stimulation and responses, apply analysis
   tools, and use the results to decide the next input. Supported dynamic input
   routines automate this cycle for their configured targets.

The stimulation editor, visualization, and analysis tools form the shared
workspace. Backend-specific integrations translate its stimulation arrays into
the input expected by a model or device and return the supported readout data.

## Interaction mechanisms

NeuroSeq currently uses several mechanisms to interact with external neural
networks and their control software:

- **File export and import:** export stimulation protocols for execution in
  another application and import its readout data afterward. Existing protocols
  can also be imported where the format is supported.
- **Direct API access:** call a simulation backend or a device-control API to
  configure and run a network and retrieve its response.
- **Combined workflows:** generate protocol files, automatically upload and run
  them through a control API, and import the resulting recordings. Holographic
  stimulation combined with ScanImage acquisition is one example.

The mechanism depends on the capabilities of the target software and hardware.
A file-based connection can therefore be useful on its own or as part of an
automated experiment.

## Current implementations

The following groups list implementations currently in use. They are starting
points for integration rather than a fixed list of supported neural substrates.

| Domain | Implementation or device | Stimulation and execution | Readout in NeuroSeq |
| --- | --- | --- | --- |
| **Computational** | **Brian2** | Run the built-in model directly from the project, using its stimulation arrays. | Simulation traces, spikes, and monitored synaptic data. See [Simulations](simulations.md). |
| | **Native Python neural networks** | Connect a Python backend that receives stimulation arrays and executes its own network code. | The backend returns an experiment with supported response data. See [Run external simulators](run_external_simulators.md). |
| **Neuromorphic** | **BrainScaleS-2** | Interactive control through a container helper and the remote hardware service; PyNN-based execution delivers stimulation to the chip. | Spikes and membrane-voltage recordings are returned to the project. See [BrainScaleS-2 delivery](brainscales2_delivery.md). |
| **Experimental (biological)** | **Patch clamp** | Export stimulation protocols for execution by patch-clamp acquisition software. | Import patch-clamp recordings, including Igor2 waves and supported CSV data. See [Export patch-clamp stimuli](export_patch_clamp_stimuli.md) and [Import Igor2 waves](importing_igor2_waves.md). |
| | **Calcium imaging / ScanImage** | Control microscope acquisition through the ScanImage MATLAB API, including starting or arming a recording. | Import completed ScanImage TIFF recordings; supported dynamic input routines extract calcium traces and inferred spikes. See [ScanImage acquisition](scanimage_acquisition.md). |
| | **Holographic stimulation / SysCon2 (Holo4D)** | Design stimulation scripts, export SysCon2 sequences, and automatically import, upload, and run them through the connected control interface. | Combine stimulation with acquisition, such as ScanImage, to collect the biological response. See [Multiphoton holographic stimulation](multiphoton_holographic_stimulation.md). |



