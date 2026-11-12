# NeuroSeq: Neuronal Sequencer

[![CI](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml/badge.svg)](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neuroseq)](https://pypi.org/project/neuroseq/)
[![Docs](https://img.shields.io/badge/docs-neuroseq.github.io-blue)](https://neuroseq.github.io/)

NeuroSeq controls the input to a neural network and reads out its output, whether the network is simulated, biological or neuromorphic. The same input protocol and the same analysis run on all three.

| Network      | Input                                                             | Readout                                     |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------- |
| Digital      | Brian2 (built in) or external simulators                          | Voltage traces, spikes, synaptic weights    |
| Biological   | Two-photon holographic optogenetics (HOLO4D/SysCon2), patch clamp | Calcium imaging (ScanImage, suite2p), Igor2 |
| Neuromorphic | BrainScaleS-2 spike sources                                       | Spikes and membrane traces                  |

![NeuroSeq GUI](https://raw.githubusercontent.com/neuroseq/neuroseq/main/docs/images/screenshot.png)

## Why

Simulations expose every input and every variable. Biological and neuromorphic networks usually do not, so the three fields developed separate methods. Modern all-optical and electrophysiological interfaces now give cell-level control and readout of living networks. NeuroSeq uses this to treat every network as a programmable system: design an input, deliver it, read the response, decide the next input.

This lets you:

- run real experiments and neuromorphic hardware almost like simulations,
- run models as if they were experiments, with realistic instrument constraints,
- test a protocol in simulation, then run it unchanged on a microscope or chip.

## Features

- **Input design**: draw, paste or script input arrays (static input), or run interactive routines that choose the next input from the last response (dynamic input, e.g. rheobase search, STDP tuning).
- **Neuron ID matching**: links input rows to real cells via Cellpose segmentation.
- **Delivery**: Brian2, external simulators, SysCon2 `.seq` scripts, ScanImage acquisition, BrainScaleS-2, patch clamp input export.
- **Readout and analysis**: import imaging (Cellpose + suite2p), Igor2 and patch clamp data; compare with simulations; extend with analysis plugins.
- **Embedded Python console** and an experimental LLM assistant.

## Install

Python 3.12 or newer:

```bash
pip install neuroseq
neuroseq
```

See the [installation guide](https://neuroseq.github.io/getting_started/installation/) (GPU, Windows) and the [developer guide](https://neuroseq.github.io/developer/developer_installation/) (uv or Conda checkout).

## Documentation

[neuroseq.github.io](https://neuroseq.github.io/). Contributions to the docs are welcome, see the [documentation guide](https://neuroseq.github.io/developer/documentation/).

## Contributing

Extensions go in as plugins where possible: new network backends, analysis methods and input protocols. They need no core changes and get a lighter review. Changes to the core architecture should be discussed with the core developers first.

- Bugs and feature requests: [GitHub issues](https://github.com/neuroseq/neuroseq/issues)
- Code: fork, branch, open a pull request. See the [contributing guide](https://neuroseq.github.io/developer/contributing/).

## License

[MIT](https://github.com/neuroseq/neuroseq/blob/main/LICENSE). Copyright © 2024-2026 Andrey Formozov. Contributors are listed in [AUTHORS.md](https://github.com/neuroseq/neuroseq/blob/main/AUTHORS.md).
