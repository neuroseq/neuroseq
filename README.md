# NeuroSeq: A unified framework for biological, neuromorphic, and computational neural networks

[![CI](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml/badge.svg)](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neuroseq)](https://pypi.org/project/neuroseq/)
[![Docs](https://img.shields.io/badge/docs-neuroseq.github.io-blue)](https://neuroseq.github.io/)

NeuroSeq is an open-source framework to bring biological neuronal circuits, neuromorphic hardware, and computational neural network models into a unified experimental environment. It integrates experimental protocol design, data acquisition and control, and methods for data analysis and parameter inference across these fundamentally different neural network implementations.

## What is a neuronal sequencer?

NeuroSeq is a first implementation of a neuronal sequencer. We define a neuronal sequencer as a software system for controlling structured sequences of inputs to neural networks and analyzing their responses. It provides a unified framework for designing, delivering, and dynamically adjusting spatiotemporal patterns of neuronal stimulation, enabling systematic exploration of neural network input–output relationships.

The concept is partially inspired by music sequencers and digital audio workstations (DAWs), which have long been used to compose, arrange, and control sequences of musical events. Similarly, NeuroSeq allows users to organize and execute precisely timed sequences of inputs, but instead of controlling musical instruments, it interfaces with neural networks. However, the analogy has its limitations as neural networks exhibit a substantially richer repertoire of dynamical responses, plasticity mechanisms, and adaptive behaviors.

From a control perspective, a neuronal sequencer treats a neural network as a dynamical system whose activity can be probed and influenced through precisely defined inputs. By integrating stimulation control, data acquisition, and methods for analysis and inference, it enables both open-loop and closed-loop experimentation, in which network responses can guide subsequent stimulation.

A neuronal sequencer provides a common experimental interface for biological neural circuits, neuromorphic hardware, and computational network models, enabling the investigation and comparison of neural dynamics, computation, and plasticity across different physical substrates.

## Graphical User Interface

**Neuronal Sequencer (NeuroSeq) provides a graphical user interface (GUI)** for defining, configuring, and executing experimental protocols across biological neural circuits, computational models, and neuromorphic hardware.

The GUI serves as a unified control environment, allowing users to **define and manage input arrays delivered to neural systems**, including stimulation patterns, timing, intensity, and spatial or temporal organization. It also provides control over experimental hardware, simulation parameters, data acquisition, and the execution of experimental sequences. Acquired data can subsequently be processed using integrated analysis and inference tools.

By bringing stimulation, system control, data acquisition, and analysis into a single interface, NeuroSeq enables users to interact with different neural substrates through a common experimental workflow.

![screenshot](https://raw.githubusercontent.com/neuroseq/neuroseq/main/docs/images/screenshot.png)

## Installation

NeuroSeq requires Python 3.12 or newer. Install the latest release from
[PyPI](https://pypi.org/project/neuroseq/), ideally into a virtual environment
or Conda environment:

```bash
pip install neuroseq
```
Then launch the GUI with:

```bash
neuroseq
```

To upgrade to a newer release, run `pip install --upgrade neuroseq`.

See the [installation guide](https://neuroseq.github.io/getting_started/installation/)
for details, including GPU setup and Cython on Windows, and the
[developer installation guide](https://neuroseq.github.io/developer/developer_installation/)
to work on NeuroSeq from a repository checkout with uv or Conda.

## Documentation

Documentation for NeuroSeq is available [online](https://neuroseq.github.io/). We are actively developing and expanding it, and community contributions are very welcome, whether to improve existing content, add tutorials and examples, or document new features. See our [documentation guide](https://neuroseq.github.io/developer/documentation/) to learn how to build it locally, preview, and contribute to the documentation.

An AI-powered assistant has also been integrated directly into the NeuroSeq application. The users can log-in with their presered accounts or API key and explore NeuroSeq functionality while application is running. The assistant enables users to ask questions, explore features, and receive contextual guidance through an interactive chatbot. Its functionality is still being refined, and user feedback is highly appreciated.

## Open-Source Development and Contributions

NeuroSeq is an **open-source project maintained within a public GitHub organization**. Everyone is welcome to propose and implement improvements, extend its functionality, and contribute to the overall reliability, stability, and performance of the application.

Our current development policy prioritizes modular extensions through plugins. In particular, contributions implementing **new neural substrates**, **analysis methods**, and **inference tools** as plugins are highly encouraged. These extensions are designed to require minimal changes to the existing codebase and therefore generally follow a simpler review and integration process.

In contrast, **modifications to the core architecture or fundamental functionality** require more substantial discussion with the core developers to ensure compatibility, maintainability, and alignment with the project's long-term development goals.

### How to Contribute

- **Report a bug or suggest a feature:** Open a new [GitHub issue](https://github.com/neuroseq/neuroseq/issues) describing the problem, proposed improvement, or requested functionality.
- **Implement a new feature:** Fork the [NeuroSeq repository](https://github.com/neuroseq/neuroseq), make the necessary modifications, and submit a pull request to initiate a review and discussion with the core developers.
- **Extend NeuroSeq through plugins:** Contributions introducing new neural substrates, analysis pipelines, or inference tools are particularly welcome and are expected to have the most straightforward integration process.
- **Improve reliability and stability:** Bug fixes, performance optimizations, testing, and improvements to existing functionality are equally valuable contributions.

We encourage open discussions and collaborative development to make NeuroSeq a flexible, reliable, and accessible framework for experimentation with biological, computational, and neuromorphic neural networks.

## License

NeuroSeq is released under the [MIT License](https://github.com/neuroseq/neuroseq/blob/main/LICENSE). Copyright © 2024-2026 Andrey Formozov.
See [AUTHORS.md](https://github.com/neuroseq/neuroseq/blob/main/AUTHORS.md) for the list of contributors.
