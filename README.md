# NeuroSeq - Neuronal Sequencer

[![CI](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml/badge.svg)](https://github.com/neuroseq/neuroseq/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neuroseq)](https://pypi.org/project/neuroseq/)
[![Docs](https://img.shields.io/badge/docs-neuroseq.github.io-blue)](https://neuroseq.github.io/)

NeuroSeq is a Qt-based desktop GUI application for computational neuroscience. It combines Brian2 spiking neural network simulation with interactive stimulus design, multi-source experimental data import, and hardware export for holographic stimulation (HOLO4D `.seq` files).

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

Documentation for the project is available [online](https://neuroseq.github.io/).
See the [documentation guide](https://neuroseq.github.io/developer/documentation/) to build or preview it locally.

## Contributing and providing feed-back

The NeuroSeq is an open-source project. Everyone can suggest and implement improvements as well as improve overall reliability and stability of its operation.

To report a bug or suggest a new feature functionality please open a new issue. To implement a new feature, please fork the repository, modify the code accordingly and open a discussion with core developers via submitting a pull request.

## License

NeuroSeq is released under the [MIT License](https://github.com/neuroseq/neuroseq/blob/main/LICENSE). Copyright © 2024-2026 Andrey Formozov.
See [AUTHORS.md](https://github.com/neuroseq/neuroseq/blob/main/AUTHORS.md) for the list of contributors.
