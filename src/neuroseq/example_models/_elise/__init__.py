"""Vendored copy of the ELiSe sequence-learning network.

Source: https://github.com/unibe-cns/temporal_sequence_learning
Upstream package: ``elise`` (version 0.3.0, commit cf790867)
Authors: Computational Neuroscience, University of Bern (Kriener, Völk,
von Hünerbein, Benitez, Senn, Petrovici, 2024)
License: MIT — see ``LICENSE`` in this directory.

Reference: Kriener et al., "ELiSe: Efficient Learning of Sequences in
Structured Recurrent Networks", 2024 (arXiv:2402.16763).

This is the minimal subset of the upstream ``elise`` package needed by
``neuroseq.example_models.elise_simulate``: the network model, optimizer,
weight builders, and rate buffer. The simulator, dataloaders, trackers,
loggers and analysis utilities are intentionally not vendored — neuroseq
provides equivalents (StimArray for stimulus, Experiment/TraceData for
data capture, ``runpy``/GUI for orchestration). To update, re-copy the
four modules from a clean upstream checkout and refresh this docstring's
commit hash.

The four files (``model.py``, ``optimizer.py``, ``rate_buffer.py``,
``weights.py``) are kept byte-identical to upstream so future sync is a
straightforward ``cp``. All attribution lives in this ``__init__.py`` and
the adjacent ``LICENSE`` file.
"""

from neuroseq.example_models._elise.model import Network
from neuroseq.example_models._elise.optimizer import SimpleUpdater
from neuroseq.example_models._elise.rate_buffer import Buffer
from neuroseq.example_models._elise.weights import DendriticWeights
from neuroseq.example_models._elise.weights import SomaticWeights

__all__ = [
    "Buffer",
    "DendriticWeights",
    "Network",
    "SimpleUpdater",
    "SomaticWeights",
]
