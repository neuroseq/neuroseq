from __future__ import annotations

import pathlib
from collections.abc import Iterator
from importlib import resources
from importlib.resources.abc import Traversable

from neuroseq.project import Project

# the repository's docs/, used when running from a checkout
_CHECKOUT_DOCS_DIR = pathlib.Path(__file__).resolve().parents[3] / "docs"


def find_docs_dir(package: Traversable | None = None) -> Traversable:
    """The docs packaged into an installed wheel, else the repository's docs/."""
    if package is None:
        package = resources.files("neuroseq")
    packaged = package / "docs"
    if packaged.is_dir():
        return packaged
    return _CHECKOUT_DOCS_DIR


DOCS_DIR = find_docs_dir()

_PREAMBLE = """\
You are NeuroSeq's in-app help assistant. NeuroSeq is a desktop GUI for
computational neuroscience: it combines Brian2 spiking neural network
simulation, interactive stimulus design, multi-source experimental data
import (Suite2p calcium imaging, Cellpose spatial footprints, patch-clamp,
Igor2 binaries, SysCon2 holographic sequences), and hardware export for
holographic stimulation.

Answer the user's question by referencing the documentation excerpts below
and the live project snapshot, inspecting project details and source code
with the read requests when needed. Prefer concrete menu paths over vague
descriptions. If the available evidence does not cover the user's question,
say so plainly rather than inventing menu items or feature names. Keep replies concise
unless the user asks for more detail.
"""


def _markdown_files(
    directory: Traversable, prefix: str = ""
) -> Iterator[tuple[str, Traversable]]:
    for entry in sorted(directory.iterdir(), key=lambda e: e.name):
        if entry.is_dir():
            yield from _markdown_files(entry, f"{prefix}{entry.name}/")
        elif entry.name.endswith(".md"):
            yield f"{prefix}{entry.name}", entry


def build_static_prompt(docs_dir: Traversable = DOCS_DIR) -> str:
    parts = [_PREAMBLE, "\n## Documentation\n"]
    try:
        md_files = list(_markdown_files(docs_dir))
    except OSError:
        md_files = []
    for rel, md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(text) > 200_000:
            continue
        parts.append(f"\n### docs/{rel}\n")
        parts.append(text)
    return "".join(parts)


def build_context_snapshot(project: Project | None) -> str:
    if project is None:
        return "## Current project\n(no project loaded)\n"
    lines = ["## Current project", ""]
    try:
        lines.append(f"- Neurons: {int(project.max_neurons)}")
        lines.append(f"- Simulation duration: {float(project.max_time_secs):.3f} s")
        lines.append(f"- Stim arrays: {sorted(project.stim_arrays.keys())}")
        experiment_names = sorted(project.experiments.keys())
        lines.append(
            f"- Experiments loaded: {experiment_names if experiment_names else 'none'}"
        )
    except Exception:
        lines.append("- (project state unavailable)")
    return "\n".join(lines) + "\n"
