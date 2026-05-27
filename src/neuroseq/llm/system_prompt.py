from __future__ import annotations

import pathlib

from neuroseq.project import Project

DOCS_DIR = pathlib.Path(__file__).resolve().parents[3] / "docs"

_PREAMBLE = """\
You are NeuroSeq's in-app help assistant. NeuroSeq is a desktop GUI for
computational neuroscience: it combines Brian2 spiking neural network
simulation, interactive stimulus design, multi-source experimental data
import (Suite2p calcium imaging, Cellpose spatial footprints, patch-clamp,
Igor2 binaries, SysCon2 holographic sequences), and hardware export for
holographic stimulation.

Answer the user's question by referencing the documentation excerpts below
and the live project snapshot. Prefer concrete menu paths over vague
descriptions. If the docs do not cover the user's question, say so plainly
rather than inventing menu items or feature names. Keep replies concise
unless the user asks for more detail.
"""


def build_static_prompt(docs_dir: pathlib.Path = DOCS_DIR) -> str:
    parts = [_PREAMBLE, "\n## Documentation\n"]
    for md_path in sorted(docs_dir.rglob("*.md")):
        try:
            if md_path.stat().st_size > 200_000:
                continue
            rel = md_path.relative_to(docs_dir).as_posix()
            parts.append(f"\n### docs/{rel}\n")
            parts.append(md_path.read_text(encoding="utf-8"))
        except OSError:
            continue
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
