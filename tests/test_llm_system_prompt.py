from __future__ import annotations

import pathlib

import brian2.numpy_ as np

from neuroseq.llm.system_prompt import DOCS_DIR
from neuroseq.llm.system_prompt import build_context_snapshot
from neuroseq.llm.system_prompt import build_static_prompt
from neuroseq.llm.system_prompt import find_docs_dir
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def test_build_static_prompt_includes_each_doc_with_header(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "a.md").write_text("Alpha", encoding="utf-8")
    subdir = tmp_path / "guide"
    subdir.mkdir()
    (subdir / "b.md").write_text("Beta", encoding="utf-8")

    prompt = build_static_prompt(tmp_path)

    assert "### docs/a.md" in prompt
    assert "Alpha" in prompt
    assert "### docs/guide/b.md" in prompt
    assert "Beta" in prompt


def test_build_static_prompt_skips_large_files(tmp_path: pathlib.Path) -> None:
    (tmp_path / "small.md").write_text("small", encoding="utf-8")
    (tmp_path / "large.md").write_text("x" * 300_000, encoding="utf-8")

    prompt = build_static_prompt(tmp_path)

    assert "small" in prompt
    assert "large.md" not in prompt


def test_build_static_prompt_skips_images(tmp_path: pathlib.Path) -> None:
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    prompt = build_static_prompt(tmp_path)

    assert "hello" in prompt
    assert "image.png" not in prompt


def test_find_docs_dir_prefers_docs_packaged_in_wheel(tmp_path: pathlib.Path) -> None:
    (tmp_path / "docs").mkdir()

    assert find_docs_dir(tmp_path) == tmp_path / "docs"


def test_find_docs_dir_falls_back_to_repository_docs(tmp_path: pathlib.Path) -> None:
    repo_docs = pathlib.Path(__file__).resolve().parents[1] / "docs"

    assert find_docs_dir(tmp_path) == repo_docs


def test_default_docs_dir_contains_the_docs() -> None:
    assert (DOCS_DIR / "index.md").is_file()
    assert "### docs/getting_started/installation.md" in build_static_prompt()


def test_build_context_snapshot_lists_project_state() -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "foo",
            StimType.Excitation,
            np.zeros((1000, 5), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    project.update_limits()

    snapshot = build_context_snapshot(project)

    assert "Neurons: 5" in snapshot
    assert "1.000 s" in snapshot
    assert "foo" in snapshot


def test_build_context_snapshot_no_project() -> None:
    assert "no project loaded" in build_context_snapshot(None)
