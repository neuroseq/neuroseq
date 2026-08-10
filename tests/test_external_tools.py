from __future__ import annotations

import sys

from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore

import neuroseq.external_tools as external_tools


def test_launch_fiji_starts_resolved_path(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    del qtbot
    calls: list[tuple[str, list[str]]] = []

    def fake_start(path, args):
        calls.append((path, args))
        return True

    monkeypatch.setattr(external_tools, "resolve_fiji_path", lambda parent: "/bin/fiji")
    monkeypatch.setattr(QtCore.QProcess, "startDetached", staticmethod(fake_start))
    assert external_tools.launch_fiji(None) is True
    assert calls == [("/bin/fiji", [])]


def test_launch_fiji_cancelled_path_does_not_start(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    del qtbot
    started: list[str] = []

    def fake_start(path, args):
        started.append(path)
        return True

    monkeypatch.setattr(external_tools, "resolve_fiji_path", lambda parent: "")
    monkeypatch.setattr(QtCore.QProcess, "startDetached", staticmethod(fake_start))
    assert external_tools.launch_fiji(None) is False
    assert started == []


def test_launch_cellpose_starts_module(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    del qtbot
    monkeypatch.setattr(
        external_tools.importlib.util, "find_spec", lambda name: object()
    )
    # warn_if_gpu_unavailable imports torch (whose CUDA init spawns its own
    # subprocess); stub it so we only observe the cellpose launch.
    monkeypatch.setattr("neuroseq.gpu.warn_if_gpu_unavailable", lambda parent: False)
    calls: list[list[str]] = []

    def fake_popen(args, **kwargs):
        calls.append(args)
        return object()

    monkeypatch.setattr(external_tools.subprocess, "Popen", fake_popen)
    assert external_tools.launch_cellpose(None) is True
    assert calls == [[sys.executable, "-m", "cellpose"]]


def test_launch_cellpose_missing_warns_and_does_not_start(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    del qtbot
    monkeypatch.setattr(external_tools.importlib.util, "find_spec", lambda name: None)
    started: list[list[str]] = []
    monkeypatch.setattr(
        external_tools.subprocess,
        "Popen",
        lambda args, **kwargs: started.append(args),
    )
    assert external_tools.launch_cellpose(None) is False
    assert started == []
