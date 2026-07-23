"""Tests for the shared PyTorch GPU-availability helpers."""

from __future__ import annotations

import sys
import types
from typing import Any


def _fake_torch(*, version: str, cuda: bool, mps: bool | None):
    """Build a stand-in ``torch`` module (setattr with a variable key keeps ruff
    and mypy happy about attributes on ModuleType)."""
    attrs: dict[str, Any] = {
        "__version__": version,
        "cuda": type("C", (), {"is_available": staticmethod(lambda: cuda)})(),
    }
    if mps is not None:
        attrs["backends"] = types.SimpleNamespace(
            mps=type("M", (), {"is_available": staticmethod(lambda: mps)})()
        )
    mod = types.ModuleType("torch")
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_torch_gpu_status_cuda(monkeypatch):
    from neuroseq.gpu import torch_gpu_status

    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(version="2.5.0+cu124", cuda=True, mps=False)
    )
    available, message = torch_gpu_status()
    assert available is True
    assert "CUDA" in message


def test_torch_gpu_status_mps(monkeypatch):
    from neuroseq.gpu import torch_gpu_status

    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(version="2.5.0", cuda=False, mps=True)
    )
    available, message = torch_gpu_status()
    assert available is True
    assert "MPS" in message


def test_torch_gpu_status_cpu_only_gives_install_hint(monkeypatch):
    from neuroseq.gpu import INSTALL_GPU_TORCH_HINT
    from neuroseq.gpu import torch_gpu_status

    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(version="2.13.0+cpu", cuda=False, mps=False)
    )
    available, message = torch_gpu_status()
    assert available is False
    assert "2.13.0+cpu" in message
    assert "CPU-only" in message
    assert INSTALL_GPU_TORCH_HINT in message
    assert "Intel" in message  # explains Intel iGPUs are unsupported


def test_torch_gpu_status_torch_missing(monkeypatch):
    from neuroseq.gpu import torch_gpu_status

    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` -> ImportError
    available, message = torch_gpu_status()
    assert available is False
    assert "not available" in message


def test_warn_if_gpu_unavailable_no_dialog_when_gpu_present(monkeypatch, mocker):
    import neuroseq.gpu as gpu

    monkeypatch.setattr(gpu, "_cpu_warning_shown", False, raising=False)
    monkeypatch.setattr(gpu, "torch_gpu_status", lambda: (True, "CUDA"))
    from qtpy import QtWidgets

    info = mocker.patch.object(QtWidgets.QMessageBox, "information")

    assert gpu.warn_if_gpu_unavailable(None) is True
    info.assert_not_called()


def test_warn_if_gpu_unavailable_shows_once(monkeypatch, mocker):
    import neuroseq.gpu as gpu

    monkeypatch.setattr(gpu, "_cpu_warning_shown", False, raising=False)
    monkeypatch.setattr(gpu, "torch_gpu_status", lambda: (False, "no gpu, do X"))
    from qtpy import QtWidgets

    info = mocker.patch.object(QtWidgets.QMessageBox, "information")

    assert gpu.warn_if_gpu_unavailable(None) is False
    assert gpu.warn_if_gpu_unavailable(None) is False  # second call suppressed
    info.assert_called_once()
