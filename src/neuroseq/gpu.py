"""PyTorch GPU-availability helpers shared by the imaging import paths.

PyTorch is a heavy, optional-at-import dependency, so it is imported lazily.
Cellpose and suite2p can use a GPU only through PyTorch's CUDA (NVIDIA) or MPS
(Apple-Silicon) backends; a CPU-only PyTorch build therefore forces CPU
segmentation/extraction, which these helpers detect and explain.
"""

from __future__ import annotations

# Command that installs a GPU-capable PyTorch build via uv's auto backend
# detection (picks the CUDA wheel matching the machine's driver, or CPU if none).
INSTALL_GPU_TORCH_HINT = "uv pip install --torch-backend=auto torch"

# Fallback for environments not managed by uv (e.g. a conda env NeuroSeq was
# installed into with plain pip) — uv's auto backend detection doesn't apply there.
INSTALL_GPU_TORCH_HINT_PIP = (
    "pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu128"
)


def torch_gpu_status() -> tuple[bool, str]:
    """Return ``(gpu_available, message)`` for the installed PyTorch.

    ``gpu_available`` is ``True`` when PyTorch exposes a GPU backend that
    Cellpose/suite2p can use (CUDA on NVIDIA, MPS on Apple Silicon). When it is
    ``False`` because a CPU-only build is installed, the message explains how to
    install a GPU build.
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch is not available in this environment."

    version = str(getattr(torch, "__version__", "unknown"))
    try:
        if torch.cuda.is_available():
            return True, f"GPU acceleration enabled: CUDA (PyTorch {version})."
    except Exception:
        pass
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        if mps is not None and mps.is_available():
            return True, f"GPU acceleration enabled: Apple MPS (PyTorch {version})."
    except Exception:
        pass

    cpu_build = "+cpu" in version or ("+cu" not in version and "+rocm" not in version)
    detail = " (a CPU-only build)" if cpu_build else ""
    message = (
        f"GPU acceleration is unavailable — the installed PyTorch ({version}) has "
        f"no usable CUDA/MPS backend{detail}, so this will run on the CPU "
        "(much slower).\n\n"
        "To enable the GPU on a machine with an NVIDIA (or Apple-Silicon) GPU, "
        "install a GPU build of PyTorch into this environment:\n"
        f"    {INSTALL_GPU_TORCH_HINT}\n\n"
        "Not using uv (e.g. NeuroSeq is installed into a conda environment)? "
        "Install directly with pip instead:\n"
        f"    {INSTALL_GPU_TORCH_HINT_PIP}\n\n"
        "Note: Cellpose can use only NVIDIA (CUDA) and Apple (MPS) GPUs — Intel "
        "integrated graphics are not supported."
    )
    return False, message


_cpu_warning_shown = False


def warn_if_gpu_unavailable(parent=None, *, once: bool = True) -> bool:
    """Pop a one-time info dialog when PyTorch has no usable GPU backend.

    Returns ``True`` if a GPU is available (nothing shown). With ``once=True``
    (the default) at most one dialog is shown per process, so repeated launches
    do not nag. Qt is imported lazily so :func:`torch_gpu_status` stays usable
    (and testable) without a running GUI.
    """
    global _cpu_warning_shown
    available, message = torch_gpu_status()
    if available:
        return True
    if once and _cpu_warning_shown:
        return False
    _cpu_warning_shown = True
    try:
        from qtpy import QtWidgets

        QtWidgets.QMessageBox.information(
            parent, "GPU acceleration unavailable", message
        )
    except Exception:
        pass
    return False
