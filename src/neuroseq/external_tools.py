"""Launch external imaging tools (Fiji/ImageJ, Cellpose) from the GUI.

Shared entry points so the main-window toolbar buttons launch the same tools
the import dialogs do. Each returns ``True`` when the tool was started and
surfaces failures via a message box on ``parent``.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.fiji import resolve_fiji_path

logger = logging.getLogger(__name__)


def launch_fiji(parent: QtWidgets.QWidget | None = None) -> bool:
    """Launch Fiji/ImageJ, prompting for its path if not yet configured."""
    fiji_path = resolve_fiji_path(parent)
    if not fiji_path:
        return False
    try:
        started = QtCore.QProcess.startDetached(fiji_path, [])
        if not started:
            raise RuntimeError(f"QProcess.startDetached({fiji_path!r}) returned False")
        return True
    except Exception as exc:
        logger.exception("Failed to launch Fiji")
        QtWidgets.QMessageBox.critical(
            parent,
            "Failed to launch Fiji",
            f"Could not launch Fiji:\n{exc}",
        )
        return False


def launch_cellpose(parent: QtWidgets.QWidget | None = None) -> bool:
    """Launch the Cellpose GUI from the current Python environment."""
    if importlib.util.find_spec("cellpose") is None:
        QtWidgets.QMessageBox.information(
            parent,
            "Install Cellpose",
            "Cellpose is not installed in the current Python environment.",
        )
        return False
    try:
        # ``-m cellpose`` starts the Cellpose GUI as a detached process.
        subprocess.Popen(
            [sys.executable, "-m", "cellpose"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # If PyTorch has no GPU backend, Cellpose's "use GPU" box is disabled;
        # explain why and how to fix it (once per session).
        from neuroseq.gpu import warn_if_gpu_unavailable

        warn_if_gpu_unavailable(parent)
        return True
    except Exception as exc:
        logger.exception("Failed to launch Cellpose")
        QtWidgets.QMessageBox.critical(
            parent,
            "Failed to launch Cellpose",
            f"Could not launch Cellpose:\n{exc}\n\n"
            "Make sure cellpose is installed in the current Python environment.",
        )
        return False
