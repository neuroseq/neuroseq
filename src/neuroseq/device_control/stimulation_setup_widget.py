"""The "Stimulation setup" pane: one tab per stimulus-delivery target, plus image
acquisition.

Each target (the SysCon2/HOLO4D rig, BrainScaleS-2 neuromorphic hardware, ...)
and the ScanImage acquisition controls keep their own panel; this widget only
hosts them so the workspace has a single pane to show, hide and place in its
splitter.
"""

from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.qt_helpers import StableSizeTabWidget

SYSCON2_TAB_TITLE = "SysCon2 (HOLO4D)"
BSS2_TAB_TITLE = "BrainScaleS-2"
SCANIMAGE_TAB_TITLE = "ScanImage (imaging)"


class StimulationSetupWidget(StableSizeTabWidget):
    """Tabbed host for the stimulus-delivery panels of one workspace."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)

    def add_setup(self, panel: QtWidgets.QWidget, title: str) -> int:
        """Append ``panel`` as a tab titled ``title``; returns its tab index."""
        return self.addTab(panel, title)

    def show_setup(self, title: str) -> bool:
        """Bring the tab titled ``title`` to the front, if present."""
        for index in range(self.count()):
            if self.tabText(index) == title:
                self.setCurrentIndex(index)
                return True
        return False
