from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets


class CellposeLegendHelperDialog(QtWidgets.QDialog):
    """Placeholder dialog for requesting Cellpose artifacts."""

    # This will request a .png and masks.npy which is a 2d array with 0s for blanks and then the nubmer ids for segmented things in the iamge 9cells - from cell2psoe e.g. 0 0 0 0 0 0 1 1 1 00 ... then next row maybe 0 0 0 0 0 0 0 0 1 1 0 0 2 etc.

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.png_file_path: Path | None = None
        self.mask_file_path: Path | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Cellpose To Neuron ID Legend Helper - Deliberately empty")
        self.setModal(True)
        self.resize(480, 220)

        # todo
