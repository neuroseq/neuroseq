from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.fpl_widget import FullScreenFPLWidget


class TestPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preview Test Window")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        preview = FullScreenFPLWidget(parent=self)
        preview.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(preview)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.setGeometry(available)
        self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowMaximized)
