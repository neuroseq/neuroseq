from __future__ import annotations

from qtpy import QtWidgets


class Gui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Formozov Project")

        grid_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(grid_layout)
        self.label = QtWidgets.QLabel("Formozov Project")
        grid_layout.addWidget(self.label)
        self.btn_ok = QtWidgets.QPushButton("&OK")
        self.btn_ok.clicked.connect(self.close)
        grid_layout.addWidget(self.btn_ok)
        self.setCentralWidget(central_widget)
