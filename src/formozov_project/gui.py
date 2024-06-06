from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from formozov_project.heatmap_widget import HeatmapWidget


class Gui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Formozov Project")
        grid_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(grid_layout)
        self.label = QtWidgets.QLabel("Formozov Project GUI prototype")
        grid_layout.addWidget(self.label)
        # for now just display a heatmap timeseries of random data:
        self.matrix_plot = HeatmapWidget(np.random.rand(100, 100, 100))
        grid_layout.addWidget(self.matrix_plot)
        self.btn_ok = QtWidgets.QPushButton("&OK")
        self.btn_ok.clicked.connect(self.close)
        grid_layout.addWidget(self.btn_ok)
        self.setCentralWidget(central_widget)
        self.resize(800, 600)
