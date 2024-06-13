from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.heatmap_widget import HeatmapWidget
from formozov_project.trace_widget import TraceWidget


@dataclass
class TempData:
    """Placeholder data to demo GUI features."""

    def __init__(self) -> None:
        n_timestamps = 200
        n_neurons = 64
        dt = 0.005
        self.timestamps = dt * np.arange(n_timestamps)
        self.matrices = np.random.rand(n_timestamps, n_neurons, n_neurons)
        self.traces = [np.random.rand(n_timestamps) for _ in range(n_neurons)]


class Gui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.data = TempData()
        self.setWindowTitle("Formozov Project")
        grid_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(grid_layout)
        self.btn_play = QtWidgets.QPushButton("&Play")
        grid_layout.addWidget(self.btn_play)
        # main GUI is left/right split between traces and heatmap
        split_top_bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.trace_widget = TraceWidget(
            traces=self.data.traces, timestamps=self.data.timestamps
        )
        split_top_bottom.addWidget(self.trace_widget)
        self.heatmap_widget = HeatmapWidget(self.data.matrices, self.data.timestamps)
        split_top_bottom.addWidget(self.heatmap_widget)
        grid_layout.addWidget(split_top_bottom)
        self.btn_ok = QtWidgets.QPushButton("&OK")
        grid_layout.addWidget(self.btn_ok)
        self.setCentralWidget(central_widget)
        self.resize(800, 600)

        # connect widgets
        self.btn_play.clicked.connect(
            lambda: self.heatmap_widget.play(
                (len(self.data.timestamps) - 1)
                / self.data.timestamps[-1]  # calculate fps
            )
        )
        self.btn_ok.clicked.connect(self.close)
        self.heatmap_widget.connect_trace_widget(self.trace_widget)
        self.trace_widget.connect_heatmap_widget(self.heatmap_widget)
