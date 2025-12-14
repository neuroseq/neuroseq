from __future__ import annotations

import logging

from qtpy import QtWidgets


class SyntheticDataDialog(QtWidgets.QDialog):
    """Dialog to generate synthetic trace and spike data and apply to widgets.

    The dialog can be constructed with references to an existing
    `TraceWidget` and `SpikeWidget`. Generated data will be applied to those
    widgets when the user clicks "Generate & Apply".
    """

    def __init__(
        self,
        trace_widget: object | None = None,
        spike_widget: object | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.trace_widget = trace_widget
        self.spike_widget = spike_widget

        self.setWindowTitle("Generate Synthetic Data")
        self.setModal(True)
        self.resize(520, 360)

        layout = QtWidgets.QVBoxLayout(self)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
