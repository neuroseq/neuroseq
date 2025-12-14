from __future__ import annotations

import logging

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.model import Model


class SyntheticDataDialog(QtWidgets.QDialog):
    """Dialog to generate synthetic trace and spike data and apply to widgets.

    The dialog can be constructed with references to an existing
    `TraceWidget` and `SpikeWidget`. Generated data will be applied to those
    widgets when the user clicks "Generate & Apply".
    """

    def __init__(
        self,
        model: Model,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = model

        self.setWindowTitle("Generate Synthetic Data")
        self.setModal(True)
        self.resize(520, 360)

        layout = QtWidgets.QVBoxLayout(self)

        # Patch clamp

        data_patch_clamp_group = QtWidgets.QGroupBox("Patch clamp")

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select traces:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_combo = QtWidgets.QComboBox()
        self.data_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(TraceData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_combo.addItem("")
        for name in data_names:
            self.data_combo.addItem(name)

        if data_names:
            self.data_combo.setCurrentIndex(1)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_combo)
        data_patch_clamp_group.setLayout(data_inner_layout)

        layout.addWidget(data_patch_clamp_group)

        ### Calcium

        data_calcium_group = QtWidgets.QGroupBox("Calcium Imaging")

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select spikes:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_combo = QtWidgets.QComboBox()
        self.data_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(SpikeData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_combo.addItem("")
        for name in data_names:
            self.data_combo.addItem(name)

        if data_names:
            self.data_combo.setCurrentIndex(1)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_combo)
        data_calcium_group.setLayout(data_inner_layout)

        layout.addWidget(data_calcium_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
