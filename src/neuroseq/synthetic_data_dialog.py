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
        self.data_patch_clamp_combo = QtWidgets.QComboBox()
        self.data_patch_clamp_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(TraceData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_patch_clamp_combo.addItem("")
        for name in data_names:
            self.data_patch_clamp_combo.addItem(name)

        if data_names:
            self.data_patch_clamp_combo.setCurrentIndex(1)

        self.data_patch_clamp_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_patch_clamp_combo)
        data_patch_clamp_group.setLayout(data_inner_layout)

        layout.addWidget(data_patch_clamp_group)

        ### Imaging

        data_imaging_group = QtWidgets.QGroupBox("Imaging")

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select spikes:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_imaging_combo = QtWidgets.QComboBox()
        self.data_imaging_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(SpikeData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_imaging_combo.addItem("")
        for name in data_names:
            self.data_imaging_combo.addItem(name)

        if data_names:
            self.data_imaging_combo.setCurrentIndex(1)

        self.data_imaging_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_imaging_combo)
        data_imaging_group.setLayout(data_inner_layout)
        layout.addWidget(data_imaging_group)

        # Data Preview
        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        # Initial preview update
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the data preview based on current selections."""
        preview_lines = []

        # Patch clamp preview
        trace_name = self.data_patch_clamp_combo.currentText()
        if trace_name:
            trace_data = self.model.get_data(trace_name, TraceData)
            if trace_data:
                preview_lines.append(f"Patch Clamp: {trace_name}")
                preview_lines.append(f"  Size of the traces: {trace_data.traces.shape}")
                # preview_lines.append(
                #    f"  Time range: {trace_data.time[0]:.2f}–{trace_data.time[-1]:.2f} ms"
                # )
                # preview_lines.append(
                #    f"  Value range: {trace_data.data.min():.3f}–{trace_data.data.max():.3f}"
                # )

        # Imaging preview
        spike_name = self.data_imaging_combo.currentText()
        if spike_name:
            spike_data = self.model.get_data(spike_name, SpikeData)
            if spike_data:
                if preview_lines:
                    preview_lines.append("")
                preview_lines.append(f"Imaging: {spike_name}")
                # preview_lines.append(f"  Neurons: {len(spike_data.neuron_ids)}")
                # total_spikes = sum(
                #     len(times) for times in spike_data.spike_times.values()
                # )
                # preview_lines.append(f"  Total spikes: {total_spikes}")
                # if spike_data.spike_times:
                #     all_spike_times = [
                #         t
                #         for times in spike_data.spike_times.values()
                #         for t in times
                #     ]
                #     if all_spike_times:
                #         preview_lines.append(
                #             f"  Time range: {min(all_spike_times):.2f}–{max(all_spike_times):.2f} ms"
                #         )

        self.preview_text.setPlainText(
            "\n".join(preview_lines) if preview_lines else "No data selected"
        )
