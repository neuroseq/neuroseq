from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
import scipy.io
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.experiment import TraceData
from neuroseq.importer_cellpose import CellposeImportResult


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp traces from .mat files."""

    def __init__(
        self,
        cellpose_import: CellposeImportResult | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.cellpose_import = cellpose_import
        self.imported_data: list[TraceData] = []
        self.experiment_name = "PatchClamp Import"
        self.append_data = False
        self.file_path: Path | None = None
        self.mat_data: dict[str, Any] | None = None
        self.overlay_widget = CellOverlayWidget("Cell overlay preview")
        self._setup_ui()
        self._load_overlay_from_cellpose()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)
        self.resize(640, 540)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Select a .mat file containing patch clamp data. Expected variable: "
            "'neur_act' with fields 'soma', 'dend', and 'bg'."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addWidget(self.overlay_widget)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        layout.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)

        self.data_source_combo = QtWidgets.QComboBox()
        self.data_source_combo.addItems(["soma", "dend", "bg"])
        options_layout.addRow("Data source:", self.data_source_combo)

        self.max_neurons_spin = QtWidgets.QSpinBox()
        self.max_neurons_spin.setMinimum(1)
        self.max_neurons_spin.setMaximum(1000)
        self.max_neurons_spin.setValue(20)
        options_layout.addRow("Max neurons to import:", self.max_neurons_spin)

        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setDecimals(2)
        self.sampling_rate_spin.setMinimum(0.1)
        self.sampling_rate_spin.setMaximum(10000.0)
        self.sampling_rate_spin.setSingleStep(1.0)
        self.sampling_rate_spin.setValue(30.0)
        self.sampling_rate_spin.setSuffix(" Hz")
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)

        mapping_layout = QtWidgets.QHBoxLayout()
        self.mapping_buttons = QtWidgets.QButtonGroup(self)
        for option in ("Linear", "Random", "Custom"):
            button = QtWidgets.QRadioButton(option)
            self.mapping_buttons.addButton(button)
            mapping_layout.addWidget(button)
        first_button = self.mapping_buttons.buttons()[0]
        first_button.setChecked(True)
        options_layout.addRow("Mapping strategy:", mapping_layout)

        layout.addWidget(options_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def _load_overlay_from_cellpose(self) -> None:
        if self.cellpose_import is None:
            self.overlay_widget.set_empty_message(
                "Import Cellpose segmentation to see the overlay here."
            )
            return

        try:
            self.overlay_widget.set_overlay(
                self.cellpose_import.overlay_image,
                self.cellpose_import.segmentation_array,
            )
        except Exception as exc:
            self.logger.warning("Could not display Cellpose overlay: %s", exc)
            self.overlay_widget.set_empty_message(
                "Cellpose overlay could not be displayed."
            )

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Patch Clamp Data File",
            "",
            "MATLAB files (*.mat);;All files (*)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, filename: str) -> None:
        try:
            self.file_path = Path(filename)
            self.file_path_label.setText(self.file_path.name)
            self.file_path_label.setStyleSheet("QLabel { color: black; }")

            mat_data = scipy.io.loadmat(
                filename, struct_as_record=False, squeeze_me=True
            )
            if "neur_act" not in mat_data:
                raise ValueError("File does not contain 'neur_act' variable")

            neur_act = mat_data["neur_act"]
            required_fields = ["soma", "dend", "bg"]
            missing = [
                field for field in required_fields if not hasattr(neur_act, field)
            ]
            if missing:
                raise ValueError(f"Missing required fields: {', '.join(missing)}")

            self.mat_data = {
                "soma": np.asarray(neur_act.soma),
                "dend": np.asarray(neur_act.dend),
                "bg": np.asarray(neur_act.bg),
            }
            self._update_preview()
            self.import_button.setEnabled(True)
        except Exception as exc:
            self.logger.exception("Error loading patch clamp file")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{exc}",
            )
            self.import_button.setEnabled(False)

    def _update_preview(self) -> None:
        if self.mat_data is None:
            self.preview_text.clear()
            return

        lines = []
        for field_name, data in self.mat_data.items():
            if not isinstance(data, np.ndarray):
                continue
            if data.ndim != 2:
                lines.append(f"{field_name}: unexpected shape {data.shape}")
                continue
            lines.append(
                f"{field_name.upper()}: {data.shape[0]} neurons × {data.shape[1]} timepoints"
            )
            lines.append(
                f"  mean {np.mean(data):.3f} ± {np.std(data):.3f}, range {np.min(data):.3f} to {np.max(data):.3f}"
            )
            lines.append("")

        self.preview_text.setPlainText("\n".join(lines).strip())

    def _get_mapping_strategy(self) -> str:
        checked = self.mapping_buttons.checkedButton()
        return checked.text() if checked is not None else "Linear"

    def import_data(self) -> None:
        if self.mat_data is None:
            return

        try:
            data_source = self.data_source_combo.currentText()
            voltage_traces = np.asarray(self.mat_data[data_source])
            if voltage_traces.ndim != 2:
                raise ValueError("Patch clamp data must be 2D (neurons × timepoints)")

            n_neurons, n_timepoints = voltage_traces.shape
            n_neurons_to_import = min(self.max_neurons_spin.value(), n_neurons)
            sampling_rate = self.sampling_rate_spin.value()
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be positive")
            dt = 1.0 / sampling_rate

            voltage_traces = voltage_traces[:n_neurons_to_import, :]
            trace_name = f"PatchClamp {data_source} ({self._get_mapping_strategy()})"
            trace_data = TraceData(
                trace_name,
                voltage_traces.astype(np.float32),
                "mV",
                dt,
            )

            self.imported_data = [trace_data]
            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {n_neurons_to_import} neurons from '{data_source}' with "
                f"{n_timepoints} timepoints at {sampling_rate:.2f} Hz.",
            )
            self.accept()
        except Exception as exc:
            self.logger.exception("Error importing patch clamp data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing data:\n{exc}",
            )
