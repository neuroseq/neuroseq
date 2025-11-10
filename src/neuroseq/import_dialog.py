from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import brian2.numpy_ as np
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.importer import Suite2pSpikePayload
from neuroseq.importer import Suite2pTracePayload
from neuroseq.importer import compute_normalized_suite2p_spikes
from neuroseq.importer import compute_normalized_suite2p_traces
from neuroseq.importer import prepare_suite2p_spikes
from neuroseq.importer import prepare_suite2p_traces


class Suite2pImportResult(TypedDict, total=False):
    suite2p_contrasting_traces: Suite2pTracePayload
    suite2p_spikes: Suite2pSpikePayload
    _replace_existing: bool


class Suite2pImportDialog(QtWidgets.QDialog):
    """Dialog for importing Suite2p traces and optional spike data."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._imported_data: Suite2pImportResult | None = None
        self.f_file_path: Path | None = None
        self.fneu_file_path: Path | None = None
        self.spks_file_path: Path | None = None
        self.f_data: np.ndarray | None = None
        self.fneu_data: np.ndarray | None = None
        self.spike_data: np.ndarray | None = None
        self._setup_ui()

    # -- UI helpers -----------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Suite2p Data")
        self.setModal(True)
        self.resize(500, 420)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Select the Suite2p fluorescence file (F.npy). "
            "Optionally include Fneu.npy for neuropil subtraction and spks.npy "
            "to import spike events at the same time."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QFormLayout(file_group)

        self.f_file_path_label = QtWidgets.QLabel("No file selected")
        self.f_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_f_button = QtWidgets.QPushButton("Browse…")
        browse_f_button.clicked.connect(self.browse_f_file)
        f_layout = QtWidgets.QHBoxLayout()
        f_layout.addWidget(self.f_file_path_label, stretch=1)
        f_layout.addWidget(browse_f_button)
        file_layout.addRow("F.npy:", f_layout)

        self.fneu_file_path_label = QtWidgets.QLabel("Optional – no file selected")
        self.fneu_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_fneu_button = QtWidgets.QPushButton("Browse…")
        browse_fneu_button.clicked.connect(self.browse_fneu_file)
        fneu_layout = QtWidgets.QHBoxLayout()
        fneu_layout.addWidget(self.fneu_file_path_label, stretch=1)
        fneu_layout.addWidget(browse_fneu_button)
        file_layout.addRow("Fneu.npy:", fneu_layout)

        self.spks_file_path_label = QtWidgets.QLabel("Optional – no file selected")
        self.spks_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_spks_button = QtWidgets.QPushButton("Browse…")
        browse_spks_button.clicked.connect(self.browse_spks_file)
        spks_layout = QtWidgets.QHBoxLayout()
        spks_layout.addWidget(self.spks_file_path_label, stretch=1)
        spks_layout.addWidget(browse_spks_button)
        file_layout.addRow("spks.npy:", spks_layout)

        layout.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.preview_text)

        self.spike_preview_text = QtWidgets.QTextEdit()
        self.spike_preview_text.setReadOnly(True)
        self.spike_preview_text.setMaximumHeight(90)
        preview_layout.addWidget(self.spike_preview_text)

        layout.addWidget(preview_group)

        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)

        self.max_neurons_spin = QtWidgets.QSpinBox()
        self.max_neurons_spin.setMinimum(1)
        self.max_neurons_spin.setMaximum(10000)
        self.max_neurons_spin.setValue(2)
        options_layout.addRow("Max neurons to import:", self.max_neurons_spin)

        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setDecimals(3)
        self.sampling_rate_spin.setMinimum(0.001)
        self.sampling_rate_spin.setMaximum(10000.0)
        self.sampling_rate_spin.setSingleStep(1.0)
        self.sampling_rate_spin.setValue(30.0)
        self.sampling_rate_spin.setSuffix(" Hz")
        self.sampling_rate_spin.setToolTip(
            "Sampling rate in Hz (base timeline unit is milliseconds)."
        )
        options_layout.addRow("Sample rate:", self.sampling_rate_spin)

        self.spike_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.spike_threshold_spin.setDecimals(3)
        self.spike_threshold_spin.setMinimum(0.0)
        self.spike_threshold_spin.setMaximum(1.0)
        self.spike_threshold_spin.setSingleStep(0.01)
        self.spike_threshold_spin.setValue(0.02)  # will be 36
        self.spike_threshold_spin.valueChanged.connect(
            lambda _: self._update_spike_preview()
        )
        options_layout.addRow("Spike threshold:", self.spike_threshold_spin)

        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing trace/spike data instead of appending"
        )
        self.replace_checkbox.setChecked(True)
        self.replace_checkbox.setToolTip(
            "When enabled, imported Suite2p traces or spikes will overwrite the "
            "current widget data rather than being concatenated to it."
        )
        options_layout.addRow(self.replace_checkbox)

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
        self._update_spike_preview()

    # -- File selection -------------------------------------------------
    def browse_f_file(self) -> None:
        filename, _ = getopenfilename(
            self, "Select Suite2p F.npy file", "", "NumPy files (*.npy);;All files (*)"
        )
        if filename:
            self.load_f_file(filename)

    def browse_fneu_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p Fneu.npy file",
            "",
            "NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self.load_fneu_file(filename)

    def browse_spks_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p spks.npy file (optional)",
            "",
            "NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self.load_spks_file(filename)

    def load_f_file(self, filename: str) -> None:
        try:
            self.f_file_path = Path(filename)
            self.f_file_path_label.setText(self.f_file_path.name)
            self.f_file_path_label.setStyleSheet("QLabel { color: black; }")
            self.f_data = np.load(filename)
            if self.f_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.f_data.ndim}D array")
            self.logger.info(
                f"Loaded F.npy ({self.f_data.shape[0]} neurons × {self.f_data.shape[1]} timepoints)"
            )
            self._check_files_loaded()
        except Exception as exc:
            self.logger.exception("Error loading F.npy file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading F.npy", f"Could not load {filename}:\n{exc}"
            )
            self.import_button.setEnabled(False)

    def load_fneu_file(self, filename: str) -> None:
        try:
            self.fneu_file_path = Path(filename)
            self.fneu_file_path_label.setText(self.fneu_file_path.name)
            self.fneu_file_path_label.setStyleSheet("QLabel { color: black; }")
            self.fneu_data = np.load(filename)
            if self.fneu_data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.fneu_data.ndim}D array")
            self.logger.info(
                f"Loaded Fneu.npy ({self.fneu_data.shape[0]} neurons × {self.fneu_data.shape[1]} timepoints)"
            )
            self._check_files_loaded()
        except Exception as exc:
            self.logger.exception("Error loading Fneu.npy file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading Fneu.npy", f"Could not load {filename}:\n{exc}"
            )
            self.import_button.setEnabled(False)

    def load_spks_file(self, filename: str) -> None:
        try:
            self.spks_file_path = Path(filename)
            self.spks_file_path_label.setText(self.spks_file_path.name)
            self.spks_file_path_label.setStyleSheet("QLabel { color: black; }")
            self.spike_data = np.load(filename)
            if self.spike_data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {self.spike_data.ndim}D array"
                )
            self.logger.info(
                f"Loaded spks.npy ({self.spike_data.shape[0]} neurons × {self.spike_data.shape[1]} timepoints)"
            )
            self._update_spike_preview()
            self._check_spike_shape_matches()
        except Exception as exc:
            self.logger.exception("Error loading spks.npy file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading spks.npy", f"Could not load {filename}:\n{exc}"
            )
            self.spks_file_path_label.setText("Optional – no file selected")
            self.spks_file_path_label.setStyleSheet("QLabel { color: gray; }")
            self.spike_data = None
            self._update_spike_preview()

    # -- Validation / preview ------------------------------------------
    def _check_files_loaded(self) -> None:
        if self.f_data is None:
            self.import_button.setEnabled(False)
            self.preview_text.clear()
            return

        if self.fneu_data is not None and self.f_data.shape != self.fneu_data.shape:
            QtWidgets.QMessageBox.critical(
                self,
                "Shape mismatch",
                f"F.npy shape {self.f_data.shape} does not match Fneu.npy shape {self.fneu_data.shape}",
            )
            self.import_button.setEnabled(False)
            return

        self.import_button.setEnabled(True)
        self._update_trace_preview()
        self._check_spike_shape_matches()

    def _check_spike_shape_matches(self) -> None:
        if self.spike_data is not None and self.f_data is not None:
            if self.spike_data.shape != self.f_data.shape:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Shape mismatch",
                    "spks.npy dimensions do not match F/Fneu data. Spike import will be disabled.",
                )
                self.spike_data = None
                self.spike_preview_text.setPlainText(
                    "Spikes disabled because shapes do not match fluorescence data."
                )
                self.spks_file_path_label.setText("Optional – no file selected")
                self.spks_file_path_label.setStyleSheet("QLabel { color: gray; }")
            else:
                self.spks_file_path_label.setStyleSheet("QLabel { color: black; }")
                self._update_spike_preview()

    def _update_trace_preview(self) -> None:
        if self.f_data is None:
            self.preview_text.clear()
            return

        try:
            normalized_traces, raw_min, raw_max = compute_normalized_suite2p_traces(
                self.f_data, self.fneu_data
            )
        except ValueError as exc:
            self.preview_text.setPlainText(str(exc))
            return

        n_neurons, n_timepoints = self.f_data.shape
        normalized_min = float(np.min(normalized_traces))
        normalized_max = float(np.max(normalized_traces))
        preview_lines = [
            f"F.npy shape: {self.f_data.shape}, range {np.min(self.f_data):.3f}–{np.max(self.f_data):.3f}",
        ]

        if self.fneu_data is not None:
            preview_lines.append(
                f"Fneu.npy shape: {self.fneu_data.shape}, range {np.min(self.fneu_data):.3f}–{np.max(self.fneu_data):.3f}"
            )
            preview_lines.append("Neuropil subtraction applied (F - 0.7 · Fneu)")
        else:
            preview_lines.append("Fneu.npy not provided – using F.npy directly")

        preview_lines.extend(
            [
                f"Contrasting traces: {n_neurons} neurons × {n_timepoints} timepoints",
                f"Pre-normalization range: {raw_min:.3f}–{raw_max:.3f}",
                f"Normalized range: {normalized_min:.3f}–{normalized_max:.3f}",
                f"Mean ± std (normalized): {np.mean(normalized_traces):.3f} ± {np.std(normalized_traces):.3f}",
            ]
        )
        self.preview_text.setPlainText("\n".join(preview_lines))

    def _update_spike_preview(self) -> None:
        if self.spike_data is None:
            self.spike_preview_text.setPlainText(
                "No spks.npy selected. Spike import is optional."
            )
            return

        threshold = self.spike_threshold_spin.value()
        normalized_spikes, raw_min, raw_max = compute_normalized_suite2p_spikes(
            self.spike_data
        )
        non_zero_count = int(np.sum(self.spike_data > 0))
        above_threshold = int(np.sum(normalized_spikes >= threshold))
        preview_lines = [
            f"spks.npy shape: {self.spike_data.shape}",
            f"Raw value range: {raw_min:.3f}–{raw_max:.3f}",
            "Normalized spike magnitudes scaled to 0–1",
            f"Non-zero entries: {non_zero_count}",
            f"Entries ≥ threshold ({threshold:.3f}): {above_threshold}",
        ]
        self.spike_preview_text.setPlainText("\n".join(preview_lines))

    # -- Import logic ---------------------------------------------------
    def import_data(self) -> None:
        if self.f_data is None:
            return

        try:
            self._imported_data = None
            sampling_rate = float(self.sampling_rate_spin.value())
            if sampling_rate <= 0.0:
                raise ValueError("Sample rate must be positive")
            dt = 1.0 / sampling_rate
            max_neurons = self.max_neurons_spin.value()
            n_neurons, n_timepoints = self.f_data.shape
            n_neurons_to_import = min(max_neurons, n_neurons)

            trace_payload = prepare_suite2p_traces(
                self.f_data,
                self.fneu_data,
                max_neurons=max_neurons,
                sampling_rate=sampling_rate,
            )

            imported_data: Suite2pImportResult = {
                "suite2p_contrasting_traces": trace_payload,
            }
            imported_data["_replace_existing"] = self.replace_checkbox.isChecked()

            spike_summary = ""
            if self.spike_data is not None:
                threshold = self.spike_threshold_spin.value()
                spike_payload = prepare_suite2p_spikes(
                    self.spike_data,
                    dt=dt,
                    threshold=threshold,
                    max_neurons=max_neurons,
                )
                imported_data["suite2p_spikes"] = spike_payload
                spike_summary = (
                    f"\nSpikes: {len(spike_payload['spikes'])} events ≥ {threshold:.3f}"
                )

            self._imported_data = imported_data
            QtWidgets.QMessageBox.information(
                self,
                "Import successful",
                "Successfully imported Suite2p data.\n"
                f"Traces: {n_neurons_to_import} neurons × {n_timepoints} timepoints."
                f"{spike_summary}",
            )
            self.accept()

        except Exception as exc:
            self.logger.exception("Error importing Suite2p data")
            QtWidgets.QMessageBox.critical(
                self, "Import Error", f"Error importing Suite2p data:\n{exc}"
            )

    def get_imported_data(self) -> Suite2pImportResult | None:
        """Return the last successfully imported data payload."""
        return self._imported_data
