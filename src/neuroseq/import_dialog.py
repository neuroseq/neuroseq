from __future__ import annotations

import logging
from pathlib import Path

import brian2.numpy_ as np
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.data import Data
from neuroseq.importer_suite2p import compute_normalized_suite2p_spikes
from neuroseq.importer_suite2p import compute_normalized_suite2p_traces
from neuroseq.importer_suite2p import prepare_suite2p_spikes
from neuroseq.importer_suite2p import prepare_suite2p_traces


class Suite2pImportDialog(QtWidgets.QDialog):
    """Dialog for importing Suite2p (npy) or CSV traces/spikes with folder auto-detect."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.experiment_name: str = "Suite2p Import"
        self.experiment_duration: float = 0.0
        self.imported_data: list[Data] = []

        self.logger = logging.getLogger(__name__)
        self.data_directory_path: Path | None = None
        self.f_file_path: Path | None = None
        self.fneu_file_path: Path | None = None
        self.spks_file_path: Path | None = None
        self.stat_file_path: Path | None = None
        self.f_data: np.ndarray | None = None
        self.fneu_data: np.ndarray | None = None
        self.spike_data: np.ndarray | None = None
        self.stat_data: np.ndarray | None = None
        self._auto_detected_files: list[str] = []
        self._setup_ui()

    # -- UI helpers -----------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Experimental Data")
        self.setModal(True)
        self.resize(500, 420)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Pick a folder that contains Suite2p exports (F/Fneu/spks/stat) "
            "or CSV equivalents."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QFormLayout(file_group)

        self.data_folder_path_label = QtWidgets.QLabel("No folder selected")
        self.data_folder_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_folder_button = QtWidgets.QPushButton("Choose Folder…")
        browse_folder_button.clicked.connect(self.browse_data_folder)
        folder_layout = QtWidgets.QHBoxLayout()
        folder_layout.addWidget(self.data_folder_path_label, stretch=1)
        folder_layout.addWidget(browse_folder_button)
        file_layout.addRow("Data folder:", folder_layout)

        self.show_individual_files_button = QtWidgets.QPushButton(
            "Show individual file imports"
        )
        self.show_individual_files_button.clicked.connect(
            self._toggle_individual_file_imports
        )
        file_layout.addRow(self.show_individual_files_button)

        self.individual_files_group = QtWidgets.QGroupBox("Individual File Selection")
        individual_files_layout = QtWidgets.QFormLayout(self.individual_files_group)

        self.f_file_path_label = QtWidgets.QLabel("No file selected")
        self.f_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_f_button = QtWidgets.QPushButton("Browse…")
        browse_f_button.clicked.connect(self.browse_f_file)
        f_layout = QtWidgets.QHBoxLayout()
        f_layout.addWidget(self.f_file_path_label, stretch=1)
        f_layout.addWidget(browse_f_button)
        individual_files_layout.addRow("F (npy/csv):", f_layout)

        self.fneu_file_path_label = QtWidgets.QLabel("Optional – no file selected")
        self.fneu_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_fneu_button = QtWidgets.QPushButton("Browse…")
        browse_fneu_button.clicked.connect(self.browse_fneu_file)
        fneu_layout = QtWidgets.QHBoxLayout()
        fneu_layout.addWidget(self.fneu_file_path_label, stretch=1)
        fneu_layout.addWidget(browse_fneu_button)
        individual_files_layout.addRow("Fneu (npy/csv):", fneu_layout)

        self.spks_file_path_label = QtWidgets.QLabel("Optional – no file selected")
        self.spks_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_spks_button = QtWidgets.QPushButton("Browse…")
        browse_spks_button.clicked.connect(self.browse_spks_file)
        spks_layout = QtWidgets.QHBoxLayout()
        spks_layout.addWidget(self.spks_file_path_label, stretch=1)
        spks_layout.addWidget(browse_spks_button)
        individual_files_layout.addRow("spks (npy/csv):", spks_layout)

        self.stat_file_path_label = QtWidgets.QLabel("Optional – no file selected")
        self.stat_file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_stat_button = QtWidgets.QPushButton("Browse…")
        browse_stat_button.clicked.connect(self.browse_stat_file)
        stat_layout = QtWidgets.QHBoxLayout()
        stat_layout.addWidget(self.stat_file_path_label, stretch=1)
        stat_layout.addWidget(browse_stat_button)
        individual_files_layout.addRow("stat.npy:", stat_layout)

        self.individual_files_group.hide()

        layout.addWidget(file_group)
        layout.addWidget(self.individual_files_group)

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
        self.spike_threshold_spin.setValue(0.02)
        self.spike_threshold_spin.valueChanged.connect(
            lambda _: self._update_spike_preview()
        )
        options_layout.addRow("Spike threshold:", self.spike_threshold_spin)

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

    def _toggle_individual_file_imports(self) -> None:
        if self.individual_files_group.isVisible():
            self.individual_files_group.hide()
            self.show_individual_files_button.setText("Show individual file imports")
            self.data_folder_path_label.setText("No folder selected")
            self.data_folder_path_label.setStyleSheet("QLabel { color: gray; }")
        else:
            self.individual_files_group.show()
            self.show_individual_files_button.setText("Hide individual file imports")

    # -- File selection -------------------------------------------------
    def browse_f_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p F file",
            "",
            "NumPy/CSV files (*.npy *.csv);;All files (*)",
        )
        if filename:
            self.load_f_file(filename)

    def browse_fneu_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p Fneu file",
            "",
            "NumPy/CSV files (*.npy *.csv);;All files (*)",
        )
        if filename:
            self.load_fneu_file(filename)

    def browse_spks_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p spks file (optional)",
            "",
            "NumPy/CSV files (*.npy *.csv);;All files (*)",
        )
        if filename:
            self.load_spks_file(filename)

    def browse_stat_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Suite2p stat.npy file (optional)",
            "",
            "NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self.load_stat_file(filename)

    def browse_data_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Suite2p folder", ""
        )
        if directory:
            self.load_data_directory(Path(directory))

    def load_data_directory(self, directory: Path) -> list[str]:
        if not isinstance(directory, Path):
            directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"{directory} is not a valid directory")

        self.data_directory_path = directory
        self.data_folder_path_label.setText(str(directory))
        self.data_folder_path_label.setStyleSheet("QLabel { color: black; }")

        detected: list[str] = []
        candidates = {
            "f": ("F.npy", "F.csv"),
            "fneu": ("Fneu.npy", "Fneu.csv"),
            "spks": ("spks.npy", "spks.csv"),
            "stat": ("stat.npy",),
        }

        for key, names in candidates.items():
            found = self._find_candidate_file(directory, names)
            if found is None:
                continue
            try:
                if key == "f":
                    self.load_f_file(str(found))
                elif key == "fneu":
                    self.load_fneu_file(str(found))
                elif key == "spks":
                    self.load_spks_file(str(found))
                elif key == "stat":
                    self.load_stat_file(str(found))
                detected.append(key)
            except Exception:
                # errors are handled inside the loader; continue scanning remaining files
                continue

        self._auto_detected_files = detected
        if not detected:
            QtWidgets.QMessageBox.information(
                self,
                "No matching files",
                "The selected folder did not contain recognizable Suite2p/CSV data files.",
            )
        else:
            pretty = ", ".join(sorted(detected))
            QtWidgets.QMessageBox.information(
                self,
                "Auto-filled data",
                f"Loaded {pretty} from {directory.name}.",
            )
        return detected

    @staticmethod
    def _find_candidate_file(directory: Path, names: tuple[str, ...]) -> Path | None:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
        lowercase_entries = {
            child.name.lower(): child
            for child in directory.iterdir()
            if child.is_file()
        }
        for name in names:
            lowercase_candidate = lowercase_entries.get(name.lower())
            if lowercase_candidate is not None:
                return lowercase_candidate
        return None

    @staticmethod
    def _load_numeric_matrix_from_file(filename: str) -> np.ndarray:
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            data = np.load(filename)
        elif suffix == ".csv":
            data = np.loadtxt(filename, delimiter=",")
        else:
            raise ValueError(
                "Unsupported file type. Please provide a .npy or .csv file."
            )

        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D array")
        return np.asarray(data, dtype=np.float32)

    def load_f_file(self, filename: str) -> None:
        try:
            self.f_file_path = Path(filename)
            self.f_file_path_label.setText(self.f_file_path.name)
            self.f_file_path_label.setStyleSheet("QLabel { color: black; }")
            self.f_data = self._load_numeric_matrix_from_file(filename)
            self.logger.info(
                f"Loaded F.npy ({self.f_data.shape[0]} neurons × {self.f_data.shape[1]} timepoints)"
            )
            # Set max_neurons to the actual number of neurons in the data
            n_neurons = self.f_data.shape[0]
            self.max_neurons_spin.setMaximum(max(10000, n_neurons))
            self.max_neurons_spin.setValue(n_neurons)
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
            self.fneu_data = self._load_numeric_matrix_from_file(filename)
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
            self.spike_data = self._load_numeric_matrix_from_file(filename)
            self.logger.info(
                f"Loaded spks.npy ({self.spike_data.shape[0]} neurons × {self.spike_data.shape[1]} timepoints)"
            )
            if self.spike_data.size > 0:
                self.logger.debug(
                    "spks.npy stats: dtype=%s range=%.4f–%.4f non_zero=%d",
                    self.spike_data.dtype,
                    float(np.min(self.spike_data)),
                    float(np.max(self.spike_data)),
                    int(np.count_nonzero(self.spike_data)),
                )
            else:
                self.logger.debug("spks.npy appears to be empty after load")
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

    def load_stat_file(self, filename: str) -> None:
        try:
            self.stat_file_path = Path(filename)
            self.stat_file_path_label.setText(self.stat_file_path.name)
            self.stat_file_path_label.setStyleSheet("QLabel { color: black; }")
            self.stat_data = np.load(filename, allow_pickle=True)
            self.logger.info(
                f"Loaded stat.npy entry with shape {getattr(self.stat_data, 'shape', 'unknown')}"
            )
        except Exception as exc:
            self.logger.exception("Error loading stat.npy file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading stat.npy", f"Could not load {filename}:\n{exc}"
            )
            self.stat_file_path_label.setText("Optional – no file selected")
            self.stat_file_path_label.setStyleSheet("QLabel { color: gray; }")
            self.stat_data = None

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
                self.logger.debug(
                    "Disabling spike import: spks shape %s does not match F/Fneu shape %s",
                    self.spike_data.shape,
                    self.f_data.shape,
                )
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
                self.logger.debug(
                    "spks.npy shape validated against F/Fneu: %s",
                    self.spike_data.shape,
                )
                self._update_spike_preview()

    def _update_trace_preview(self) -> None:
        if self.f_data is None:
            self.preview_text.clear()
            return

        try:
            normalized_traces, raw_mins, raw_maxs = compute_normalized_suite2p_traces(
                self.f_data, self.fneu_data
            )
        except ValueError as exc:
            self.preview_text.setPlainText(str(exc))
            return

        n_neurons, n_timepoints = self.f_data.shape
        normalized_min = float(np.min(normalized_traces))
        normalized_max = float(np.max(normalized_traces))
        overall_raw_min = float(np.min(raw_mins))
        overall_raw_max = float(np.max(raw_maxs))
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
                f"Pre-normalization range (overall): {overall_raw_min:.3f}–{overall_raw_max:.3f}",
                "Per-neuron normalization applied to preserve individual dynamics",
                f"Normalized range: {normalized_min:.3f}–{normalized_max:.3f}",
                f"Mean ± std (normalized): {np.mean(normalized_traces):.3f} ± {np.std(normalized_traces):.3f}",
            ]
        )
        if self.stat_data is not None:
            stat_len = getattr(self.stat_data, "shape", None)
            preview_lines.append(
                f"stat.npy entries detected (preview only): {stat_len}"
            )
        self.preview_text.setPlainText("\n".join(preview_lines))

    def _update_spike_preview(self) -> None:
        if self.spike_data is None:
            self.spike_preview_text.setPlainText(
                "No spks.npy selected. Spike import is optional."
            )
            return

        threshold = self.spike_threshold_spin.value()
        normalized_spikes, raw_mins, raw_maxs = compute_normalized_suite2p_spikes(
            self.spike_data
        )
        non_zero_count = int(np.sum(self.spike_data > 0))
        above_threshold = int(np.sum(normalized_spikes >= threshold))
        overall_raw_min = float(np.min(raw_mins))
        overall_raw_max = float(np.max(raw_maxs))
        self.logger.debug(
            "Spike preview: threshold=%.3f raw_range=%.4f–%.4f non_zero=%d above_threshold=%d",
            threshold,
            overall_raw_min,
            overall_raw_max,
            non_zero_count,
            above_threshold,
        )
        preview_lines = [
            f"spks.npy shape: {self.spike_data.shape}",
            f"Raw value range (overall): {overall_raw_min:.3f}–{overall_raw_max:.3f}",
            "Per-neuron normalization applied to preserve individual dynamics",
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
            sampling_rate = float(self.sampling_rate_spin.value())
            if sampling_rate <= 0.0:
                raise ValueError("Sample rate must be positive")
            dt = 1.0 / sampling_rate
            max_neurons = self.max_neurons_spin.value()
            n_neurons, n_timepoints = self.f_data.shape
            self.experiment_duration = n_timepoints * dt
            n_neurons_to_import = min(max_neurons, n_neurons)

            self.logger.debug(
                "Importing Suite2p data: sample_rate=%.3fHz dt=%.6f max_neurons=%d f_shape=%s spks_shape=%s",
                sampling_rate,
                dt,
                max_neurons,
                self.f_data.shape,
                getattr(self.spike_data, "shape", None),
            )

            traces = prepare_suite2p_traces(
                self.f_data,
                self.fneu_data,
                max_neurons=max_neurons,
                sampling_rate=sampling_rate,
            )

            self.imported_data = [traces]

            spike_summary = ""
            if self.spike_data is not None:
                threshold = self.spike_threshold_spin.value()
                spike_data = prepare_suite2p_spikes(
                    self.spike_data,
                    dt=dt,
                    threshold=threshold,
                    max_neurons=max_neurons,
                )
                self.imported_data.append(spike_data)
                self.logger.debug(
                    "Spike import ready: %d events at threshold %.3f",
                    len(spike_data.spike_indices),
                    threshold,
                )
                spike_summary = f"\nSpikes: {len(spike_data.spike_indices)} events ≥ {threshold:.3f}"

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
