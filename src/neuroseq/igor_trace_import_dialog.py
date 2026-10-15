from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import brian2.numpy_ as np
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.importer_igor2 import NEUROSEQ_IGOR_TARGET_DT_SECONDS
from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import create_trace_data_from_igor_wave
from neuroseq.importer_igor2 import estimate_resampled_sample_count
from neuroseq.importer_igor2 import import_igor_wave_into_trace_data
from neuroseq.importer_igor2 import is_pxp_file
from neuroseq.importer_igor2 import load_csv_trace_table
from neuroseq.importer_igor2 import load_igor_pxp_file
from neuroseq.importer_igor2 import load_igor_wave_file
from neuroseq.importer_igor2 import prepare_igor_wave_for_neuroseq
from neuroseq.importer_patchclamp import PeakDetectionParams
from neuroseq.spike_detection import detect_spikes_voltage_trace
from neuroseq.stim_timing import format_milliseconds
from neuroseq.stim_timing import stim_dt_matches

if TYPE_CHECKING:
    from neuroseq.importer_igor2 import PreparedIgorWave
    from neuroseq.project import Project

_SPIKE_DETECTION_DEFAULTS = PeakDetectionParams(
    peak_height=3.0,
    peak_prominence=0.5,
    min_interval=0.001,
    peak_distance=10,
    peak_width=1,
)


class IgorTraceImportDialog(QtWidgets.QDialog):
    """Import Igor2/Sutter patch-clamp TRACE waves into project experiments."""

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
        *,
        csv_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.csv_only = csv_only
        self.loaded_wave: IgorWave | None = None
        self.csv_waves: list[IgorWave] = []
        self.file_path: str = ""
        self.selected_experiment_name: str = ""
        self.selected_trace_name: str = ""
        self._auto_trace_name = ""
        self._auto_units = ""
        self._auto_experiment_name = ""
        self.loaded_pxp_waves: dict[str, IgorWave] = {}
        self._imported_pxp_keys: set[str] = set()
        self._had_successful_import: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Igor2 Patch-Clamp Trace")
        if self.csv_only:
            self.setWindowTitle("Import CSV / Text patch clamp output")
        self.setModal(True)
        self.resize(860, 780)

        root_layout = QtWidgets.QVBoxLayout(self)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content_widget)
        self.scroll_area.setWidget(content_widget)
        root_layout.addWidget(self.scroll_area)

        intro = QtWidgets.QLabel(
            "Import Igor2/Sutter patch-clamp trace waves from .ibw, .pxp, or the "
            "simple NeuroSeq-compatible .csv format. Each wave is stored as a "
            "single-neuron TraceData inside an Experiment. Packed experiments can "
            "contain many waves; import as many as you need from the same file, "
            "then click Close when done."
        )
        intro.setWordWrap(True)
        if self.csv_only:
            intro.setText(
                "Import CSV or text output with one trace per column and one sample per row. "
                "An optional first row names the traces. Set the shared source sampling "
                "rate in Hz before importing. Choose the recorded signal units below."
            )
        layout.addWidget(intro)

        self.file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(self.file_group)
        self.file_label = QtWidgets.QLabel("No Igor wave selected")
        self.file_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_label, stretch=1)
        file_layout.addWidget(browse_button)
        layout.addWidget(self.file_group)

        self.wave_picker_group = QtWidgets.QGroupBox("Wave Selection")
        wave_picker_layout = QtWidgets.QVBoxLayout(self.wave_picker_group)
        self.pxp_wave_list = QtWidgets.QListWidget()
        self.pxp_wave_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.pxp_wave_list.currentItemChanged.connect(self._on_pxp_wave_selected)
        wave_picker_layout.addWidget(self.pxp_wave_list)
        self.wave_picker_group.setVisible(False)
        layout.addWidget(self.wave_picker_group)

        preview_group = QtWidgets.QGroupBox("Wave Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.multi_sweep_info = QtWidgets.QLabel("")
        self.multi_sweep_info.setWordWrap(True)
        self.multi_sweep_info.setObjectName("multi_sweep_info")
        self.multi_sweep_info.setStyleSheet(
            "QLabel { color: #555; font-weight: bold; }"
        )
        self.multi_sweep_info.setVisible(False)
        preview_layout.addWidget(self.multi_sweep_info)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        self.preview_text.setPlaceholderText(
            "Preview appears here after selecting an Igor wave."
        )
        preview_layout.addWidget(self.preview_text)
        self.wave_plot = MatplotlibWidget()
        self.wave_plot.setMinimumHeight(200)
        self.wave_plot.setMaximumHeight(260)
        self.wave_plot.setVisible(False)
        original_draw_idle = self.wave_plot.canvas._draw_idle

        def _safe_draw_idle(canvas) -> None:
            try:
                original_draw_idle()
            except RuntimeError as exc:
                if "deleted" not in str(exc).lower():
                    raise

        self.wave_plot.canvas._draw_idle = MethodType(
            _safe_draw_idle,
            self.wave_plot.canvas,
        )
        preview_layout.addWidget(self.wave_plot)
        layout.addWidget(preview_group)

        self.source_group = QtWidgets.QGroupBox("Source Wave Settings")
        source_layout = QtWidgets.QFormLayout(self.source_group)
        self.source_dt_spin = QtWidgets.QDoubleSpinBox()
        self.source_dt_spin.setDecimals(12)
        self.source_dt_spin.setRange(0.000001, 3600.0)
        self.source_dt_spin.setSingleStep(0.0001)
        self.source_dt_spin.setValue(NEUROSEQ_IGOR_TARGET_DT_SECONDS)
        self.source_dt_spin.setSuffix(" s")
        self.source_dt_spin.setReadOnly(True)
        self.source_dt_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.source_dt_spin.valueChanged.connect(self._on_timing_settings_changed)
        source_layout.addRow("Source sample interval:", self.source_dt_spin)
        self.sampling_rate_spin = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spin.setDecimals(6)
        self.sampling_rate_spin.setRange(0.001, 1000000.0)
        self.sampling_rate_spin.setValue(1000.0)
        self.sampling_rate_spin.setSuffix(" Hz")
        self.sampling_rate_spin.valueChanged.connect(
            lambda rate: self.source_dt_spin.setValue(1.0 / rate)
        )
        self.sampling_rate_label = QtWidgets.QLabel("Source sampling rate:")
        source_layout.addRow(self.sampling_rate_label, self.sampling_rate_spin)
        self.sampling_rate_label.setVisible(self.csv_only)
        self.sampling_rate_spin.setVisible(self.csv_only)
        self.match_source_grid_checkbox = QtWidgets.QCheckBox(
            "Adjust NeuroSeq trace grid to source/file grid"
        )
        self.match_source_grid_checkbox.toggled.connect(
            self._on_timing_settings_changed
        )
        source_layout.addRow(self.match_source_grid_checkbox)
        self.target_dt_label = QtWidgets.QLabel(
            f"{NEUROSEQ_IGOR_TARGET_DT_SECONDS:.3f} s (fixed NeuroSeq grid)"
        )
        source_layout.addRow("Imported sample interval:", self.target_dt_label)
        layout.addWidget(self.source_group)

        self.target_group = QtWidgets.QGroupBox("Import Target")
        target_layout = QtWidgets.QVBoxLayout(self.target_group)

        self.create_new_radio = QtWidgets.QRadioButton("Create new trace data")
        self.import_existing_radio = QtWidgets.QRadioButton(
            "Import into existing trace data"
        )
        self.create_new_radio.setChecked(True)
        self.create_new_radio.toggled.connect(self._update_mode_state)
        self.import_existing_radio.toggled.connect(self._update_mode_state)
        target_layout.addWidget(self.create_new_radio)
        target_layout.addWidget(self.import_existing_radio)

        self.new_target_group = QtWidgets.QGroupBox("New Trace Data")
        new_target_layout = QtWidgets.QFormLayout(self.new_target_group)
        self.new_experiment_combo = QtWidgets.QComboBox()
        self.new_experiment_combo.currentTextChanged.connect(
            self._on_new_experiment_target_changed
        )
        new_target_layout.addRow("Experiment:", self.new_experiment_combo)
        self.new_experiment_name_label = QtWidgets.QLabel("New experiment name:")
        self.new_experiment_name_edit = QtWidgets.QLineEdit()
        self.new_experiment_name_edit.textChanged.connect(
            self._update_import_button_state
        )
        new_target_layout.addRow(
            self.new_experiment_name_label, self.new_experiment_name_edit
        )
        self.new_trace_name_edit = QtWidgets.QLineEdit()
        self.new_trace_name_edit.textChanged.connect(self._update_import_button_state)
        new_target_layout.addRow("Trace data name:", self.new_trace_name_edit)
        self.new_neuron_id_spin = QtWidgets.QSpinBox()
        self.new_neuron_id_spin.setRange(0, 10_000)
        new_target_layout.addRow("Neuron ID:", self.new_neuron_id_spin)
        self.new_units_edit = QtWidgets.QLineEdit()
        new_target_layout.addRow("Units:", self.new_units_edit)
        target_layout.addWidget(self.new_target_group)

        self.existing_target_group = QtWidgets.QGroupBox("Existing Trace Data")
        existing_target_layout = QtWidgets.QFormLayout(self.existing_target_group)
        self.existing_experiment_combo = QtWidgets.QComboBox()
        self.existing_experiment_combo.currentTextChanged.connect(
            self._on_existing_experiment_changed
        )
        existing_target_layout.addRow("Experiment:", self.existing_experiment_combo)
        self.existing_trace_combo = QtWidgets.QComboBox()
        self.existing_trace_combo.currentTextChanged.connect(
            self._on_existing_trace_changed
        )
        existing_target_layout.addRow("Trace data:", self.existing_trace_combo)
        self.existing_neuron_id_spin = QtWidgets.QSpinBox()
        self.existing_neuron_id_spin.setRange(0, 10_000)
        existing_target_layout.addRow("Target neuron ID:", self.existing_neuron_id_spin)
        self.existing_extend_checkbox = QtWidgets.QCheckBox(
            "Extend target trace data to fit imported duration"
        )
        self.existing_extend_checkbox.setChecked(True)
        existing_target_layout.addRow(self.existing_extend_checkbox)
        self.existing_replace_checkbox = QtWidgets.QCheckBox(
            "Replace existing row with the same neuron ID"
        )
        self.existing_replace_checkbox.setChecked(True)
        existing_target_layout.addRow(self.existing_replace_checkbox)
        target_layout.addWidget(self.existing_target_group)
        layout.addWidget(self.target_group)

        self.spike_group = QtWidgets.QGroupBox("Spike detection")
        spike_layout = QtWidgets.QVBoxLayout(self.spike_group)
        self.spike_detect_checkbox = QtWidgets.QCheckBox(
            "Detect spikes (CC voltage traces only — leave unchecked for VC current)"
        )
        self.spike_detect_checkbox.setChecked(False)
        self.spike_detect_checkbox.setToolTip(
            "If enabled, detected spikes are stored as SpikeData in the same "
            "experiment using the default peak-detection parameters."
        )
        spike_layout.addWidget(self.spike_detect_checkbox)
        layout.addWidget(self.spike_group)

        self.last_import_label = QtWidgets.QLabel("")
        self.last_import_label.setWordWrap(True)
        self.last_import_label.setStyleSheet("QLabel { color: #2a7a2a; }")
        layout.addWidget(self.last_import_label)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self.info_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_file)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        root_layout.addLayout(button_layout)

        self._populate_target_experiments()
        self._update_mode_state()
        self._update_preview()

    def _target_dt_seconds(self) -> float:
        if self.match_source_grid_checkbox.isChecked():
            return float(self.source_dt_spin.value())
        return NEUROSEQ_IGOR_TARGET_DT_SECONDS

    def _update_target_dt_label(self) -> None:
        target_dt = self._target_dt_seconds()
        if self.match_source_grid_checkbox.isChecked():
            self.target_dt_label.setText(f"{target_dt:.9f} s (match source/file grid)")
        else:
            self.target_dt_label.setText(f"{target_dt:.3f} s (fixed NeuroSeq grid)")

    def _on_timing_settings_changed(self) -> None:
        with QtCore.QSignalBlocker(self.sampling_rate_spin):
            self.sampling_rate_spin.setValue(1.0 / self.source_dt_spin.value())
        self._update_target_dt_label()
        self._update_preview()
        self._update_import_button_state()

    def _experiments_with_trace_data(self) -> list[str]:
        return [
            experiment_name
            for experiment_name, experiment in self.project.experiments.items()
            if any(isinstance(data, TraceData) for data in experiment.data)
        ]

    def _trace_data_names(self, experiment_name: str) -> list[str]:
        experiment = self.project.experiments.get(experiment_name)
        if experiment is None:
            return []
        return [data.name for data in experiment.data if isinstance(data, TraceData)]

    def _populate_target_experiments(self) -> None:
        current_new_target = self.new_experiment_combo.currentText()
        current_existing_experiment = self.existing_experiment_combo.currentText()
        current_existing_trace = self.existing_trace_combo.currentText()

        all_experiment_names = list(self.project.experiments.keys())
        trace_experiment_names = self._experiments_with_trace_data()

        with QtCore.QSignalBlocker(self.new_experiment_combo):
            self.new_experiment_combo.clear()
            self.new_experiment_combo.addItem("(New experiment)")
            self.new_experiment_combo.addItems(all_experiment_names)
            if (
                current_new_target
                and self.new_experiment_combo.findText(current_new_target) >= 0
            ):
                self.new_experiment_combo.setCurrentText(current_new_target)
            else:
                self.new_experiment_combo.setCurrentIndex(0)

        with QtCore.QSignalBlocker(self.existing_experiment_combo):
            self.existing_experiment_combo.clear()
            if trace_experiment_names:
                self.existing_experiment_combo.addItems(trace_experiment_names)
                if current_existing_experiment in trace_experiment_names:
                    self.existing_experiment_combo.setCurrentText(
                        current_existing_experiment
                    )
                else:
                    self.existing_experiment_combo.setCurrentIndex(0)
                self.existing_experiment_combo.setEnabled(True)
                self.import_existing_radio.setEnabled(True)
            else:
                self.existing_experiment_combo.addItem("(No trace data available)")
                self.existing_experiment_combo.setEnabled(False)
                self.import_existing_radio.setEnabled(False)
                self.create_new_radio.setChecked(True)

        with QtCore.QSignalBlocker(self.existing_trace_combo):
            self.existing_trace_combo.clear()
            self.existing_trace_combo.setEnabled(bool(trace_experiment_names))

        self._on_new_experiment_target_changed(self.new_experiment_combo.currentText())
        if trace_experiment_names:
            self._on_existing_experiment_changed(
                self.existing_experiment_combo.currentText(),
                preferred_trace_name=current_existing_trace,
            )
        else:
            self.existing_trace_combo.clear()
            self.existing_trace_combo.addItem("(No trace data available)")
            self.existing_trace_combo.setEnabled(False)
            self._update_preview()
            self._update_import_button_state()

    def _on_new_experiment_target_changed(self, current: str) -> None:
        is_new_experiment = current == "(New experiment)"
        self.new_experiment_name_label.setVisible(is_new_experiment)
        self.new_experiment_name_edit.setVisible(is_new_experiment)
        self._update_import_button_state()

    def _on_existing_experiment_changed(
        self,
        experiment_name: str,
        *,
        preferred_trace_name: str | None = None,
    ) -> None:
        trace_names = self._trace_data_names(experiment_name)
        with QtCore.QSignalBlocker(self.existing_trace_combo):
            self.existing_trace_combo.clear()
            if trace_names:
                self.existing_trace_combo.addItems(trace_names)
                if preferred_trace_name in trace_names:
                    self.existing_trace_combo.setCurrentText(preferred_trace_name)
                else:
                    self.existing_trace_combo.setCurrentIndex(0)
                self.existing_trace_combo.setEnabled(True)
            else:
                self.existing_trace_combo.addItem("(No trace data available)")
                self.existing_trace_combo.setEnabled(False)
        self._on_existing_trace_changed(self.existing_trace_combo.currentText())

    def _on_existing_trace_changed(self, _trace_name: str) -> None:
        self._update_preview()
        self._update_import_button_state()

    def _on_pxp_wave_selected(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        if current is None:
            self.loaded_wave = None
        else:
            path_key = current.text().removeprefix("✓ ")
            self.loaded_wave = self.loaded_pxp_waves.get(path_key)
            if self.loaded_wave is not None and self.loaded_wave.source_dt is not None:
                self.source_dt_spin.setValue(float(self.loaded_wave.source_dt))
            else:
                self.source_dt_spin.setValue(NEUROSEQ_IGOR_TARGET_DT_SECONDS)
            previous_auto_name = self._auto_trace_name
            self._auto_trace_name = self.loaded_wave.name if self.loaded_wave else ""
            current_name = self.new_trace_name_edit.text().strip()
            if not current_name or current_name == previous_auto_name:
                self.new_trace_name_edit.setText(self._auto_trace_name)
            previous_auto_units = self._auto_units
            self._auto_units = (
                (self.loaded_wave.units or "") if self.loaded_wave else ""
            )
            current_units = self.new_units_edit.text().strip()
            if not current_units or current_units == previous_auto_units:
                self.new_units_edit.setText(self._auto_units)
            self._set_auto_experiment_name(
                self.loaded_wave.name if self.loaded_wave else ""
            )
        self._update_preview()
        self._update_import_button_state()

    def _set_auto_experiment_name(self, base: str) -> None:
        """Suggest a new experiment named after the wave (issue #619), unless
        the user typed a name of their own."""
        name = base
        number = 2
        while name and name in self.project.experiments:
            name = f"{base} {number}"
            number += 1
        previous_auto_name = self._auto_experiment_name
        self._auto_experiment_name = name
        current_name = self.new_experiment_name_edit.text().strip()
        if not current_name or current_name == previous_auto_name:
            self.new_experiment_name_edit.setText(name)

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select CSV or text output" if self.csv_only else "Select Igor wave",
            "",
            "CSV / Text files (*.csv *.txt)"
            if self.csv_only
            else "Wave files (*.ibw *.pxp *.csv *.txt);;Igor Binary Wave (*.ibw);;Igor Packed Experiment (*.pxp);;CSV / Text files (*.csv *.txt);;All files (*)",
        )
        if filename:
            self._load_file(filename)

    def load_file(self, filename: str) -> None:
        """Load ``filename`` as if it had been browsed for, e.g. a file
        dropped onto the main window (issue #597)."""
        self._load_file(filename)

    def _load_file(self, filename: str) -> None:
        self.csv_waves = []
        self.new_trace_name_edit.setReadOnly(False)
        self.import_existing_radio.setEnabled(bool(self._experiments_with_trace_data()))
        self.loaded_pxp_waves = {}
        self.pxp_wave_list.clear()
        self.wave_picker_group.setVisible(False)
        if is_pxp_file(filename):
            self._load_pxp_file(filename)
            return
        try:
            if Path(filename).suffix.lower() in {".csv", ".txt"}:
                self.csv_waves = load_csv_trace_table(filename)
                wave = self.csv_waves[0]
            else:
                wave = load_igor_wave_file(filename)
        except Exception as exc:
            self.logger.exception("Failed to load Igor wave %s", filename)
            self.loaded_wave = None
            self.file_path = ""
            self.file_label.setText("No Igor wave selected")
            self.file_label.setStyleSheet("QLabel { color: gray; }")
            self._update_preview()
            self._update_import_button_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Could not load Igor wave:\n{exc}",
            )
            return

        previous_auto_name = self._auto_trace_name
        self._auto_trace_name = wave.name
        current_name = self.new_trace_name_edit.text().strip()
        if not current_name or current_name == previous_auto_name:
            self.new_trace_name_edit.setText(self._auto_trace_name)

        previous_auto_units = self._auto_units
        self._auto_units = wave.units or ""
        current_units = self.new_units_edit.text().strip()
        if not current_units or current_units == previous_auto_units:
            self.new_units_edit.setText(self._auto_units)

        self.loaded_wave = wave
        self._set_auto_experiment_name(
            Path(filename).stem if len(self.csv_waves) > 1 else wave.name
        )
        if len(self.csv_waves) > 1:
            self.create_new_radio.setChecked(True)
            self.import_existing_radio.setEnabled(False)
            self.new_trace_name_edit.setText(", ".join(w.name for w in self.csv_waves))
            self.new_trace_name_edit.setReadOnly(True)
        is_csv = wave.source_format == "csv"
        self.sampling_rate_label.setVisible(is_csv)
        self.sampling_rate_spin.setVisible(is_csv)
        self.file_path = filename
        self.file_label.setText(filename)
        self.file_label.setStyleSheet("QLabel { color: black; }")
        if wave.source_dt is not None:
            self.source_dt_spin.setValue(float(wave.source_dt))
        else:
            self.source_dt_spin.setValue(NEUROSEQ_IGOR_TARGET_DT_SECONDS)
        self._update_preview()
        self._update_import_button_state()

    def _load_pxp_file(self, filename: str) -> None:
        try:
            report = load_igor_pxp_file(filename)
        except Exception as exc:
            self.logger.exception("Failed to load Igor packed experiment %s", filename)
            self.loaded_wave = None
            self.loaded_pxp_waves = {}
            self.file_path = ""
            self.file_label.setText("No Igor wave selected")
            self.file_label.setStyleSheet("QLabel { color: gray; }")
            self.pxp_wave_list.clear()
            self.wave_picker_group.setVisible(False)
            self._update_preview()
            self._update_import_button_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Could not load Igor packed experiment:\n{exc}",
            )
            return

        waves = report.waves
        self.loaded_pxp_waves = waves
        self._imported_pxp_keys = set()
        self.last_import_label.setText("")
        self.file_path = filename
        self.file_label.setText(filename)
        self.file_label.setStyleSheet("QLabel { color: black; }")

        self.pxp_wave_list.clear()
        for path_key in sorted(waves.keys()):
            self.pxp_wave_list.addItem(path_key)
        self.wave_picker_group.setVisible(True)
        if report.skipped:
            summary = (
                f"{len(report.skipped)} record(s) in '{Path(filename).name}' could not "
                "be parsed and were skipped. The other waves loaded normally. "
                "Open the application log for details."
            )
            QtWidgets.QMessageBox.warning(self, "Some waves were skipped", summary)
            for skipped in report.skipped:
                self.logger.warning(
                    "Skipped record #%d (%s): %s",
                    skipped.record_index,
                    skipped.record_type,
                    skipped.reason,
                )

        if len(waves) == 1:
            self.pxp_wave_list.setCurrentRow(0)
        else:
            self.loaded_wave = None
            self._update_preview()
            self._update_import_button_state()

    def _build_preview_text(self, wave: IgorWave) -> str:
        source_dt = float(self.source_dt_spin.value())
        target_dt = self._target_dt_seconds()
        imported_samples = estimate_resampled_sample_count(
            int(wave.values.size),
            source_dt,
            target_dt=target_dt,
        )
        lines = [
            f"Wave name={wave.name}",
            f"Source format={wave.source_format}",
            f"Source samples={int(wave.values.size)}",
            (
                f"Detected source dt={wave.source_dt:.9f} s"
                if wave.source_dt is not None
                else "Detected source dt=(not stored in file)"
            ),
            f"Import source dt={source_dt:.9f} s",
            f"Imported sample interval={target_dt:.9f} s",
            f"Imported samples on target grid={imported_samples}",
            f"Wave units={wave.units or '(none)'}",
            f"Axis units={wave.axis_units or '(none)'}",
            f"Value range={float(np.min(wave.values)):.6g} .. {float(np.max(wave.values)):.6g}",
        ]
        if wave.note:
            lines.extend(["", f"Wave note={wave.note}"])
        return "\n".join(lines)

    def _update_multi_sweep_info(self) -> None:
        wave = self.loaded_wave
        if wave is None or wave.n_sweeps <= 1 or wave.samples_per_sweep is None:
            self.multi_sweep_info.setVisible(False)
            self.multi_sweep_info.setText("")
            return
        source_dt = float(self.source_dt_spin.value())
        target_dt = self._target_dt_seconds()
        target_samples = estimate_resampled_sample_count(
            int(wave.values.size), source_dt, target_dt=target_dt
        )
        total_duration_s = float(wave.values.size) * source_dt
        self.multi_sweep_info.setText(
            f"Multi-sweep IBW: {wave.n_sweeps} sweeps × "
            f"{wave.samples_per_sweep} source samples concatenated "
            f"({wave.values.size} total source samples → "
            f"{target_samples} samples at the {target_dt * 1000.0:g} ms "
            f"import grid = {total_duration_s:g} s). "
            "Inter-sweep recording gaps are not preserved."
        )
        self.multi_sweep_info.setVisible(True)

    def _update_mode_state(self) -> None:
        create_new = self.create_new_radio.isChecked()
        self.new_target_group.setEnabled(create_new)
        self.existing_target_group.setEnabled(not create_new)
        self._update_preview()
        self._update_import_button_state()

    def _update_wave_plot(self, wave: IgorWave | None) -> None:
        fig = self.wave_plot.getFigure()
        fig.clf()
        if wave is None:
            self.wave_plot.setVisible(False)
            self.wave_plot.draw()
            return

        source_dt = float(self.source_dt_spin.value())
        values = np.asarray(wave.values, dtype=float).ravel()
        max_display_pts = 4000
        step = max(1, len(values) // max_display_pts)
        t = np.arange(0, len(values), step) * source_dt
        y = values[::step]

        ax = fig.add_subplot(111)
        ax.plot(t, y, linewidth=0.8, color="#1f77b4")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(wave.units or "a.u.", fontsize=8)
        ax.set_title(wave.name, fontsize=9)
        ax.tick_params(labelsize=7)
        fig.tight_layout(pad=0.5)
        self.wave_plot.setVisible(True)
        self.wave_plot.draw()

    def _cleanup_wave_plot(self) -> None:
        canvas = self.wave_plot.canvas
        canvas._draw_pending = False
        canvas._is_drawing = False

    def _update_preview(self) -> None:
        self._update_target_dt_label()
        self._update_wave_plot(self.loaded_wave)
        self._update_multi_sweep_info()
        if self.loaded_wave is None:
            self.preview_text.setPlainText("")
            if not self._experiments_with_trace_data():
                self.info_label.setText(
                    "Select an Igor wave to preview it. Only the new-trace import "
                    "mode is currently available because the project has no trace "
                    "data yet."
                )
            else:
                self.info_label.setText("Select an Igor wave to preview it.")
            return

        self.preview_text.setPlainText(self._build_preview_text(self.loaded_wave))
        if len(self.csv_waves) > 1:
            self.preview_text.appendPlainText(
                f"CSV table: {len(self.csv_waves)} traces; columns: "
                + ", ".join(w.name for w in self.csv_waves)
                + ". All columns will be imported with consecutive neuron IDs "
                "starting at the selected neuron ID. Plot shows the first column."
            )
        source_dt = float(self.source_dt_spin.value())
        target_dt = self._target_dt_seconds()
        info_lines: list[str] = []
        if abs(source_dt - target_dt) > 1e-9:
            info_lines.append(
                "This wave will be resampled to the selected NeuroSeq trace grid "
                "using zero-order hold. Fast sub-ms changes may be lost."
            )
        elif self.match_source_grid_checkbox.isChecked():
            info_lines.append(
                "This import will preserve the source/file timestep by adjusting the "
                "NeuroSeq trace grid."
            )
        else:
            info_lines.append("This wave already matches NeuroSeq's 1 ms timeline.")
        if self.loaded_wave.source_dt is None:
            info_lines.append(
                "The source file does not store the sample interval. Review the "
                "source sample interval before importing."
            )
        selected_trace = self._resolve_existing_trace_data(
            self.existing_experiment_combo.currentText(),
            self.existing_trace_combo.currentText(),
        )
        if (
            self.import_existing_radio.isChecked()
            and selected_trace is not None
            and not stim_dt_matches(float(selected_trace.dt), target_dt)
        ):
            info_lines.append(
                "The selected target trace data will be remapped from "
                f"{format_milliseconds(float(selected_trace.dt) * 1000.0)} ms to "
                f"{format_milliseconds(target_dt * 1000.0)} ms before import."
            )
        self.info_label.setText("\n".join(info_lines))

    def _update_import_button_state(self) -> None:
        enabled = self.loaded_wave is not None
        if enabled and self.loaded_pxp_waves:
            current_item = self.pxp_wave_list.currentItem()
            if current_item is not None:
                raw_key = current_item.text().removeprefix("✓ ")
                if raw_key in self._imported_pxp_keys:
                    enabled = False
        if enabled and self.create_new_radio.isChecked():
            enabled = self.new_trace_name_edit.text().strip() != ""
            if (
                enabled
                and self.new_experiment_combo.currentText() == "(New experiment)"
            ):
                enabled = self.new_experiment_name_edit.text().strip() != ""
        if enabled and self.import_existing_radio.isChecked():
            enabled = (
                self._resolve_existing_trace_data(
                    self.existing_experiment_combo.currentText(),
                    self.existing_trace_combo.currentText(),
                )
                is not None
            )
        self.import_button.setEnabled(enabled)

    def _resolve_new_target_experiment_name(self) -> str:
        current = self.new_experiment_combo.currentText()
        if current == "(New experiment)":
            name = self.new_experiment_name_edit.text().strip()
            if not name:
                raise ValueError("Experiment name cannot be empty")
            if name in self.project.experiments:
                raise ValueError(f"Experiment '{name}' already exists")
            return name
        return current

    def _commit_new_trace_data(
        self,
        experiment_name: str,
        trace_data: TraceData,
    ) -> None:
        if experiment_name in self.project.experiments:
            if any(
                isinstance(data, TraceData) and data.name == trace_data.name
                for data in self.project.experiments[experiment_name].data
            ):
                raise ValueError(
                    f"Trace data '{trace_data.name}' already exists "
                    f"in experiment '{experiment_name}'"
                )
            self.project.add_data_to_experiment(experiment_name, trace_data)
            return

        experiment = Experiment(
            name=experiment_name,
            duration=float(trace_data.traces.shape[1]) * float(trace_data.dt),
            timepoints=[],
            data=[trace_data],
        )
        self.project.add_experiment(experiment)

    def _resolve_existing_trace_data(
        self,
        experiment_name: str,
        trace_name: str,
    ) -> TraceData | None:
        experiment = self.project.experiments.get(experiment_name)
        if experiment is None:
            return None
        for data in experiment.data:
            if isinstance(data, TraceData) and data.name == trace_name:
                return data
        return None

    def _refresh_experiment_timing(
        self,
        experiment_name: str,
        trace_data: TraceData,
    ) -> None:
        experiment = self.project.experiments.get(experiment_name)
        if experiment is None:
            return
        new_duration = float(trace_data.traces.shape[1]) * float(trace_data.dt)
        if new_duration > experiment.duration:
            experiment.duration = new_duration
            if new_duration not in experiment.timepoints:
                experiment.timepoints.append(new_duration)
            experiment.timepoints.sort()
        self.project.update_limits()

    def _confirm_and_remap_trace_dt(
        self,
        target_trace: TraceData,
        experiment_name: str,
        new_dt: float,
    ) -> bool:
        msg = (
            "Changing the trace data timestep will remap trace data "
            f"'{target_trace.name}' from "
            f"{format_milliseconds(float(target_trace.dt) * 1000.0)} ms "
            f"to {format_milliseconds(new_dt * 1000.0)} ms. "
            "Trace timing may shift by up to one sample. Continue?"
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            "Change trace data timestep",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False

        from neuroseq.importer_igor2 import resample_wave_zero_order_hold

        if target_trace.traces.shape[0] == 0:
            new_sample_count = estimate_resampled_sample_count(
                int(target_trace.traces.shape[1]),
                float(target_trace.dt),
                target_dt=float(new_dt),
            )
            target_trace.traces = np.zeros((0, new_sample_count), dtype=np.float32)
        else:
            new_rows = [
                resample_wave_zero_order_hold(
                    target_trace.traces[i],
                    source_dt=float(target_trace.dt),
                    target_dt=float(new_dt),
                )
                for i in range(target_trace.traces.shape[0])
            ]
            target_trace.traces = np.stack(new_rows, axis=0).astype(
                np.float32, copy=False
            )
        target_trace.dt = float(new_dt)
        self._refresh_experiment_timing(experiment_name, target_trace)
        return True

    def _run_spike_detection(
        self,
        prepared_wave: PreparedIgorWave,
        *,
        experiment_name: str,
        neuron_id: int,
    ) -> int:
        spike_data = detect_spikes_voltage_trace(
            prepared_wave.values,
            neuron_id=neuron_id,
            dt=float(prepared_wave.target_dt),
            detection_method="peak",
            peak_height_threshold=_SPIKE_DETECTION_DEFAULTS.peak_height,
            peak_prominence=_SPIKE_DETECTION_DEFAULTS.peak_prominence,
            min_spike_interval=_SPIKE_DETECTION_DEFAULTS.min_interval,
            peak_distance=_SPIKE_DETECTION_DEFAULTS.peak_distance,
            peak_width=_SPIKE_DETECTION_DEFAULTS.peak_width,
        )
        spike_data.name = "Igor Patch Clamp Spikes"
        self.project.add_data_to_experiment(experiment_name, spike_data)
        return int(len(spike_data.spike_times))

    def _promote_cancel_to_close(self) -> None:
        """Promote Cancel to Close once the dialog has modified the project."""
        if self.cancel_button.text() == "Close":
            return
        self.cancel_button.setText("Close")
        with suppress(TypeError, RuntimeError):
            self.cancel_button.clicked.disconnect(self.reject)
        self.cancel_button.clicked.connect(self.accept)

    def _import_file(self) -> None:
        if self.loaded_wave is None:
            return

        try:
            if len(self.csv_waves) > 1:
                self._import_csv_table()
                return
            target_dt = self._target_dt_seconds()
            prepared_wave = prepare_igor_wave_for_neuroseq(
                self.loaded_wave,
                source_dt_override=float(self.source_dt_spin.value()),
                target_dt=target_dt,
            )

            created_new_experiment = False
            if self.create_new_radio.isChecked():
                experiment_name = self._resolve_new_target_experiment_name()
                created_new_experiment = experiment_name not in self.project.experiments
                trace_name = self.new_trace_name_edit.text().strip()
                if not trace_name:
                    raise ValueError("Trace data name cannot be empty")
                neuron_id = int(self.new_neuron_id_spin.value())
                units = self.new_units_edit.text().strip()

                trace_data = create_trace_data_from_igor_wave(
                    prepared_wave,
                    trace_name=trace_name,
                    neuron_id=neuron_id,
                    units=units,
                )
                self._commit_new_trace_data(experiment_name, trace_data)
                self.selected_experiment_name = experiment_name
                self.selected_trace_name = trace_name
                report_headline = (
                    f"Imported '{prepared_wave.name}' into new trace data "
                    f"'{trace_name}' in experiment '{experiment_name}'."
                )
                if (
                    prepared_wave.n_sweeps > 1
                    and prepared_wave.samples_per_sweep is not None
                ):
                    report_headline += (
                        f" Source wave contained {prepared_wave.n_sweeps} sweeps × "
                        f"{prepared_wave.samples_per_sweep} samples, concatenated on import."
                    )
                spike_neuron_id = neuron_id
            else:
                experiment_name = self.existing_experiment_combo.currentText()
                trace_name = self.existing_trace_combo.currentText()
                target_trace = self._resolve_existing_trace_data(
                    experiment_name,
                    trace_name,
                )
                if target_trace is None:
                    raise ValueError("No target trace data selected")

                if not stim_dt_matches(float(target_trace.dt), prepared_wave.target_dt):
                    if not self._confirm_and_remap_trace_dt(
                        target_trace,
                        experiment_name,
                        prepared_wave.target_dt,
                    ):
                        return
                    target_trace = self._resolve_existing_trace_data(
                        experiment_name,
                        trace_name,
                    )
                    if target_trace is None:
                        raise ValueError("No target trace data selected")

                report = import_igor_wave_into_trace_data(
                    target_trace,
                    prepared_wave,
                    neuron_id=int(self.existing_neuron_id_spin.value()),
                    replace_existing=self.existing_replace_checkbox.isChecked(),
                    extend_to_fit=self.existing_extend_checkbox.isChecked(),
                )
                report.target_experiment_name = experiment_name
                self._refresh_experiment_timing(experiment_name, target_trace)
                self.selected_experiment_name = experiment_name
                self.selected_trace_name = trace_name
                report_headline = (
                    f"Imported '{prepared_wave.name}' into '{trace_name}' in "
                    f"experiment '{experiment_name}' at neuron "
                    f"{report.target_neuron_id}."
                )
                if report.n_sweeps > 1 and report.samples_per_sweep is not None:
                    report_headline += (
                        f" Source wave contained {report.n_sweeps} sweeps × "
                        f"{report.samples_per_sweep} samples, concatenated on import."
                    )
                spike_neuron_id = int(self.existing_neuron_id_spin.value())

            spikes_detected: int | None = None
            if self.spike_detect_checkbox.isChecked():
                spikes_detected = self._run_spike_detection(
                    prepared_wave,
                    experiment_name=self.selected_experiment_name,
                    neuron_id=spike_neuron_id,
                )

            is_pxp_session = bool(self.loaded_pxp_waves)
            status_text = report_headline
            if spikes_detected is not None:
                status_text += f" Detected {spikes_detected} spike(s)."

            if is_pxp_session:
                current_item = self.pxp_wave_list.currentItem()
                if current_item is not None:
                    path_key = current_item.text().removeprefix("✓ ")
                    self._imported_pxp_keys.add(path_key)
                    current_item.setText(f"✓ {path_key}")
                    current_item.setFlags(
                        current_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                self.last_import_label.setText(status_text)
                self.pxp_wave_list.clearSelection()
                self.pxp_wave_list.setCurrentItem(None)
                self.loaded_wave = None
                self.new_trace_name_edit.clear()
                self.new_units_edit.clear()
                self._auto_trace_name = ""
                self._auto_units = ""
                self._populate_target_experiments()
                if created_new_experiment:
                    self.new_experiment_combo.setCurrentText(
                        self.selected_experiment_name
                    )
                self._had_successful_import = True
                self._promote_cancel_to_close()
                self._update_preview()
                self._update_import_button_state()
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Complete",
                    status_text,
                )
                self.accept()
        except Exception as exc:
            self.logger.exception("Failed to import Igor trace wave")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import Igor trace wave:\n{exc}",
            )

    def _import_csv_table(self) -> None:
        experiment_name = self._resolve_new_target_experiment_name()
        existing = self.project.experiments.get(experiment_name)
        existing_names = (
            {data.name for data in existing.data if isinstance(data, TraceData)}
            if existing is not None
            else set()
        )
        if existing_names.intersection(wave.name for wave in self.csv_waves):
            raise ValueError("A CSV trace name already exists in the target experiment")
        target_dt = self._target_dt_seconds()
        prepared_waves = [
            prepare_igor_wave_for_neuroseq(
                wave,
                source_dt_override=float(self.source_dt_spin.value()),
                target_dt=target_dt,
            )
            for wave in self.csv_waves
        ]
        first_id = int(self.new_neuron_id_spin.value())
        traces = [
            create_trace_data_from_igor_wave(
                wave,
                trace_name=wave.name,
                neuron_id=first_id + index,
                units=self.new_units_edit.text().strip(),
            )
            for index, wave in enumerate(prepared_waves)
        ]
        for index, trace in enumerate(traces):
            self._commit_new_trace_data(experiment_name, trace)
            if self.spike_detect_checkbox.isChecked():
                self._run_spike_detection(
                    prepared_waves[index],
                    experiment_name=experiment_name,
                    neuron_id=first_id + index,
                )
        self.selected_experiment_name = experiment_name
        self.selected_trace_name = traces[0].name
        QtWidgets.QMessageBox.information(
            self,
            "Import Complete",
            f"Imported {len(traces)} CSV column traces into '{experiment_name}'.",
        )
        self.accept()

    def done(self, result: int) -> None:
        self._cleanup_wave_plot()
        super().done(result)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._cleanup_wave_plot()
        super().closeEvent(event)
