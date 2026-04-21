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

from neuroseq.importer_igor2 import NEUROSEQ_IGOR_TARGET_DT_SECONDS
from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import create_stim_array_from_igor_wave
from neuroseq.importer_igor2 import estimate_resampled_sample_count
from neuroseq.importer_igor2 import import_igor_wave_into_stim_array
from neuroseq.importer_igor2 import is_pxp_file
from neuroseq.importer_igor2 import load_igor_pxp_file
from neuroseq.importer_igor2 import load_igor_wave_file
from neuroseq.importer_igor2 import prepare_igor_wave_for_neuroseq
from neuroseq.stim import StimType
from neuroseq.stim_timing import format_milliseconds
from neuroseq.stim_timing import stim_dt_matches

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray


class IgorImportDialog(QtWidgets.QDialog):
    """Dialog for importing single-wave Igor stim traces into NeuroSeq."""

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project = project
        self.loaded_wave: IgorWave | None = None
        self.file_path: str = ""
        self.selected_stim_name: str = ""
        self._auto_stim_name = ""
        self._auto_units = ""
        self.loaded_pxp_waves: dict[str, IgorWave] = {}
        self._imported_pxp_keys: set[str] = set()
        self._had_successful_import: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Igor2 Wave")
        self.setModal(True)
        self.resize(860, 700)

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
            "Import Igor2/Sutter stimulation waves from .ibw, .pxp, or the simple "
            "NeuroSeq-compatible .csv format. Packed experiments can contain many "
            "waves; import as many as you need from the same file, then click Close "
            "when done. Imported waves are placed on NeuroSeq's 1 ms timeline using "
            "zero-order-hold resampling when needed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        self.file_label = QtWidgets.QLabel("No Igor wave selected")
        self.file_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_label, stretch=1)
        file_layout.addWidget(browse_button)
        layout.addWidget(file_group)

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

        source_group = QtWidgets.QGroupBox("Source Wave Settings")
        source_layout = QtWidgets.QFormLayout(source_group)
        self.source_dt_spin = QtWidgets.QDoubleSpinBox()
        self.source_dt_spin.setDecimals(9)
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
        self.match_source_grid_checkbox = QtWidgets.QCheckBox(
            "Adjust NeuroSeq stim grid to source/file grid"
        )
        self.match_source_grid_checkbox.toggled.connect(
            self._on_timing_settings_changed
        )
        source_layout.addRow(self.match_source_grid_checkbox)
        self.target_dt_label = QtWidgets.QLabel(
            f"{NEUROSEQ_IGOR_TARGET_DT_SECONDS:.3f} s (fixed NeuroSeq grid)"
        )
        source_layout.addRow("Imported sample interval:", self.target_dt_label)
        layout.addWidget(source_group)

        target_group = QtWidgets.QGroupBox("Import Target")
        target_layout = QtWidgets.QVBoxLayout(target_group)

        self.create_new_radio = QtWidgets.QRadioButton("Create new stim array")
        self.import_existing_radio = QtWidgets.QRadioButton(
            "Import into existing stim array"
        )
        self.create_new_radio.setChecked(True)
        self.create_new_radio.toggled.connect(self._update_mode_state)
        self.import_existing_radio.toggled.connect(self._update_mode_state)
        target_layout.addWidget(self.create_new_radio)
        target_layout.addWidget(self.import_existing_radio)

        self.new_target_group = QtWidgets.QGroupBox("New Stim Array")
        new_target_layout = QtWidgets.QFormLayout(self.new_target_group)
        self.new_name_edit = QtWidgets.QLineEdit()
        self.new_name_edit.textChanged.connect(self._update_import_button_state)
        new_target_layout.addRow("Stim name:", self.new_name_edit)
        self.new_type_combo = QtWidgets.QComboBox()
        self.new_type_combo.addItem("Excitatory", StimType.Excitation)
        self.new_type_combo.addItem("Inhibitory", StimType.Inhibition)
        self.new_type_combo.addItem("External", StimType.External)
        self.new_type_combo.setCurrentIndex(2)
        new_target_layout.addRow("Stim type:", self.new_type_combo)
        self.new_units_edit = QtWidgets.QLineEdit()
        new_target_layout.addRow("Units:", self.new_units_edit)
        target_layout.addWidget(self.new_target_group)

        self.existing_target_group = QtWidgets.QGroupBox("Existing Stim Array")
        existing_target_layout = QtWidgets.QFormLayout(self.existing_target_group)
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        existing_target_layout.addRow("Target stim array:", self.target_combo)
        self.neuron_index_spin = QtWidgets.QSpinBox()
        self.neuron_index_spin.setRange(0, 0)
        existing_target_layout.addRow("Target neuron index:", self.neuron_index_spin)
        self.extend_checkbox = QtWidgets.QCheckBox(
            "Extend target stim array to fit imported duration"
        )
        self.extend_checkbox.setChecked(True)
        existing_target_layout.addRow(self.extend_checkbox)
        self.replace_checkbox = QtWidgets.QCheckBox(
            "Replace target neuron values instead of adding to them"
        )
        self.replace_checkbox.setChecked(True)
        existing_target_layout.addRow(self.replace_checkbox)
        target_layout.addWidget(self.existing_target_group)
        layout.addWidget(target_group)

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

        self._populate_target_stims()
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
        self._update_target_dt_label()
        self._update_preview()
        self._update_import_button_state()

    def _populate_target_stims(self) -> None:
        self.target_combo.clear()
        stim_names = list(self.project.stim_arrays.keys())
        if not stim_names:
            self.target_combo.addItem("(No stim arrays available)")
            self.target_combo.setEnabled(False)
            self.import_existing_radio.setEnabled(False)
            self.create_new_radio.setChecked(True)
            self.neuron_index_spin.setRange(0, 0)
            self.neuron_index_spin.setValue(0)
            return

        self.target_combo.addItems(stim_names)
        self.target_combo.setEnabled(True)
        self.import_existing_radio.setEnabled(True)
        self._on_target_changed(self.target_combo.currentText())

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
            previous_auto_name = self._auto_stim_name
            self._auto_stim_name = self.loaded_wave.name if self.loaded_wave else ""
            current_name = self.new_name_edit.text().strip()
            if not current_name or current_name == previous_auto_name:
                self.new_name_edit.setText(self._auto_stim_name)
            previous_auto_units = self._auto_units
            self._auto_units = (
                (self.loaded_wave.units or "") if self.loaded_wave else ""
            )
            current_units = self.new_units_edit.text().strip()
            if not current_units or current_units == previous_auto_units:
                self.new_units_edit.setText(self._auto_units)
        self._update_preview()
        self._update_import_button_state()

    def _selected_stim_array(self) -> StimArray | None:
        target_name = self.target_combo.currentText()
        if not target_name or target_name.startswith("("):
            return None
        return self.project.stim_arrays.get(target_name)

    def _on_target_changed(self, target_name: str) -> None:
        stim_array = self.project.stim_arrays.get(target_name)
        if stim_array is None:
            self.neuron_index_spin.setRange(0, 0)
            self.neuron_index_spin.setValue(0)
            self._update_import_button_state()
            return
        max_neuron_index = max(0, int(stim_array.values.shape[1]) - 1)
        self.neuron_index_spin.setRange(0, max_neuron_index)
        if self.neuron_index_spin.value() > max_neuron_index:
            self.neuron_index_spin.setValue(max_neuron_index)
        self._update_preview()
        self._update_import_button_state()

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Igor wave",
            "",
            "Igor files (*.ibw *.pxp *.csv);;Igor Binary Wave (*.ibw);;Igor Packed Experiment (*.pxp);;CSV file (*.csv);;All files (*)",
        )
        if filename:
            self._load_file(filename)

    def _load_file(self, filename: str) -> None:
        self.loaded_pxp_waves = {}
        self.pxp_wave_list.clear()
        self.wave_picker_group.setVisible(False)
        if is_pxp_file(filename):
            self._load_pxp_file(filename)
            return
        try:
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

        previous_auto_name = self._auto_stim_name
        self._auto_stim_name = wave.name
        current_name = self.new_name_edit.text().strip()
        if not current_name or current_name == previous_auto_name:
            self.new_name_edit.setText(self._auto_stim_name)

        previous_auto_units = self._auto_units
        self._auto_units = wave.units or ""
        current_units = self.new_units_edit.text().strip()
        if not current_units or current_units == previous_auto_units:
            self.new_units_edit.setText(self._auto_units)

        self.loaded_wave = wave
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
        if self.loaded_wave is None:
            self.preview_text.setPlainText("")
            if not self.project.stim_arrays:
                self.info_label.setText(
                    "Select an Igor wave to preview it. Only the new-stim import mode is "
                    "currently available because the project has no stim arrays yet."
                )
            else:
                self.info_label.setText("Select an Igor wave to preview it.")
            return

        self.preview_text.setPlainText(self._build_preview_text(self.loaded_wave))
        source_dt = float(self.source_dt_spin.value())
        target_dt = self._target_dt_seconds()
        info_lines: list[str] = []
        if abs(source_dt - target_dt) > 1e-9:
            info_lines.append(
                "This wave will be resampled to the selected NeuroSeq stim grid using zero-order hold. "
                "Fast sub-ms changes may be lost."
            )
        elif self.match_source_grid_checkbox.isChecked():
            info_lines.append(
                "This import will preserve the source/file timestep by adjusting the NeuroSeq stim grid."
            )
        else:
            info_lines.append("This wave already matches NeuroSeq's 1 ms timeline.")
        if self.loaded_wave.source_dt is None:
            info_lines.append(
                "The source file does not store the sample interval. Review the source sample interval before importing."
            )
        selected_stim = self._selected_stim_array()
        if (
            self.import_existing_radio.isChecked()
            and selected_stim is not None
            and not stim_dt_matches(float(selected_stim.dt), target_dt)
        ):
            info_lines.append(
                "The selected target stim array will be remapped from "
                f"{format_milliseconds(float(selected_stim.dt) * 1000.0)} ms to "
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
            enabled = self.new_name_edit.text().strip() != ""
        if enabled and self.import_existing_radio.isChecked():
            target = self._selected_stim_array()
            enabled = target is not None
        self.import_button.setEnabled(enabled)

    def _current_new_stim_type(self) -> StimType:
        current_data = self.new_type_combo.currentData()
        if isinstance(current_data, StimType):
            return current_data
        return StimType.External

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
            target_dt = self._target_dt_seconds()
            prepared_wave = prepare_igor_wave_for_neuroseq(
                self.loaded_wave,
                source_dt_override=float(self.source_dt_spin.value()),
                target_dt=target_dt,
            )

            _dt_changed = not stim_dt_matches(target_dt, self.project.dt)
            if _dt_changed:
                affected = self.project.remappable_data_description(target_dt)
                if affected:
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        "Change project sample interval",
                        f"Importing at {format_milliseconds(target_dt * 1000.0)} ms "
                        "will change the project sample interval from "
                        f"{format_milliseconds(self.project.dt * 1000.0)} ms to "
                        f"{format_milliseconds(target_dt * 1000.0)} ms and remap "
                        f"{affected} using zero-order hold. Continue?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        return
                self.project.remap_all_data_to_dt(target_dt)

            if self.create_new_radio.isChecked():
                stim_name = self.new_name_edit.text().strip()
                if not stim_name:
                    raise ValueError("Stim name cannot be empty")
                if stim_name in self.project.stim_arrays:
                    raise ValueError(
                        f"A stimulation data entity named '{stim_name}' already exists"
                    )
                stim_array = create_stim_array_from_igor_wave(
                    prepared_wave,
                    stim_name=stim_name,
                    stim_type=self._current_new_stim_type(),
                    units=self.new_units_edit.text().strip(),
                )
                self.project.add_stim_array(stim_array)
                self.selected_stim_name = stim_name
                info_lines = [
                    f"Imported '{prepared_wave.name}' into new stim array '{stim_name}'.",
                    f"Source samples: {prepared_wave.original_sample_count}",
                    f"Imported samples: {prepared_wave.values.shape[0]}",
                    "Target neuron count: 1",
                ]
                if prepared_wave.resampled:
                    info_lines.append(
                        f"Resampled from {prepared_wave.source_dt:.9f} s to {prepared_wave.target_dt:.3f} s using zero-order hold."
                    )
            else:
                target_stim = self._selected_stim_array()
                if target_stim is None:
                    raise ValueError("No target stimulation data entity selected")
                remapped_target_dt = _dt_changed
                if not stim_dt_matches(float(target_stim.dt), prepared_wave.target_dt):
                    continue_with_remap = QtWidgets.QMessageBox.question(
                        self,
                        "Change stim array timestep",
                        "Changing the stimulation timestep will remap stimulus array "
                        f"'{target_stim.name}' from "
                        f"{format_milliseconds(float(target_stim.dt) * 1000.0)} ms "
                        f"to {format_milliseconds(prepared_wave.target_dt * 1000.0)} ms. "
                        "Stimulus timing may shift by up to one sample. Continue?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if continue_with_remap != QtWidgets.QMessageBox.StandardButton.Yes:
                        return
                    self.project.update_stim_array_properties(
                        target_stim.name,
                        n_neurons=int(target_stim.values.shape[1]),
                        stim_type=target_stim.stim_type,
                        units=target_stim.units,
                        dt=prepared_wave.target_dt,
                    )
                    target_stim = self.project.stim_arrays[target_stim.name]
                    remapped_target_dt = True
                report = import_igor_wave_into_stim_array(
                    target_stim,
                    prepared_wave,
                    neuron_index=self.neuron_index_spin.value(),
                    replace_existing=self.replace_checkbox.isChecked(),
                    extend_to_fit=self.extend_checkbox.isChecked(),
                )
                self.selected_stim_name = target_stim.name
                info_lines = [
                    f"Imported '{prepared_wave.name}' into '{target_stim.name}' neuron {report.target_neuron_index}.",
                    f"Source samples: {report.source_samples}",
                    f"Imported samples: {report.imported_samples}",
                ]
                if report.truncated_samples > 0:
                    info_lines.append(
                        f"Truncated imported samples: {report.truncated_samples}"
                    )
                if remapped_target_dt:
                    info_lines.append(
                        f"Adjusted target stim grid to {report.target_dt:.9f} s before import."
                    )
                if report.resampled:
                    info_lines.append(
                        f"Resampled from {report.source_dt:.9f} s to {report.target_dt:.3f} s using zero-order hold."
                    )

            is_pxp_session = bool(self.loaded_pxp_waves)

            if is_pxp_session:
                current_item = self.pxp_wave_list.currentItem()
                if current_item is not None:
                    path_key = current_item.text().removeprefix("✓ ")
                    self._imported_pxp_keys.add(path_key)
                    current_item.setText(f"✓ {path_key}")
                    current_item.setFlags(
                        current_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
                    )

                self.last_import_label.setText(info_lines[0])
                self.pxp_wave_list.clearSelection()
                self.pxp_wave_list.setCurrentItem(None)
                self.loaded_wave = None
                self.new_name_edit.clear()
                self.new_units_edit.clear()
                self._auto_stim_name = ""
                self._auto_units = ""
                self._populate_target_stims()

                self._had_successful_import = True
                self._promote_cancel_to_close()
                self._update_preview()
                self._update_import_button_state()
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Complete",
                    "\n".join(info_lines),
                )
                self.accept()
        except Exception as exc:
            self.logger.exception("Failed to import Igor wave")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import Igor wave:\n{exc}",
            )

    def done(self, result: int) -> None:
        self._cleanup_wave_plot()
        super().done(result)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._cleanup_wave_plot()
        super().closeEvent(event)
