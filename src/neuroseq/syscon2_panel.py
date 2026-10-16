"""GUI panel for live SysCon2 control: connection, sequence manager, Holo4D, and laser.

Only talks to SysCon2 on the same machine (see the project plan for the deferred
network-bridge transport). Hazardous actions (emission, intensity, sequence run) go through a
confirmation dialog first.
"""

from __future__ import annotations

import logging
import pathlib
import re
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.export.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.export.syscon2_export_service import SysconExportError
from neuroseq.export.syscon2_export_service import SysconExportResult
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.export.syscon2_export_service import default_export_directory
from neuroseq.export.syscon2_export_service import default_target_directory
from neuroseq.export.syscon2_export_service import export_stim_array
from neuroseq.stim import StimArray
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.stimulation_generator_dialog import next_default_export_name
from neuroseq.syscon2_client import LightsourceStatus
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_connection_manager import SysCon2ConnectionManager
from neuroseq.syscon2_stimulation_host import SysCon2StimulationHost

if TYPE_CHECKING:
    from neuroseq.gui_settings import GuiSettings
    from neuroseq.project import Project

logger = logging.getLogger(__name__)

_START_CONDITIONS = ["auto", "trigger1", "trigger2"]
_TRIGGER_BEHAVIORS = ["rising", "falling"]


class _NoScrollSpinBox(QtWidgets.QSpinBox):
    """Only reacts to the wheel while focused, so scrolling the panel's QScrollArea
    doesn't silently change the value when the cursor merely passes over it.
    """

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        elif event is not None:
            event.ignore()


class _NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        elif event is not None:
            event.ignore()


class _NoScrollComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        elif event is not None:
            event.ignore()


class _SequencePreview(QtWidgets.QScrollArea):
    """A bounded image viewport with scrolling and explicit zoom controls."""

    zoom_changed = QtCore.Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._source = QtGui.QPixmap()
        self._zoom = 0.0
        self._fit_mode = True
        self._message = "No preview yet."
        self.setMinimumSize(200, 140)
        self.setMaximumHeight(280)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.setWidgetResizable(False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label = QtWidgets.QLabel(self._message)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.setWidget(self.image_label)

    def setText(self, text: str) -> None:
        self._source = QtGui.QPixmap()
        self._message = text
        self.image_label.setPixmap(QtGui.QPixmap())
        self.image_label.setText(text)

    def text(self) -> str:
        return self._message

    def pixmap(self) -> QtGui.QPixmap:
        return self.image_label.pixmap()

    def set_image(self, path: str) -> None:
        self._source = QtGui.QPixmap(path)
        if self._source.isNull():
            self.setText(f"Could not load preview image:\n{path}")
            return
        self._zoom = 0.0
        self._fit_mode = True
        self.fit_to_view()

    def zoom_by(self, factor: float) -> None:
        if self._source.isNull():
            return
        self._fit_mode = False
        self._zoom = min(8.0, max(0.1, self._zoom * factor))
        self._fit_image()
        self.zoom_changed.emit(self._zoom)

    def fit_to_view(self) -> None:
        if self._source.isNull():
            return
        self._fit_mode = True
        available = self.viewport().size()
        self._zoom = min(
            available.width() / self._source.width(),
            available.height() / self._source.height(),
            1.0,
        )
        self._fit_image()
        self.zoom_changed.emit(self._zoom)

    def _fit_image(self) -> None:
        if not self._source.isNull():
            self.image_label.setText("")
            self.image_label.setPixmap(
                self._source.scaled(
                    max(1, round(self._source.width() * self._zoom)),
                    max(1, round(self._source.height() * self._zoom)),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.image_label.resize(self.image_label.pixmap().size())

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_view()


_READY_CHECK_TIMEOUT_MS = 10_000
_TRIGGER_HINT_MS = 5_000


class SysCon2Panel(QtWidgets.QWidget):
    #: Targets in the most recent preview whose import was acknowledged by SysCon.
    #: This does not indicate remote selection, upload, or live illumination.
    targets_changed = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        manager: SysCon2ConnectionManager | None = None,
        settings: GuiSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        #: Remembers the export folder between sessions, if given.
        self._settings = settings
        self._manager = manager if manager is not None else SysCon2ConnectionManager()
        self._project: Project | None = None
        self._connected = False
        self._import_serial = 0
        self._preview_request: str | None = None
        self._preview_source = ""
        self._pending_selections: set[str] = set()
        #: Generated overview images by .seq file name, for previewing selections.
        self._known_previews: dict[str, str] = {}
        #: File name of a just-imported sequence to select once the list refreshes.
        self._auto_select_name: str | None = None
        self._known_sources: dict[str, str] = {}
        self._pending_imports: dict[str, frozenset[int] | None] = {}
        self._import_files = tempfile.TemporaryDirectory(prefix="neuroseq-import-")
        self._lightsources: dict[str, list[str]] = {}
        self._execution_cancel = threading.Event()
        self._discover_for_status = False
        self._execution_serial = 0
        self._execution_phase = "idle"
        self._execution_options = ("auto", "rising")
        self._execution_elapsed = QtCore.QElapsedTimer()
        self._progress_message = "No run requested."
        self._hardware_state = ""
        self._ready_elapsed = QtCore.QElapsedTimer()
        self._progress_timer = QtCore.QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._update_progress_text)
        self._current_stim_provider: Callable[[], StimArray | None] | None = None
        #: Export options of the last successful dialog export; reused by the one-click button.
        self._export_settings: SysconExportSettings | None = None
        self._active_dialog: StimulationGeneratorDialog | None = None
        self._result_handlers: dict[str, Callable[[object], None]] = {
            "sequence_manager.get_sequences": self._on_sequences,
            "sequence_manager.import": self._on_sequence_imported,
            "sequence_manager.delete": self._on_sequence_imported,
            "sequence_manager.select": self._on_sequence_selected,
            "laser.get_lightsources": self._on_lightsources,
            "laser.get_status": self._on_lightsource_status,
        }

        self._build_ui()
        self._wire_manager_signals()
        self._update_connection_ui()

        #: Hardware counterpart of the Brian2 simulation host for dynamic input routines.
        self.stimulation_host = SysCon2StimulationHost(
            self._manager.submit,
            settings_provider=lambda: self._export_settings,
            target_directory_provider=lambda: self.target_directory_edit.text().strip(),
            parent=self,
        )
        self._manager.command_finished.connect(
            self.stimulation_host.on_command_finished
        )
        self._manager.command_failed.connect(self.stimulation_host.on_command_failed)
        self.stimulation_host.block_sent.connect(self._on_host_block_sent)

    def set_project(self, project: Project | None) -> None:
        if project is self._project:
            return
        self._project = project
        self._export_settings = None
        self._reset_import_status()
        self.preview_label.setText("No preview yet.")

    def set_current_stim_provider(
        self, provider: Callable[[], StimArray | None]
    ) -> None:
        """Tell the panel how to look up the stim array currently shown in the editor."""
        self._current_stim_provider = provider

    @property
    def export_settings(self) -> SysconExportSettings | None:
        return self._export_settings

    def shutdown(self) -> None:
        """Disconnect and stop the background connection thread. Call on app shutdown."""
        self._execution_cancel.set()
        self._manager.shutdown()
        self._import_files.cleanup()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self.shutdown()
        super().closeEvent(event)

    # ---- UI construction ----

    def _build_ui(self) -> None:
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_sequence_manager_group())
        layout.addWidget(self._build_preview_group())
        layout.addWidget(self._build_holo4d_group())
        layout.addWidget(self._build_laser_group())
        layout.addStretch(1)

        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )

    def _build_connection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Connection")
        layout = QtWidgets.QHBoxLayout(group)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self._manager.connect_to_syscon2)
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(self._manager.disconnect_from_syscon2)
        self.status_label = QtWidgets.QLabel("Disconnected")
        layout.addWidget(self.btn_connect)
        layout.addWidget(self.btn_disconnect)
        layout.addWidget(self.status_label, stretch=1)
        return group

    def _build_sequence_manager_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Sequence Manager")
        layout = QtWidgets.QVBoxLayout(group)

        self.btn_send_to_syscon2 = QtWidgets.QPushButton("Export with settings…")
        self.btn_send_to_syscon2.setToolTip(
            "Open the stimulation export dialog and, once it generates a script, "
            "import it if connected. Import does not select, upload, or run it."
        )
        self.btn_send_to_syscon2.clicked.connect(self._on_send_to_syscon2_clicked)
        layout.addWidget(self.btn_send_to_syscon2)

        self.btn_export_current = QtWidgets.QPushButton(
            "Export current array to stimulation setup"
        )
        self.btn_export_current.setToolTip(
            "Export the input array currently shown in the editor with the settings of "
            "the last 'Export with settings…' export, and import it into the stimulation setup. Opens the "
            "export dialog first if there are no settings yet."
        )
        self.btn_export_current.clicked.connect(self._on_export_current_clicked)
        # Next to the export button, so changing only the power needs no trip
        # through the export dialog (issue #616).
        self.target_power_spin = _NoScrollDoubleSpinBox()
        self.target_power_spin.setRange(0.0, 1000.0)
        self.target_power_spin.setDecimals(2)
        self.target_power_spin.setSuffix(" mW")
        self.target_power_spin.setSingleStep(0.5)
        self.target_power_spin.setValue(1.0)
        self.target_power_spin.setToolTip(
            "Target power of the export: used by the button beside it and filled "
            "into 'Export with settings…'. Set to 0 to disable power control."
        )
        self.target_power_spin.valueChanged.connect(self._on_target_power_changed)
        export_row = QtWidgets.QHBoxLayout()
        export_row.addWidget(self.btn_export_current, stretch=1)
        export_row.addWidget(QtWidgets.QLabel("Target power:"))
        export_row.addWidget(self.target_power_spin)
        layout.addLayout(export_row)

        self.sequence_list = QtWidgets.QListWidget()
        self.sequence_list.setMaximumHeight(120)
        self.sequence_list.currentRowChanged.connect(
            lambda _: self._update_sequence_buttons()
        )
        layout.addWidget(self.sequence_list)
        self.selection_label = QtWidgets.QLabel(
            "Remote selection not confirmed. Highlight a row, then Select."
        )
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        refresh_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_sequences = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_sequences.clicked.connect(self._on_refresh_sequences_clicked)
        self.btn_select_sequence = QtWidgets.QPushButton("Select")
        self.btn_select_sequence.clicked.connect(self._on_select_sequence_clicked)
        self.btn_delete_sequence = QtWidgets.QPushButton("Delete")
        self.btn_delete_sequence.clicked.connect(self._on_delete_sequence_clicked)
        refresh_row.addWidget(self.btn_refresh_sequences)
        refresh_row.addWidget(self.btn_select_sequence)
        refresh_row.addWidget(self.btn_delete_sequence)
        layout.addLayout(refresh_row)

        import_row = QtWidgets.QHBoxLayout()
        self.import_path_edit = QtWidgets.QLineEdit()
        self.import_path_edit.setPlaceholderText("Local .seq file to import")
        self.btn_browse_import = QtWidgets.QPushButton("Browse…")
        self.btn_browse_import.clicked.connect(self._on_browse_import_clicked)
        import_row.addWidget(self.import_path_edit, stretch=1)
        import_row.addWidget(self.btn_browse_import)
        layout.addLayout(import_row)

        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("Target directory:"))
        self.target_directory_edit = QtWidgets.QLineEdit(default_target_directory())
        self.target_directory_edit.setToolTip(
            "Folder on the stimulation PC that imported scripts go to; by "
            "default one per day, named after the date."
        )
        target_row.addWidget(self.target_directory_edit, stretch=1)
        self.btn_import_sequence = QtWidgets.QPushButton("Import")
        self.btn_import_sequence.clicked.connect(self._on_import_sequence_clicked)
        target_row.addWidget(self.btn_import_sequence)
        layout.addLayout(target_row)

        return group

    def _build_preview_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Preview")
        layout = QtWidgets.QVBoxLayout(group)
        self.preview_status_label = QtWidgets.QLabel("No generated export.")
        self.preview_status_label.setWordWrap(True)
        self.preview_status_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.preview_status_label)
        self.btn_toggle_preview = QtWidgets.QToolButton()
        self.btn_toggle_preview.setText("Show preview")
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_toggle_preview.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.btn_toggle_preview.toggled.connect(self._on_preview_toggled)
        layout.addWidget(self.btn_toggle_preview)
        self.preview_container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(self.preview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = _SequencePreview()
        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.addStretch(1)
        self.btn_preview_zoom_out = QtWidgets.QToolButton()
        self.btn_preview_zoom_out.setText("−")
        self.btn_preview_zoom_out.setToolTip("Zoom out")
        self.btn_preview_zoom_out.clicked.connect(
            lambda: self.preview_label.zoom_by(1 / 1.25)
        )
        self.preview_zoom_label = QtWidgets.QLabel("Fit")
        self.preview_zoom_label.setMinimumWidth(44)
        self.preview_zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_preview_zoom_in = QtWidgets.QToolButton()
        self.btn_preview_zoom_in.setText("+")
        self.btn_preview_zoom_in.setToolTip("Zoom in")
        self.btn_preview_zoom_in.clicked.connect(
            lambda: self.preview_label.zoom_by(1.25)
        )
        self.btn_preview_fit = QtWidgets.QToolButton()
        self.btn_preview_fit.setText("Fit")
        self.btn_preview_fit.clicked.connect(self.preview_label.fit_to_view)
        zoom_row.addWidget(self.btn_preview_zoom_out)
        zoom_row.addWidget(self.preview_zoom_label)
        zoom_row.addWidget(self.btn_preview_zoom_in)
        zoom_row.addWidget(self.btn_preview_fit)
        container_layout.addLayout(zoom_row)
        self.preview_label.zoom_changed.connect(
            lambda zoom: self.preview_zoom_label.setText(f"{zoom:.0%}")
        )
        container_layout.addWidget(self.preview_label)
        self.preview_container.setVisible(False)
        layout.addWidget(self.preview_container)
        return group

    def _on_preview_toggled(self, checked: bool) -> None:
        self.preview_container.setVisible(checked)
        self.btn_toggle_preview.setText("Hide preview" if checked else "Show preview")
        self.btn_toggle_preview.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )
        if checked:
            self.preview_label.fit_to_view()

    def _build_holo4d_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Holo4D")
        layout = QtWidgets.QVBoxLayout(group)

        self.btn_toggle_holo4d_advanced = QtWidgets.QToolButton()
        self.btn_toggle_holo4d_advanced.setText("Advanced options")
        self.btn_toggle_holo4d_advanced.setCheckable(True)
        self.btn_toggle_holo4d_advanced.setChecked(False)
        self.btn_toggle_holo4d_advanced.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_toggle_holo4d_advanced.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.btn_toggle_holo4d_advanced.toggled.connect(
            self._on_holo4d_advanced_toggled
        )
        layout.addWidget(self.btn_toggle_holo4d_advanced)

        self.holo4d_advanced_panel = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(self.holo4d_advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        self.holo4d_state_label = QtWidgets.QLabel("State: unknown")
        layout.addWidget(self.holo4d_state_label)

        form = QtWidgets.QFormLayout()
        self.runs_spinbox = _NoScrollSpinBox()
        self.runs_spinbox.setRange(0, 1_000_000)
        self.runs_spinbox.setSpecialValueText("Endless (0)")
        self.runs_spinbox.setValue(1)
        form.addRow("Runs:", self.runs_spinbox)
        self.start_condition_combo = _NoScrollComboBox()
        self.start_condition_combo.addItems(_START_CONDITIONS)
        form.addRow("Start condition:", self.start_condition_combo)
        self.trigger_behavior_combo = _NoScrollComboBox()
        self.trigger_behavior_combo.addItems(_TRIGGER_BEHAVIORS)
        form.addRow("Trigger behavior:", self.trigger_behavior_combo)
        advanced_layout.addLayout(form)

        self.holo4d_advanced_panel.setVisible(False)
        layout.addWidget(self.holo4d_advanced_panel)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_holo4d_upload = QtWidgets.QPushButton("Upload")
        self.btn_holo4d_upload.clicked.connect(self._on_upload_clicked)
        self.btn_holo4d_run = QtWidgets.QPushButton("Run")
        self.btn_holo4d_run.clicked.connect(self._on_run_clicked)
        self.btn_holo4d_stop = QtWidgets.QPushButton("Stop")
        self.btn_holo4d_stop.clicked.connect(self._on_holo4d_stop_clicked)
        button_row.addWidget(self.btn_holo4d_upload)
        button_row.addWidget(self.btn_holo4d_run)
        button_row.addWidget(self.btn_holo4d_stop)
        button_row.addStretch()
        layout.addLayout(button_row)
        self.execution_label = QtWidgets.QLabel(self._progress_message)
        self.execution_label.setWordWrap(True)
        layout.addWidget(self.execution_label)
        self.execution_progress = QtWidgets.QProgressBar()
        self.execution_progress.setRange(0, 1)
        self.execution_progress.setValue(0)
        self.execution_progress.setTextVisible(False)
        layout.addWidget(self.execution_progress)

        return group

    def _build_laser_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Laser")
        layout = QtWidgets.QVBoxLayout(group)

        refresh_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_lightsources = QtWidgets.QPushButton("Refresh Lightsources")
        self.btn_refresh_lightsources.clicked.connect(
            self._on_refresh_lightsources_clicked
        )
        self.btn_refresh_lightsources.setParent(group)
        self.btn_refresh_lightsources.hide()
        layout.addLayout(refresh_row)

        form = QtWidgets.QFormLayout()
        self.device_combo = _NoScrollComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        self.device_combo.setParent(group)
        self.device_combo.hide()
        self.description_combo = _NoScrollComboBox()
        self.description_combo.setPlaceholderText("940 nm — Get Status to discover")
        form.addRow("Lightsource:", self.description_combo)
        self.intensity_spinbox = _NoScrollDoubleSpinBox()
        self.intensity_spinbox.setRange(0.0, 1_000_000.0)
        self.intensity_spinbox.setDecimals(3)
        self.intensity_spinbox.setParent(group)
        self.intensity_spinbox.hide()
        layout.addLayout(form)

        self.btn_toggle_laser_advanced = QtWidgets.QToolButton()
        self.btn_toggle_laser_advanced.setText("Advanced laser controls")
        self.btn_toggle_laser_advanced.setCheckable(True)
        self.btn_toggle_laser_advanced.setChecked(False)
        self.btn_toggle_laser_advanced.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.btn_toggle_laser_advanced.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.btn_toggle_laser_advanced.toggled.connect(self._on_laser_advanced_toggled)
        self.btn_toggle_laser_advanced.setParent(group)
        self.btn_toggle_laser_advanced.hide()

        self.laser_advanced_panel = QtWidgets.QWidget()
        advanced_row = QtWidgets.QHBoxLayout(self.laser_advanced_panel)
        advanced_row.setContentsMargins(0, 0, 0, 0)
        self.btn_emission_on = QtWidgets.QPushButton("Emission ON")
        self.btn_emission_on.clicked.connect(
            lambda: self._on_set_emission_clicked(True)
        )
        self.btn_emission_off = QtWidgets.QPushButton("Emission OFF")
        self.btn_emission_off.clicked.connect(
            lambda: self._on_set_emission_clicked(False)
        )
        self.btn_shutter_open = QtWidgets.QPushButton("Open Shutter")
        self.btn_shutter_open.clicked.connect(
            lambda: self._on_set_shutter_clicked(True)
        )
        self.btn_shutter_close = QtWidgets.QPushButton("Close Shutter")
        self.btn_shutter_close.clicked.connect(
            lambda: self._on_set_shutter_clicked(False)
        )
        for btn in (
            self.btn_emission_on,
            self.btn_emission_off,
            self.btn_shutter_open,
            self.btn_shutter_close,
        ):
            advanced_row.addWidget(btn)
        self.laser_advanced_panel.setVisible(False)
        layout.addWidget(self.laser_advanced_panel)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_set_intensity = QtWidgets.QPushButton("Set Intensity")
        self.btn_set_intensity.clicked.connect(self._on_set_intensity_clicked)
        self.btn_refresh_status = QtWidgets.QPushButton("Get Status")
        self.btn_refresh_status.clicked.connect(self._on_refresh_status_clicked)
        self.btn_set_intensity.setParent(group)
        self.btn_set_intensity.hide()
        button_row.addWidget(self.btn_refresh_status)
        layout.addLayout(button_row)

        self.lightsource_status_label = QtWidgets.QLabel("")
        layout.addWidget(self.lightsource_status_label)

        return group

    # ---- connection manager signal handling ----

    def _wire_manager_signals(self) -> None:
        self._manager.connected.connect(self._on_connected)
        self._manager.disconnected.connect(self._on_disconnected)
        self._manager.state_changed.connect(self._on_state_changed)
        self._manager.state_poll_failed.connect(self._on_state_poll_failed)
        self._manager.command_finished.connect(self._on_command_finished)
        self._manager.command_failed.connect(self._on_command_failed)

    def _on_connected(self) -> None:
        self._connected = True
        self._update_connection_ui()

    def _on_disconnected(self, message: str) -> None:
        self._connected = False
        self._discover_for_status = False
        self.description_combo.clear()
        self.lightsource_status_label.setText("Status unknown (disconnected).")
        self._execution_cancel.set()
        self._execution_serial += 1
        self._set_execution_progress(
            "idle", "Disconnected; hardware execution status unknown."
        )
        self._reset_import_status()
        self.sequence_list.clear()
        self.selection_label.setText("Remote selection not confirmed.")
        self._update_connection_ui()
        if message:
            self.status_label.setText(f"Disconnected: {message}")
            self.status_label.setStyleSheet("QLabel { color: red; }")

    def _update_connection_ui(self) -> None:
        self._update_sequence_buttons()
        idle = self._connected and self._execution_phase == "idle"
        self.btn_holo4d_upload.setEnabled(idle)
        self.btn_holo4d_run.setEnabled(idle)
        self.btn_holo4d_stop.setEnabled(self._connected)
        self.btn_refresh_status.setEnabled(self._connected)
        can_connect_here = sys.platform == "win32"
        self.btn_connect.setEnabled(can_connect_here and not self._connected)
        self.btn_disconnect.setEnabled(self._connected)
        if not can_connect_here:
            self.btn_connect.setToolTip(
                "Stimulation setup control requires neuroseq to run on the same Windows PC as the stimulation setup."
            )
        if self._connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        elif not self.status_label.text().startswith("Disconnected:"):
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("QLabel { color: gray; }")

    def _update_sequence_buttons(self) -> None:
        self.btn_refresh_sequences.setEnabled(self._connected)
        self.btn_import_sequence.setEnabled(self._connected)
        has_selection = self._selected_sequence_path() is not None
        can_change_selection = (
            self._connected and has_selection and self._execution_phase == "idle"
        )
        self.btn_select_sequence.setEnabled(can_change_selection)
        self.btn_delete_sequence.setEnabled(can_change_selection)

    def _reset_import_status(self) -> None:
        self._auto_select_name = None
        self._pending_imports.clear()
        self._pending_selections.clear()
        self._preview_request = None
        self.preview_status_label.setText(
            "Import status unknown. No current confirmation."
        )
        self.targets_changed.emit(frozenset())

    def _on_state_poll_failed(self, message: str) -> None:
        self.holo4d_state_label.setText(f"State: unavailable ({message})")
        self._hardware_state = f"unavailable ({message})"
        self._update_progress_text()

    def _on_state_changed(self, state: dict) -> None:
        holo4d_state = state.get("Holo4D")
        if holo4d_state is not None:
            self.holo4d_state_label.setText(
                f"State: {holo4d_state}"
                if holo4d_state
                else "State: unrecognised reply (see log)"
            )
            self._advance_execution(str(holo4d_state))

    def _on_command_finished(self, name: str, result: object) -> None:
        if name.startswith("holo4d.execute."):
            self._on_execution_finished(name, result)
            return
        if name.startswith("sequence_manager.import."):
            if name not in self._pending_imports:
                return
            targets = self._pending_imports.pop(name)
            if name == self._preview_request:
                self.preview_status_label.setText(
                    f"Import confirmed: {self._preview_source}; selecting it, then Upload and Run. "
                    "The image is a generated preview, not hardware readback."
                )
                self.targets_changed.emit(targets or frozenset())
                self._auto_select_name = self._preview_source
            self._on_refresh_sequences_clicked()
            return
        if name.startswith("sequence_manager.select."):
            if name in self._pending_selections:
                self._pending_selections.remove(name)
                self._on_sequence_selected(result)
            return
        handler = self._result_handlers.get(name)
        if handler is not None:
            handler(result)

    def _on_command_failed(self, name: str, message: str) -> None:
        if name == "laser.get_lightsources":
            self._discover_for_status = False
        if name.startswith("holo4d.execute."):
            if not name.startswith(f"holo4d.execute.{self._execution_serial}."):
                return
            if name.endswith(".stop") and "could not stop" in message.lower():
                # SysCon could not stop anything, so the sequence has most likely ended.
                self._set_execution_progress(
                    "idle",
                    "Nothing to stop: the stimulation setup could not stop a sequence, "
                    "so it has probably already ended "
                    f"(last hardware state: {self._hardware_state or 'unknown'}).",
                )
                return
            self._set_execution_progress(
                "idle", f"Command failed: {message}. Check hardware state."
            )
        if name.startswith("sequence_manager.select."):
            if name not in self._pending_selections:
                return
            self._pending_selections.remove(name)
            self.selection_label.setText(f"Selection failed: {message}")
        if name.startswith("sequence_manager.import."):
            if name not in self._pending_imports:
                return
            self._pending_imports.pop(name)
            if name == self._preview_request:
                self.preview_status_label.setText(
                    f"Import failed: {message}. Generated preview only."
                )
        QtWidgets.QMessageBox.warning(
            self, "Stimulation setup command failed", f"{name}: {message}"
        )

    def _submit(self, name: str, fn: Callable[[SysCon2Client], object]) -> None:
        self._manager.submit(name, fn)

    def _confirm(self, message: str) -> bool:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    # ---- Sequence Manager ----

    def _on_refresh_sequences_clicked(self) -> None:
        self._submit(
            "sequence_manager.get_sequences",
            lambda client: client.sequence_manager.get_sequences(),
        )

    def _on_sequences(self, result: object) -> None:
        selected = self._selected_sequence_path()
        sequences = result if isinstance(result, list) else []
        self.sequence_list.clear()
        self.sequence_list.addItems(sequences)
        if selected in sequences:
            self.sequence_list.setCurrentRow(sequences.index(selected))
        self._update_sequence_buttons()
        self._select_imported_sequence()

    def _select_imported_sequence(self) -> None:
        """Select the sequence this panel just imported, so only Upload and Run remain."""
        name, self._auto_select_name = self._auto_select_name, None
        if not name or not self._connected or self._execution_phase != "idle":
            return
        key = name.lower()
        matches = [
            row
            for row in range(self.sequence_list.count())
            if pathlib.PureWindowsPath(
                self.sequence_list.item(row).text().replace("/", "\\")
            ).name.lower()
            == key
        ]
        if not matches:
            return
        target = self.target_directory_edit.text().strip().lower().replace("/", "\\")
        in_target = [
            row
            for row in matches
            if self.sequence_list.item(row)
            .text()
            .lower()
            .replace("/", "\\")
            .startswith(target)
        ]
        self.sequence_list.setCurrentRow((in_target or matches)[-1])
        self._on_select_sequence_clicked()

    def _on_sequence_imported(self, _result: object) -> None:
        self.selection_label.setText("Remote selection not confirmed.")
        self._on_refresh_sequences_clicked()

    def _remember_preview(self, seq_path: str, png_path: str) -> None:
        key = pathlib.Path(seq_path).name.lower()
        self._known_previews[key] = png_path
        self._known_sources[key] = seq_path

    def _on_sequence_selected(self, result: object) -> None:
        self.selection_label.setText(
            f"Selection confirmed: {result}. Use Upload, then Run, to apply it to Holo4D."
        )

    def _selected_sequence_path(self) -> str | None:
        item = self.sequence_list.currentItem()
        return item.text() if item is not None else None

    def _on_select_sequence_clicked(self) -> None:
        path = self._selected_sequence_path()
        if not path:
            return

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.select_sequence(path)
            return path

        self._show_preview_for_sequence(path)
        self._import_serial += 1
        name = f"sequence_manager.select.{self._import_serial}"
        self._pending_selections.add(name)
        self.selection_label.setText(f"Selection pending: {path}")
        self._submit(name, task)

    def _remote_sequence_names(self) -> list[str]:
        return [
            self.sequence_list.item(row).text()
            for row in range(self.sequence_list.count())
        ]

    def _show_preview_for_sequence(self, remote_path: str) -> None:
        """Show a preview for a sequence chosen in the manager.

        Uses the overview generated at export time when available; otherwise renders
        one from a local copy of the ``.seq`` file (the remote name is the imported file's
        name, so a local file with that name is looked up in known sources and the output folder).
        """
        name = pathlib.PureWindowsPath(remote_path.replace("/", "\\")).name
        key = name.lower()
        png = self._known_previews.get(key)
        if png is not None and pathlib.Path(png).is_file():
            self._show_preview(png)
            return
        output_dir = pathlib.Path(self.stimulation_host.output_dir)
        candidates = []
        known = self._known_sources.get(key)
        if known:
            candidates.append(pathlib.Path(known))
        candidates.append(output_dir / name)
        typed = self.import_path_edit.text().strip()
        if typed and pathlib.Path(typed).name.lower() == key:
            candidates.append(pathlib.Path(typed))
        for seq in candidates:
            if not seq.is_file():
                continue
            existing = seq.with_name(f"{seq.stem}_overview.png")
            if existing.is_file():
                self._known_previews[key] = str(existing)
                self._show_preview(str(existing))
                return
            try:
                folder = pathlib.Path(self._import_files.name) / "previews"
                folder.mkdir(exist_ok=True)
                out = generate_syscon2_timeline_overview_png(
                    seq.read_text(encoding="utf-8", errors="replace"),
                    folder / f"{seq.stem}_{len(self._known_previews)}_overview.png",
                )
            except (ValueError, OSError):
                continue
            self._known_previews[key] = str(out)
            self._show_preview(str(out))
            return
        self.preview_label.setText(
            f"No local copy of {name} found to preview. Put it in the export folder "
            f"({output_dir}) or enter its path in the import field."
        )

    def _on_delete_sequence_clicked(self) -> None:
        path = self._selected_sequence_path()
        if not path:
            return
        if not self._confirm(
            f"Delete sequence {path!r} from the stimulation setup's sequence manager?"
        ):
            return

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.delete_sequence(path)
            return None

        self._submit("sequence_manager.delete", task)

    def _on_browse_import_clicked(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select .seq file to import",
            "",
            "Stimulation sequence files (*.seq);;All files (*)",
        )
        if filename:
            self.import_path_edit.setText(filename)

    def _on_import_sequence_clicked(self) -> None:
        path = self.import_path_edit.text().strip()
        if path:
            self.preview_label.setText(
                "No generated preview associated with this file."
            )
            self._import_sequence(path, None)

    def _on_send_to_syscon2_clicked(self) -> None:
        if self._project is None:
            QtWidgets.QMessageBox.warning(
                self, "No project", "No active project to export a stimulus from."
            )
            return
        current = self._current_stim_provider() if self._current_stim_provider else None
        dialog = StimulationGeneratorDialog(
            self._project,
            parent=self.window(),
            current_stim_name=current.name if current else None,
            output_folder=self._export_directory(),
        )
        dialog.reserved_names = self._remote_sequence_names
        if self._export_settings is not None:
            dialog.apply_settings(self._export_settings)
        dialog.target_power_spin.setValue(self.target_power_spin.value())
        dialog.sequence_generated.connect(self._on_dialog_sequence_generated)
        self._active_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._active_dialog = None
        if self._settings is not None and dialog.output_folder is not None:
            self._settings.syscon_export_directory = str(dialog.output_folder)

    def _export_directory(self) -> pathlib.Path:
        """The folder used last, or a default one (issue #614)."""
        stored = self._settings.syscon_export_directory if self._settings else ""
        return pathlib.Path(stored) if stored else default_export_directory()

    def _on_target_power_changed(self, value: float) -> None:
        if self._export_settings is not None:
            self._export_settings = replace(
                self._export_settings, target_power_mw=float(value)
            )

    def _show_export_settings(self, settings: SysconExportSettings) -> None:
        self._export_settings = settings
        self.target_power_spin.blockSignals(True)
        self.target_power_spin.setValue(settings.target_power_mw)
        self.target_power_spin.blockSignals(False)

    def _on_export_current_clicked(self) -> None:
        if self._project is None:
            QtWidgets.QMessageBox.warning(
                self, "No project", "No active project to export a stimulus from."
            )
            return
        if self._export_settings is None:
            self._on_send_to_syscon2_clicked()
            return
        stim_array = (
            self._current_stim_provider()
            if self._current_stim_provider is not None
            else None
        )
        if stim_array is None:
            QtWidgets.QMessageBox.warning(
                self, "No input array", "Select an input array in the editor first."
            )
            return
        try:
            result = export_stim_array(
                self._project,
                stim_array,
                self._export_settings,
                self.stimulation_host.output_dir,
                next_default_export_name(
                    self.stimulation_host.output_dir,
                    reserved=self._remote_sequence_names(),
                ),
                confirm_warnings=lambda warnings: self._confirm(
                    "Review export changes:\n" + "\n".join(warnings)
                ),
            )
        except (SysconExportError, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._on_dialog_sequence_generated(
            str(result.seq_path),
            str(result.overview_png_path),
            result.targeted_stim_ids,
        )

    def _on_dialog_sequence_generated(
        self,
        seq_path: str,
        overview_png_path: str,
        targeted_stim_ids: frozenset[int],
    ) -> None:
        if self._active_dialog is not None:
            self._show_export_settings(self._active_dialog.export_settings())
        self._show_preview(overview_png_path)
        self._remember_preview(seq_path, overview_png_path)
        self.import_path_edit.setText(seq_path)
        self._import_sequence(seq_path, targeted_stim_ids)

    def _import_sequence(self, seq_path: str, targets: frozenset[int] | None) -> None:
        self._preview_request = None
        self.targets_changed.emit(frozenset())
        self.preview_status_label.setText(
            f"Generated/local file only; not imported: {seq_path}"
        )
        if not self._connected:
            return
        target = self.target_directory_edit.text().strip()
        try:
            source = pathlib.Path(seq_path).resolve(strict=True)
            self._known_sources[source.name.lower()] = str(source)
            remote = pathlib.PureWindowsPath(target)
            if source.suffix.lower() != ".seq" or not source.is_file():
                raise ValueError("Select an existing .seq file.")
            if remote.drive or remote.root or ".." in remote.parts:
                raise ValueError(
                    "Target directory must be relative to the SysCon experiment directory."
                )
            target.encode("ascii")
            source.name.encode("ascii")
            self._import_serial += 1
            name = f"sequence_manager.import.{self._import_serial}"
            folder = pathlib.Path(self._import_files.name) / str(self._import_serial)
            folder.mkdir()
            snapshot = folder / source.name
            shutil.copyfile(source, snapshot)
            str(snapshot).encode("ascii")
        except (ValueError, OSError) as exc:
            self.preview_status_label.setText(f"Not imported: {exc}")
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc))
            return
        self._pending_imports[name] = targets
        self._preview_request = name
        self._preview_source = source.name
        self.preview_status_label.setText(
            f"Import pending: {source.name}. Generated preview only."
        )

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.import_sequence(str(snapshot), target)
            return None

        self._submit(name, task)

    def show_generated_preview(
        self,
        seq_path: str,
        png_path: str,
        _targets: object,
        settings: SysconExportSettings | None = None,
    ) -> None:
        if settings is not None:
            self._show_export_settings(settings)
        self._show_preview(png_path)
        self._remember_preview(seq_path, png_path)
        self.import_path_edit.setText(seq_path)
        self._preview_request = None
        self.targets_changed.emit(frozenset())
        self.preview_status_label.setText(
            f"Generated preview only; not imported: {seq_path}"
        )

    def _on_host_block_sent(self, result: object) -> None:
        if not isinstance(result, SysconExportResult):
            return
        self._preview_request = None
        self._show_preview(str(result.overview_png_path))
        self._remember_preview(str(result.seq_path), str(result.overview_png_path))
        self.import_path_edit.setText(str(result.seq_path))
        self.preview_status_label.setText(
            f"Import confirmed: {result.seq_path.name}. Select it, then Upload and Run."
        )
        self.targets_changed.emit(result.targeted_stim_ids)
        if self._connected:
            self._on_refresh_sequences_clicked()

    def _show_preview(self, png_path: str) -> None:
        self.preview_label.set_image(png_path)

    # ---- Holo4D ----

    def _on_holo4d_advanced_toggled(self, checked: bool) -> None:
        self.holo4d_advanced_panel.setVisible(checked)
        self.btn_toggle_holo4d_advanced.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def _set_execution_progress(self, phase: str, message: str) -> None:
        self._execution_phase = phase
        self._progress_message = message
        busy = phase != "idle"
        self.execution_progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.execution_progress.setValue(
                1 if message.startswith("Sequence finished") else 0
            )
            self._progress_timer.stop()
        else:
            self._progress_timer.start()
        self.btn_holo4d_upload.setEnabled(self._connected and not busy)
        self.btn_holo4d_run.setEnabled(self._connected and not busy)
        self._update_sequence_buttons()
        self._update_progress_text()

    def _update_progress_text(self) -> None:
        if (
            self._execution_phase == "ready_check"
            and self._ready_elapsed.isValid()
            and self._ready_elapsed.elapsed() > _READY_CHECK_TIMEOUT_MS
        ):
            state = self._hardware_state or "unknown"
            self._set_execution_progress(
                "idle",
                f"Upload acknowledged but the hardware state is still '{state}'. "
                "Check the stimulation setup; Run is available.",
            )
            return
        busy = self._execution_phase != "idle"
        if (
            self._execution_phase == "awaiting_busy"
            and self._hardware_state == "idle"
            and self._execution_elapsed.isValid()
            and self._execution_elapsed.elapsed() > _TRIGGER_HINT_MS
        ):
            self._progress_message = (
                f"Run accepted; hardware stays Idle, waiting for {self._execution_options[0]}. "
                "If the trigger already fired, a short sequence may have finished "
                "before Busy was observed; press Stop to reset."
            )
        elapsed = (
            f" ({self._execution_elapsed.elapsed() / 1000:.1f} s)"
            if busy and self._execution_elapsed.isValid()
            else ""
        )
        hardware = (
            f" [hardware state: {self._hardware_state}]"
            if busy and self._hardware_state
            else ""
        )
        self.execution_label.setText(self._progress_message + elapsed + hardware)

    def _on_upload_clicked(self) -> None:
        if not self._connected or self._execution_phase != "idle":
            return
        runs = self.runs_spinbox.value()
        if runs == 0 and not self._confirm(
            "Upload for endless repetition? The sequence will repeat until stopped."
        ):
            return
        self._execution_serial += 1
        self._execution_cancel = threading.Event()
        self._execution_elapsed.start()
        self._set_execution_progress("uploading", "Uploading sequence…")
        self._submit(
            f"holo4d.execute.{self._execution_serial}.upload",
            lambda client: client.holo4d.upload_sequence(runs),
        )

    def _on_run_clicked(self) -> None:
        if not self._connected or self._execution_phase != "idle":
            return
        if not self._confirm("Run the currently uploaded sequence on Holo4D now?"):
            return
        self._execution_serial += 1
        cancelled = self._execution_cancel = threading.Event()
        start = self.start_condition_combo.currentText()
        edge = self.trigger_behavior_combo.currentText()
        self._execution_options = (start, edge)
        self._execution_elapsed.start()
        self._set_execution_progress("starting", "Starting sequence…")
        self._submit(
            f"holo4d.execute.{self._execution_serial}.run",
            lambda client: (
                None if cancelled.is_set() else client.holo4d.run_sequence(start, edge)
            ),
        )

    def _on_execution_finished(self, name: str, result: object) -> None:
        if not name.startswith(f"holo4d.execute.{self._execution_serial}."):
            return
        if name.endswith(".upload") and self._execution_phase == "uploading":
            self._ready_elapsed.start()
            self._set_execution_progress(
                "ready_check", "Upload acknowledged; waiting for hardware readiness…"
            )
            self._submit(
                f"holo4d.execute.{self._execution_serial}.state",
                lambda client: client.holo4d.get_state(),
            )
        elif name.endswith(".state"):
            self._advance_execution(str(result))
        elif name.endswith(".run") and self._execution_phase == "starting":
            trigger = self._execution_options[0]
            self._set_execution_progress(
                "awaiting_busy",
                "Run accepted; "
                + (
                    "waiting for hardware Busy state…"
                    if trigger == "auto"
                    else f"waiting for {trigger} / hardware Busy state…"
                ),
            )
        elif name.endswith(".stop"):
            self._set_execution_progress("idle", "Stop acknowledged.")

    def _advance_execution(self, state: str) -> None:
        state = state.lower().removeprefix("state:").strip()
        if state != self._hardware_state:
            logger.info(
                "Holo4D state %r -> %r (execution phase %s)",
                self._hardware_state,
                state,
                self._execution_phase,
            )
        self._hardware_state = state
        if self._execution_phase == "ready_check":
            if state in {"", "busy", "disconnected"} or state.startswith("upload"):
                self._progress_message = f"Upload acknowledged; waiting for hardware (state: {state or '?'})…"
                self._update_progress_text()
            else:
                # SysCon acknowledged the upload, and the hardware is not uploading or busy,
                # so the sequence is ready ("idle" is the usual state; others are accepted
                # rather than leaving the panel stuck on a state name we do not know).
                self._set_execution_progress(
                    "idle",
                    f"Upload complete (hardware state: {state or 'unknown'}). Press Run to start.",
                )
        elif self._execution_phase in {"awaiting_busy", "running"} and state == "busy":
            self._set_execution_progress("running", "Sequence running (hardware Busy).")
        elif self._execution_phase == "running" and state == "idle":
            self._set_execution_progress(
                "idle", "Sequence finished (hardware returned to Idle)."
            )
        elif self._execution_phase == "awaiting_busy" and state == "idle":
            # A short auto-start run may finish between polls; do not claim measured completion.
            if self._execution_options[0] == "auto":
                self._set_execution_progress(
                    "idle",
                    "Run accepted; hardware is Idle. A short run may have finished between polls.",
                )
        elif self._execution_phase != "idle" and state == "disconnected":
            self._execution_cancel.set()
            self._execution_serial += 1
            self._set_execution_progress(
                "idle", "Hardware disconnected; execution status unknown."
            )

    def _on_holo4d_stop_clicked(self) -> None:
        # Invalidate any pending upload completion so Stop can never be followed by Run.
        self._execution_cancel.set()
        self._execution_serial += 1
        self._set_execution_progress("stopping", "Stop requested…")
        self._submit(
            f"holo4d.execute.{self._execution_serial}.stop",
            lambda client: client.holo4d.stop_sequence(),
        )

    # ---- Laser ----

    def _on_laser_advanced_toggled(self, checked: bool) -> None:
        self.laser_advanced_panel.setVisible(checked)
        self.btn_toggle_laser_advanced.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def _on_refresh_lightsources_clicked(self) -> None:
        def task(client: SysCon2Client) -> object:
            return client.laser.get_lightsources()

        self._submit("laser.get_lightsources", task)

    def _on_lightsources(self, result: object) -> None:
        self._lightsources = result if isinstance(result, dict) else {}
        previous = self.description_combo.currentData()
        self.description_combo.clear()
        for device, descriptions in sorted(self._lightsources.items()):
            for description in descriptions:
                label = (
                    re.sub(
                        r"\bdummy(?:_\d+)*\s*", "", description, flags=re.IGNORECASE
                    ).strip()
                    or "Laser"
                )
                self.description_combo.addItem(label, (device, description))
        preferred = self.description_combo.findData(previous) if previous else -1
        if preferred < 0:
            preferred = next(
                (
                    i
                    for i in range(self.description_combo.count())
                    if "940" in self.description_combo.itemText(i)
                ),
                0,
            )
        self.description_combo.setCurrentIndex(preferred)
        if self._discover_for_status:
            self._discover_for_status = False
            if self.description_combo.count():
                self._on_refresh_status_clicked()
            else:
                self.lightsource_status_label.setText(
                    "No lightsources reported by SysCon2."
                )

    def _on_device_changed(self, _device_sn: str) -> None:
        # Device identity is kept as item data in the single lightsource selector.
        pass

    def _selected_device_and_description(self) -> tuple[str, str] | None:
        selection = self.description_combo.currentData()
        return (
            selection if isinstance(selection, tuple) and len(selection) == 2 else None
        )

    def _on_set_emission_clicked(self, on: bool) -> None:
        selection = self._selected_device_and_description()
        if selection is None:
            return
        device_sn, description = selection
        verb = "ON" if on else "OFF"
        if not self._confirm(
            f"Turn laser emission {verb} for {description} ({device_sn})?"
        ):
            return

        def task(client: SysCon2Client) -> object:
            client.laser.set_emission(device_sn, description, on)
            return None

        self._submit("laser.set_emission", task)

    def _on_set_shutter_clicked(self, open_: bool) -> None:
        selection = self._selected_device_and_description()
        if selection is None:
            return
        device_sn, description = selection
        if open_ and not self._confirm(
            f"Open the laser shutter for {description} ({device_sn})?"
        ):
            return

        def task(client: SysCon2Client) -> object:
            client.laser.set_shutter(device_sn, open_)
            return None

        self._submit("laser.set_shutter", task)

    def _on_set_intensity_clicked(self) -> None:
        selection = self._selected_device_and_description()
        if selection is None:
            return
        device_sn, description = selection
        intensity = self.intensity_spinbox.value()
        if not self._confirm(
            f"Set intensity to {intensity} for {description} ({device_sn})?"
        ):
            return

        def task(client: SysCon2Client) -> object:
            client.laser.set_intensity(device_sn, description, intensity)
            return None

        self._submit("laser.set_intensity", task)

    def _on_refresh_status_clicked(self) -> None:
        if not self.description_combo.count():
            self._discover_for_status = True
            self._on_refresh_lightsources_clicked()
            return
        selection = self._selected_device_and_description()
        if selection is None:
            return
        device_sn, description = selection

        def task(client: SysCon2Client) -> object:
            return client.laser.get_lightsource_status(device_sn, description)

        self._submit("laser.get_status", task)

    def _on_lightsource_status(self, result: object) -> None:
        if not isinstance(result, LightsourceStatus):
            return
        self.lightsource_status_label.setText(
            f"Connected: {result.connected} | Emission: {result.emission} | "
            f"Shutter open: {result.shutter_open} | Intensity: {result.intensity} | "
            f"Mode: {result.mode}"
        )
