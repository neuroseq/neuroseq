"""GUI panel for live SysCon2 control: connection, sequence manager, Holo4D, and laser.

Only talks to SysCon2 on the same machine (see the project plan for the deferred
network-bridge transport). Hazardous actions (emission, intensity, sequence run) go through a
confirmation dialog first.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.stim import StimArray
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.syscon2_client import LightsourceStatus
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_connection_manager import SysCon2ConnectionManager
from neuroseq.syscon2_export_service import SysconExportError
from neuroseq.syscon2_export_service import SysconExportSettings
from neuroseq.syscon2_export_service import export_stim_array
from neuroseq.syscon2_stimulation_host import SysCon2StimulationHost

if TYPE_CHECKING:
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


class SysCon2Panel(QtWidgets.QWidget):
    #: Emitted with the frozenset[int] of stim ids targeted by whichever
    #: sequence was most recently generated and sent to SysCon2 - lets a
    #: caller (gui.py) reflect it elsewhere, e.g. as crosses in the Spatial
    #: Footprints panel. Not a live hardware readout (SysCon2's SCI protocol
    #: has no "which spots are illuminated right now" query) - just "who's in
    #: the sequence currently loaded there."
    targets_changed = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        manager: SysCon2ConnectionManager | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._manager = manager if manager is not None else SysCon2ConnectionManager()
        self._project: Project | None = None
        self._connected = False
        self._lightsources: dict[str, list[str]] = {}
        self._current_stim_provider: Callable[[], StimArray | None] | None = None
        #: Export options of the last successful dialog export; reused by the one-click button.
        self._export_settings: SysconExportSettings | None = None
        self._active_dialog: StimulationGeneratorDialog | None = None
        self._result_handlers: dict[str, Callable[[object], None]] = {
            "sequence_manager.get_sequences": self._on_sequences,
            "sequence_manager.import": self._on_sequence_imported,
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

    def set_project(self, project: Project | None) -> None:
        if project is self._project:
            return
        self._project = project
        self._export_settings = None

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
        self._manager.shutdown()

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

        self.btn_send_to_syscon2 = QtWidgets.QPushButton("Send to SysCon2…")
        self.btn_send_to_syscon2.setToolTip(
            "Open the stimulation export dialog and, once it generates a script, "
            "push it straight into SysCon2's sequence manager."
        )
        self.btn_send_to_syscon2.clicked.connect(self._on_send_to_syscon2_clicked)
        layout.addWidget(self.btn_send_to_syscon2)

        self.sequence_list = QtWidgets.QListWidget()
        layout.addWidget(self.sequence_list)

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
        self.target_directory_edit = QtWidgets.QLineEdit("neuroseq")
        target_row.addWidget(self.target_directory_edit, stretch=1)
        self.btn_import_sequence = QtWidgets.QPushButton("Import")
        self.btn_import_sequence.clicked.connect(self._on_import_sequence_clicked)
        target_row.addWidget(self.btn_import_sequence)
        layout.addLayout(target_row)

        return group

    def _build_preview_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Preview")
        layout = QtWidgets.QVBoxLayout(group)
        self.preview_label = QtWidgets.QLabel("No preview yet.")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        return group

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
        advanced_layout.addWidget(self.holo4d_state_label)

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
        self.btn_holo4d_upload.clicked.connect(self._on_holo4d_upload_clicked)
        self.btn_holo4d_run = QtWidgets.QPushButton("Run")
        self.btn_holo4d_run.clicked.connect(self._on_holo4d_run_clicked)
        self.btn_holo4d_stop = QtWidgets.QPushButton("Stop")
        self.btn_holo4d_stop.clicked.connect(self._on_holo4d_stop_clicked)
        button_row.addWidget(self.btn_holo4d_upload)
        button_row.addWidget(self.btn_holo4d_run)
        button_row.addWidget(self.btn_holo4d_stop)
        layout.addLayout(button_row)

        return group

    def _build_laser_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Laser")
        layout = QtWidgets.QVBoxLayout(group)

        refresh_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_lightsources = QtWidgets.QPushButton("Refresh Lightsources")
        self.btn_refresh_lightsources.clicked.connect(
            self._on_refresh_lightsources_clicked
        )
        refresh_row.addWidget(self.btn_refresh_lightsources)
        layout.addLayout(refresh_row)

        self.btn_export_current = QtWidgets.QPushButton(
            "Export current array to SysCon2"
        )
        self.btn_export_current.setToolTip(
            "Export the input array currently shown in the editor with the settings of "
            "the last 'Send to SysCon2...' export, and import it into SysCon2. Opens the "
            "export dialog first if there are no settings yet."
        )
        self.btn_export_current.clicked.connect(self._on_export_current_clicked)
        layout.addWidget(self.btn_export_current)

        form = QtWidgets.QFormLayout()
        self.device_combo = _NoScrollComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        form.addRow("Device:", self.device_combo)
        self.description_combo = _NoScrollComboBox()
        form.addRow("Lightsource:", self.description_combo)
        self.intensity_spinbox = _NoScrollDoubleSpinBox()
        self.intensity_spinbox.setRange(0.0, 1_000_000.0)
        self.intensity_spinbox.setDecimals(3)
        form.addRow("Intensity:", self.intensity_spinbox)
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
        layout.addWidget(self.btn_toggle_laser_advanced)

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
        button_row.addWidget(self.btn_set_intensity)
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
        self._manager.command_finished.connect(self._on_command_finished)
        self._manager.command_failed.connect(self._on_command_failed)

    def _on_connected(self) -> None:
        self._connected = True
        self._update_connection_ui()

    def _on_disconnected(self, message: str) -> None:
        self._connected = False
        self._update_connection_ui()
        if message:
            self.status_label.setText(f"Disconnected: {message}")
            self.status_label.setStyleSheet("QLabel { color: red; }")

    def _update_connection_ui(self) -> None:
        can_connect_here = sys.platform == "win32"
        self.btn_connect.setEnabled(can_connect_here and not self._connected)
        self.btn_disconnect.setEnabled(self._connected)
        if not can_connect_here:
            self.btn_connect.setToolTip(
                "SysCon2 control requires neuroseq to run on the same Windows PC as SysCon2."
            )
        if self._connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        elif not self.status_label.text().startswith("Disconnected:"):
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("QLabel { color: gray; }")

    def _on_state_changed(self, state: dict) -> None:
        holo4d_state = state.get("Holo4D")
        if holo4d_state is not None:
            self.holo4d_state_label.setText(f"State: {holo4d_state}")

    def _on_command_finished(self, name: str, result: object) -> None:
        handler = self._result_handlers.get(name)
        if handler is not None:
            handler(result)

    def _on_command_failed(self, name: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(
            self, "SysCon2 command failed", f"{name}: {message}"
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
        sequences = result if isinstance(result, list) else []
        self.sequence_list.clear()
        self.sequence_list.addItems(sequences)

    def _on_sequence_imported(self, _result: object) -> None:
        self._on_refresh_sequences_clicked()

    def _selected_sequence_path(self) -> str | None:
        item = self.sequence_list.currentItem()
        return item.text() if item is not None else None

    def _on_select_sequence_clicked(self) -> None:
        path = self._selected_sequence_path()
        if not path:
            return

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.select_sequence(path)
            return None

        self._submit("sequence_manager.select", task)

    def _on_delete_sequence_clicked(self) -> None:
        path = self._selected_sequence_path()
        if not path:
            return
        if not self._confirm(
            f"Delete sequence {path!r} from SysCon2's sequence manager?"
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
            "SysCon2 sequence files (*.seq);;All files (*)",
        )
        if filename:
            self.import_path_edit.setText(filename)

    def _on_import_sequence_clicked(self) -> None:
        path = self.import_path_edit.text().strip()
        target = self.target_directory_edit.text().strip()
        if not path:
            return

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.import_sequence(path, target)
            return None

        self._submit("sequence_manager.import", task)

    def _on_send_to_syscon2_clicked(self) -> None:
        if self._project is None:
            QtWidgets.QMessageBox.warning(
                self, "No project", "No active project to export a stimulus from."
            )
            return
        dialog = StimulationGeneratorDialog(self._project, parent=self.window())
        if self._export_settings is not None:
            dialog.apply_settings(self._export_settings)
        dialog.sequence_generated.connect(self._on_dialog_sequence_generated)
        self._active_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._active_dialog = None

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
                f"{stim_array.name}_current",
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
            self._export_settings = self._active_dialog.export_settings()
        self._show_preview(overview_png_path)
        self.targets_changed.emit(targeted_stim_ids)
        target = self.target_directory_edit.text().strip()

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.import_sequence(seq_path, target)
            return None

        self._submit("sequence_manager.import", task)

    def _show_preview(self, png_path: str) -> None:
        pixmap = QtGui.QPixmap(png_path)
        if pixmap.isNull():
            self.preview_label.setText(f"Could not load preview image:\n{png_path}")
            return
        self.preview_label.setPixmap(
            pixmap.scaledToWidth(400, QtCore.Qt.TransformationMode.SmoothTransformation)
        )

    # ---- Holo4D ----

    def _on_holo4d_advanced_toggled(self, checked: bool) -> None:
        self.holo4d_advanced_panel.setVisible(checked)
        self.btn_toggle_holo4d_advanced.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def _on_holo4d_upload_clicked(self) -> None:
        runs = self.runs_spinbox.value()
        if runs == 0 and not self._confirm(
            "Upload for endless repetition? The sequence will repeat until stopped."
        ):
            return

        def task(client: SysCon2Client) -> object:
            client.holo4d.upload_sequence(runs)
            return None

        self._submit("holo4d.upload", task)

    def _on_holo4d_run_clicked(self) -> None:
        start_condition = self.start_condition_combo.currentText()
        trigger_behavior = self.trigger_behavior_combo.currentText()
        if not self._confirm("Run the currently uploaded sequence on Holo4D now?"):
            return

        def task(client: SysCon2Client) -> object:
            client.holo4d.run_sequence(start_condition, trigger_behavior)
            return None

        self._submit("holo4d.run", task)

    def _on_holo4d_stop_clicked(self) -> None:
        def task(client: SysCon2Client) -> object:
            client.holo4d.stop_sequence()
            return None

        self._submit("holo4d.stop", task)

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
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItems(sorted(self._lightsources))
        self.device_combo.blockSignals(False)
        self._on_device_changed(self.device_combo.currentText())

    def _on_device_changed(self, device_sn: str) -> None:
        self.description_combo.clear()
        self.description_combo.addItems(self._lightsources.get(device_sn, []))

    def _selected_device_and_description(self) -> tuple[str, str] | None:
        device_sn = self.device_combo.currentText()
        description = self.description_combo.currentText()
        if not device_sn or not description:
            return None
        return device_sn, description

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
