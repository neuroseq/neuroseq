"""GUI panel for live SysCon2 control: connection, sequence manager, Holo4D, and laser.

Only talks to SysCon2 on the same machine (see the project plan for the deferred
network-bridge transport). Hazardous actions (emission, intensity, sequence run) go through a
confirmation dialog first.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.syscon2_client import LightsourceStatus
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_connection_manager import SysCon2ConnectionManager

logger = logging.getLogger(__name__)

_START_CONDITIONS = ["auto", "trigger1", "trigger2"]
_TRIGGER_BEHAVIORS = ["rising", "falling"]


class SysCon2Panel(QtWidgets.QWidget):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        manager: SysCon2ConnectionManager | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._manager = manager if manager is not None else SysCon2ConnectionManager()
        self._connected = False
        self._lightsources: dict[str, list[str]] = {}
        self._result_handlers: dict[str, Callable[[object], None]] = {
            "sequence_manager.get_sequences": self._on_sequences,
            "laser.get_lightsources": self._on_lightsources,
            "laser.get_status": self._on_lightsource_status,
        }

        self._build_ui()
        self._wire_manager_signals()
        self._update_connection_ui()

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

    def _build_holo4d_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Holo4D")
        layout = QtWidgets.QVBoxLayout(group)

        self.holo4d_state_label = QtWidgets.QLabel("State: unknown")
        layout.addWidget(self.holo4d_state_label)

        form = QtWidgets.QFormLayout()
        self.runs_spinbox = QtWidgets.QSpinBox()
        self.runs_spinbox.setRange(0, 1_000_000)
        self.runs_spinbox.setSpecialValueText("Endless (0)")
        form.addRow("Runs:", self.runs_spinbox)
        self.start_condition_combo = QtWidgets.QComboBox()
        self.start_condition_combo.addItems(_START_CONDITIONS)
        form.addRow("Start condition:", self.start_condition_combo)
        self.trigger_behavior_combo = QtWidgets.QComboBox()
        self.trigger_behavior_combo.addItems(_TRIGGER_BEHAVIORS)
        form.addRow("Trigger behavior:", self.trigger_behavior_combo)
        layout.addLayout(form)

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

        form = QtWidgets.QFormLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        form.addRow("Device:", self.device_combo)
        self.description_combo = QtWidgets.QComboBox()
        form.addRow("Lightsource:", self.description_combo)
        self.intensity_spinbox = QtWidgets.QDoubleSpinBox()
        self.intensity_spinbox.setRange(0.0, 1_000_000.0)
        self.intensity_spinbox.setDecimals(3)
        form.addRow("Intensity:", self.intensity_spinbox)
        layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
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
        self.btn_set_intensity = QtWidgets.QPushButton("Set Intensity")
        self.btn_set_intensity.clicked.connect(self._on_set_intensity_clicked)
        self.btn_refresh_status = QtWidgets.QPushButton("Get Status")
        self.btn_refresh_status.clicked.connect(self._on_refresh_status_clicked)
        for btn in (
            self.btn_emission_on,
            self.btn_emission_off,
            self.btn_shutter_open,
            self.btn_shutter_close,
            self.btn_set_intensity,
            self.btn_refresh_status,
        ):
            button_row.addWidget(btn)
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

    # ---- Holo4D ----

    def _on_holo4d_upload_clicked(self) -> None:
        runs = self.runs_spinbox.value()

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
        device_sn, _description = selection

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
