"""Acquisition controls embedded above the calcium-video canvas."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.scanimage_connection import ScanImageConnection


class ScanImageControls(QtWidgets.QGroupBox):
    record_requested = QtCore.Signal(object)
    live_changed = QtCore.Signal(bool)
    channel_changed = QtCore.Signal(int)
    recording_selected = QtCore.Signal(str)

    def __init__(self, connection: ScanImageConnection, parent=None) -> None:
        super().__init__("ScanImage acquisition", parent)
        self.connection = connection
        self._last_frame_at: float | None = None
        self._waiting_since: float | None = None
        self._last_frame_number = None
        layout = QtWidgets.QVBoxLayout(self)
        settings = QtCore.QSettings()
        connection_row = QtWidgets.QHBoxLayout()
        self.python_edit = QtWidgets.QLineEdit(
            settings.value("scanimage/python", "", type=str)
        )
        self.python_edit.setPlaceholderText("MATLAB-compatible Python executable")
        self.python_edit.setToolTip(
            "Python containing the MATLAB Engine matching your MATLAB release, numpy and tifffile"
        )
        browse = QtWidgets.QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse_python)
        self.session_edit = QtWidgets.QLineEdit("NeuroSeqScanImage")
        self.session_edit.setToolTip(
            "Name passed to matlab.engine.shareEngine in ScanImage's MATLAB window"
        )
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect)
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(lambda: connection.command("disconnect"))
        for widget in (
            self.python_edit,
            browse,
            self.session_edit,
            self.connect_button,
            self.disconnect_button,
        ):
            connection_row.addWidget(widget)
        layout.addLayout(connection_row)
        form = QtWidgets.QFormLayout()
        self.directory_edit = QtWidgets.QLineEdit(
            settings.value("scanimage/directory", "", type=str)
        )
        directory_row = QtWidgets.QHBoxLayout()
        directory_row.addWidget(self.directory_edit)
        directory_button = QtWidgets.QToolButton()
        directory_button.setText("…")
        directory_button.clicked.connect(self._browse_directory)
        directory_row.addWidget(directory_button)
        form.addRow("Recording folder", directory_row)
        self.name_edit = QtWidgets.QLineEdit("Acquisition")
        form.addRow("Experiment name", self.name_edit)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(1, 10000000)
        self.frames_spin.setValue(300)
        form.addRow("Frames", self.frames_spin)
        self.external_check = QtWidgets.QCheckBox(
            "Wait for external acquisition trigger"
        )
        self.external_check.setToolTip(
            "Configure the input terminal and polarity in ScanImage. This does not start SysCon."
        )
        form.addRow(self.external_check)
        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(1, 3600)
        self.timeout_spin.setValue(60)
        self.timeout_spin.setSuffix(" s")
        form.addRow("First-frame timeout", self.timeout_spin)
        layout.addLayout(form)
        self.recordings_combo = QtWidgets.QComboBox()
        self.recordings_combo.addItem("Recorded acquisitions", "")
        self.recordings_combo.activated.connect(
            lambda _: self.recording_selected.emit(
                self.recordings_combo.currentData() or ""
            )
        )
        layout.addWidget(self.recordings_combo)
        actions = QtWidgets.QHBoxLayout()
        self.live_check = QtWidgets.QCheckBox("Live view")
        self.live_check.toggled.connect(self.live_changed.emit)
        self.live_check.toggled.connect(
            lambda checked: setattr(connection, "live_preview", checked)
        )
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(1, 32)
        self.channel_spin.valueChanged.connect(self._channel)
        self.preview_button = QtWidgets.QPushButton("Preview (Focus)")
        self.preview_button.clicked.connect(self._preview)
        self.record_button = QtWidgets.QPushButton("Record / Arm")
        self.record_button.clicked.connect(self._record)
        self.abort_button = QtWidgets.QPushButton("Abort acquisition")
        self.abort_button.clicked.connect(lambda: connection.command("abort"))
        for action_widget in (
            self.live_check,
            QtWidgets.QLabel("Channel"),
            self.channel_spin,
            self.preview_button,
            self.record_button,
            self.abort_button,
        ):
            actions.addWidget(action_widget)
        layout.addLayout(actions)
        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.frame_label = QtWidgets.QLabel("No live frame received")
        layout.addWidget(self.frame_label)
        connection.connected_changed.connect(self._connected)
        connection.status_changed.connect(self._status)
        connection.frame_received.connect(self._frame)
        connection.error.connect(self.show_message)
        self._connected(False)
        self._freshness_timer = QtCore.QTimer(self)
        self._freshness_timer.setInterval(1000)
        self._freshness_timer.timeout.connect(self._freshness)
        self._freshness_timer.start()

    def _browse_python(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "MATLAB-compatible Python executable"
        )
        if path:
            self.python_edit.setText(path)

    def set_recordings(self, records: list[dict]) -> None:
        self.recordings_combo.clear()
        self.recordings_combo.addItem("Recorded acquisitions", "")
        for record in records:
            if "recording" in record:
                self.recordings_combo.addItem(
                    f"{record['name']} · {record['run_id'][:8]} · {record['status']}",
                    record["run_id"],
                )

    def _browse_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder")
        if path:
            self.directory_edit.setText(path)

    def _connect(self) -> None:
        python = self.python_edit.text().strip()
        if not Path(python).is_file():
            self.show_message(
                "Select a Python executable containing MATLAB Engine, numpy and tifffile."
            )
            return
        QtCore.QSettings().setValue("scanimage/python", python)
        self.connect_button.setEnabled(False)
        self.status_label.setText("Connecting to the named MATLAB session…")
        self.connection.connect_session(python, self.session_edit.text().strip())

    def _connected(self, connected: bool) -> None:
        self.connect_button.setEnabled(not connected)
        self.python_edit.setEnabled(not connected)
        self.session_edit.setEnabled(not connected)
        for widget in (
            self.disconnect_button,
            self.preview_button,
            self.record_button,
            self.abort_button,
        ):
            widget.setEnabled(connected)
        if not connected:
            self.live_check.setChecked(False)
            self._waiting_since = None

    def _channel(self, channel: int) -> None:
        self.connection.channel = channel
        self.channel_changed.emit(channel)

    def _preview(self) -> None:
        if self.connection.command("preview"):
            self.live_check.setChecked(True)
            self._last_frame_number = None

    def _record(self) -> None:
        directory = Path(self.directory_edit.text().strip())
        if not self.directory_edit.text().strip() or not directory.is_dir():
            self.show_message("Select an existing recording folder on this PC.")
            return
        run_id = uuid.uuid4().hex
        QtCore.QSettings().setValue("scanimage/directory", str(directory))
        self.record_requested.emit(
            {
                "run_id": run_id,
                "directory": str(directory / run_id),
                "frames": self.frames_spin.value(),
                "external_trigger": self.external_check.isChecked(),
                "name": self.name_edit.text().strip() or "Acquisition",
            }
        )

    def _status(self, status: dict) -> None:
        state = status.get("acq_state", "unknown")
        run = status.get("run", {})
        self.status_label.setText(
            f"ScanImage {status.get('scanimage_version', '')}: {state}; acquisition: {run.get('state', 'idle')}"
        )
        idle = state == "idle"
        for widget in (self.record_button, self.preview_button, self.disconnect_button):
            widget.setEnabled(idle and self.connection.run_id is None)
        self.abort_button.setEnabled(not idle and status.get("can_abort", False))
        if (
            run.get("state") == "waiting_for_first_frame"
            and run.get("id") == self.connection.run_id
        ):
            if self._waiting_since is None:
                self._waiting_since = time.monotonic()
        else:
            self._waiting_since = None

    def _frame(self, frame: dict) -> None:
        number = frame.get("number")
        if number is None or number != self._last_frame_number:
            self._last_frame_at = time.monotonic()
            self._last_frame_number = number
        self._freshness()

    def _freshness(self) -> None:
        if self._last_frame_at is not None:
            age = time.monotonic() - self._last_frame_at
            self.frame_label.setText(
                f"Preview frame {self._last_frame_number or '—'} · received {age:.0f} s ago"
            )
        if (
            self._waiting_since is not None
            and time.monotonic() - self._waiting_since > self.timeout_spin.value()
        ):
            self._waiting_since = None
            self.connection.command("abort")
            self.status_label.setText("First-frame timeout; abort requested.")

    def show_message(self, message: str) -> None:
        """Show a user message independently of periodic acquisition status."""
        self.message_label.setText(message)
