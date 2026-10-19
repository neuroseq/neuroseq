"""ScanImage acquisition controls, hosted in the Image acquisition dialog."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.gui_settings import GuiSettings


class ScanImageControls(QtWidgets.QWidget):
    record_requested = QtCore.Signal(object)
    live_changed = QtCore.Signal(bool)
    channel_changed = QtCore.Signal(int)

    def __init__(
        self,
        connection: ScanImageConnection,
        settings: GuiSettings,
        parent=None,
        *,
        owns_run: Callable[[str | None], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.connection = connection
        # Only the controls whose workspace started a run arm and act on its
        # first-frame timeout; every workspace sees the shared connection.
        self._owns_run = owns_run or (lambda run_id: run_id is not None)
        # Remembered across sessions in NeuroSeq's config file, like the Fiji path.
        self.settings = settings
        self._last_frame_at: float | None = None
        self._waiting_since: float | None = None
        self._last_frame_number = None
        self._frame_timing: dict | None = None
        layout = QtWidgets.QVBoxLayout(self)
        connection_row = QtWidgets.QHBoxLayout()
        self.python_edit = QtWidgets.QLineEdit(settings.scanimage_python_path)
        self.python_edit.setPlaceholderText("MATLAB-compatible Python executable")
        self.python_edit.setToolTip(
            "Python containing the MATLAB Engine matching your MATLAB release, numpy "
            "and tifffile. Remembered as the default for next time."
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
            settings.scanimage_recording_directory
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
        # One checkbox per ScanImage channel, filled in once connected.
        self.save_channels_widget = QtWidgets.QWidget()
        self._save_channels_layout = QtWidgets.QHBoxLayout(self.save_channels_widget)
        self._save_channels_layout.setContentsMargins(0, 0, 0, 0)
        self.save_channel_checks: list[QtWidgets.QCheckBox] = []
        self._save_channels_hint = QtWidgets.QLabel("Connect to choose")
        self._save_channels_layout.addWidget(self._save_channels_hint)
        self._save_channels_layout.addStretch(1)
        self._save_channels_chosen = False
        self.save_channels_widget.setToolTip(
            "Channels saved in the recording; several channels are saved "
            "together. Starts as set in ScanImage, which keeps its own choice."
        )
        form.addRow("Save channels", self.save_channels_widget)
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
        actions = QtWidgets.QHBoxLayout()
        self.live_check = QtWidgets.QCheckBox("Live view")
        self.live_check.setChecked(connection.live_preview)
        # The checkbox edits the connection's shared preview state; every
        # workspace's checkbox and video follow it via live_preview_changed.
        self.live_check.toggled.connect(self._live_toggled)
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(1, 32)
        self.channel_spin.valueChanged.connect(self._channel)
        self.average_spin = QtWidgets.QSpinBox()
        self.average_spin.setRange(0, 50)
        self.average_spin.setSpecialValueText("as ScanImage")
        self.average_spin.setToolTip(
            "Average the last frames of the preview, like ScanImage's display "
            "rolling average. Recordings are not affected."
        )
        self.average_spin.valueChanged.connect(
            lambda value: setattr(connection, "preview_average", value or None)
        )
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
            QtWidgets.QLabel("Average"),
            self.average_spin,
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
        # Errors carry MATLAB file/line details worth pasting into a bug report.
        for label in (self.message_label, self.status_label, self.frame_label):
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        connection.connected_changed.connect(self._connected)
        connection.status_changed.connect(self._status)
        connection.frame_received.connect(self._frame)
        connection.error.connect(self.show_message)
        connection.live_preview_changed.connect(self._live_preview_changed)
        self._update_connected_ui(connection.connected)
        self._freshness_timer = QtCore.QTimer(self)
        self._freshness_timer.setInterval(1000)
        self._freshness_timer.timeout.connect(self._freshness)
        self._freshness_timer.start()

    def dispose(self) -> None:
        """Stop following the shared connection once the workspace closes."""
        self._freshness_timer.stop()
        for signal, slot in (
            (self.connection.connected_changed, self._connected),
            (self.connection.status_changed, self._status),
            (self.connection.frame_received, self._frame),
            (self.connection.error, self.show_message),
            (self.connection.live_preview_changed, self._live_preview_changed),
        ):
            with contextlib.suppress(TypeError, RuntimeError):
                signal.disconnect(slot)

    def _live_toggled(self, checked: bool) -> None:
        self.connection.live_preview = checked

    def _live_preview_changed(self, live: bool) -> None:
        with QtCore.QSignalBlocker(self.live_check):
            self.live_check.setChecked(live)
        self.live_changed.emit(live)

    def _browse_python(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "MATLAB-compatible Python executable"
        )
        if path:
            self.python_edit.setText(path)
            self.settings.scanimage_python_path = path

    def _browse_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder")
        if path:
            self.directory_edit.setText(path)
            self.settings.scanimage_recording_directory = path

    def _connect(self) -> None:
        python = self.python_edit.text().strip()
        if not Path(python).is_file():
            self.show_message(
                "Select a Python executable containing MATLAB Engine, numpy and tifffile."
            )
            return
        self.settings.scanimage_python_path = python
        self.connect_button.setEnabled(False)
        self.status_label.setText("Connecting to the named MATLAB session…")
        self.connection.connect_session(python, self.session_edit.text().strip())

    def _connected(self, connected: bool) -> None:
        self._update_connected_ui(connected)
        if not connected:
            self.connection.live_preview = False
            self._waiting_since = None

    def _update_connected_ui(self, connected: bool) -> None:
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

    def _channel(self, channel: int) -> None:
        self.connection.channel = channel
        self.channel_changed.emit(channel)

    def _preview(self) -> None:
        if self.connection.command("preview"):
            self.live_check.setChecked(True)
            self._last_frame_number = None

    def saved_channels(self) -> list[int]:
        return [
            number
            for number, check in enumerate(self.save_channel_checks, start=1)
            if check.isChecked()
        ]

    def _show_save_channels(self, status: dict) -> None:
        available = status.get("channels_available")
        if not isinstance(available, int) or available < 1:
            return
        if len(self.save_channel_checks) != available:
            for check in self.save_channel_checks:
                check.deleteLater()
            self.save_channel_checks = []
            self._save_channels_hint.hide()
            for number in range(1, available + 1):
                check = QtWidgets.QCheckBox(str(number))
                check.toggled.connect(self._save_channels_toggled)
                self._save_channels_layout.insertWidget(number - 1, check)
                self.save_channel_checks.append(check)
        if not self._save_channels_chosen:
            # Follow ScanImage's selection until the user picks channels here.
            saved = status.get("saved_channels", [])
            saved = {int(saved)} if isinstance(saved, int) else {int(c) for c in saved}
            for number, check in enumerate(self.save_channel_checks, start=1):
                check.blockSignals(True)
                check.setChecked(number in saved)
                check.blockSignals(False)

    def _save_channels_toggled(self, _checked: bool) -> None:
        self._save_channels_chosen = True

    def _record(self) -> None:
        directory = Path(self.directory_edit.text().strip())
        if not self.directory_edit.text().strip() or not directory.is_dir():
            self.show_message("Select an existing recording folder on this PC.")
            return
        channels = self.saved_channels()
        if self.save_channel_checks and not channels:
            self.show_message("Tick at least one channel to save.")
            return
        run_id = uuid.uuid4().hex
        self.settings.scanimage_recording_directory = str(directory)
        self.record_requested.emit(
            {
                "run_id": run_id,
                "directory": str(directory / run_id),
                "frames": self.frames_spin.value(),
                "external_trigger": self.external_check.isChecked(),
                "name": self.name_edit.text().strip() or "Acquisition",
                **({"channels": channels} if channels else {}),
            }
        )

    def _status(self, status: dict) -> None:
        self._show_save_channels(status)
        state = status.get("acq_state", "unknown")
        run = status.get("run", {})
        text = f"ScanImage {status.get('scanimage_version', '')}: {state}; acquisition: {run.get('state', 'idle')}"
        if status.get("preview_problem"):
            # Why no preview frame came, e.g. a display read ScanImage rejected.
            text += f"\nNo preview frame: {status['preview_problem']}"
        self.status_label.setText(text)
        idle = state == "idle"
        for widget in (self.record_button, self.preview_button, self.disconnect_button):
            widget.setEnabled(idle and self.connection.run_id is None)
        self.abort_button.setEnabled(not idle and status.get("can_abort", False))
        if (
            run.get("state") == "waiting_for_first_frame"
            and run.get("id") == self.connection.run_id
            and self._owns_run(run.get("id"))
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
        self._frame_timing = frame
        self._freshness()

    def _freshness(self) -> None:
        if self._last_frame_at is not None:
            age = time.monotonic() - self._last_frame_at
            text = f"Preview frame {self._last_frame_number or '—'} · received {age:.0f} s ago"
            timing = self._frame_timing or {}
            shape = timing.get("pixels", np.empty((0, 0))).shape
            parts = []
            if timing.get("rate_hz"):
                parts.append(f"{timing['rate_hz']:.1f} frames/s")
            if timing.get("roundtrip_ms") is not None:
                parts.append(f"{timing['roundtrip_ms']:.0f} ms per request")
            if timing.get("averaged", 1) > 1:
                parts.append(f"mean of {timing['averaged']} frames")
            if timing.get("matlab_ms") is not None:
                parts.append(f"{timing['matlab_ms']:.0f} ms in MATLAB")
            if shape[0]:
                parts.append(f"{shape[1]}×{shape[0]} px")
            if parts:
                text += "\n" + " · ".join(parts)
            self.frame_label.setText(text)
        if (
            self._waiting_since is not None
            and time.monotonic() - self._waiting_since > self.timeout_spin.value()
        ):
            self._waiting_since = None
            if not self._owns_run(self.connection.run_id):
                return
            self.connection.command("abort")
            self.status_label.setText("First-frame timeout; abort requested.")

    def show_message(self, message: str) -> None:
        """Show a user message independently of periodic acquisition status."""
        self.message_label.setText(message)
