"""ScanImage acquisition controls: the "ScanImage (imaging)" tab of the Stimulation setup pane.

Controls only: NeuroSeq does not stream frames (issues #628, #651); ScanImage shows its own
display. Each recording is one TIFF in the recording folder, named after the date and a
recording number, with a small JSON metadata file beside it.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.device_control.scanimage_files import next_recording_number
from neuroseq.device_control.scanimage_files import recording_file_name
from neuroseq.device_control.scanimage_files import recording_stem
from neuroseq.gui_settings import GuiSettings


class ScanImageControls(QtWidgets.QWidget):
    record_requested = QtCore.Signal(object)
    channel_changed = QtCore.Signal(int)
    #: The user picked another recording folder (the path as a string).
    directory_changed = QtCore.Signal(str)

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
        self._waiting_since: float | None = None
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
        self.directory_edit.setToolTip(
            "Every recording is saved directly in this folder as one TIFF named "
            "<date>_<number>.tif, with its metadata in <date>_<number>.json."
        )
        self.directory_edit.editingFinished.connect(self._directory_edited)
        directory_row = QtWidgets.QHBoxLayout()
        directory_row.addWidget(self.directory_edit)
        directory_button = QtWidgets.QToolButton()
        directory_button.setText("…")
        directory_button.clicked.connect(self._browse_directory)
        directory_row.addWidget(directory_button)
        form.addRow("Recording folder", directory_row)
        self.next_file_label = QtWidgets.QLabel()
        form.addRow("Next file", self.next_file_label)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Same as the file name")
        self.name_edit.setToolTip("Name of the recording's data group in the project")
        form.addRow("Experiment name", self.name_edit)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(1, 10000000)
        self.frames_spin.setValue(300)
        self.frames_spin.setToolTip(
            "Frames of a manual recording. Dynamic input routines set the number "
            "of frames of each block themselves."
        )
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
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(1, 32)
        self.channel_spin.setToolTip(
            "Channel of the calcium indicator: shown for recordings in the Imaging "
            "tab and measured by dynamic input routines."
        )
        self.channel_spin.valueChanged.connect(self.channel_changed)
        form.addRow("Functional channel", self.channel_spin)
        self.external_check = QtWidgets.QCheckBox(
            "Wait for external acquisition trigger"
        )
        self.external_check.setToolTip(
            "Configure the input terminal and polarity in ScanImage. Dynamic input "
            "routines always wait for the trigger sent by the stimulation sequence."
        )
        form.addRow(self.external_check)
        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(1, 3600)
        self.timeout_spin.setValue(60)
        self.timeout_spin.setSuffix(" s")
        form.addRow("First-frame timeout", self.timeout_spin)
        layout.addLayout(form)
        actions = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Focus")
        self.preview_button.setToolTip(
            "Start ScanImage's Focus mode; the image is shown in ScanImage."
        )
        self.preview_button.clicked.connect(self._preview)
        self.record_button = QtWidgets.QPushButton("Record / Arm")
        self.record_button.clicked.connect(self._record)
        self.abort_button = QtWidgets.QPushButton("Abort acquisition")
        self.abort_button.clicked.connect(lambda: connection.command("abort"))
        for action_widget in (
            self.preview_button,
            self.record_button,
            self.abort_button,
        ):
            actions.addWidget(action_widget)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        # Errors carry MATLAB file/line details worth pasting into a bug report.
        for label in (self.message_label, self.status_label):
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        connection.connected_changed.connect(self._connected)
        connection.status_changed.connect(self._status)
        connection.error.connect(self.show_message)
        connection.run_finished.connect(self._update_next_file)
        self._update_connected_ui(connection.connected)
        self._update_next_file()
        self._timeout_timer = QtCore.QTimer(self)
        self._timeout_timer.setInterval(1000)
        self._timeout_timer.timeout.connect(self._check_first_frame_timeout)
        self._timeout_timer.start()

    def dispose(self) -> None:
        """Stop following the shared connection once the workspace closes."""
        self._timeout_timer.stop()
        for signal, slot in (
            (self.connection.connected_changed, self._connected),
            (self.connection.status_changed, self._status),
            (self.connection.error, self.show_message),
            (self.connection.run_finished, self._update_next_file),
        ):
            with contextlib.suppress(TypeError, RuntimeError):
                signal.disconnect(slot)

    def set_directory(self, directory: str) -> None:
        """Show the recording folder of the workspace's project."""
        self.directory_edit.setText(directory)
        self._update_next_file()

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
            self._directory_edited()

    def _directory_edited(self) -> None:
        path = self.directory_edit.text().strip()
        if path:
            self.settings.scanimage_recording_directory = path
            self.directory_changed.emit(path)
        self._update_next_file()

    def _update_next_file(self, *_args: object) -> None:
        text = self.directory_edit.text().strip()
        if not text:
            self.next_file_label.setText("Select a recording folder")
            return
        stem = recording_stem()
        number = next_recording_number(Path(text), stem)
        self.next_file_label.setText(recording_file_name(stem, number))

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

    def _preview(self) -> None:
        self.connection.command("preview")

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

    def recording_request(
        self, *, frames: int, external_trigger: bool, name: str = ""
    ) -> dict:
        """Arguments of a ``record`` command; raises ``ValueError`` with a user message."""
        text = self.directory_edit.text().strip()
        directory = Path(text)
        if not text or not directory.is_dir():
            raise ValueError("Select an existing recording folder on this PC.")
        channels = self.saved_channels()
        if self.save_channel_checks and not channels:
            raise ValueError("Tick at least one channel to save.")
        stem = recording_stem()
        number = next_recording_number(directory, stem)
        self.settings.scanimage_recording_directory = str(directory)
        return {
            "run_id": uuid.uuid4().hex,
            "directory": str(directory),
            "stem": stem,
            "counter": number,
            "frames": int(frames),
            "external_trigger": bool(external_trigger),
            "name": name or Path(recording_file_name(stem, number)).stem,
            **({"channels": channels} if channels else {}),
        }

    def _record(self) -> None:
        try:
            request = self.recording_request(
                frames=self.frames_spin.value(),
                external_trigger=self.external_check.isChecked(),
                name=self.name_edit.text().strip(),
            )
        except ValueError as exc:
            self.show_message(str(exc))
            return
        self.record_requested.emit(request)

    def _status(self, status: dict) -> None:
        self._show_save_channels(status)
        state = status.get("acq_state", "unknown")
        run = status.get("run", {})
        self.status_label.setText(
            f"ScanImage {status.get('scanimage_version', '')}: {state}; "
            f"acquisition: {run.get('state', 'idle')}"
        )
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

    def _check_first_frame_timeout(self) -> None:
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
