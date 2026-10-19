"""Workspace-local ScanImage acquisition and live-image lifecycle."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Callable
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.device_control.scanimage_controls import ScanImageControls

if TYPE_CHECKING:
    from neuroseq.device_control.scanimage_connection import ScanImageConnection
    from neuroseq.export.syscon2_export_service import SysconExportSettings
    from neuroseq.gui_settings import GuiSettings
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget


class WorkspaceAcquisitionController(QtCore.QObject):
    """Own acquisition state without owning the shared physical connection."""

    recording_added = QtCore.Signal()

    def __init__(
        self,
        connection: ScanImageConnection,
        settings: GuiSettings,
        video_widget: CalciumVideoWidget,
        *,
        project: Callable[[], Project],
        current_stim: Callable[[], StimArray | None],
        export_settings: Callable[[], SysconExportSettings | None],
        video_widgets: Callable[[], Sequence[CalciumVideoWidget]],
        show_video: Callable[[], None],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.connection = connection
        self.video_widget = video_widget
        self._project = project
        self._current_stim = current_stim
        self._export_settings = export_settings
        self._video_widgets = video_widgets
        self._show_video = show_video
        self.run_ids: set[str] = set()
        self._live_frames_connected = False
        self._disposed = False
        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("Image acquisition")
        self.controls = ScanImageControls(connection, settings, self.dialog)
        QtWidgets.QVBoxLayout(self.dialog).addWidget(self.controls)
        self.controls.live_changed.connect(self.set_scanimage_live)
        self.controls.channel_changed.connect(self.set_scanimage_channel)
        self.controls.record_requested.connect(self.start_scanimage_recording)
        connection.status_changed.connect(self.scanimage_status)
        connection.recording_ready.connect(self.scanimage_recording_ready)
        connection.run_finished.connect(self._forget_run)
        self.set_active(True)

    @property
    def acquisition_in_progress(self) -> bool:
        """Whether a run started here is still being tracked by a live connection.

        A run whose helper died stays tracked by the connection so a reconnect
        can recover it, but nothing can progress while disconnected, so it
        does not keep the workspace open.
        """
        return self.connection.connected and self.connection.run_id in self.run_ids

    def set_active(self, active: bool) -> None:
        """Only the visible workspace consumes shared live frames."""
        if self._disposed or active == self._live_frames_connected:
            return
        if active:
            self.connection.frame_received.connect(self.video_widget.show_live_frame)
        else:
            self.connection.frame_received.disconnect(self.video_widget.show_live_frame)
        self._live_frames_connected = active

    def dispose(self) -> None:
        """Disconnect this workspace without stopping the shared device."""
        if self._disposed:
            return
        self.set_active(False)
        self._disposed = True
        self.controls.dispose()
        self.dialog.close()
        self.connection.status_changed.disconnect(self.scanimage_status)
        self.connection.recording_ready.disconnect(self.scanimage_recording_ready)
        self.connection.run_finished.disconnect(self._forget_run)

    def _forget_run(self, run_id: str) -> None:
        self.run_ids.discard(run_id)

    def start_scanimage_recording(self, arguments: dict) -> None:
        if self.connection.run_id is not None:
            self.controls.show_message("An acquisition is already being finalized.")
            return
        project = self._project()
        record = dict(arguments)
        record.update(
            status="requested",
            requested_at=datetime.now(UTC).isoformat(),
            synchronization="unverified",
        )
        stim = self._current_stim()
        if stim is not None:
            values = np.ascontiguousarray(stim.values)
            record["stimulus_snapshot"] = {
                "name": stim.name,
                "dt": stim.dt,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "data": base64.b64encode(values.tobytes()).decode(),
            }
        settings = self._export_settings()
        if settings is not None:
            record["syscon_export_settings"] = json.loads(
                json.dumps(dataclasses.asdict(settings), default=str)
            )
        if self.connection.command("record", {**arguments, "manifest": record}):
            project.imaging_acquisitions.append(record)
            project.mark_dirty()
            self.run_ids.add(arguments["run_id"])
            self.controls.live_check.setChecked(True)

    def scanimage_status(self, status: dict) -> None:
        run = status.get("run", {})
        if run.get("id") not in self.run_ids:
            return
        project = self._project()
        record = next(
            r for r in project.imaging_acquisitions if r["run_id"] == run["id"]
        )
        previous = record.get("status")
        record["status"] = run.get("state", "unknown")
        if "error" in run:
            record["error"] = run["error"]
        elif record["status"] in {"recording", "completed", "aborted"}:
            record.pop("error", None)
        for source, target in (
            ("matlab_version", "matlab_version"),
            ("scanimage_version", "scanimage_version"),
            ("settings", "scanimage_settings"),
            ("files", "files"),
        ):
            if source in status:
                record[target] = status[source]
        if record["status"] != previous:
            project.mark_dirty()

    def open_image_acquisition_dialog(self) -> None:
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def set_scanimage_live(self, live: bool) -> None:
        self.video_widget.set_live(live)
        if live:
            self._show_video()
        if not live:
            self.set_scanimage_channel(self.controls.channel_spin.value())

    def set_scanimage_channel(self, channel: int) -> None:
        index = self._project().calcium_scanimage_index
        if index is not None and not self.controls.live_check.isChecked():
            channel = self.recorded_scanimage_channel(index, channel)
        for widget in self._video_widgets():
            widget.set_recorded_channel(channel)

    def recorded_scanimage_channel(self, index: dict | None, channel: int) -> int:
        if index is not None and channel not in index["channels"]:
            selected = index["channels"][0]
            self.controls.show_message(
                f"Channel {channel} was not saved in this recording; showing channel {selected}."
            )
            self.controls.channel_spin.blockSignals(True)
            self.controls.channel_spin.setValue(selected)
            self.controls.channel_spin.blockSignals(False)
            self.connection.channel = selected
            return selected
        return channel

    def scanimage_recording_ready(self, result: dict) -> None:
        if result["run_id"] not in self.run_ids:
            return
        self.run_ids.remove(result["run_id"])
        project = self._project()
        record = next(
            r for r in project.imaging_acquisitions if r["run_id"] == result["run_id"]
        )
        record.update(
            status=result["status"],
            recording=result,
            finished_at=datetime.now(UTC).isoformat(),
        )
        record.pop("error", None)
        name = record["name"]
        if name in project.experiments:
            name = f"{name} ({result['run_id'][:8]})"
        project.experiments[name] = Experiment(
            name=name, duration=result["duration_s"], timepoints=[], data=[]
        )
        record["experiment_name"] = name
        project.add_calcium_video(
            [pathlib.Path(p) for p in result["paths"]],
            result["fps"],
            name=name,
            run_id=result["run_id"],
        )
        self.controls.show_message(f"Recording {name} added to the imaging data.")
        project.mark_dirty()
        project.update_limits()
        self.controls.live_check.setChecked(False)
        self.recording_added.emit()
        # Movie availability does not imply extracted traces are ready. Do not
        # call SysCon2StimulationHost.acquisition_imported here.
