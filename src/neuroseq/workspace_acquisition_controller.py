"""Workspace-local ScanImage acquisition lifecycle.

Also records the blocks of dynamic input routines for
:class:`~neuroseq.device_control.syscon2_stimulation_host.SysCon2StimulationHost`: each block
is a triggered recording whose ΔF/F traces are extracted with the project's Cellpose masks
and deconvolved into spikes with OASIS.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import pathlib
from collections.abc import Callable
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.background_jobs import BackgroundIOWorker
from neuroseq.background_jobs import BackgroundJob
from neuroseq.data import Experiment
from neuroseq.device_control.scanimage_controls import ScanImageControls
from neuroseq.importers.importer_scanimage_traces import deconvolve_dff
from neuroseq.importers.importer_scanimage_traces import extract_recording_dff

if TYPE_CHECKING:
    from neuroseq.data import SpikeData
    from neuroseq.data import TraceData
    from neuroseq.device_control.scanimage_connection import ScanImageConnection
    from neuroseq.device_control.syscon2_stimulation_host import BlockRecording
    from neuroseq.export.syscon2_export_service import SysconExportSettings
    from neuroseq.gui_settings import GuiSettings
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget

logger = logging.getLogger(__name__)

#: Run states after which a block recording yields no data.
_FAILED_STATES = {"failed", "aborted", "interrupted", "finalization_failed", "unknown"}
#: Recording folder key in ``Project.device_folders``.
RECORDING_FOLDER = "scanimage_recording"


def stimulus_summary(stim: StimArray) -> dict:
    """What identifies a stimulus in recording metadata, without its samples."""
    values = np.ascontiguousarray(stim.values)
    return {
        "name": stim.name,
        "dt": stim.dt,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


class WorkspaceAcquisitionController(QtCore.QObject):
    """Own acquisition state without owning the shared physical connection."""

    recording_added = QtCore.Signal()
    # BlockAcquisition signals, see syscon2_stimulation_host.
    block_armed = QtCore.Signal(str)
    block_recorded = QtCore.Signal(str, object)
    block_recording_failed = QtCore.Signal(str, str)

    def __init__(
        self,
        connection: ScanImageConnection,
        settings: GuiSettings,
        *,
        project: Callable[[], Project],
        current_stim: Callable[[], StimArray | None],
        export_settings: Callable[[], SysconExportSettings | None],
        video_widgets: Callable[[], Sequence[CalciumVideoWidget]],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.connection = connection
        self._project = project
        self._current_stim = current_stim
        self._export_settings = export_settings
        self._video_widgets = video_widgets
        self.run_ids: set[str] = set()
        #: Block recordings of dynamic input routines, by run id.
        self._blocks: dict[str, BlockRecording] = {}
        self._armed: set[str] = set()
        #: Recorded blocks whose traces are being extracted.
        self._extracting: set[str] = set()
        #: Functional (calcium) channel of each block recording, from the ScanImage tab.
        self._block_channels: dict[str, int] = {}
        #: Stimulus samples written beside a recording that has not started yet.
        self._stimulus_files: dict[str, pathlib.Path] = {}
        self._extraction = BackgroundJob(self)
        self._disposed = False
        #: Shown as the "ScanImage (imaging)" tab of the Stimulation setup pane.
        self.controls = ScanImageControls(
            connection, settings, owns_run=self.run_ids.__contains__
        )
        self.controls.channel_changed.connect(self.set_scanimage_channel)
        self.controls.record_requested.connect(self.start_scanimage_recording)
        self.controls.directory_changed.connect(self._remember_directory)
        connection.status_changed.connect(self.scanimage_status)
        connection.recording_ready.connect(self.scanimage_recording_ready)
        connection.run_finished.connect(self._forget_run)
        self.project_changed()

    @property
    def acquisition_in_progress(self) -> bool:
        """Whether a run started here is still being tracked by a live connection,
        or a recorded block's traces are still being extracted.

        A run whose helper died stays tracked by the connection so a reconnect
        can recover it, but nothing can progress while disconnected, so it
        does not keep the workspace open.
        """
        return self._extraction.busy or (
            self.connection.connected and self.connection.run_id in self.run_ids
        )

    def dispose(self) -> None:
        """Disconnect this workspace without stopping the shared device."""
        if self._disposed:
            return
        self._disposed = True
        self.controls.dispose()
        self.connection.status_changed.disconnect(self.scanimage_status)
        self.connection.recording_ready.disconnect(self.scanimage_recording_ready)
        self.connection.run_finished.disconnect(self._forget_run)

    def project_changed(self) -> None:
        """Show the recording folder of the (new) project, if it has one."""
        folder = self._project().device_folders.get(RECORDING_FOLDER)
        if folder:
            self.controls.set_directory(folder)

    def _remember_directory(self, directory: str) -> None:
        self._project().set_device_folder(RECORDING_FOLDER, directory)

    def _record(self, run_id: str) -> dict | None:
        """The project's record of ``run_id``; ``None`` if the project was replaced."""
        return next(
            (
                record
                for record in self._project().imaging_acquisitions
                if record.get("run_id") == run_id
            ),
            None,
        )

    def _forget_run(self, run_id: str) -> None:
        self.run_ids.discard(run_id)
        self._armed.discard(run_id)
        self._stimulus_files.pop(run_id, None)
        if run_id in self._extracting:
            # The connection ends the run right after recording_ready; the block
            # completes once its traces are extracted.
            return
        self._block_channels.pop(run_id, None)
        if self._blocks.pop(run_id, None) is not None:
            self.block_recording_failed.emit(
                run_id, "The recording ended without producing data."
            )

    def start_scanimage_recording(
        self,
        arguments: dict,
        block: BlockRecording | None = None,
        functional_channel: int | None = None,
    ) -> bool:
        if self.connection.run_id is not None:
            self.controls.show_message("An acquisition is already being finalized.")
            return False
        project = self._project()
        record = dict(arguments)
        record.update(
            status="requested",
            requested_at=datetime.now(UTC).isoformat(),
            # A routine's block is triggered by its stimulation sequence; manual
            # recordings have no verified relation to any stimulus.
            synchronization="stimulation_trigger" if block else "unverified",
        )
        stim = block.stim_block if block is not None else self._current_stim()
        samples_path: pathlib.Path | None = None
        if stim is not None:
            record["stimulus"] = stimulus_summary(stim)
            # The samples themselves beside the TIFF, so the recording folder alone
            # reproduces the stimulus even if the project is never saved.
            samples_path = pathlib.Path(arguments["directory"]) / (
                f"{arguments['stem']}_{int(arguments['counter']):05d}_stimulus.npy"
            )
            try:
                np.save(samples_path, np.ascontiguousarray(stim.values))
            except OSError as exc:
                self.controls.show_message(f"Could not save the stimulus: {exc}")
                return False
            record["stimulus"]["samples_file"] = samples_path.name
        if block is not None:
            record["block"] = {
                "label": block.label,
                "duration_s": block.duration_s,
                "tail_s": block.tail_s,
                "functional_channel": functional_channel,
                # The exact script this recording captured.
                "seq_path": block.seq_path,
                "syscon_sequence": block.syscon_sequence,
            }
        settings = self._export_settings()
        if settings is not None:
            record["syscon_export_settings"] = {
                key: value
                for key, value in dataclasses.asdict(settings).items()
                if key != "config_text"
            }
        record = json.loads(json.dumps(record, default=str))
        if not self.connection.command("record", {**arguments, "manifest": record}):
            if samples_path is not None:
                samples_path.unlink(missing_ok=True)
            return False
        project.imaging_acquisitions.append(record)
        project.mark_dirty()
        self.run_ids.add(arguments["run_id"])
        if samples_path is not None:
            self._stimulus_files[arguments["run_id"]] = samples_path
        return True

    def scanimage_status(self, status: dict) -> None:
        run = status.get("run", {})
        run_id = run.get("id")
        if run_id not in self.run_ids:
            return
        state = run.get("state", "unknown")
        samples_path = self._stimulus_files.get(run_id)
        if samples_path is not None and state == "failed":
            # ScanImage refused to start: no recording for these samples to describe.
            samples_path.unlink(missing_ok=True)
            del self._stimulus_files[run_id]
        elif state != "starting":
            self._stimulus_files.pop(run_id, None)
        if run_id in self._blocks:
            if state in {"waiting_for_first_frame", "recording"} and (
                run_id not in self._armed
            ):
                self._armed.add(run_id)
                self.block_armed.emit(run_id)
            elif state in _FAILED_STATES:
                self._fail_block(
                    run_id, run.get("error") or f"ScanImage reported '{state}'."
                )
        record = self._record(run_id)
        if record is None:
            logger.warning("No project record for ScanImage run %s", run_id)
            return
        previous = record.get("status")
        record["status"] = state
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
            self._project().mark_dirty()

    def set_scanimage_channel(self, channel: int) -> None:
        index = self._project().calcium_scanimage_index
        if index is not None:
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
            return selected
        return channel

    def scanimage_recording_ready(self, result: dict) -> None:
        run_id = result["run_id"]
        if run_id not in self.run_ids:
            return
        self.run_ids.remove(run_id)
        project = self._project()
        record = self._record(run_id)
        if record is None:
            # The project was replaced during the recording; keep the files only.
            logger.warning("Recording %s finished for a project no longer open", run_id)
            self._fail_block(run_id, "The project changed during the recording.")
            return
        record.update(
            status=result["status"],
            recording=result,
            finished_at=datetime.now(UTC).isoformat(),
        )
        record.pop("error", None)
        name = record["name"]
        if name in project.experiments:
            name = f"{name} ({run_id[:8]})"
        project.add_experiment(
            Experiment(name=name, duration=result["duration_s"], timepoints=[], data=[])
        )
        record["experiment_name"] = name
        project.add_calcium_video(
            [pathlib.Path(p) for p in result["paths"]],
            result["fps"],
            name=name,
            run_id=run_id,
        )
        self.controls.show_message(f"Recording {name} added to the imaging data.")
        project.mark_dirty()
        project.update_limits()
        self.recording_added.emit()
        block = self._blocks.get(run_id)
        if block is None:
            return
        if result["status"] != "completed":
            self._fail_block(run_id, f"The recording was {result['status']}.")
            return
        self._extract_block(run_id, block, result, name)

    def _fail_block(self, run_id: str, message: str) -> None:
        self._block_channels.pop(run_id, None)
        if self._blocks.pop(run_id, None) is not None:
            self.block_recording_failed.emit(run_id, message)

    def _extract_block(
        self, run_id: str, block: BlockRecording, result: dict, experiment: str
    ) -> None:
        if self._extraction.busy:
            self._fail_block(run_id, "Another recording is still being extracted.")
            return
        project = self._project()
        cellpose = project.cellpose_data
        paths = list(result["paths"])
        fps = float(result["fps"])
        channels = list(result["channels"])
        channel = self._block_channels.pop(run_id, self.controls.channel_spin.value())

        def task(_report: Callable[[int], None]) -> tuple[TraceData, SpikeData]:
            trace = extract_recording_dff(
                paths,
                fps=fps,
                saved_channels=channels,
                functional_channel=channel,
                cellpose_data=cellpose,
                stim_block=block.stim_block,
            )
            return trace, deconvolve_dff(trace, threshold=block.spike_threshold)

        def finished(result: object) -> None:
            trace, spikes = cast(tuple["TraceData", "SpikeData"], result)
            self._extracting.discard(run_id)
            if self._blocks.pop(run_id, None) is None:
                return
            current = self._project()
            if experiment not in current.experiments:
                self.block_recording_failed.emit(
                    run_id, "The recording's data group was removed."
                )
                return
            current.add_data_to_experiment(experiment, trace)
            current.add_data_to_experiment(experiment, spikes)
            self.recording_added.emit()
            self.block_recorded.emit(
                run_id,
                {
                    "trace": f"{experiment} :: {trace.name}",
                    "spikes": f"{experiment} :: {spikes.name}",
                    "experiment": experiment,
                },
            )

        def failed(message: str) -> None:
            self._extracting.discard(run_id)
            self._fail_block(run_id, f"Trace extraction failed: {message}")

        self._extracting.add(run_id)
        self.controls.show_message(
            f"Extracting ΔF/F traces and spikes of {experiment}…"
        )
        self._extraction.start(
            BackgroundIOWorker(task),
            on_progress=lambda _value: None,
            on_finished=finished,
            on_failed=failed,
        )

    # ---- BlockAcquisition, for the SysCon2 stimulation host ----

    def block_recording_unavailable(self) -> str | None:
        if not self.connection.connected:
            return "Connect to ScanImage first (Stimulation setup > ScanImage)."
        if self.connection.run_id is not None or self._extraction.busy:
            return "ScanImage is still busy with another recording."
        if self._project().cellpose_data is None:
            return (
                "Load a Cellpose segmentation of the field of view first: it defines "
                "the neurons whose calcium responses are measured."
            )
        if not self._frame_rate():
            return "ScanImage has not reported its frame rate yet."
        return None

    def _frame_rate(self) -> float | None:
        fps = self.connection.last_status.get("fps")
        return float(fps) if isinstance(fps, (int, float)) and fps > 0 else None

    def arm_block(self, recording: BlockRecording) -> str:
        reason = self.block_recording_unavailable()
        if reason is not None:
            raise ValueError(reason)
        fps = self._frame_rate()
        assert fps is not None
        frames = max(1, math.ceil((recording.duration_s + recording.tail_s) * fps))
        channel = self.controls.channel_spin.value()
        saved = self.controls.saved_channels()
        if saved and channel not in saved:
            raise ValueError(
                f"The functional channel {channel} is not among the saved channels "
                f"({', '.join(map(str, saved))}) in the ScanImage tab."
            )
        arguments = self.controls.recording_request(
            frames=frames, external_trigger=True
        )
        run_id = arguments["run_id"]
        self._blocks[run_id] = recording
        self._block_channels[run_id] = channel
        if not self.start_scanimage_recording(arguments, recording, channel):
            self._blocks.pop(run_id, None)
            self._block_channels.pop(run_id, None)
            raise ValueError("ScanImage did not accept the recording.")
        return run_id

    def abort_block(self, run_id: str) -> None:
        self._blocks.pop(run_id, None)
        self._block_channels.pop(run_id, None)
        if self.connection.run_id == run_id:
            self.connection.command("abort")
