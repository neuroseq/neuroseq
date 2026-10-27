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
import time
from collections.abc import Callable
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import cast

import brian2.numpy_ as np
import tifffile
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.background_jobs import BackgroundIOWorker
from neuroseq.background_jobs import BackgroundJob
from neuroseq.data import Experiment
from neuroseq.device_control.scanimage_controls import ScanImageControls
from neuroseq.device_control.scanimage_files import STIMULUS_SUFFIX
from neuroseq.device_control.scanimage_files import stimulus_file_name
from neuroseq.importers.importer_scanimage_traces import deconvolve_dff
from neuroseq.importers.importer_scanimage_traces import extract_recording_dff
from neuroseq.importers.importer_scanimage_traces import functional_channel_mean

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
#: Slack beyond a block's length before an armed recording that has not finished
#: is aborted (e.g. ScanImage never received the trigger).
BLOCK_DEADLINE_MARGIN_S = 15.0
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
    #: The mean image of a "Grab FOV" recording was saved (its path).
    fov_ready = QtCore.Signal(str)
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
        #: When each block's recording started waiting for its trigger.
        self._armed_at: dict[str, float] = {}
        #: Deadline of each armed block recording, until its recording is ready.
        self._deadlines: dict[str, QtCore.QTimer] = {}
        #: Recorded blocks whose traces are being extracted.
        self._extracting: set[str] = set()
        #: Functional (calcium) channel of each block recording, from the ScanImage tab.
        self._block_channels: dict[str, int] = {}
        #: Stimulus samples written beside a recording that has not started yet.
        self._stimulus_files: dict[str, pathlib.Path] = {}
        #: Last reported state of each run started here.
        self._run_states: dict[str, str] = {}
        #: "Grab FOV" recordings, averaged into one image for cell segmentation, with
        #: the functional channel chosen when the grab was requested.
        self._fov_runs: dict[str, int] = {}
        #: Running trace extractions, one job each: a routine's next block must not
        #: wait for (or fail because of) a previous, possibly abandoned, one.
        self._extractions: dict[str, BackgroundJob] = {}
        self._disposed = False
        #: Shown as the "ScanImage (imaging)" tab of the Stimulation setup pane.
        self.controls = ScanImageControls(
            connection, settings, owns_run=self.run_ids.__contains__
        )
        self.controls.channel_changed.connect(self.set_scanimage_channel)
        self.controls.record_requested.connect(self.start_scanimage_recording)
        self.controls.fov_requested.connect(self.start_fov_recording)
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
        return any(job.busy for job in self._extractions.values()) or (
            self.connection.connected and self.connection.run_id in self.run_ids
        )

    def dispose(self) -> None:
        """Disconnect this workspace without stopping the shared device."""
        if self._disposed:
            return
        self._disposed = True
        for run_id in list(self._deadlines):
            self._stop_deadline(run_id)
        self.controls.dispose()
        self.connection.status_changed.disconnect(self.scanimage_status)
        self.connection.recording_ready.disconnect(self.scanimage_recording_ready)
        self.connection.run_finished.disconnect(self._forget_run)

    def project_changed(self) -> None:
        """Show the recording folder of the (new) project; one without a folder of
        its own starts from the folder last used in any project (the default for
        new projects), not from whatever the previous project showed."""
        folder = self._project().device_folders.get(RECORDING_FOLDER)
        self.controls.set_directory(
            folder or self.controls.settings.scanimage_recording_directory
        )

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
        self._armed_at.pop(run_id, None)
        self._remove_metadata_without_recording(run_id)
        if self._fov_runs.pop(run_id, None) is not None:
            self.controls.show_message("The FOV recording ended without data.")
        if run_id in self._extracting:
            # The connection ends the run right after recording_ready; the block
            # completes once its traces are extracted.
            return
        self._block_channels.pop(run_id, None)
        if self._blocks.pop(run_id, None) is not None:
            self.block_recording_failed.emit(
                run_id, "The recording ended without producing data."
            )

    def _remove_metadata_without_recording(self, run_id: str) -> None:
        """A run that ended without a TIFF (refused, or aborted before its first
        frame) leaves no recording for its stimulus and JSON to describe. Runs
        with an unknown outcome (interrupted) keep them for recovery."""
        samples_path = self._stimulus_files.pop(run_id, None)
        state = self._run_states.pop(run_id, None)
        if samples_path is None or state not in {"failed", "aborted"}:
            return
        prefix = samples_path.name.removesuffix(STIMULUS_SUFFIX)
        if any(samples_path.parent.glob(f"{prefix}*.tif")):
            return
        samples_path.unlink(missing_ok=True)
        samples_path.with_name(f"{prefix}.json").unlink(missing_ok=True)

    def start_fov_recording(self, arguments: dict) -> bool:
        """Record a few frames whose mean image becomes the FOV to segment."""
        if not self.start_scanimage_recording(
            arguments, purpose="fov_for_segmentation"
        ):
            return False
        self._fov_runs[arguments["run_id"]] = self.controls.channel_spin.value()
        self.controls.show_message("Recording the field of view…")
        return True

    def start_scanimage_recording(
        self,
        arguments: dict,
        block: BlockRecording | None = None,
        functional_channel: int | None = None,
        *,
        purpose: str | None = None,
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
        if purpose is not None:
            record["purpose"] = purpose
        # An FOV snapshot is taken without stimulation: no stimulus to describe.
        stim = (
            block.stim_block
            if block is not None
            else None
            if purpose is not None
            else self._current_stim()
        )
        samples_path: pathlib.Path | None = None
        if stim is not None:
            record["stimulus"] = stimulus_summary(stim)
            # The samples themselves beside the TIFF, so the recording folder alone
            # reproduces the stimulus even if the project is never saved.
            samples_path = pathlib.Path(arguments["directory"]) / stimulus_file_name(
                arguments["stem"], int(arguments["counter"])
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
        # A routine's block: the options it was actually delivered with.
        settings = (
            block.export_settings
            if block is not None and block.export_settings is not None
            else self._export_settings()
        )
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
        # The project records the folder its recordings went to.
        self._remember_directory(str(arguments["directory"]))
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
        previous_state = self._run_states.get(run_id)
        self._run_states[run_id] = state
        if run_id in self._blocks:
            if state in {"waiting_for_first_frame", "recording"} and (
                run_id not in self._armed
            ):
                self._armed.add(run_id)
                self._armed_at[run_id] = time.monotonic()
                self._start_deadline(run_id, self._blocks[run_id])
                self.block_armed.emit(run_id)
            elif state in _FAILED_STATES:
                self._fail_block(
                    run_id,
                    run.get("error")
                    or self._block_failure_message(run_id, state, previous_state),
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
            # Only the display falls back: the Functional channel stays what the
            # user chose, since dynamic input routines measure it.
            self.controls.show_message(
                f"Channel {channel} was not saved in this recording; showing channel {selected}."
            )
            return selected
        return channel

    def scanimage_recording_ready(self, result: dict) -> None:
        run_id = result["run_id"]
        self._stop_deadline(run_id)
        if run_id not in self.run_ids:
            return
        self.run_ids.remove(run_id)
        project = self._project()
        record = self._record(run_id)
        if record is None:
            # The project was replaced during the recording; keep the files only.
            logger.warning("Recording %s finished for a project no longer open", run_id)
            if self._fov_runs.pop(run_id, None) is not None:
                self.controls.show_message(
                    "The project changed during the FOV recording; its TIFF is kept "
                    "in the recording folder but was not segmented."
                )
            self._fail_block(run_id, "The project changed during the recording.")
            return
        record.update(
            status=result["status"],
            recording=result,
            finished_at=datetime.now(UTC).isoformat(),
        )
        record.pop("error", None)
        if run_id in self._fov_runs:
            channel = self._fov_runs.pop(run_id)
            self._save_fov_image(run_id, result, record, channel)
            project.mark_dirty()
            return
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

    def _save_fov_image(
        self, run_id: str, result: dict, record: dict, channel: int
    ) -> None:
        """Average ``channel`` and save it beside the recording, in the background
        like a block's trace extraction."""
        paths = [pathlib.Path(p) for p in result["paths"]]
        channels = list(result["channels"])
        target = paths[0].with_name(f"{paths[0].stem}_mean.tif")

        def task(_report: Callable[[int], None]) -> str:
            image = functional_channel_mean(
                paths, saved_channels=channels, functional_channel=channel
            )
            tifffile.imwrite(target, image)
            return str(target)

        def finished(_path: object) -> None:
            record["fov_image"] = str(target)
            self._project().mark_dirty()
            self.controls.show_message(
                f"FOV image saved: {target.name} (mean of channel {channel}); "
                "segmenting it with Cellpose."
            )
            self.fov_ready.emit(str(target))

        def failed(message: str) -> None:
            self.controls.show_message(f"Could not make the FOV image: {message}")

        self.controls.show_message("Averaging the field of view…")
        self._start_job(run_id, task, finished, failed)

    def _start_job(
        self,
        run_id: str,
        task: Callable[[Callable[[int], None]], object],
        finished: Callable[[object], None],
        failed: Callable[[str], None],
    ) -> None:
        """Run ``task`` for recording ``run_id`` in its own background job."""
        job = BackgroundJob(self)
        self._extractions[run_id] = job
        job.idle.connect(lambda: self._extraction_done(run_id))
        job.start(
            BackgroundIOWorker(task),
            on_progress=lambda _value: None,
            on_finished=finished,
            on_failed=failed,
        )

    def _start_deadline(self, run_id: str, recording: BlockRecording) -> None:
        """Abort the block if its recording has not finished well after its length:
        a recording that started without the trigger would otherwise wait forever."""
        limit_s = recording.duration_s + recording.tail_s + BLOCK_DEADLINE_MARGIN_S
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._deadline_passed(run_id, limit_s))
        self._deadlines[run_id] = timer
        timer.start(int(limit_s * 1000))

    def _stop_deadline(self, run_id: str) -> None:
        timer = self._deadlines.pop(run_id, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()

    def _deadline_passed(self, run_id: str, limit_s: float) -> None:
        self._stop_deadline(run_id)
        if run_id not in self._blocks:
            return
        if self.connection.run_id == run_id:
            self.connection.command("abort")
        self._fail_block(
            run_id,
            f"ScanImage did not finish the recording within {limit_s:.0f} s. Check "
            "that SysCon2's TTL Out 1 reaches ScanImage's acquisition start trigger.",
        )

    def _block_failure_message(
        self, run_id: str, state: str, previous_state: str | None
    ) -> str:
        if state != "aborted":
            return f"ScanImage reported '{state}'."
        armed_at = self._armed_at.get(run_id)
        after = (
            f" {time.monotonic() - armed_at:.1f} s after it was armed"
            if armed_at is not None
            else ""
        )
        if previous_state == "recording":
            return (
                f"ScanImage aborted the recording{after}, after its first frame. "
                "Check MATLAB's command window for a ScanImage error."
            )
        return (
            f"ScanImage aborted the recording{after}, before its first frame. "
            "Either no trigger reached its acquisition start trigger from SysCon2's "
            "TTL Out 1, the ScanImage tab's first-frame timeout passed, or Abort was "
            "pressed in ScanImage. Check MATLAB's command window for a ScanImage error."
        )

    def _fail_block(self, run_id: str, message: str) -> None:
        self._stop_deadline(run_id)
        self._block_channels.pop(run_id, None)
        if self._blocks.pop(run_id, None) is not None:
            self.block_recording_failed.emit(run_id, message)

    def _extract_block(
        self, run_id: str, block: BlockRecording, result: dict, experiment: str
    ) -> None:
        # The project recorded into, even if another one is open when this finishes.
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
            if experiment not in project.experiments:
                self.block_recording_failed.emit(
                    run_id, "The recording's data group was removed."
                )
                return
            try:
                project.add_data_to_experiment(experiment, trace)
                project.add_data_to_experiment(experiment, spikes)
            except Exception as exc:
                # The routine must hear about it, or it waits for this block forever.
                logger.exception("Could not add the traces of recording %s", run_id)
                self.block_recording_failed.emit(
                    run_id, f"The traces could not be added: {exc}"
                )
                return
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
        self._start_job(run_id, task, finished, failed)

    def _extraction_done(self, run_id: str) -> None:
        job = self._extractions.pop(run_id, None)
        if job is not None:
            job.deleteLater()

    # ---- BlockAcquisition, for the SysCon2 stimulation host ----

    def block_recording_unavailable(self) -> str | None:
        if not self.connection.connected:
            return "Connect to ScanImage first (Stimulation setup > ScanImage)."
        if self.connection.run_id is not None:
            return "ScanImage is still busy with another recording."
        if self._project().cellpose_data is None:
            return (
                "Load a Cellpose segmentation of the field of view first: it defines "
                "the neurons whose calcium responses are measured."
            )
        if not self._frame_rate():
            return "ScanImage has not reported its frame rate yet."
        # Checked here too, so a bad setup is refused before anything is uploaded.
        folder = self.controls.directory_edit.text().strip()
        if not folder or not pathlib.Path(folder).is_dir():
            return "Select an existing recording folder in the ScanImage tab."
        channel = self.controls.channel_spin.value()
        saved = self.controls.saved_channels()
        if saved and channel not in saved:
            return (
                f"The functional channel {channel} is not among the saved channels "
                f"({', '.join(map(str, saved))}) in the ScanImage tab."
            )
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
        # block_recording_unavailable() checked that this channel is saved.
        channel = self.controls.channel_spin.value()
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
        self._stop_deadline(run_id)
        self._blocks.pop(run_id, None)
        self._block_channels.pop(run_id, None)
        if self.connection.run_id == run_id:
            self.connection.command("abort")
