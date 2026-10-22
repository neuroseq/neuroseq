"""Deliver stim-array blocks to cells in a dish through SysCon2 (SCI), optionally recording
each block with ScanImage.

This is the hardware counterpart of the Brian2 simulation host that the closed-loop routines
(rheobase, STDP...) drive: a routine appends a block to the stim array, then asks the host to
"run" it. Two modes:

- **Manual**: export the block as a ``.seq`` and import it into SysCon2; the experimenter
  runs it, acquires and imports the calcium data, and the routine resumes via
  :meth:`SysCon2StimulationHost.acquisition_imported`.
  Phases: ``idle -> sending -> awaiting_acquisition -> idle``.
- **Automatic** (issue #651): import, select and upload the block, arm a ScanImage recording
  that waits for the external trigger sent by the sequence, run the sequence, and resume the
  routine once the recording's ΔF/F traces are in the project. Blocks needing no data
  (e.g. STDP pairings) are run without recording.
  Phases: ``idle -> preparing -> arming -> running -> idle``.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from itertools import count
from typing import Protocol

from qtpy import QtCore

from neuroseq.device_control.syscon2_client import SysCon2Client
from neuroseq.export.syscon2_export_service import SysconExportError
from neuroseq.export.syscon2_export_service import SysconExportResult
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.export.syscon2_export_service import export_stim_array
from neuroseq.stimulation.stim import StimArray

logger = logging.getLogger(__name__)

IMPORT_COMMAND_PREFIX = "stim_host.import"
COMMAND_PREFIX = "stim_host"
#: How long SysCon2 may take to report an uploaded sequence ready to run.
UPLOAD_TIMEOUT_S = 60.0
#: Extra time allowed for an unrecorded block to finish after its nominal duration.
RUN_MARGIN_S = 30.0
_POLL_S = 0.1


@dataclass(frozen=True)
class BlockRecording:
    """What to record for one automatically delivered block."""

    label: str
    #: The block delivered during the recording, starting at its first frame.
    stim_block: StimArray
    #: Recording time after the end of the block (calcium decay), in seconds.
    tail_s: float
    #: ScanImage channel carrying the calcium indicator.
    functional_channel: int

    @property
    def duration_s(self) -> float:
        return float(self.stim_block.values.shape[0]) * float(self.stim_block.dt)


class BlockAcquisition(Protocol):
    """Records blocks for the host (implemented by the workspace's ScanImage controller).

    Signals: ``block_armed(str run_id)`` once the recording waits for its trigger,
    ``block_recorded(str run_id, object payload)`` once its traces are in the project
    (``payload["trace"]`` is the ΔF/F dataset name), ``block_recording_failed(str run_id,
    str message)``.
    """

    block_armed: QtCore.SignalInstance
    block_recorded: QtCore.SignalInstance
    block_recording_failed: QtCore.SignalInstance

    def block_recording_unavailable(self) -> str | None:
        """Why a block cannot be recorded now, or ``None`` when it can."""

    def arm_block(self, recording: BlockRecording) -> str:
        """Start a triggered recording; returns its run id. Raises ``ValueError``."""

    def abort_block(self, run_id: str) -> None: ...


class _Cancelled(Exception):
    pass


@dataclass
class _Block:
    index: int
    owner: object
    result: SysconExportResult
    duration_s: float
    automatic: bool
    recording: BlockRecording | None
    cancel: threading.Event
    remote_path: str = ""
    run_id: str | None = None
    run_submitted: bool = False

    def command(self, step: str) -> str:
        return f"{COMMAND_PREFIX}.{self.index}.{step}"


def remote_sequence_path(sequences: list[str], name: str, target: str) -> str:
    """The sequence-manager entry of an imported ``name``, preferring ``target``."""
    key = name.lower()
    matches = [
        entry
        for entry in sequences
        if pathlib.PureWindowsPath(entry.replace("/", "\\")).name.lower() == key
    ]
    prefix = target.lower().replace("/", "\\")
    in_target = [
        entry
        for entry in matches
        if entry.lower().replace("/", "\\").startswith(prefix)
    ]
    if not matches:
        raise RuntimeError(f"SysCon2 does not list the imported sequence {name}.")
    return (in_target or matches)[-1]


def _ready_to_run(state: str) -> bool:
    """Same readiness rule as the panel's Upload button."""
    state = state.lower().removeprefix("state:").strip()
    return (
        bool(state)
        and state not in {"busy", "disconnected"}
        and not (state.startswith("upload"))
    )


class SysCon2StimulationHost(QtCore.QObject):
    """Exports blocks of a stim array and pushes them to SysCon2.

    ``submit`` is ``SysCon2ConnectionManager.submit``; ``command_finished`` /
    ``command_failed`` are the manager's signals (kept as plain callables so the host is
    testable without a thread or a pipe).

    Every signal carries the ``owner`` passed to :meth:`send_block`, so a routine only
    reacts to its own blocks (GUI thread):

    - ``block_sent(owner, SysconExportResult)`` -- SysCon2 imported the block.
    - ``block_failed(owner, str)`` -- the block could not be delivered or recorded.
    - ``block_completed(owner, object)`` -- the block ran; ``None`` after
      :meth:`acquisition_imported`, else the recording's payload (see
      :class:`BlockAcquisition`), or ``{}`` for an automatic block without recording.
    - ``progress(str)`` -- a status line for the user.
    """

    block_sent = QtCore.Signal(object, object)
    block_failed = QtCore.Signal(object, str)
    block_completed = QtCore.Signal(object, object)
    progress = QtCore.Signal(str)

    def __init__(
        self,
        submit: Callable[[str, Callable[[SysCon2Client], object]], None],
        *,
        settings_provider: Callable[[], SysconExportSettings | None],
        settings_setter: Callable[[SysconExportSettings], None] | None = None,
        target_directory_provider: Callable[[], str],
        confirm_warnings: Callable[[list[str]], bool] | None = None,
        is_connected: Callable[[], bool] = lambda: True,
        output_dir: pathlib.Path | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._settings_provider = settings_provider
        self._settings_setter = settings_setter
        self._target_directory_provider = target_directory_provider
        self._confirm_warnings = confirm_warnings
        self._is_connected = is_connected
        #: Export warnings the user already accepted, so a routine asks once, not per block.
        self._accepted_warnings: set[str] = set()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        if output_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="neuroseq_syscon_blocks_")
            output_dir = pathlib.Path(self._tmpdir.name)
        self._output_dir = output_dir
        self._counter = count(1)
        self._block: _Block | None = None
        self._acquisition: BlockAcquisition | None = None
        self.phase = "idle"

    @property
    def output_dir(self) -> pathlib.Path:
        return self._output_dir

    @property
    def owner(self) -> object | None:
        """The routine whose block is in flight."""
        return self._block.owner if self._block is not None else None

    def set_acquisition(self, acquisition: BlockAcquisition | None) -> None:
        """Attach the recorder used by automatic blocks."""
        if self._acquisition is not None:
            self._acquisition.block_armed.disconnect(self._on_block_armed)
            self._acquisition.block_recorded.disconnect(self._on_block_recorded)
            self._acquisition.block_recording_failed.disconnect(
                self._on_block_recording_failed
            )
        self._acquisition = acquisition
        if acquisition is not None:
            acquisition.block_armed.connect(self._on_block_armed)
            acquisition.block_recorded.connect(self._on_block_recorded)
            acquisition.block_recording_failed.connect(self._on_block_recording_failed)

    def export_settings(self) -> SysconExportSettings | None:
        """The export options blocks are delivered with."""
        return self._settings_provider()

    def set_export_settings(self, settings: SysconExportSettings) -> None:
        """Change the export options, e.g. from a routine's hardware settings (#651)."""
        if self._settings_setter is not None:
            self._settings_setter(settings)

    def recording_unavailable(self) -> str | None:
        """Why automatic blocks cannot be recorded now, or ``None``."""
        if self._acquisition is None:
            return "No image acquisition is available."
        return self._acquisition.block_recording_unavailable()

    def reset_accepted_warnings(self) -> None:
        self._accepted_warnings.clear()

    # ---- routine-facing API ----

    def send_block(
        self,
        project: object,
        stim_array: StimArray,
        start_sample: int,
        stop_sample: int,
        label: str = "block",
        *,
        preserve_amplitude: bool = False,
        owner: object = None,
        automatic: bool = False,
        recording: BlockRecording | None = None,
    ) -> bool:
        """Export ``stim_array[start:stop]`` and import it into SysCon2.

        ``preserve_amplitude`` exports the array's values as a fraction of the target power
        instead of forcing every active sample to full power (needed for amplitude sweeps).
        ``automatic`` also uploads and runs the block, recording it when ``recording`` is
        given; otherwise the user runs it and calls :meth:`acquisition_imported`.

        Returns ``False`` (after emitting ``block_failed``) if it could not be started.
        """
        if self.phase != "idle":
            self.block_failed.emit(owner, "A previous block is still in flight.")
            return False
        if recording is not None and not automatic:
            raise ValueError("Only automatic blocks can be recorded.")
        if not self._is_connected():
            self.block_failed.emit(owner, "Connect to SysCon2 first.")
            return False
        if recording is not None:
            reason = self.recording_unavailable()
            if reason is not None:
                self.block_failed.emit(owner, reason)
                return False
        settings = self._settings_provider()
        if settings is None:
            self.block_failed.emit(
                owner,
                "No export settings yet - run 'Export with settings…' once first.",
            )
            return False
        if preserve_amplitude and settings.force_binary_intensity:
            settings = replace(settings, force_binary_intensity=False)
        index = next(self._counter)
        try:
            result = export_stim_array(
                project,  # type: ignore[arg-type]
                stim_array,
                settings,
                self._output_dir,
                f"{label}_{index:03d}",
                start_sample=start_sample,
                stop_sample=stop_sample,
                confirm_warnings=self._confirm_new_warnings,
            )
        except (SysconExportError, ValueError, OSError) as exc:
            self.block_failed.emit(owner, str(exc))
            return False

        block = _Block(
            index=index,
            owner=owner,
            result=result,
            duration_s=(stop_sample - start_sample) * float(stim_array.dt),
            automatic=automatic,
            recording=recording,
            cancel=threading.Event(),
        )
        self._block = block
        seq_path = str(result.seq_path)
        target = self._target_directory_provider()
        if not automatic:

            def import_task(client: SysCon2Client) -> object:
                client.sequence_manager.import_sequence(seq_path, target)
                return None

            self.phase = "sending"
            self._submit(f"{IMPORT_COMMAND_PREFIX}.{index}", import_task)
            return True

        cancel = block.cancel
        name = result.seq_path.name

        def prepare_task(client: SysCon2Client) -> object:
            client.sequence_manager.import_sequence(seq_path, target)
            remote = remote_sequence_path(
                client.sequence_manager.get_sequences(), name, target
            )
            client.sequence_manager.select_sequence(remote)
            client.holo4d.upload_sequence(1)
            deadline = time.monotonic() + UPLOAD_TIMEOUT_S
            state = ""
            while not _ready_to_run(state := client.holo4d.get_state()):
                if cancel.is_set():
                    raise _Cancelled
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The uploaded sequence was not ready after {UPLOAD_TIMEOUT_S:.0f} s "
                        f"(state: {state or 'unknown'})."
                    )
                time.sleep(_POLL_S)
            return remote

        self.phase = "preparing"
        self.progress.emit(f"Importing and uploading {name}…")
        self._submit(block.command("prepare"), prepare_task)
        return True

    def acquisition_imported(self) -> None:
        """The user has run the block, acquired and imported the calcium data."""
        if self.phase != "awaiting_acquisition" or self._block is None:
            return
        owner = self._block.owner
        self._clear()
        self.block_completed.emit(owner, None)

    def reawait(self, owner: object = None) -> None:
        """Keep waiting for :meth:`acquisition_imported` on the block that just completed,
        e.g. because the user selected a stale dataset. Manual blocks only."""
        if self.phase != "idle" or self._last_manual is None:
            return
        if owner is not None and self._last_manual.owner is not owner:
            return
        self._block = self._last_manual
        self.phase = "awaiting_acquisition"

    def cancel(self, owner: object = None) -> None:
        """Abandon the block in flight; with ``owner``, only if it is that routine's.

        A SysCon2 command already submitted still completes on the hardware, but its
        result is ignored; a running sequence is stopped and a recording aborted.
        """
        block = self._block
        if block is None or (owner is not None and block.owner is not owner):
            return
        block.cancel.set()
        if block.run_submitted:
            self._submit(
                block.command("stop"), lambda client: client.holo4d.stop_sequence()
            )
        if block.run_id is not None and self._acquisition is not None:
            self._acquisition.abort_block(block.run_id)
        self._clear()

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ---- internals ----

    _last_manual: _Block | None = None

    def _clear(self) -> None:
        block, self._block = self._block, None
        self._last_manual = block if block is not None and not block.automatic else None
        self.phase = "idle"

    def _confirm_new_warnings(self, warnings: list[str]) -> bool:
        new = [
            warning for warning in warnings if warning not in self._accepted_warnings
        ]
        if not new:
            return True
        if self._confirm_warnings is None or not self._confirm_warnings(new):
            return False
        self._accepted_warnings.update(new)
        return True

    def _fail(self, message: str) -> None:
        block = self._block
        if block is None:
            return
        if block.run_id is not None and self._acquisition is not None:
            self._acquisition.abort_block(block.run_id)
        if block.run_submitted:
            self._submit(
                block.command("stop"), lambda client: client.holo4d.stop_sequence()
            )
        self._clear()
        self.progress.emit(f"Block failed: {message}")
        self.block_failed.emit(block.owner, message)

    def _complete(self, payload: object) -> None:
        block = self._block
        if block is None:
            return
        self._clear()
        self.progress.emit(f"Block {block.result.seq_path.name} done.")
        self.block_completed.emit(block.owner, payload)

    def _run(self, block: _Block) -> None:
        """Start the uploaded sequence; without a recording, wait until it has ended."""
        cancel = block.cancel
        wait_s = None if block.recording is not None else block.duration_s

        def run_task(client: SysCon2Client) -> object:
            if cancel.is_set():
                return None
            client.holo4d.run_sequence("auto", "rising")
            if wait_s is None:
                return None
            started = time.monotonic()
            deadline = started + wait_s + RUN_MARGIN_S
            seen_busy = False
            while not cancel.is_set():
                state = client.holo4d.get_state().lower()
                if state == "busy":
                    seen_busy = True
                elif state == "idle" and (
                    seen_busy or time.monotonic() - started > wait_s + 1.0
                ):
                    return None
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The sequence did not finish within {wait_s + RUN_MARGIN_S:.0f} s."
                    )
                time.sleep(_POLL_S)
            return None

        block.run_submitted = True
        self.phase = "running"
        self.progress.emit(
            f"Running {block.result.seq_path.name}"
            + (" while ScanImage records…" if block.recording is not None else "…")
        )
        self._submit(block.command("run"), run_task)

    # ---- acquisition signal handlers ----

    def _on_block_armed(self, run_id: str) -> None:
        block = self._block
        if block is None or block.run_id != run_id or self.phase != "arming":
            return
        self._run(block)

    def _on_block_recorded(self, run_id: str, payload: object) -> None:
        block = self._block
        if block is None or block.run_id != run_id:
            return
        self._complete(payload)

    def _on_block_recording_failed(self, run_id: str, message: str) -> None:
        block = self._block
        if block is None or block.run_id != run_id:
            return
        block.run_id = None  # already ended; nothing to abort
        self._fail(f"Recording failed: {message}")

    # ---- manager signal handlers (wired by the owner) ----

    def _current(self, name: str) -> _Block | None:
        block = self._block
        if block is None:
            return None
        if name == f"{IMPORT_COMMAND_PREFIX}.{block.index}" and not block.automatic:
            return block
        if name.startswith(f"{COMMAND_PREFIX}.{block.index}."):
            return block
        return None

    def on_command_finished(self, name: str, result: object) -> None:
        block = self._current(name)
        if block is None:
            return
        if not block.automatic:
            self.phase = "awaiting_acquisition"
            self.block_sent.emit(block.owner, block.result)
            return
        step = name.rsplit(".", 1)[-1]
        if step == "prepare" and self.phase == "preparing":
            block.remote_path = str(result)
            self.block_sent.emit(block.owner, block.result)
            if block.recording is None:
                self._run(block)
                return
            assert self._acquisition is not None
            try:
                block.run_id = self._acquisition.arm_block(block.recording)
            except ValueError as exc:
                self._fail(f"Could not start the recording: {exc}")
                return
            self.phase = "arming"
            self.progress.emit("Waiting for ScanImage to arm the triggered recording…")
        elif step == "run" and self.phase == "running" and block.recording is None:
            self._complete({})

    def on_command_failed(self, name: str, message: str) -> None:
        block = self._current(name)
        if block is None:
            return
        if name.endswith(".stop"):
            return
        self._fail(message)
