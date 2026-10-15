"""Deliver stim-array blocks to cells in a dish through SysCon2 (SCI).

This is the hardware counterpart of the Brian2 simulation host that the closed-loop routines
(rheobase, STDP...) drive: a routine appends a block to the stim array, then asks the host to
"run" it. Here, running means exporting that block as a ``.seq`` and importing it into SysCon2;
the experimenter then acquires and imports the calcium data, and the routine resumes via
:meth:`SysCon2StimulationHost.acquisition_imported` (semi-automatic for now; a ScanImage-driven
import, see issue #589, would call the same method).

Phases per block: ``idle -> sending -> awaiting_acquisition -> idle``.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
from collections.abc import Callable
from dataclasses import replace
from itertools import count

from qtpy import QtCore

from neuroseq.stim import StimArray
from neuroseq.syscon2_client import SysCon2Client
from neuroseq.syscon2_export_service import SysconExportError
from neuroseq.syscon2_export_service import SysconExportResult
from neuroseq.syscon2_export_service import SysconExportSettings
from neuroseq.syscon2_export_service import export_stim_array

logger = logging.getLogger(__name__)

IMPORT_COMMAND_PREFIX = "stim_host.import"


class SysCon2StimulationHost(QtCore.QObject):
    """Exports blocks of a stim array and pushes them to SysCon2.

    ``submit`` is ``SysCon2ConnectionManager.submit``; ``command_finished`` /
    ``command_failed`` are the manager's signals (kept as plain callables so the host is
    testable without a thread or a pipe).

    Signals (GUI thread):

    - ``block_sent(object)`` -- the block's :class:`SysconExportResult`, once SysCon2 has
      imported it. The routine should now tell the user to run + acquire.
    - ``block_failed(str)`` -- export or SCI import failed; nothing was left half-sent.
    - ``block_completed()`` -- :meth:`acquisition_imported` was called after a sent block.
    """

    block_sent = QtCore.Signal(object)
    block_failed = QtCore.Signal(str)
    block_completed = QtCore.Signal()

    def __init__(
        self,
        submit: Callable[[str, Callable[[SysCon2Client], object]], None],
        *,
        settings_provider: Callable[[], SysconExportSettings | None],
        target_directory_provider: Callable[[], str],
        output_dir: pathlib.Path | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._settings_provider = settings_provider
        self._target_directory_provider = target_directory_provider
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        if output_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="neuroseq_syscon_blocks_")
            output_dir = pathlib.Path(self._tmpdir.name)
        self._output_dir = output_dir
        self._counter = count(1)
        self._pending_command: str | None = None
        self._pending_result: SysconExportResult | None = None
        self.phase = "idle"

    @property
    def output_dir(self) -> pathlib.Path:
        return self._output_dir

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
    ) -> bool:
        """Export ``stim_array[start:stop]`` and import it into SysCon2.

        ``preserve_amplitude`` exports the array's values as a fraction of the target power
        instead of forcing every active sample to full power (needed for amplitude sweeps).

        Returns ``False`` (after emitting ``block_failed``) if it could not be started.
        """
        if self.phase != "idle":
            self.block_failed.emit("A previous block is still in flight.")
            return False
        settings = self._settings_provider()
        if settings is None:
            self.block_failed.emit(
                "No export settings yet - run Send to SysCon2... once first."
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
            )
        except (SysconExportError, ValueError, OSError) as exc:
            self.block_failed.emit(str(exc))
            return False

        command = f"{IMPORT_COMMAND_PREFIX}.{index}"
        seq_path = str(result.seq_path)
        target = self._target_directory_provider()

        def task(client: SysCon2Client) -> object:
            client.sequence_manager.import_sequence(seq_path, target)
            return None

        self._pending_command = command
        self._pending_result = result
        self.phase = "sending"
        self._submit(command, task)
        return True

    def acquisition_imported(self) -> None:
        """The user has run the block, acquired and imported the calcium data."""
        if self.phase != "awaiting_acquisition":
            return
        self.phase = "idle"
        self.block_completed.emit()

    def cancel(self) -> None:
        self._pending_command = None
        self._pending_result = None
        self.phase = "idle"

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ---- manager signal handlers (wired by the owner) ----

    def on_command_finished(self, name: str, _result: object) -> None:
        if name != self._pending_command:
            return
        result = self._pending_result
        self._pending_command = None
        self._pending_result = None
        self.phase = "awaiting_acquisition"
        self.block_sent.emit(result)

    def on_command_failed(self, name: str, message: str) -> None:
        if name != self._pending_command:
            return
        self._pending_command = None
        self._pending_result = None
        self.phase = "idle"
        self.block_failed.emit(message)
