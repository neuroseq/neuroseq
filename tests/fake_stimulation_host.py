"""A stand-in for SysCon2StimulationHost, shared by the routine dialog tests."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore

from neuroseq.export.syscon2_export_service import SysconExportSettings


class FakeStimulationHost(QtCore.QObject):
    """Records what a routine sends. Manual blocks are "imported" at once; automatic
    ones stay running until the test calls :meth:`complete`."""

    block_sent = QtCore.Signal(object, object)
    block_failed = QtCore.Signal(object, str)
    block_completed = QtCore.Signal(object, object)
    progress = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.phase = "idle"
        self.sent: list[tuple[int, int, bool]] = []
        self.recordings: list[object] = []
        self.automatic: list[bool] = []
        self.run_names: list[str | None] = []
        self.owner: object = None
        self.cancelled = 0
        self.unavailable: str | None = None
        self.settings = SysconExportSettings(config_text="")

    def export_settings(self) -> SysconExportSettings:
        return self.settings

    def set_export_settings(self, settings: SysconExportSettings) -> None:
        self.settings = settings

    def run_directory(self, run_name: str) -> Path:
        return Path("runs") / run_name

    def recording_unavailable(self) -> str | None:
        return self.unavailable

    def send_block(
        self,
        project,
        stim,
        start,
        stop,
        label="block",
        *,
        preserve_amplitude=False,
        owner=None,
        automatic=False,
        recording=None,
        run_name=None,
    ):
        self.sent.append((start, stop, preserve_amplitude))
        self.run_names.append(run_name)
        self.recordings.append(recording)
        self.automatic.append(automatic)
        self.owner = owner
        if automatic:
            self.phase = "running"
            return True
        self.phase = "awaiting_acquisition"
        self.block_sent.emit(owner, None)
        return True

    def complete(self, payload: object) -> None:
        """Finish an automatic block, as after its recording was extracted."""
        self.phase = "idle"
        self.block_completed.emit(self.owner, payload)

    def acquisition_imported(self) -> None:
        if self.phase != "awaiting_acquisition":
            return
        self.phase = "idle"
        self.block_completed.emit(self.owner, None)

    def reawait(self, owner: object = None) -> None:
        self.phase = "awaiting_acquisition"

    def cancel(self, owner: object = None) -> None:
        if owner is not None and owner is not self.owner:
            return
        self.cancelled += 1
        self.phase = "idle"
