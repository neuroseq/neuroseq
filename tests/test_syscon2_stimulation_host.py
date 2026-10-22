from __future__ import annotations

import brian2.numpy_ as np
import pytest
from qtpy import QtCore

import neuroseq.device_control.syscon2_stimulation_host as host_module
from neuroseq.device_control.syscon2_stimulation_host import BlockRecording
from neuroseq.device_control.syscon2_stimulation_host import SysCon2StimulationHost
from neuroseq.device_control.syscon2_stimulation_host import remote_sequence_path
from neuroseq.export.syscon2_export_service import SysconExportError
from neuroseq.export.syscon2_export_service import SysconExportResult
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.export.syscon2_export_service import slice_stim_array
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


@pytest.fixture()
def stim() -> StimArray:
    return StimArray("a", StimType.External, np.ones((10, 2)), "", 0.001)


def _host(submitted, settings, tmp_path, **kwargs):
    return SysCon2StimulationHost(
        lambda name, fn: submitted.append((name, fn)),
        settings_provider=lambda: settings,
        target_directory_provider=lambda: "target",
        output_dir=tmp_path,
        **kwargs,
    )


def test_slice_stim_array_copies_range(stim: StimArray) -> None:
    part = slice_stim_array(stim, 2, 5)
    assert part.values.shape == (3, 2)
    part.values[:] = 0
    assert stim.values.min() == 1


def test_slice_stim_array_rejects_bad_range(stim: StimArray) -> None:
    with pytest.raises(SysconExportError):
        slice_stim_array(stim, 5, 5)


def test_send_block_without_settings_fails(stim: StimArray, tmp_path, qtbot) -> None:
    host = _host([], None, tmp_path)
    with qtbot.waitSignal(host.block_failed):
        assert not host.send_block(Project(), stim, 0, 5)
    assert host.phase == "idle"


def test_send_block_full_cycle(stim: StimArray, tmp_path, qtbot, mocker) -> None:
    submitted: list = []
    host = _host(submitted, SysconExportSettings(config_text=""), tmp_path)
    result = SysconExportResult(
        tmp_path / "b.seq", tmp_path / "b.png", frozenset({0}), mocker.Mock()
    )
    export = mocker.patch.object(host_module, "export_stim_array", return_value=result)
    client = mocker.Mock()

    assert host.send_block(Project(), stim, 2, 6, label="probe")
    assert host.phase == "sending"
    assert export.call_args.kwargs["start_sample"] == 2
    assert export.call_args.kwargs["stop_sample"] == 6
    name, task = submitted[0]
    task(client)
    client.sequence_manager.import_sequence.assert_called_once_with(
        str(result.seq_path), "target"
    )

    with qtbot.waitSignal(host.block_sent) as sent:
        host.on_command_finished(name, None)
    assert sent.args == [None, result]
    assert host.phase == "awaiting_acquisition"

    with qtbot.waitSignal(host.block_completed) as completed:
        host.acquisition_imported()
    assert completed.args == [None, None]
    assert host.phase == "idle"


def test_import_failure_resets_phase(stim: StimArray, tmp_path, qtbot, mocker) -> None:
    submitted: list = []
    host = _host(submitted, SysconExportSettings(config_text=""), tmp_path)
    mocker.patch.object(host_module, "export_stim_array", return_value=mocker.Mock())
    host.send_block(Project(), stim, 0, 5)
    with qtbot.waitSignal(host.block_failed) as failed:
        host.on_command_failed(submitted[0][0], "boom")
    assert failed.args == [None, "boom"]
    assert host.phase == "idle"


def test_second_block_while_in_flight_is_rejected(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host = _host([], SysconExportSettings(config_text=""), tmp_path)
    mocker.patch.object(host_module, "export_stim_array", return_value=mocker.Mock())
    host.send_block(Project(), stim, 0, 5)
    with qtbot.waitSignal(host.block_failed):
        assert not host.send_block(Project(), stim, 0, 5)


def test_unrelated_command_results_are_ignored(
    stim: StimArray, tmp_path, mocker
) -> None:
    host = _host([], SysconExportSettings(config_text=""), tmp_path)
    mocker.patch.object(host_module, "export_stim_array", return_value=mocker.Mock())
    host.send_block(Project(), stim, 0, 5)
    host.on_command_finished("sequence_manager.import", None)
    assert host.phase == "sending"


def test_send_block_preserve_amplitude_unforces_binary(
    stim: StimArray, tmp_path, mocker
) -> None:
    host = _host([], SysconExportSettings(config_text=""), tmp_path)
    export = mocker.patch.object(
        host_module, "export_stim_array", return_value=mocker.Mock()
    )
    host.send_block(Project(), stim, 0, 5, preserve_amplitude=True)
    assert export.call_args.args[2].force_binary_intensity is False


class _FakeAcquisition(QtCore.QObject):
    block_armed = QtCore.Signal(str)
    block_recorded = QtCore.Signal(str, object)
    block_recording_failed = QtCore.Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.unavailable: str | None = None
        self.armed: list[BlockRecording] = []
        self.aborted: list[str] = []

    def block_recording_unavailable(self) -> str | None:
        return self.unavailable

    def arm_block(self, recording: BlockRecording) -> str:
        self.armed.append(recording)
        return f"run{len(self.armed)}"

    def abort_block(self, run_id: str) -> None:
        self.aborted.append(run_id)


class _FakeClient:
    """Records SysCon2 calls; Holo4D reports the states queued in ``states``."""

    def __init__(self, sequences: list[str], states: list[str]) -> None:
        self.calls: list[tuple] = []
        self.states = list(states)
        outer = self

        class _Manager:
            def import_sequence(self, path, target):
                outer.calls.append(("import", path, target))

            def get_sequences(self):
                return sequences

            def select_sequence(self, path):
                outer.calls.append(("select", path))

        class _Holo4D:
            def upload_sequence(self, runs):
                outer.calls.append(("upload", runs))

            def get_state(self):
                return outer.states.pop(0) if len(outer.states) > 1 else outer.states[0]

            def run_sequence(self, start, edge):
                outer.calls.append(("run", start, edge))

            def stop_sequence(self):
                outer.calls.append(("stop",))

        self.sequence_manager = _Manager()
        self.holo4d = _Holo4D()


def _automatic_host(tmp_path, mocker, *, acquisition=True):
    submitted: list = []
    host = _host(submitted, SysconExportSettings(config_text=""), tmp_path)
    result = SysconExportResult(
        tmp_path / "probe_001.seq", tmp_path / "b.png", frozenset({0}), mocker.Mock()
    )
    mocker.patch.object(host_module, "export_stim_array", return_value=result)
    mocker.patch.object(host_module, "_POLL_S", 0.0)
    fake = _FakeAcquisition() if acquisition else None
    host.set_acquisition(fake)
    return host, submitted, fake


def _recording(stim: StimArray) -> BlockRecording:
    return BlockRecording("probe", slice_stim_array(stim, 0, 5), 2.0, 1)


def _run(submitted, host, client) -> None:
    """Execute the last submitted SysCon2 task and report it like the manager."""
    name, task = submitted[-1]
    try:
        value = task(client)
    except Exception as exc:  # the manager reports any task error
        host.on_command_failed(name, str(exc))
        return
    host.on_command_finished(name, value)


def test_automatic_block_is_uploaded_recorded_and_run(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    owner = object()
    client = _FakeClient(
        ["other\\probe_001.seq", "target\\probe_001.seq"], ["Uploading", "Idle"]
    )

    assert host.send_block(
        Project(), stim, 0, 5, owner=owner, automatic=True, recording=_recording(stim)
    )
    assert host.phase == "preparing"
    with qtbot.waitSignal(host.block_sent) as sent:
        _run(submitted, host, client)
    assert sent.args[0] is owner
    assert client.calls == [
        ("import", str(tmp_path / "probe_001.seq"), "target"),
        ("select", "target\\probe_001.seq"),
        ("upload", 1),
    ]
    assert host.phase == "arming"
    assert acquisition.armed[0].tail_s == 2.0

    acquisition.block_armed.emit("other-run")  # not this block's recording
    assert host.phase == "arming"
    acquisition.block_armed.emit("run1")
    assert host.phase == "running"
    _run(submitted, host, client)
    assert client.calls[-1] == ("run", "auto", "rising")
    assert host.phase == "running"  # until the recording is in

    with qtbot.waitSignal(host.block_completed) as completed:
        acquisition.block_recorded.emit("run1", {"trace": "Rec :: dF/F"})
    assert completed.args == [owner, {"trace": "Rec :: dF/F"}]
    assert host.phase == "idle"


def test_automatic_block_without_recording_waits_for_the_sequence(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    client = _FakeClient(["target\\probe_001.seq"], ["Idle", "Busy", "Busy", "Idle"])
    host.send_block(Project(), stim, 0, 5, automatic=True)
    _run(submitted, host, client)
    assert host.phase == "running"
    assert not acquisition.armed
    with qtbot.waitSignal(host.block_completed) as completed:
        _run(submitted, host, client)
    assert completed.args == [None, {}]
    assert host.phase == "idle"


def test_recording_failure_fails_the_block_and_stops_the_sequence(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    client = _FakeClient(["target\\probe_001.seq"], ["Idle"])
    host.send_block(Project(), stim, 0, 5, automatic=True, recording=_recording(stim))
    _run(submitted, host, client)
    acquisition.block_armed.emit("run1")
    with qtbot.waitSignal(host.block_failed) as failed:
        acquisition.block_recording_failed.emit("run1", "first-frame timeout")
    assert "first-frame timeout" in failed.args[1]
    assert host.phase == "idle"
    assert acquisition.aborted == []  # the recording already ended
    _run(submitted, host, client)
    assert client.calls[-1] == ("stop",)


def test_recording_unavailable_refuses_the_block(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    acquisition.unavailable = "Connect to ScanImage first."
    with qtbot.waitSignal(host.block_failed) as failed:
        assert not host.send_block(
            Project(), stim, 0, 5, automatic=True, recording=_recording(stim)
        )
    assert failed.args[1] == "Connect to ScanImage first."
    assert not submitted


def test_missing_remote_sequence_fails_the_block(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    host, submitted, _acquisition = _automatic_host(tmp_path, mocker)
    host.send_block(Project(), stim, 0, 5, automatic=True)
    with qtbot.waitSignal(host.block_failed) as failed:
        _run(submitted, host, _FakeClient([], ["Idle"]))
    assert "does not list" in failed.args[1]
    assert host.phase == "idle"


def test_cancel_by_another_owner_is_ignored(stim: StimArray, tmp_path, mocker) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    owner = object()
    host.send_block(
        Project(), stim, 0, 5, owner=owner, automatic=True, recording=_recording(stim)
    )
    host.cancel(owner=object())
    assert host.phase == "preparing"
    host.cancel(owner=owner)
    assert host.phase == "idle"
    # The upload still completes on the hardware, but nothing follows from it.
    _run(submitted, host, _FakeClient(["target\\probe_001.seq"], ["Idle"]))
    assert host.phase == "idle"
    assert not acquisition.armed


def test_cancel_while_recording_aborts_and_stops(
    stim: StimArray, tmp_path, mocker
) -> None:
    host, submitted, acquisition = _automatic_host(tmp_path, mocker)
    client = _FakeClient(["target\\probe_001.seq"], ["Idle"])
    host.send_block(Project(), stim, 0, 5, automatic=True, recording=_recording(stim))
    _run(submitted, host, client)
    acquisition.block_armed.emit("run1")
    host.cancel()
    assert acquisition.aborted == ["run1"]
    assert submitted[-1][0].endswith(".stop")
    assert host.phase == "idle"


def test_not_connected_refuses_the_block(stim: StimArray, tmp_path, qtbot) -> None:
    host = _host(
        [], SysconExportSettings(config_text=""), tmp_path, is_connected=lambda: False
    )
    with qtbot.waitSignal(host.block_failed) as failed:
        assert not host.send_block(Project(), stim, 0, 5)
    assert "Connect to SysCon2" in failed.args[1]


def test_accepted_export_warnings_are_not_asked_again(
    stim: StimArray, tmp_path, mocker
) -> None:
    asked: list[list[str]] = []

    def confirm_warnings(warnings: list[str]) -> bool:
        asked.append(warnings)
        return True

    host = _host(
        [],
        SysconExportSettings(config_text=""),
        tmp_path,
        confirm_warnings=confirm_warnings,
    )
    export = mocker.patch.object(
        host_module, "export_stim_array", return_value=mocker.Mock()
    )
    host.send_block(Project(), stim, 0, 5)
    confirm = export.call_args.kwargs["confirm_warnings"]
    assert confirm(["No power LUT"])
    assert confirm(["No power LUT"])
    assert confirm(["No power LUT", "Pulses shifted"])
    assert asked == [["No power LUT"], ["Pulses shifted"]]
    host.reset_accepted_warnings()
    assert confirm(["No power LUT"])
    assert len(asked) == 3


def test_reawait_restores_the_last_manual_block(
    stim: StimArray, tmp_path, qtbot, mocker
) -> None:
    submitted: list = []
    host = _host(submitted, SysconExportSettings(config_text=""), tmp_path)
    mocker.patch.object(host_module, "export_stim_array", return_value=mocker.Mock())
    owner = object()
    host.send_block(Project(), stim, 0, 5, owner=owner)
    host.on_command_finished(submitted[0][0], None)
    host.acquisition_imported()
    host.reawait(object())
    assert host.phase == "idle"
    host.reawait(owner)
    assert host.phase == "awaiting_acquisition"
    with qtbot.waitSignal(host.block_completed) as completed:
        host.acquisition_imported()
    assert completed.args[0] is owner


def test_remote_sequence_path_prefers_the_target_folder() -> None:
    entries = ["a\\x.seq", "target\\X.SEQ", "target/y.seq"]
    assert remote_sequence_path(entries, "x.seq", "target") == "target\\X.SEQ"
    assert remote_sequence_path(entries, "x.seq", "elsewhere") == "target\\X.SEQ"
    with pytest.raises(RuntimeError):
        remote_sequence_path(entries, "z.seq", "target")
