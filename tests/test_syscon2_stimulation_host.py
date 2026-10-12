from __future__ import annotations

import brian2.numpy_ as np
import pytest

import neuroseq.syscon2_stimulation_host as host_module
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.syscon2_export_service import SysconExportError
from neuroseq.syscon2_export_service import SysconExportResult
from neuroseq.syscon2_export_service import SysconExportSettings
from neuroseq.syscon2_export_service import slice_stim_array
from neuroseq.syscon2_stimulation_host import SysCon2StimulationHost


@pytest.fixture()
def stim() -> StimArray:
    return StimArray("a", StimType.External, np.ones((10, 2)), "", 0.001)


def _host(submitted, settings, tmp_path):
    return SysCon2StimulationHost(
        lambda name, fn: submitted.append((name, fn)),
        settings_provider=lambda: settings,
        target_directory_provider=lambda: "target",
        output_dir=tmp_path,
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
    assert export.call_args.kwargs == {"start_sample": 2, "stop_sample": 6}
    name, task = submitted[0]
    task(client)
    client.sequence_manager.import_sequence.assert_called_once_with(
        str(result.seq_path), "target"
    )

    with qtbot.waitSignal(host.block_sent) as sent:
        host.on_command_finished(name, None)
    assert sent.args == [result]
    assert host.phase == "awaiting_acquisition"

    with qtbot.waitSignal(host.block_completed):
        host.acquisition_imported()
    assert host.phase == "idle"


def test_import_failure_resets_phase(stim: StimArray, tmp_path, qtbot, mocker) -> None:
    submitted: list = []
    host = _host(submitted, SysconExportSettings(config_text=""), tmp_path)
    mocker.patch.object(host_module, "export_stim_array", return_value=mocker.Mock())
    host.send_block(Project(), stim, 0, 5)
    with qtbot.waitSignal(host.block_failed) as failed:
        host.on_command_failed(submitted[0][0], "boom")
    assert failed.args == ["boom"]
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
