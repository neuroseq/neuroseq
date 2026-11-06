import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qtpy import QtWidgets

import neuroseq.workspace_acquisition_controller as controller_module
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.device_control.syscon2_stimulation_host import BlockRecording
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.workspace_acquisition_controller import WorkspaceAcquisitionController
from tests.wait_timeout import WAIT_TIMEOUT_MS


@pytest.fixture
def setup(qtbot, tmp_config_dir, tmp_path, mocker):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    connection = ScanImageConnection(parent)
    project = Project()
    project.device_folders["scanimage_recording"] = str(tmp_path)
    stim = StimArray("Stim", StimType.External, np.ones((10, 2)), "", 0.001)
    controller = WorkspaceAcquisitionController(
        connection,
        GuiSettings(),
        project=lambda: project,
        current_stim=lambda: stim,
        export_settings=lambda: SysconExportSettings(
            config_text="long template", target_power_mw=3.0
        ),
        video_widgets=lambda: [],
        parent=parent,
    )
    command = mocker.patch.object(connection, "command", return_value=True)
    return SimpleNamespace(
        parent=parent,
        connection=connection,
        controller=controller,
        project=project,
        command=command,
        folder=tmp_path,
    )


def _block(seconds: float = 1.0) -> BlockRecording:
    stim = StimArray(
        "Stim", StimType.External, np.ones((int(seconds * 1000), 2)), "", 0.001
    )
    return BlockRecording(
        "rheobase",
        stim,
        tail_s=2.0,
        seq_path="D:/runs/rheobase_001.seq",
        syscon_sequence="neuroseq\\rheobase_001.seq",
    )


def _ready_for_blocks(s) -> None:
    s.connection.connected = True
    s.connection._last_status = {"fps": 30.0}
    s.project.cellpose_data = Mock(
        stimulus_to_cellpose_mapping=None,
        segmentation_array=np.array([[-1, 0]]),
        mapping_strategy=None,
    )


def test_disposal_is_idempotent(qtbot, tmp_config_dir):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    connection = ScanImageConnection(parent)
    status_receivers = connection.receivers(connection.status_changed)
    ready_receivers = connection.receivers(connection.recording_ready)
    controller = WorkspaceAcquisitionController(
        connection,
        GuiSettings(),
        project=Project,
        current_stim=lambda: None,
        export_settings=lambda: None,
        video_widgets=lambda: [],
        parent=parent,
    )
    controller.dispose()
    controller.dispose()
    # Both the controller and its controls stop following the connection.
    assert connection.receivers(connection.status_changed) == status_receivers
    assert connection.receivers(connection.recording_ready) == ready_receivers


def test_recording_metadata_is_small_and_readable(setup):
    s = setup
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    assert s.controller.start_scanimage_recording(request)
    manifest = s.command.call_args.args[1]["manifest"]
    # No base64 samples (#628): the stimulus is identified, not embedded.
    assert set(manifest["stimulus"]) == {
        "name",
        "dt",
        "shape",
        "dtype",
        "sha256",
        "samples_file",
    }
    # The samples sit beside the TIFF instead (review finding 10).
    samples = s.folder / manifest["stimulus"]["samples_file"]
    assert samples.name == f"{request['stem']}_{request['counter']:05d}_stimulus.npy"
    np.testing.assert_array_equal(np.load(samples), np.ones((10, 2)))
    assert manifest["synchronization"] == "unverified"
    assert "config_text" not in manifest["syscon_export_settings"]
    assert manifest["syscon_export_settings"]["target_power_mw"] == 3.0
    assert len(json.dumps(manifest)) < 2000
    assert s.project.imaging_acquisitions == [manifest]


def test_status_of_a_run_the_project_no_longer_has_is_ignored(setup):
    s = setup
    s.controller.run_ids.add("gone")
    s.controller.scanimage_status({"run": {"id": "gone", "state": "recording"}})
    s.controller.scanimage_recording_ready(
        {"run_id": "gone", "status": "completed", "paths": [], "fps": 30}
    )
    assert s.project.experiments == {}


def test_recording_folder_belongs_to_the_project(setup, tmp_path):
    s = setup
    controls = s.controller.controls
    assert controls.directory_edit.text() == str(s.folder)
    other = tmp_path / "other"
    other.mkdir()
    controls.directory_edit.setText(str(other))
    controls._directory_edited()
    assert s.project.device_folders["scanimage_recording"] == str(other)


def test_block_recording_needs_connection_rate_and_segmentation(setup):
    s = setup
    controller = s.controller
    assert "Connect to ScanImage" in controller.block_recording_unavailable()
    s.connection.connected = True
    assert "Cellpose" in controller.block_recording_unavailable()
    s.project.cellpose_data = Mock()
    assert "frame rate" in controller.block_recording_unavailable()
    s.connection._last_status = {"fps": 30.0}
    assert controller.block_recording_unavailable() is None
    with pytest.raises(ValueError):
        s.connection.connected = False
        controller.arm_block(_block())


def test_block_is_armed_triggered_and_its_traces_extracted(setup, qtbot, mocker):
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    extract = mocker.patch.object(
        controller_module, "extract_recording_dff", return_value=trace
    )
    # OASIS itself is tested with the extraction; its first (numba) call can be
    # slow, and this test is about the controller's wiring.
    spikes = SpikeData("Deconvolved spikes", np.array([0]), np.array([1.0]))
    deconvolve = mocker.patch.object(
        controller_module, "deconvolve_dff", return_value=spikes
    )
    controller = s.controller

    run_id = controller.arm_block(_block(1.0))

    arguments = s.command.call_args.args[1]
    assert arguments["external_trigger"] is True
    assert arguments["frames"] == 90  # (1 s block + 2 s tail) at 30 Hz
    assert arguments["directory"] == str(s.folder)
    manifest = arguments["manifest"]
    assert manifest["synchronization"] == "stimulation_trigger"
    assert manifest["block"]["label"] == "rheobase"
    assert manifest["block"]["seq_path"] == "D:/runs/rheobase_001.seq"
    assert manifest["block"]["syscon_sequence"] == "neuroseq\\rheobase_001.seq"
    with qtbot.waitSignal(controller.block_armed, timeout=WAIT_TIMEOUT_MS) as armed:
        controller.scanimage_status(
            {"run": {"id": run_id, "state": "waiting_for_first_frame"}}
        )
    assert armed.args == [run_id]
    with qtbot.assertNotEmitted(controller.block_armed):
        controller.scanimage_status({"run": {"id": run_id, "state": "recording"}})

    tif = str(s.folder / f"{arguments['stem']}_00001.tif")
    with qtbot.waitSignal(
        controller.block_recorded, timeout=WAIT_TIMEOUT_MS
    ) as recorded:
        controller.scanimage_recording_ready(
            {
                "run_id": run_id,
                "status": "completed",
                "paths": [tif],
                "fps": 30.0,
                "duration_s": 3.0,
                "channels": [1, 2],
            }
        )
    name = arguments["name"]
    assert recorded.args == [
        run_id,
        {
            "trace": f"{name} :: dF/F",
            "spikes": f"{name} :: Deconvolved spikes",
            "experiment": name,
            "block_trace": trace,
            "block_spikes": spikes,
        },
    ]
    data = s.project.experiments[name].data
    assert trace in data
    assert spikes in data
    assert extract.call_args.kwargs["saved_channels"] == [1, 2]
    assert extract.call_args.kwargs["functional_channel"] == 1
    assert extract.call_args.kwargs["remove_artefact"] is True
    assert extract.call_args.kwargs["method"] == "suite2p"
    assert deconvolve.call_args.args == (trace,)
    assert deconvolve.call_args.kwargs == {"threshold": 0.1}
    assert not controller.acquisition_in_progress


@pytest.mark.parametrize("state", ["failed", "aborted", "interrupted"])
def test_failed_block_recording_is_reported(setup, qtbot, state):
    s = setup
    _ready_for_blocks(s)
    run_id = s.controller.arm_block(_block())
    with qtbot.waitSignal(
        s.controller.block_recording_failed, timeout=WAIT_TIMEOUT_MS
    ) as failed:
        s.controller.scanimage_status(
            {"run": {"id": run_id, "state": state, "error": "no trigger"}}
        )
    assert failed.args == [run_id, "no trigger"]


@pytest.mark.parametrize(
    ("before", "expected"),
    [
        ("waiting_for_first_frame", "before its first frame"),
        ("recording", "after its first frame"),
    ],
)
def test_aborted_block_says_when_scanimage_aborted(setup, qtbot, before, expected):
    s = setup
    _ready_for_blocks(s)
    failures = []
    s.controller.block_recording_failed.connect(lambda *args: failures.append(args))
    run_id = s.controller.arm_block(_block())
    s.controller.scanimage_status({"run": {"id": run_id, "state": before}})
    s.controller.scanimage_status({"run": {"id": run_id, "state": "aborted"}})
    # The block waits for the saved TIFF: a full one counts as completed.
    assert not failures
    assert run_id not in s.controller._deadlines
    s.controller.scanimage_status(
        {"run": {"id": run_id, "state": "finalization_failed"}}
    )
    assert len(failures) == 1
    message = failures[0][1]
    assert expected in message
    assert "s after it was armed" in message
    # 1 s block + 2 s tail at 30 Hz.
    assert "recording of 90 frames, 3.0 s at 30.00 Hz" in message


def test_armed_block_that_never_finishes_fails_at_its_deadline(
    setup, qtbot, monkeypatch
):
    s = setup
    _ready_for_blocks(s)
    # Block 0.1 s + tail 2 s: a margin of -2 s leaves a 0.1 s deadline.
    monkeypatch.setattr(controller_module, "BLOCK_DEADLINE_MARGIN_S", -2.0)
    run_id = s.controller.arm_block(_block(0.1))
    s.connection.run_id = run_id
    with qtbot.waitSignal(
        s.controller.block_recording_failed, timeout=WAIT_TIMEOUT_MS
    ) as failed:
        s.controller.scanimage_status(
            {"run": {"id": run_id, "state": "waiting_for_first_frame"}}
        )
    assert failed.args[0] == run_id
    assert "TTL Out 1" in failed.args[1]
    assert s.command.call_args.args[0] == "abort"
    # The aborted recording arriving afterwards does not fail the block again.
    with qtbot.assertNotEmitted(s.controller.block_recording_failed):
        s.controller.scanimage_status({"run": {"id": run_id, "state": "aborted"}})


def test_aborted_block_stops_its_deadline(setup):
    s = setup
    _ready_for_blocks(s)
    run_id = s.controller.arm_block(_block(0.1))
    s.controller.scanimage_status(
        {"run": {"id": run_id, "state": "waiting_for_first_frame"}}
    )
    assert run_id in s.controller._deadlines
    s.controller.abort_block(run_id)
    assert run_id not in s.controller._deadlines


def test_extraction_failure_is_reported(setup, qtbot, mocker):
    s = setup
    _ready_for_blocks(s)
    mocker.patch.object(
        controller_module,
        "extract_recording_dff",
        side_effect=ValueError("masks do not match"),
    )
    run_id = s.controller.arm_block(_block())
    with qtbot.waitSignal(
        s.controller.block_recording_failed, timeout=WAIT_TIMEOUT_MS
    ) as failed:
        s.controller.scanimage_recording_ready(
            {
                "run_id": run_id,
                "status": "completed",
                "paths": [],
                "fps": 30.0,
                "duration_s": 3.0,
                "channels": [1],
            }
        )
    assert "masks do not match" in failed.args[1]


def test_aborting_a_block_aborts_its_recording(setup):
    s = setup
    _ready_for_blocks(s)
    run_id = s.controller.arm_block(_block())
    s.connection.run_id = run_id
    s.controller.abort_block(run_id)
    s.command.assert_called_with("abort")
    s.connection.run_id = None


def _ready_result(run_id: str, arguments: dict, folder) -> dict:
    return {
        "run_id": run_id,
        "status": "completed",
        "paths": [str(folder / f"{arguments['stem']}_00001.tif")],
        "fps": 30.0,
        "duration_s": 3.0,
        "channels": [1, 2],
    }


def test_block_completes_when_the_run_ends_right_after_recording_ready(
    setup, qtbot, mocker
):
    """The connection emits run_finished immediately after recording_ready; the
    block must survive until its traces are extracted (review finding 1)."""
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    mocker.patch.object(controller_module, "extract_recording_dff", return_value=trace)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    failed = []
    s.controller.block_recording_failed.connect(lambda *args: failed.append(args))
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    with qtbot.waitSignal(s.controller.block_recorded, timeout=WAIT_TIMEOUT_MS):
        s.connection.recording_ready.emit(_ready_result(run_id, arguments, s.folder))
        s.connection.run_finished.emit(run_id)
    assert not failed
    assert not s.controller._blocks


def test_aborted_block_with_a_complete_recording_is_extracted(setup, qtbot, mocker):
    """ScanImage can fire acqAbort at the end of a triggered Grab."""
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    mocker.patch.object(controller_module, "extract_recording_dff", return_value=trace)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    failed = []
    s.controller.block_recording_failed.connect(lambda *args: failed.append(args))
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    s.controller.scanimage_status({"run": {"id": run_id, "state": "recording"}})
    s.controller.scanimage_status({"run": {"id": run_id, "state": "aborted"}})
    with qtbot.waitSignal(s.controller.block_recorded, timeout=WAIT_TIMEOUT_MS):
        s.connection.recording_ready.emit(_ready_result(run_id, arguments, s.folder))
        s.connection.run_finished.emit(run_id)
    assert not failed


def test_aborted_block_with_a_partial_recording_fails(setup, qtbot):
    s = setup
    _ready_for_blocks(s)
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    s.controller.scanimage_status({"run": {"id": run_id, "state": "recording"}})
    s.controller.scanimage_status({"run": {"id": run_id, "state": "aborted"}})
    result = _ready_result(run_id, arguments, s.folder) | {"status": "aborted"}
    with qtbot.waitSignal(
        s.controller.block_recording_failed, timeout=WAIT_TIMEOUT_MS
    ) as failed:
        s.connection.recording_ready.emit(result)
    assert "ScanImage aborted the recording" in failed.args[1]


def test_blocks_of_a_run_make_one_continuous_trace_and_spike_dataset(
    setup, qtbot, mocker
):
    s = setup
    _ready_for_blocks(s)
    mocker.patch.object(
        controller_module,
        "extract_recording_dff",
        side_effect=lambda *_a, **_k: TraceData(
            "dF/F", np.ones((1, 90)), np.array([0]), "dF/F", 1 / 30
        ),
    )
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        side_effect=lambda *_a, **_k: SpikeData(
            "Deconvolved spikes", np.array([0]), np.array([0.5])
        ),
    )
    experiments_before = set(s.project.experiments)
    payloads = []
    for offset in (0.0, 2.0):
        block = dataclasses.replace(_block(), group="rheobase_run", offset_s=offset)
        run_id = s.controller.arm_block(block)
        arguments = s.command.call_args.args[1]
        with qtbot.waitSignal(
            s.controller.block_recorded, timeout=WAIT_TIMEOUT_MS
        ) as recorded:
            s.controller.scanimage_recording_ready(
                _ready_result(run_id, arguments, s.folder)
            )
        payloads.append(recorded.args[1])
    # One data group for the run; no group per recording.
    assert set(s.project.experiments) - experiments_before == {"rheobase_run"}
    assert payloads[1]["trace"] == "rheobase_run :: dF/F"
    assert payloads[1]["block_trace"].traces.shape == (1, 90)
    trace = s.project.get_data("rheobase_run :: dF/F", TraceData)
    # The second block (at 2 s) overwrites the first recording's tail.
    assert trace.traces.shape == (1, 60 + 90)
    spikes = s.project.get_data("rheobase_run :: Deconvolved spikes", SpikeData)
    np.testing.assert_allclose(spikes.spike_times, [0.5, 2.5])


def test_block_measures_the_functional_channel_of_the_scanimage_tab(
    setup, qtbot, mocker
):
    s = setup
    _ready_for_blocks(s)
    s.controller.controls.channel_spin.setValue(2)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    extract = mocker.patch.object(
        controller_module, "extract_recording_dff", return_value=trace
    )
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    assert arguments["manifest"]["block"]["functional_channel"] == 2
    s.controller.controls.channel_spin.setValue(1)  # changed after arming
    with qtbot.waitSignal(s.controller.block_recorded, timeout=WAIT_TIMEOUT_MS):
        s.controller.scanimage_recording_ready(
            _ready_result(run_id, arguments, s.folder)
        )
    assert extract.call_args.kwargs["functional_channel"] == 2


def test_block_needs_its_functional_channel_saved(setup):
    s = setup
    _ready_for_blocks(s)
    controls = s.controller.controls
    controls._status({"acq_state": "idle", "channels_available": 2})
    controls.save_channel_checks[1].setChecked(True)
    controls.save_channel_checks[0].setChecked(False)
    controls.channel_spin.setValue(1)
    with pytest.raises(ValueError, match="not among the saved channels"):
        s.controller.arm_block(_block())
    s.command.assert_not_called()


def test_stimulus_samples_are_removed_when_scanimage_refuses(setup):
    s = setup
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    assert s.controller.start_scanimage_recording(request)
    samples = s.folder / f"{request['stem']}_{request['counter']:05d}_stimulus.npy"
    assert samples.is_file()
    s.controller.scanimage_status(
        {"run": {"id": request["run_id"], "state": "failed", "error": "busy"}}
    )
    s.connection.run_finished.emit(request["run_id"])
    assert not samples.exists()


@pytest.mark.parametrize(
    ("state", "tiff", "kept"),
    [
        ("aborted", False, False),  # e.g. first-frame timeout: no recording
        ("aborted", True, True),  # aborted mid-recording: a partial TIFF exists
        ("interrupted", False, True),  # outcome unknown: keep for recovery
        ("completed", True, True),
    ],
)
def test_stimulus_and_metadata_stay_only_with_a_recording(setup, state, tiff, kept):
    s = setup
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    s.controller.start_scanimage_recording(request)
    prefix = f"{request['stem']}_{request['counter']:05d}"
    samples = s.folder / f"{prefix}_stimulus.npy"
    metadata = s.folder / f"{prefix}.json"
    metadata.write_text("{}")  # written by the ScanImage helper
    if tiff:
        (s.folder / f"{prefix}.tif").write_bytes(b"")
    run_id = request["run_id"]
    s.controller.scanimage_status({"run": {"id": run_id, "state": "recording"}})
    s.controller.scanimage_status({"run": {"id": run_id, "state": state}})
    s.connection.run_finished.emit(run_id)
    assert samples.exists() == kept
    assert metadata.exists() == kept


def test_rejected_record_command_leaves_no_stimulus_file(setup):
    s = setup
    s.command.return_value = False
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    assert not s.controller.start_scanimage_recording(request)
    assert not list(s.folder.glob("*_stimulus.npy"))


def test_showing_a_recording_does_not_change_the_functional_channel(setup):
    """Review: the display may fall back, the measured channel must not."""
    s = setup
    controls = s.controller.controls
    controls.channel_spin.setValue(2)
    shown = s.controller.recorded_scanimage_channel({"channels": [1]}, 2)
    assert shown == 1
    assert controls.channel_spin.value() == 2


def test_next_block_does_not_wait_for_an_earlier_extraction(setup, qtbot, mocker):
    """Review: an abandoned block's extraction must not block or fail the next."""
    import threading

    s = setup
    _ready_for_blocks(s)
    release = threading.Event()
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)

    def slow_extract(*_args, **_kwargs):
        release.wait(WAIT_TIMEOUT_MS / 1000)
        return trace

    mocker.patch.object(controller_module, "extract_recording_dff", slow_extract)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    first = s.controller.arm_block(_block())
    first_arguments = s.command.call_args.args[1]
    s.controller.scanimage_recording_ready(
        _ready_result(first, first_arguments, s.folder)
    )
    s.controller.abort_block(first)  # the routine stopped meanwhile
    assert s.controller.block_recording_unavailable() is None
    second = s.controller.arm_block(_block())
    second_arguments = s.command.call_args.args[1]
    recorded = []
    s.controller.block_recorded.connect(lambda run_id, _p: recorded.append(run_id))
    s.controller.scanimage_recording_ready(
        _ready_result(second, second_arguments, s.folder)
    )
    assert s.controller.acquisition_in_progress
    release.set()
    qtbot.waitUntil(
        lambda: not s.controller.acquisition_in_progress, timeout=WAIT_TIMEOUT_MS
    )
    assert recorded == [second]  # the abandoned first block reports nothing


def test_traces_go_to_the_project_that_was_recorded_into(setup, qtbot, mocker):
    """Review: opening another project during extraction must not redirect data."""
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    mocker.patch.object(controller_module, "extract_recording_dff", return_value=trace)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    recorded_into = s.project
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    s.controller.scanimage_recording_ready(_ready_result(run_id, arguments, s.folder))
    other = Project()
    other.add_experiment(controller_module.Experiment(arguments["name"], 3.0, [], []))
    s.controller._project = lambda: other
    qtbot.waitUntil(
        lambda: not s.controller.acquisition_in_progress, timeout=WAIT_TIMEOUT_MS
    )
    assert trace in recorded_into.experiments[arguments["name"]].data
    assert other.experiments[arguments["name"]].data == []


def test_block_metadata_records_the_settings_it_was_delivered_with(setup):
    import dataclasses

    s = setup
    _ready_for_blocks(s)
    delivered = SysconExportSettings(config_text="cfg", target_power_mw=42.0)
    block = dataclasses.replace(_block(), export_settings=delivered)
    s.controller.arm_block(block)
    manifest = s.command.call_args.args[1]["manifest"]
    assert manifest["syscon_export_settings"]["target_power_mw"] == 42.0


def test_a_project_without_a_folder_starts_from_the_default_and_keeps_it(
    setup, tmp_path
):
    """Review: a new project shows the default folder, not the previous project's,
    and records the folder its recordings go to."""
    s = setup
    controls = s.controller.controls
    default = tmp_path / "default"
    default.mkdir()
    controls.settings.scanimage_recording_directory = str(default)
    other = tmp_path / "previous"
    other.mkdir()
    controls.set_directory(str(other))
    fresh = Project()
    s.controller._project = lambda: fresh
    s.controller.project_changed()
    assert controls.directory_edit.text() == str(default)
    request = controls.recording_request(frames=4, external_trigger=False)
    s.controller.start_scanimage_recording(request)
    assert fresh.device_folders["scanimage_recording"] == str(default)


def test_leaving_the_folder_field_unchanged_does_not_dirty_the_project(setup):
    s = setup
    s.project.dirty = False
    s.controller.controls._directory_edited()
    assert not s.project.dirty


def test_failing_to_add_the_traces_fails_the_block(setup, qtbot, mocker):
    """Review: an error while adding the data must not leave the routine waiting."""
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)
    mocker.patch.object(controller_module, "extract_recording_dff", return_value=trace)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    mocker.patch.object(
        s.project, "add_data_to_experiment", side_effect=KeyError("group gone")
    )
    run_id = s.controller.arm_block(_block())
    arguments = s.command.call_args.args[1]
    with qtbot.waitSignal(
        s.controller.block_recording_failed, timeout=WAIT_TIMEOUT_MS
    ) as failed:
        s.controller.scanimage_recording_ready(
            _ready_result(run_id, arguments, s.folder)
        )
    assert "could not be added" in failed.args[1]


def test_grab_fov_records_averages_and_hands_the_image_to_segmentation(setup, qtbot):
    import tifffile

    s = setup
    controls = s.controller.controls
    controls.fov_frames_spin.setValue(30)
    controls.channel_spin.setValue(2)
    with qtbot.waitSignal(controls.fov_requested, timeout=WAIT_TIMEOUT_MS) as requested:
        controls.fov_button.setEnabled(True)
        controls.fov_button.click()
    arguments = s.command.call_args.args[1]
    assert requested.args[0]["run_id"] == arguments["run_id"]
    assert (arguments["frames"], arguments["external_trigger"]) == (30, False)
    assert arguments["name"].startswith("FOV ")
    manifest = arguments["manifest"]
    assert manifest["purpose"] == "fov_for_segmentation"
    assert "stimulus" not in manifest  # an FOV snapshot is not stimulated
    assert not list(s.folder.glob("*_stimulus.npy"))

    prefix = f"{arguments['stem']}_{arguments['counter']:05d}"
    tiff = s.folder / f"{prefix}.tif"
    pages = []
    for _ in range(30):
        pages += [np.full((4, 5), 9, np.int16), np.full((4, 5), 50, np.int16)]
    tifffile.imwrite(tiff, np.stack(pages))
    with qtbot.waitSignal(s.controller.fov_ready, timeout=WAIT_TIMEOUT_MS) as ready:
        s.controller.scanimage_recording_ready(
            {
                "run_id": arguments["run_id"],
                "status": "completed",
                "paths": [str(tiff)],
                "fps": 30.0,
                "duration_s": 1.0,
                "channels": [1, 2],
            }
        )
    image_path = s.folder / f"{prefix}_mean.tif"
    assert ready.args == [str(image_path)]
    np.testing.assert_allclose(tifffile.imread(image_path), 50.0)  # channel 2
    # A snapshot for segmentation, not a recording to analyse.
    assert s.project.experiments == {}
    assert s.project.calcium_videos == []
    record = s.project.imaging_acquisitions[-1]
    assert record["fov_image"] == str(image_path)


def test_failed_fov_recording_says_so(setup):
    s = setup
    request = s.controller.controls.recording_request(frames=30, external_trigger=False)
    s.controller.start_fov_recording(request)
    s.controller.scanimage_status({"run": {"id": request["run_id"], "state": "failed"}})
    s.connection.run_finished.emit(request["run_id"])
    assert "without data" in s.controller.controls.message_label.text()


def test_routine_start_is_refused_without_folder_or_saved_channel(setup, tmp_path):
    """Review: refused before any block is uploaded, not in arm_block."""
    s = setup
    _ready_for_blocks(s)
    controls = s.controller.controls
    controls.set_directory(str(tmp_path / "missing"))
    assert "recording folder" in s.controller.block_recording_unavailable()
    controls.set_directory(str(s.folder))
    controls._status({"acq_state": "idle", "channels_available": 2})
    controls.save_channel_checks[1].setChecked(True)
    controls.save_channel_checks[0].setChecked(False)
    controls.channel_spin.setValue(1)
    assert "not among the saved channels" in s.controller.block_recording_unavailable()
    controls.channel_spin.setValue(2)
    assert s.controller.block_recording_unavailable() is None


def test_fov_uses_the_channel_chosen_when_it_was_grabbed(setup, qtbot):
    import tifffile

    s = setup
    controls = s.controller.controls
    controls.channel_spin.setValue(1)
    request = controls.recording_request(frames=2, external_trigger=False)
    s.controller.start_fov_recording(request)
    controls.channel_spin.setValue(2)  # changed while recording
    tiff = s.folder / f"{request['stem']}_{request['counter']:05d}.tif"
    tifffile.imwrite(
        tiff, np.stack([np.full((3, 4), v, np.int16) for v in (4, 9, 4, 9)])
    )
    with qtbot.waitSignal(s.controller.fov_ready, timeout=WAIT_TIMEOUT_MS) as ready:
        s.controller.scanimage_recording_ready(
            {
                "run_id": request["run_id"],
                "status": "completed",
                "paths": [str(tiff)],
                "fps": 30.0,
                "duration_s": 0.1,
                "channels": [1, 2],
            }
        )
    np.testing.assert_allclose(tifffile.imread(ready.args[0]), 4.0)  # channel 1


def test_fov_interrupted_by_a_project_change_says_why(setup):
    s = setup
    request = s.controller.controls.recording_request(frames=2, external_trigger=False)
    s.controller.start_fov_recording(request)
    s.controller._project = Project  # another project now
    s.controller.scanimage_recording_ready(
        {"run_id": request["run_id"], "status": "completed", "paths": [], "fps": 30}
    )
    s.connection.run_finished.emit(request["run_id"])
    message = s.controller.controls.message_label.text()
    assert "project changed" in message and "without data" not in message


def test_fov_auto_brightness_choice_is_remembered(setup):
    from neuroseq.gui_settings import GuiSettings

    check = setup.controller.controls.fov_auto_brightness_check
    assert not check.isChecked()
    check.setChecked(True)
    assert GuiSettings().fov_auto_brightness is True


def test_slow_suite2p_extraction_is_noted_and_the_block_carries_on(
    setup, qtbot, mocker
):
    import threading

    s = setup
    _ready_for_blocks(s)
    release = threading.Event()
    trace = TraceData("dF/F", np.zeros((1, 90)), np.array([0]), "dF/F", 1 / 30)

    def slow_suite2p(*_args, method, on_registered, **_kwargs):
        assert method == "suite2p"
        release.wait(WAIT_TIMEOUT_MS / 1000)
        on_registered({"reference": "provided", "max_shift_px": 2.25})
        return trace

    mocker.patch.object(controller_module, "extract_recording_dff", slow_suite2p)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([]), np.array([])),
    )
    block = dataclasses.replace(_block(), extraction="suite2p", extraction_slow_s=0.05)
    run_id = s.controller.arm_block(block)
    arguments = s.command.call_args.args[1]
    events: list[tuple] = []
    s.controller.block_recording_note.connect(
        lambda rid, message: events.append(("note", rid, message))
    )
    s.controller.block_recorded.connect(lambda rid, _p: events.append(("done", rid)))
    s.controller.block_recording_failed.connect(lambda *a: events.append(("failed",)))

    def notes() -> list[str]:
        return [event[2] for event in events if event[0] == "note"]

    try:
        s.controller.scanimage_recording_ready(
            _ready_result(run_id, arguments, s.folder)
        )
        qtbot.waitUntil(
            lambda: any("still waiting" in note for note in notes()),
            timeout=WAIT_TIMEOUT_MS,
        )
        assert (
            "more than 0 s; still waiting" in s.controller.controls.message_label.text()
        )
    finally:
        release.set()
    qtbot.waitUntil(lambda: ("done", run_id) in events, timeout=WAIT_TIMEOUT_MS)
    assert ("failed",) not in events
    logged = notes()
    assert logged[0].startswith("Recording complete: ")
    assert logged[1] == (
        "Extracting ΔF/F traces with Suite2p (motion correction, neuropil "
        "subtraction); stimulation artefact removed."
    )
    assert logged[-2].startswith("Traces extracted in ")
    # The motion correction is reported before the block completes.
    assert logged[-1] == (
        "Motion correction: largest shift 2.2 px, aligned to the segmented FOV image."
    )
    assert events[-1] == ("done", run_id)


def test_registration_note_explains_a_block_aligned_to_itself() -> None:
    note = controller_module.registration_note(
        {"reference": "movie", "max_shift_px": 1.0}
    )
    assert "1.0 px" in note and "drift since the FOV was grabbed" in note


def test_extraction_steps_are_reported_for_the_log_and_progress_bar(
    setup, qtbot, mocker
):
    s = setup
    _ready_for_blocks(s)
    trace = TraceData("dF/F", np.zeros((2, 90)), np.array([0, 1]), "dF/F", 1 / 30)

    def staged_extract(*_args, progress, **_kwargs):
        progress("Converting TIFF to binary", 0.1)
        progress("Converting TIFF to binary", 0.2)  # same stage: logged once
        progress("Extracting fluorescence traces", 0.5)
        return trace

    mocker.patch.object(controller_module, "extract_recording_dff", staged_extract)
    mocker.patch.object(
        controller_module,
        "deconvolve_dff",
        return_value=SpikeData("Deconvolved spikes", np.array([1]), np.array([0.5])),
    )
    notes: list[str] = []
    progress: list[tuple[str, int]] = []
    ended: list[str] = []
    s.controller.block_recording_note.connect(lambda _rid, note: notes.append(note))
    s.controller.block_extraction_progress.connect(
        lambda _rid, stage, percent: progress.append((stage, percent))
    )
    s.controller.block_extraction_ended.connect(ended.append)
    block = dataclasses.replace(
        _block(), extraction="roi_average", remove_artefact=False
    )
    run_id = s.controller.arm_block(block)
    arguments = s.command.call_args.args[1]
    s.controller.scanimage_status(
        {"run": {"id": run_id, "state": "waiting_for_first_frame"}}
    )
    assert notes == [
        "ScanImage armed: 90 frames at 30.00 Hz, waiting for SysCon2's trigger."
    ]
    s.controller.scanimage_recording_ready(_ready_result(run_id, arguments, s.folder))
    qtbot.waitUntil(lambda: ended == [run_id], timeout=WAIT_TIMEOUT_MS)

    assert progress[0] == ("Starting", 0)
    assert ("Extracting fluorescence traces", 45) in progress
    assert ("Deconvolving ΔF/F into spikes (OASIS)", 95) in progress
    stages = [
        note.split("]: ", 1)[1]
        for note in notes
        if note.startswith("Trace extraction [")
    ]
    assert stages == [
        "Converting TIFF to binary",
        "Extracting fluorescence traces",
        "Deconvolving ΔF/F into spikes (OASIS)",
    ]
    assert (
        "Extracting ΔF/F traces with ROI average; stimulation artefact kept." in notes
    )
    assert notes[-1] == (
        "Traces extracted in " + notes[-1].split("Traces extracted in ", 1)[1]
    )
    assert "2 cells, 90 frames; 1 deconvolved spike(s)." in notes[-1]
