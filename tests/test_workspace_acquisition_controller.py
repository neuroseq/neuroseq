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
    with qtbot.waitSignal(controller.block_armed) as armed:
        controller.scanimage_status(
            {"run": {"id": run_id, "state": "waiting_for_first_frame"}}
        )
    assert armed.args == [run_id]
    with qtbot.assertNotEmitted(controller.block_armed):
        controller.scanimage_status({"run": {"id": run_id, "state": "recording"}})

    tif = str(s.folder / f"{arguments['stem']}_00001.tif")
    with qtbot.waitSignal(controller.block_recorded, timeout=5000) as recorded:
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
        },
    ]
    data = s.project.experiments[name].data
    assert trace in data
    assert spikes in data
    assert extract.call_args.kwargs["saved_channels"] == [1, 2]
    assert extract.call_args.kwargs["functional_channel"] == 1
    assert deconvolve.call_args.args == (trace,)
    assert deconvolve.call_args.kwargs == {"threshold": 0.1}
    assert not controller.acquisition_in_progress


@pytest.mark.parametrize("state", ["failed", "aborted", "interrupted"])
def test_failed_block_recording_is_reported(setup, qtbot, state):
    s = setup
    _ready_for_blocks(s)
    run_id = s.controller.arm_block(_block())
    with qtbot.waitSignal(s.controller.block_recording_failed) as failed:
        s.controller.scanimage_status(
            {"run": {"id": run_id, "state": state, "error": "no trigger"}}
        )
    assert failed.args == [run_id, "no trigger"]


def test_extraction_failure_is_reported(setup, qtbot, mocker):
    s = setup
    _ready_for_blocks(s)
    mocker.patch.object(
        controller_module,
        "extract_recording_dff",
        side_effect=ValueError("masks do not match"),
    )
    run_id = s.controller.arm_block(_block())
    with qtbot.waitSignal(s.controller.block_recording_failed, timeout=5000) as failed:
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
    with qtbot.waitSignal(s.controller.block_recorded, timeout=5000):
        s.connection.recording_ready.emit(_ready_result(run_id, arguments, s.folder))
        s.connection.run_finished.emit(run_id)
    assert not failed
    assert not s.controller._blocks


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
    with qtbot.waitSignal(s.controller.block_recorded, timeout=5000):
        s.controller.scanimage_recording_ready(
            _ready_result(run_id, arguments, s.folder)
        )
    assert extract.call_args.kwargs["functional_channel"] == 2


def test_block_needs_its_functional_channel_saved(setup):
    s = setup
    _ready_for_blocks(s)
    controls = s.controller.controls
    controls._status(
        {"acq_state": "idle", "channels_available": 2, "saved_channels": 2}
    )
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
    assert not samples.exists()


def test_stimulus_samples_stay_once_the_recording_started(setup):
    s = setup
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    s.controller.start_scanimage_recording(request)
    samples = s.folder / f"{request['stem']}_{request['counter']:05d}_stimulus.npy"
    s.controller.scanimage_status(
        {"run": {"id": request["run_id"], "state": "waiting_for_first_frame"}}
    )
    s.controller.scanimage_status({"run": {"id": request["run_id"], "state": "failed"}})
    assert samples.is_file()


def test_rejected_record_command_leaves_no_stimulus_file(setup):
    s = setup
    s.command.return_value = False
    request = s.controller.controls.recording_request(frames=4, external_trigger=False)
    assert not s.controller.start_scanimage_recording(request)
    assert not list(s.folder.glob("*_stimulus.npy"))
