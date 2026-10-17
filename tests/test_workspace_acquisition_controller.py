from unittest.mock import Mock

from qtpy import QtWidgets

from neuroseq.device_control.scanimage_connection import ScanImageConnection
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.workspace_acquisition_controller import WorkspaceAcquisitionController


def test_live_subscription_and_disposal_are_idempotent(qtbot, tmp_config_dir):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    connection = ScanImageConnection(parent)
    video = Mock()
    controller = WorkspaceAcquisitionController(
        connection,
        GuiSettings(),
        video,
        project=Project,
        current_stim=lambda: None,
        export_settings=lambda: None,
        video_widgets=lambda: [video],
        show_video=lambda: None,
        parent=parent,
    )
    frame = {"number": 1}
    controller.set_active(True)
    connection.frame_received.emit(frame)
    video.show_live_frame.assert_called_once_with(frame)

    controller.set_active(False)
    controller.set_active(False)
    connection.frame_received.emit(frame)
    assert video.show_live_frame.call_count == 1

    controller.set_active(True)
    connection.frame_received.emit(frame)
    assert video.show_live_frame.call_count == 2
    status_receivers = connection.receivers(connection.status_changed)
    ready_receivers = connection.receivers(connection.recording_ready)
    controller.open_image_acquisition_dialog()
    assert controller.dialog.isVisible()

    controller.dispose()
    controller.dispose()
    controller.set_active(True)
    connection.frame_received.emit(frame)
    assert video.show_live_frame.call_count == 2
    assert not controller.dialog.isVisible()
    assert connection.receivers(connection.status_changed) == status_receivers - 1
    assert connection.receivers(connection.recording_ready) == ready_receivers - 1
    assert not connection._closing
