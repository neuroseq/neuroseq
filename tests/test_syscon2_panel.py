from __future__ import annotations

import sys
from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import neuroseq.device_control.syscon2_connection_manager as connection_manager_module
import neuroseq.device_control.syscon2_panel as syscon2_panel_module
from neuroseq.device_control.syscon2_connection_manager import SysCon2ConnectionManager
from neuroseq.device_control.syscon2_panel import SysCon2Panel
from neuroseq.device_control.syscon2_protocol import encode_string
from neuroseq.export.syscon2_export_service import SysconExportError
from neuroseq.export.syscon2_export_service import SysconExportResult
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.export.syscon2_export_service import default_export_directory
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.wait_timeout import WAIT_TIMEOUT_MS


class FakeTransport:
    """Queues scripted responses per ``send()``; falls back to ``default_response`` once
    exhausted so a background poll tick sneaking in doesn't crash a test.
    """

    def __init__(
        self,
        responses: list[bytes] | None = None,
        default_response: bytes | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._default_response = default_response
        self.sent: list[bytes] = []
        self.closed = False
        self._buffer = b""

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        if self._responses:
            self._buffer += self._responses.pop(0)
        elif self._default_response is not None:
            self._buffer += self._default_response

    def recv(self, n: int) -> bytes:
        assert len(self._buffer) >= n, "not enough buffered response data"
        chunk, self._buffer = self._buffer[:n], self._buffer[n:]
        return chunk

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def panel(qtbot: QtBot):
    # A very long poll interval keeps the background GetState timer out of tests that script
    # exact response sequences for the command each test triggers explicitly.
    manager = SysCon2ConnectionManager(poll_interval_ms=3_600_000)
    widget = SysCon2Panel(manager=manager)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def _connect(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, transport
) -> None:
    # Drives the manager directly rather than clicking btn_connect: that button is disabled
    # off-Windows by design (see test_connect_button_disabled_off_windows), and these tests
    # run on Linux CI -- they exercise the manager/panel signal wiring, not the button itself.
    monkeypatch.setattr(
        connection_manager_module, "NamedPipeTransport", lambda: transport
    )
    with qtbot.waitSignal(panel._manager.connected, timeout=WAIT_TIMEOUT_MS):
        panel._manager.connect_to_syscon2()


def test_connect_updates_status_and_button_state(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connect(panel, qtbot, monkeypatch, FakeTransport())

    assert panel.status_label.text() == "Connected"
    assert not panel.btn_connect.isEnabled()
    assert panel.btn_disconnect.isEnabled()


def test_disconnect_button_resets_status(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTransport()
    _connect(panel, qtbot, monkeypatch, fake)

    with qtbot.waitSignal(panel._manager.disconnected, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_disconnect, QtCore.Qt.MouseButton.LeftButton)

    assert panel.status_label.text() == "Disconnected"
    assert fake.closed is True


@pytest.mark.skipif(
    sys.platform == "win32", reason="asserts the non-Windows disabled state"
)
def test_connect_button_disabled_off_windows(panel: SysCon2Panel) -> None:
    assert not panel.btn_connect.isEnabled()
    assert "Windows" in panel.btn_connect.toolTip()


def test_refresh_sequences_populates_list(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connect(
        panel,
        qtbot,
        monkeypatch,
        FakeTransport([encode_string("a.seq\r\nb.seq\r\n")]),
    )

    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)

    items = [
        panel.sequence_list.item(i).text() for i in range(panel.sequence_list.count())
    ]
    assert items == ["a.seq", "b.seq"]


def test_select_sequence_sends_command(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTransport([encode_string("a.seq\r\n"), encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)

    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)
    panel.sequence_list.setCurrentRow(0)

    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_select_sequence, QtCore.Qt.MouseButton.LeftButton)

    assert b"SelectSequence" in fake.sent[-1]
    assert b"a.seq" in fake.sent[-1]


def test_delete_sequence_requires_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    fake = FakeTransport([encode_string("a.seq\r\n"), encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)
    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)
    panel.sequence_list.setCurrentRow(0)

    question = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    )
    qtbot.mouseClick(panel.btn_delete_sequence, QtCore.Qt.MouseButton.LeftButton)
    question.assert_called_once()
    assert not any(b"DeleteSequence" in frame for frame in fake.sent)

    question.return_value = QtWidgets.QMessageBox.StandardButton.Yes
    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_delete_sequence, QtCore.Qt.MouseButton.LeftButton)
    assert any(b"DeleteSequence" in frame for frame in fake.sent)


def test_holo4d_run_requires_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    fake = FakeTransport(
        [encode_string("OK"), encode_string("State:Idle"), encode_string("OK")]
    )
    _connect(panel, qtbot, monkeypatch, fake)

    mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    )
    qtbot.mouseClick(panel.btn_holo4d_run, QtCore.Qt.MouseButton.LeftButton)
    assert fake.sent == []

    mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_holo4d_run, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: any(b"RunSequence" in frame for frame in fake.sent),
        timeout=WAIT_TIMEOUT_MS,
    )


def test_laser_emission_uses_selected_device_after_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    lightsources_text = "SN1:Chameleon\r\n"
    fake = FakeTransport([encode_string(lightsources_text), encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)

    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(
            panel.btn_refresh_lightsources, QtCore.Qt.MouseButton.LeftButton
        )
    assert panel._selected_device_and_description() == ("SN1", "Chameleon")
    assert panel.description_combo.currentText() == "Chameleon"

    mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_emission_on, QtCore.Qt.MouseButton.LeftButton)

    sent = fake.sent[-1]
    assert b"SetEmission" in sent
    assert b"SN1" in sent
    assert b"Chameleon" in sent


def test_command_failure_shows_warning(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    fake = FakeTransport([encode_string("Error: device busy")])
    _connect(panel, qtbot, monkeypatch, fake)

    warning = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning"
    )
    with qtbot.waitSignal(panel._manager.command_failed, timeout=WAIT_TIMEOUT_MS):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)

    warning.assert_called_once()
    assert "device busy" in warning.call_args.args[-1]


def test_send_to_syscon2_without_project_shows_warning(
    panel: SysCon2Panel, qtbot: QtBot, mocker
) -> None:
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    warning = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning"
    )

    qtbot.mouseClick(panel.btn_send_to_syscon2, QtCore.Qt.MouseButton.LeftButton)

    warning.assert_called_once()
    dialog_cls.assert_not_called()


def test_send_to_syscon2_opens_dialog_with_current_project(
    panel: SysCon2Panel, qtbot: QtBot, mocker
) -> None:
    project = Project()
    panel.set_project(project)
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")

    qtbot.mouseClick(panel.btn_send_to_syscon2, QtCore.Qt.MouseButton.LeftButton)

    dialog_cls.assert_called_once_with(
        project,
        parent=panel.window(),
        current_stim_name=None,
        output_folder=default_export_directory(),
    )
    dialog_cls.return_value.sequence_generated.connect.assert_called_once()
    dialog_cls.return_value.exec.assert_called_once()


def test_dialog_sequence_generated_shows_preview_and_imports_and_refreshes(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake = FakeTransport([encode_string("OK"), encode_string("pushed.seq\r\n")])
    _connect(panel, qtbot, monkeypatch, fake)

    png_path = tmp_path / "preview_overview.png"
    QtGui.QPixmap(10, 10).save(str(png_path))
    (tmp_path / "pushed.seq").write_text("sequence contents")
    seq_path = str(tmp_path / "pushed.seq")

    with qtbot.waitSignal(panel._manager.command_finished, timeout=WAIT_TIMEOUT_MS):
        panel._on_dialog_sequence_generated(seq_path, str(png_path), frozenset({1, 2}))

    pixmap = panel.preview_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()

    assert b"ImportSequence" in fake.sent[0]
    assert b"pushed.seq" in fake.sent[0]
    assert "Import confirmed" in panel.preview_status_label.text()

    qtbot.waitUntil(lambda: panel.sequence_list.count() == 1, timeout=WAIT_TIMEOUT_MS)
    assert panel.sequence_list.item(0).text() == "pushed.seq"


def test_dialog_sequence_generated_emits_targets_changed(
    panel: SysCon2Panel, qtbot: QtBot
) -> None:
    """Feature (#582): the Spatial Footprints panel learns which neurons a
    newly generated/sent sequence targets via this signal."""
    with qtbot.waitSignal(panel.targets_changed, timeout=WAIT_TIMEOUT_MS) as blocker:
        panel._on_dialog_sequence_generated(
            "unused.seq", "unused.png", frozenset({3, 5})
        )

    assert blocker.args == [frozenset()]
    assert "not imported" in panel.preview_status_label.text()


def test_dialog_sequence_generated_shows_error_for_unloadable_image(
    panel: SysCon2Panel, qtbot: QtBot, tmp_path
) -> None:
    bogus_path = tmp_path / "not_an_image.png"
    bogus_path.write_text("not a png")

    panel._show_preview(str(bogus_path))

    assert "Could not load preview" in panel.preview_label.text()


def _wheel_event(delta_y: int = 120):
    return QtGui.QWheelEvent(
        QtCore.QPointF(0, 0),
        QtCore.QPointF(0, 0),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, delta_y),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_scrolling_over_unfocused_spinbox_does_not_change_its_value(
    panel: SysCon2Panel, qtbot: QtBot
) -> None:
    panel.runs_spinbox.setValue(5)
    assert not panel.runs_spinbox.hasFocus()

    panel.runs_spinbox.wheelEvent(_wheel_event())

    assert panel.runs_spinbox.value() == 5


def test_scrolling_over_focused_spinbox_still_changes_its_value(
    panel: SysCon2Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel.runs_spinbox.setValue(5)
    monkeypatch.setattr(panel.runs_spinbox, "hasFocus", lambda: True)

    panel.runs_spinbox.wheelEvent(_wheel_event())

    assert panel.runs_spinbox.value() == 6


def test_scrolling_over_unfocused_combo_box_does_not_change_selection(
    panel: SysCon2Panel, qtbot: QtBot
) -> None:
    panel.start_condition_combo.setCurrentIndex(0)
    assert not panel.start_condition_combo.hasFocus()

    panel.start_condition_combo.wheelEvent(_wheel_event())

    assert panel.start_condition_combo.currentIndex() == 0


def test_holo4d_advanced_options_hidden_by_default_and_toggle_reveals_them(
    panel: SysCon2Panel, qtbot: QtBot
) -> None:
    with qtbot.waitExposed(panel, timeout=WAIT_TIMEOUT_MS):
        panel.show()
    assert not panel.holo4d_advanced_panel.isVisible()
    assert not panel.btn_toggle_holo4d_advanced.isChecked()

    panel.btn_toggle_holo4d_advanced.setChecked(True)

    assert panel.holo4d_advanced_panel.isVisible()
    assert panel.btn_toggle_holo4d_advanced.arrowType() == QtCore.Qt.ArrowType.DownArrow

    panel.btn_toggle_holo4d_advanced.setChecked(False)

    assert not panel.holo4d_advanced_panel.isVisible()
    assert (
        panel.btn_toggle_holo4d_advanced.arrowType() == QtCore.Qt.ArrowType.RightArrow
    )


def test_laser_advanced_controls_hidden_by_default_and_toggle_reveals_them(
    panel: SysCon2Panel, qtbot: QtBot
) -> None:
    with qtbot.waitExposed(panel, timeout=WAIT_TIMEOUT_MS):
        panel.show()
    assert not panel.laser_advanced_panel.isVisible()
    assert not panel.btn_toggle_laser_advanced.isChecked()

    panel.btn_toggle_laser_advanced.setChecked(True)

    assert panel.laser_advanced_panel.isVisible()
    assert panel.btn_toggle_laser_advanced.arrowType() == QtCore.Qt.ArrowType.DownArrow

    panel.btn_toggle_laser_advanced.setChecked(False)

    assert not panel.laser_advanced_panel.isVisible()
    assert panel.btn_toggle_laser_advanced.arrowType() == QtCore.Qt.ArrowType.RightArrow


def _settings() -> SysconExportSettings:
    return SysconExportSettings(config_text="nohash\nversion=3\n")


def test_export_current_without_settings_opens_dialog(
    panel: SysCon2Panel, qtbot: QtBot, mocker
) -> None:
    panel.set_project(Project())
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")

    qtbot.mouseClick(panel.btn_export_current, QtCore.Qt.MouseButton.LeftButton)

    dialog_cls.return_value.exec.assert_called_once()


def test_dialog_export_remembers_settings_and_preloads_next_dialog(
    panel: SysCon2Panel, mocker
) -> None:
    panel.set_project(Project())
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    dialog = dialog_cls.return_value
    dialog.export_settings.return_value = _settings()
    dialog.exec.side_effect = lambda: panel._on_dialog_sequence_generated(
        "x.seq", "x.png", frozenset()
    )

    panel.open_generator_dialog()
    assert panel.export_settings == _settings()

    panel.open_generator_dialog()
    dialog.apply_settings.assert_called_once_with(_settings())


@pytest.mark.parametrize("send_to_syscon", [True, False])
def test_generator_dialog_imports_only_when_sending_to_syscon(
    panel: SysCon2Panel, mocker, tmp_path, send_to_syscon: bool
) -> None:
    """The File > Export menu writes files only; it never uploads to SysCon2."""
    panel.set_project(Project())
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    import_sequence = mocker.patch.object(panel, "_import_sequence")
    mocker.patch.object(panel, "_show_preview")

    panel.open_generator_dialog(send_to_syscon=send_to_syscon)
    slot = dialog_cls.return_value.sequence_generated.connect.call_args.args[0]
    slot(str(tmp_path / "x.seq"), str(tmp_path / "x.png"), frozenset({1}))

    assert import_sequence.called is send_to_syscon
    assert panel.import_path_edit.text() == str(tmp_path / "x.seq")


def test_export_current_uses_remembered_settings_and_imports(
    panel: SysCon2Panel, mocker, tmp_path
) -> None:
    panel.set_project(Project())
    panel._export_settings = _settings()
    stim = StimArray("a", StimType.External, np.ones((4, 2)), "", 0.001)
    panel.set_current_stim_provider(lambda: stim)
    result = SysconExportResult(
        tmp_path / "a.seq", tmp_path / "a.png", frozenset({0, 1}), mocker.Mock()
    )
    export = mocker.patch.object(
        syscon2_panel_module, "export_stim_array", return_value=result
    )
    handler = mocker.patch.object(panel, "_on_dialog_sequence_generated")

    panel._on_export_current_clicked()

    assert export.call_args.args[1:3] == (stim, _settings())
    handler.assert_called_once_with(
        str(result.seq_path), str(result.overview_png_path), frozenset({0, 1})
    )


def test_export_current_failure_warns_and_does_not_import(
    panel: SysCon2Panel, mocker
) -> None:
    panel.set_project(Project())
    panel._export_settings = _settings()
    stim = StimArray("a", StimType.External, np.zeros((4, 2)), "", 0.001)
    panel.set_current_stim_provider(lambda: stim)
    warning = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning"
    )
    handler = mocker.patch.object(panel, "_on_dialog_sequence_generated")

    panel._on_export_current_clicked()

    warning.assert_called_once()
    handler.assert_not_called()


def test_upload_defaults_to_one_run(panel, mocker):
    panel._on_connected()
    mocker.patch.object(panel, "_confirm", return_value=True)
    submit = mocker.patch.object(panel, "_submit")
    panel.btn_holo4d_upload.click()
    client = mocker.Mock()
    submit.call_args.args[1](client)
    client.holo4d.upload_sequence.assert_called_once_with(1)


def test_endless_upload_requires_explicit_confirmation(panel, mocker):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    confirm = mocker.patch.object(panel, "_confirm", return_value=False)
    panel.runs_spinbox.setValue(0)
    panel.btn_holo4d_upload.click()
    submit.assert_not_called()
    assert "endless" in confirm.call_args.args[0]
    confirm.return_value = True
    panel.btn_holo4d_upload.click()
    client = mocker.Mock()
    submit.call_args.args[1](client)
    client.holo4d.upload_sequence.assert_called_once_with(0)


def test_shutter_open_requires_confirmation_but_close_is_immediate(panel, mocker):
    submit = mocker.patch.object(panel, "_submit")
    confirm = mocker.patch.object(panel, "_confirm", return_value=False)
    panel._on_lightsources({"SN1": ["Laser"]})
    panel.btn_shutter_open.click()
    confirm.assert_called_once()
    submit.assert_not_called()
    confirm.return_value = True
    panel.btn_shutter_open.click()
    client = mocker.Mock()
    submit.call_args.args[1](client)
    client.laser.set_shutter.assert_called_once_with("SN1", True)
    confirm.reset_mock()
    confirm.return_value = False
    panel.btn_shutter_close.click()
    submit.call_args.args[1](client)
    assert client.laser.set_shutter.call_args.args == ("SN1", False)
    confirm.assert_not_called()


def test_refreshing_same_project_preserves_export_settings(panel):
    project = Project()
    panel.set_project(project)
    settings = SysconExportSettings(config_text="config")
    panel._export_settings = settings
    panel.set_project(project)
    assert panel.export_settings is settings
    panel.set_project(Project())
    assert panel.export_settings is None
    # Kept with the project, so switching back restores them.
    panel.set_project(project)
    assert panel.export_settings is settings


def test_routines_get_default_settings_with_the_panel_power(panel):
    project = Project()
    project.syscon2_export_template_text = "template"
    panel.set_project(project)
    panel.target_power_spin.setValue(7.5)
    settings = panel.stimulation_host.export_settings()
    assert settings.config_text == "template"
    assert settings.target_power_mw == 7.5
    assert panel.export_settings is None  # nothing stored until chosen

    changed = []
    panel.export_settings_changed.connect(lambda: changed.append(True))
    panel.stimulation_host.set_export_settings(
        SysconExportSettings(config_text="x", target_power_mw=12.0)
    )
    assert project.syscon_export_settings.target_power_mw == 12.0
    assert panel.target_power_spin.value() == 12.0
    assert project.dirty
    assert changed


def test_export_folder_is_remembered_per_project(panel, mocker, tmp_path):
    first, second = Project(), Project()
    first.device_folders["syscon_export"] = str(tmp_path / "first")
    panel.set_project(first)
    assert panel._export_directory() == tmp_path / "first"
    panel.set_project(second)
    assert panel._export_directory() != tmp_path / "first"
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    dialog_cls.return_value.output_folder = tmp_path / "second"
    panel.open_generator_dialog()
    assert second.device_folders["syscon_export"] == str(tmp_path / "second")
    assert first.device_folders["syscon_export"] == str(tmp_path / "first")


def test_import_ack_tracks_latest_preview_and_snapshots_file(panel, mocker, tmp_path):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    targets = mocker.Mock()
    panel.targets_changed.connect(targets)
    source = tmp_path / "same.seq"
    source.write_text("first export")
    panel._import_sequence(str(source), frozenset({1}))
    first, first_task = submit.call_args.args
    source.write_text("second export")
    panel._import_sequence(str(source), frozenset({2}))
    second = submit.call_args.args[0]
    assert first != second
    client = mocker.Mock()
    first_task(client)
    snapshot = client.sequence_manager.import_sequence.call_args.args[0]
    assert Path(snapshot).read_text() == "first export"
    panel._on_command_finished(first, None)
    assert "pending" in panel.preview_status_label.text()
    targets.assert_called_with(frozenset())
    panel._on_command_finished(second, None)
    targets.assert_called_with(frozenset({2}))
    assert "Import confirmed" in panel.preview_status_label.text()


def test_confirmed_import_auto_selects_the_new_sequence(panel, mocker, tmp_path):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    source = tmp_path / "export_seq_x_1.seq"
    source.write_text("seq")
    panel._import_sequence(str(source), frozenset({1}))
    name = submit.call_args.args[0]
    panel._on_command_finished(name, None)
    assert submit.call_args.args[0] == "sequence_manager.get_sequences"

    panel._on_sequences(["old.seq", "neuroseq\\export_seq_x_1.seq"])
    select_name, task = submit.call_args.args
    assert select_name.startswith("sequence_manager.select.")
    assert panel.sequence_list.currentItem().text() == "neuroseq\\export_seq_x_1.seq"
    client = mocker.Mock()
    task(client)
    client.sequence_manager.select_sequence.assert_called_once_with(
        "neuroseq\\export_seq_x_1.seq"
    )
    # Only the import that just finished triggers a selection, not later refreshes.
    submit.reset_mock()
    panel._on_sequences(["old.seq", "neuroseq\\export_seq_x_1.seq"])
    submit.assert_not_called()


def test_failed_and_stale_imports_do_not_confirm_targets(panel, mocker, tmp_path):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    source = tmp_path / "export.seq"
    source.write_text("seq")
    panel._import_sequence(str(source), frozenset({1}))
    name = submit.call_args.args[0]
    panel._on_command_failed(name, "rejected")
    assert "Import failed" in panel.preview_status_label.text()
    panel._import_sequence(str(source), frozenset({1}))
    name = submit.call_args.args[0]
    panel._on_disconnected("lost connection")
    panel._on_command_finished(name, None)
    assert "No current confirmation" in panel.preview_status_label.text()
    assert not panel.btn_import_sequence.isEnabled()


@pytest.mark.parametrize(
    "target", ["C:\\outside", "../outside", "/outside", "non-ascii-ä"]
)
def test_import_rejects_invalid_target_directory(panel, mocker, tmp_path, target):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    source = tmp_path / "export.seq"
    source.write_text("seq")
    panel.target_directory_edit.setText(target)
    panel._import_sequence(str(source), frozenset({1}))
    submit.assert_not_called()
    assert "Not imported" in panel.preview_status_label.text()


def test_sequence_refresh_preserves_highlight_and_delete_refreshes(panel, mocker):
    panel._on_connected()
    panel._on_sequences(["one.seq", "two.seq"])
    panel.sequence_list.setCurrentRow(1)
    panel._on_sequences(["two.seq", "one.seq"])
    assert panel._selected_sequence_path() == "two.seq"
    submit = mocker.patch.object(panel, "_submit")
    panel._on_command_finished("sequence_manager.delete", None)
    assert submit.call_args.args[0] == "sequence_manager.get_sequences"


def test_export_dialog_preselects_editor_array(panel, mocker):
    panel.set_project(Project())
    stim = StimArray("selected", StimType.External, np.zeros((64, 1)), "", 0.0005208125)
    panel.set_current_stim_provider(lambda: stim)
    dialog = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    panel._on_export_current_clicked()
    assert dialog.call_args.kwargs["current_stim_name"] == "selected"


def test_selection_status_requires_ack_and_ignores_disconnected_reply(panel, mocker):
    panel._on_connected()
    panel._on_sequences(["one.seq"])
    panel.sequence_list.setCurrentRow(0)
    submit = mocker.patch.object(panel, "_submit")
    panel._on_select_sequence_clicked()
    name, task = submit.call_args.args
    assert "pending" in panel.selection_label.text()
    result = task(mocker.Mock())
    panel._on_command_finished(name, result)
    assert "Selection confirmed: one.seq" in panel.selection_label.text()
    panel._on_select_sequence_clicked()
    name = submit.call_args.args[0]
    panel._on_command_failed(name, "rejected")
    assert "Selection failed" in panel.selection_label.text()
    panel._on_select_sequence_clicked()
    name = submit.call_args.args[0]
    panel._on_disconnected("")
    panel._on_command_finished(name, "one.seq")
    assert "not confirmed" in panel.selection_label.text()


def test_panel_text_uses_stimulation_setup_wording(panel):
    assert "SysCon2" not in panel.btn_export_current.text()
    assert "SysCon2" not in panel.btn_export_current.toolTip()
    assert "SysCon2" not in panel.btn_connect.toolTip()


def test_quick_export_uses_dated_numbered_name(panel, mocker, tmp_path):
    panel.set_project(Project())
    panel._export_settings = _settings()
    panel.set_current_stim_provider(
        lambda: StimArray("a", StimType.External, np.ones((4, 2)), "", 0.001)
    )
    result = SysconExportResult(
        tmp_path / "a.seq", tmp_path / "a.png", frozenset(), mocker.Mock()
    )
    export = mocker.patch.object(
        syscon2_panel_module, "export_stim_array", return_value=result
    )
    mocker.patch.object(panel, "_on_dialog_sequence_generated")
    panel._on_export_current_clicked()
    name = export.call_args.args[4]
    assert name.startswith("export_seq_") and name.endswith("_1")


def _begin_upload_run(panel, mocker):
    panel._on_connected()
    mocker.patch.object(panel, "_confirm", return_value=True)
    submit = mocker.patch.object(panel, "_submit")
    panel._on_upload_clicked()
    return submit, submit.call_args.args[0]


def _finish_upload(panel, submit, mocker):
    """Complete an upload (ack + Idle) and press Run; returns (run_name, run_task)."""
    panel._on_command_finished(submit.call_args.args[0], None)
    panel._on_command_finished(submit.call_args.args[0], "Idle")
    assert panel.btn_holo4d_run.isEnabled()
    panel._on_run_clicked()
    return submit.call_args.args


def test_upload_waits_for_ready_then_run_tracks_actual_states(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    assert "Uploading" in panel.execution_label.text()
    assert not panel.btn_holo4d_upload.isEnabled()
    assert not panel.btn_holo4d_run.isEnabled()
    panel._on_command_finished(upload, None)
    state_request = submit.call_args.args[0]
    panel._on_command_finished(state_request, "Uploading")
    assert submit.call_count == 2
    panel._on_state_changed({"Holo4D": "Idle"})
    assert "Press Run" in panel.execution_label.text()
    assert panel.btn_holo4d_upload.isEnabled()
    assert panel.btn_holo4d_run.isEnabled()
    assert submit.call_count == 2  # upload alone never starts the run
    panel._on_run_clicked()
    run, task = submit.call_args.args
    client = mocker.Mock()
    task(client)
    client.holo4d.run_sequence.assert_called_once_with("auto", "rising")
    panel._on_command_finished(run, None)
    panel._on_state_changed({"Holo4D": "Busy"})
    assert "running" in panel.execution_label.text()
    panel._on_state_changed({"Holo4D": "Idle"})
    assert "finished" in panel.execution_label.text()
    assert panel.btn_holo4d_run.isEnabled()


def test_upload_ack_with_unexpected_state_name_does_not_stick(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    panel._on_command_finished(upload, None)
    panel._on_state_changed({"Holo4D": "Uploading"})
    assert not panel.btn_holo4d_run.isEnabled()
    assert "state: uploading" in panel.execution_label.text()
    panel._on_state_changed({"Holo4D": "Ready"})
    assert panel.btn_holo4d_run.isEnabled()
    assert "Press Run" in panel.execution_label.text()


def test_upload_ready_check_times_out_with_state_message(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    panel._on_command_finished(upload, None)
    panel._on_state_changed({"Holo4D": "Busy"})
    mocker.patch.object(panel._ready_elapsed, "elapsed", return_value=11_000)
    panel._update_progress_text()
    assert panel.btn_holo4d_run.isEnabled()
    assert "still 'busy'" in panel.execution_label.text()


def test_stop_failure_with_nothing_running_returns_to_idle(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    run, _ = _finish_upload(panel, submit, mocker)
    panel._on_command_finished(run, None)
    panel._on_state_changed({"Holo4D": "Busy"})
    assert "[hardware state: busy]" in panel.execution_label.text()
    warning = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning"
    )
    panel._on_holo4d_stop_clicked()
    stop = submit.call_args.args[0]
    panel._on_command_failed(stop, "Could not stop current sequence")
    assert panel.btn_holo4d_run.isEnabled()
    assert "Nothing to stop" in panel.execution_label.text()
    assert panel.execution_progress.maximum() != 0
    warning.assert_not_called()


def test_triggered_run_that_finishes_unobserved_explains_how_to_reset(panel, mocker):
    panel.start_condition_combo.setCurrentText("trigger1")
    submit, upload = _begin_upload_run(panel, mocker)
    run, _ = _finish_upload(panel, submit, mocker)
    panel._on_command_finished(run, None)
    panel._on_state_changed({"Holo4D": "Idle"})
    mocker.patch.object(panel._execution_elapsed, "elapsed", return_value=6_000)
    panel._update_progress_text()
    text = panel.execution_label.text()
    assert "press Stop to reset" in text and "hardware state: idle" in text


def test_failed_state_poll_is_visible(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    run, _ = _finish_upload(panel, submit, mocker)
    panel._on_command_finished(run, None)
    panel._manager.state_poll_failed.emit("Error: device busy")
    assert "unavailable (Error: device busy)" in panel.holo4d_state_label.text()
    assert "unavailable" in panel.execution_label.text()


def test_run_requires_confirmation(panel, mocker):
    panel._on_connected()
    submit = mocker.patch.object(panel, "_submit")
    mocker.patch.object(panel, "_confirm", return_value=False)
    panel.btn_holo4d_run.click()
    submit.assert_not_called()


def test_stop_during_upload_never_starts_run(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    panel._on_holo4d_stop_clicked()
    stop = submit.call_args.args[0]
    panel._on_command_finished(upload, None)
    assert submit.call_count == 2
    panel._on_command_finished(stop, None)
    assert "Stop acknowledged" in panel.execution_label.text()


def test_stop_cancels_run_already_queued(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    _, run_task = _finish_upload(panel, submit, mocker)
    panel._on_holo4d_stop_clicked()
    client = mocker.Mock()
    run_task(client)
    client.holo4d.run_sequence.assert_not_called()


def test_upload_failure_does_not_run(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    panel._on_command_failed(upload, "device rejected upload")
    assert submit.call_count == 1
    assert "failed" in panel.execution_label.text()
    assert panel.btn_holo4d_upload.isEnabled()


def test_trigger_wait_is_not_reported_as_completed(panel, mocker):
    panel.start_condition_combo.setCurrentText("trigger1")
    submit, upload = _begin_upload_run(panel, mocker)
    run, _ = _finish_upload(panel, submit, mocker)
    panel._on_command_finished(run, None)
    panel._on_state_changed({"Holo4D": "Idle"})
    assert "waiting for trigger1" in panel.execution_label.text()
    assert not panel.btn_holo4d_run.isEnabled()


def test_laser_controls_hidden_and_single_selector_defaults_940(panel, qtbot):
    panel._on_lightsources({"dummy_SN": ["dummy 1100 nm", "dummy 940 nm"]})
    with qtbot.waitExposed(panel, timeout=WAIT_TIMEOUT_MS):
        panel.show()
    assert panel.description_combo.currentText() == "940 nm"
    assert panel._selected_device_and_description() == ("dummy_SN", "dummy 940 nm")
    for widget in (
        panel.device_combo,
        panel.intensity_spinbox,
        panel.btn_set_intensity,
        panel.btn_toggle_laser_advanced,
        panel.laser_advanced_panel,
        panel.btn_refresh_lightsources,
    ):
        assert not widget.isVisible()
    assert panel.btn_refresh_status.isVisible()
    for button in panel.findChildren(QtWidgets.QPushButton):
        assert (
            button.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Fixed
        )


def test_generated_and_routine_exports_display_resizable_preview(
    panel, tmp_path, mocker, qtbot
):
    path = tmp_path / "preview.png"
    source = QtGui.QPixmap(1000, 400)
    source.fill(QtCore.Qt.GlobalColor.red)
    source.save(str(path))
    submit = mocker.patch.object(panel, "_submit")
    panel._on_dialog_sequence_generated("example.seq", str(path), frozenset({1}))
    with qtbot.waitExposed(panel, timeout=WAIT_TIMEOUT_MS):
        panel.show()
    panel.btn_toggle_preview.click()
    panel.resize(460, 850)
    qtbot.waitUntil(
        lambda: panel.preview_label.pixmap() is not None, timeout=WAIT_TIMEOUT_MS
    )
    assert not panel.preview_label.pixmap().isNull()
    assert panel.preview_label.pixmap().width() <= panel.preview_label.width()
    assert isinstance(panel.preview_label, QtWidgets.QScrollArea)
    assert panel.preview_label.maximumHeight() == 280
    fitted_size = panel.preview_label.pixmap().size()
    panel.preview_label.zoom_by(2.0)
    assert panel.preview_label.pixmap().width() > fitted_size.width()
    assert panel.preview_label.horizontalScrollBar().maximum() > 0
    panel.preview_label.fit_to_view()
    assert (
        panel.preview_label.pixmap().width() <= panel.preview_label.viewport().width()
    )
    submit.assert_not_called()
    assert "not imported" in panel.preview_status_label.text()
    result = SysconExportResult(
        tmp_path / "block.seq", path, frozenset({2}), mocker.Mock()
    )
    targets = mocker.Mock()
    panel.targets_changed.connect(targets)
    panel.stimulation_host.block_sent.emit(None, result)
    assert "Import confirmed: block.seq" in panel.preview_status_label.text()
    targets.assert_called_with(frozenset({2}))


def test_get_status_discovers_and_uses_raw_identifiers(panel, mocker):
    submit = mocker.patch.object(panel, "_submit")
    panel._on_refresh_status_clicked()
    assert submit.call_args.args[0] == "laser.get_lightsources"
    panel._on_command_finished("laser.get_lightsources", {"dummy_SN": ["dummy 940 nm"]})
    name, task = submit.call_args.args
    assert name == "laser.get_status"
    client = mocker.Mock()
    task(client)
    client.laser.get_lightsource_status.assert_called_once_with(
        "dummy_SN", "dummy 940 nm"
    )


def test_short_run_reports_idle_without_claiming_observed_completion(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    run, _ = _finish_upload(panel, submit, mocker)
    panel._on_command_finished(run, None)
    panel._on_state_changed({"Holo4D": "Idle"})
    assert "may have finished between polls" in panel.execution_label.text()
    assert panel.btn_holo4d_run.isEnabled()


def test_disconnect_cancels_pending_run_and_ignores_old_ack(panel, mocker):
    submit, upload = _begin_upload_run(panel, mocker)
    name, task = _finish_upload(panel, submit, mocker)
    panel._on_disconnected("lost pipe")
    task(mocker.Mock())
    panel._on_command_finished(name, None)
    assert "unknown" in panel.execution_label.text()
    assert not panel.btn_holo4d_run.isEnabled()


def test_preview_hidden_by_default_and_toggle_expands_it(panel, qtbot):
    with qtbot.waitExposed(panel, timeout=WAIT_TIMEOUT_MS):
        panel.show()
    assert not panel.preview_container.isVisible()
    assert panel.btn_toggle_preview.text() == "Show preview"
    panel.btn_toggle_preview.click()
    assert panel.preview_container.isVisible()
    assert panel.btn_toggle_preview.text() == "Hide preview"
    panel.btn_toggle_preview.click()
    assert not panel.preview_container.isVisible()


def test_selecting_sequence_shows_its_generated_preview(
    panel: SysCon2Panel, qtbot: QtBot, tmp_path
) -> None:
    from qtpy import QtGui

    png = tmp_path / "a_overview.png"
    image = QtGui.QImage(40, 20, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("red"))
    assert image.save(str(png))
    panel._remember_preview(str(tmp_path / "a.seq"), str(png))

    panel._show_preview_for_sequence("Experiment\\A.seq")
    assert not panel.preview_label._source.isNull()

    panel._show_preview_for_sequence("unknown.seq")
    assert "No local copy" in panel.preview_label.text()


def test_selecting_sequence_renders_preview_from_local_seq_file(
    panel: SysCon2Panel, tmp_path
) -> None:
    from tests.test_importer_syscon2 import _sample_holo4d_layout_seq_text

    (tmp_path / "Sequence2.seq").write_text(_sample_holo4d_layout_seq_text())
    panel.import_path_edit.setText(str(tmp_path / "Sequence2.seq"))

    panel._show_preview_for_sequence("Experiment\\Sequence2.seq")

    assert not panel.preview_label._source.isNull()


def test_scripts_go_to_a_dated_folder_on_the_syscon_pc(panel: SysCon2Panel) -> None:
    """Issue #614: one folder per experiment day instead of all in "neuroseq"."""
    import datetime

    from neuroseq.export.syscon2_export_service import default_target_directory

    assert default_target_directory(datetime.date(2026, 10, 15)) == (
        "neuroseq\\2026-10-15"
    )
    assert panel.target_directory_edit.text() == default_target_directory()


def test_export_dialog_starts_in_the_folder_used_last(
    qtbot: QtBot, mocker, tmp_path
) -> None:
    """Issue #614: no need to pick the output folder for every export."""
    from types import SimpleNamespace

    settings = SimpleNamespace(syscon_export_directory="")
    widget = SysCon2Panel(
        manager=SysCon2ConnectionManager(poll_interval_ms=3_600_000),
        settings=settings,  # type: ignore[arg-type]
    )
    qtbot.addWidget(widget)
    widget.set_project(Project())
    dialog_cls = mocker.patch.object(syscon2_panel_module, "StimulationGeneratorDialog")
    dialog_cls.return_value.output_folder = tmp_path / "chosen"

    widget.open_generator_dialog()
    assert dialog_cls.call_args.kwargs["output_folder"] == default_export_directory()
    assert settings.syscon_export_directory == str(tmp_path / "chosen")

    widget.open_generator_dialog()
    assert dialog_cls.call_args.kwargs["output_folder"] == tmp_path / "chosen"
    widget.shutdown()


def test_target_power_beside_the_export_button(
    panel: SysCon2Panel, mocker, tmp_path
) -> None:
    """Issue #616: changing only the power reuses the other export settings."""
    project = Project()
    panel.set_project(project)
    stim = StimArray("stim", StimType.Excitation, np.ones((10, 2)), "mW", 0.001)
    panel.set_current_stim_provider(lambda: stim)
    panel._show_export_settings(
        SysconExportSettings(config_text="cfg", spiralling=True, target_power_mw=5.0)
    )
    assert panel.target_power_spin.value() == pytest.approx(5.0)

    panel.target_power_spin.setValue(12.5)
    export = mocker.patch.object(
        syscon2_panel_module, "export_stim_array", side_effect=SysconExportError("x")
    )
    mocker.patch("neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning")
    panel._on_export_current_clicked()

    used = export.call_args.args[2]
    assert used.target_power_mw == pytest.approx(12.5)
    assert used.spiralling  # the other settings are kept


def test_project_without_settings_starts_from_the_default_power(panel):
    """Review finding 5: the previous project's power must not carry over."""
    first = Project()
    first.syscon_export_settings = SysconExportSettings(
        config_text="x", target_power_mw=60.0
    )
    panel.set_project(first)
    assert panel.target_power_spin.value() == 60.0
    second = Project()
    second.dirty = False
    panel.set_project(second)
    assert panel.target_power_spin.value() == SysconExportSettings("").target_power_mw
    assert panel.stimulation_host.export_settings().target_power_mw == 1.0
    assert second.syscon_export_settings is None
    assert not second.dirty


def test_host_announces_settings_changed_in_the_panel(panel):
    panel.set_project(Project())
    changed = []
    panel.stimulation_host.export_settings_changed.connect(lambda: changed.append(1))
    panel.target_power_spin.setValue(12.0)
    assert changed


def test_routine_blocks_count_as_execution_and_stop_reaches_them(panel, mocker):
    """Review: the panel must not offer Upload/Run, or let the workspace close,
    while a routine's block uses the hardware; its Stop ends that block."""
    panel.set_project(Project())
    panel._on_connected()
    host = panel.stimulation_host
    abort = mocker.patch.object(host, "abort")
    host.phase = "running"
    panel._update_connection_ui()
    assert panel.execution_in_progress
    assert not panel.btn_holo4d_upload.isEnabled()
    assert not panel.btn_holo4d_run.isEnabled()
    mocker.patch.object(panel, "_submit")
    panel._on_holo4d_stop_clicked()
    abort.assert_called_once()
    # A manual block waiting for the user's import does not use the hardware.
    host.phase = "awaiting_acquisition"
    panel._set_execution_progress("idle", "")
    assert not panel.execution_in_progress
    assert panel.btn_holo4d_upload.isEnabled()
    host.phase = "idle"


def test_routine_block_failures_do_not_pop_up(panel, mocker):
    """Rig test: a routine's failed run also showed "stim_host.1.run: Sequence not
    uploaded" and "stim_host.1.stop: ..." dialogs; the routine's log reports them."""
    warning = mocker.patch(
        "neuroseq.device_control.syscon2_panel.QtWidgets.QMessageBox.warning"
    )
    panel._on_command_failed("stim_host.1.run", "Sequence not uploaded")
    panel._on_command_failed("stim_host.1.stop", "Could not stop current sequence")
    warning.assert_not_called()
    panel._on_command_failed("sequence_manager.get_sequences", "pipe closed")
    warning.assert_called_once()
