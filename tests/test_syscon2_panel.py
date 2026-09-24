from __future__ import annotations

import sys

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

import neuroseq.syscon2_connection_manager as connection_manager_module
from neuroseq.syscon2_connection_manager import SysCon2ConnectionManager
from neuroseq.syscon2_panel import SysCon2Panel
from neuroseq.syscon2_protocol import encode_string


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
    with qtbot.waitSignal(panel._manager.connected, timeout=2000):
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

    with qtbot.waitSignal(panel._manager.disconnected, timeout=2000):
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

    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
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

    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)
    panel.sequence_list.setCurrentRow(0)

    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(panel.btn_select_sequence, QtCore.Qt.MouseButton.LeftButton)

    assert b"SelectSequence" in fake.sent[-1]
    assert b"a.seq" in fake.sent[-1]


def test_delete_sequence_requires_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    fake = FakeTransport([encode_string("a.seq\r\n"), encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)
    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)
    panel.sequence_list.setCurrentRow(0)

    question = mocker.patch(
        "neuroseq.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    )
    qtbot.mouseClick(panel.btn_delete_sequence, QtCore.Qt.MouseButton.LeftButton)
    question.assert_called_once()
    assert not any(b"DeleteSequence" in frame for frame in fake.sent)

    question.return_value = QtWidgets.QMessageBox.StandardButton.Yes
    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(panel.btn_delete_sequence, QtCore.Qt.MouseButton.LeftButton)
    assert any(b"DeleteSequence" in frame for frame in fake.sent)


def test_holo4d_run_requires_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    fake = FakeTransport([encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)

    mocker.patch(
        "neuroseq.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    )
    qtbot.mouseClick(panel.btn_holo4d_run, QtCore.Qt.MouseButton.LeftButton)
    assert fake.sent == []

    mocker.patch(
        "neuroseq.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(panel.btn_holo4d_run, QtCore.Qt.MouseButton.LeftButton)
    assert b"RunSequence" in fake.sent[-1]


def test_laser_emission_uses_selected_device_after_confirmation(
    panel: SysCon2Panel, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    lightsources_text = "SN1:Chameleon\r\n"
    fake = FakeTransport([encode_string(lightsources_text), encode_string("OK")])
    _connect(panel, qtbot, monkeypatch, fake)

    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
        qtbot.mouseClick(
            panel.btn_refresh_lightsources, QtCore.Qt.MouseButton.LeftButton
        )
    assert panel.device_combo.currentText() == "SN1"
    assert panel.description_combo.currentText() == "Chameleon"

    mocker.patch(
        "neuroseq.syscon2_panel.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    with qtbot.waitSignal(panel._manager.command_finished, timeout=2000):
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

    warning = mocker.patch("neuroseq.syscon2_panel.QtWidgets.QMessageBox.warning")
    with qtbot.waitSignal(panel._manager.command_failed, timeout=2000):
        qtbot.mouseClick(panel.btn_refresh_sequences, QtCore.Qt.MouseButton.LeftButton)

    warning.assert_called_once()
    assert "device busy" in warning.call_args.args[-1]
