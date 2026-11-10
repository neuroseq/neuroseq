from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.chat_widget import ChatWidget
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.wait_timeout import WAIT_TIMEOUT_MS


class FakeBackend(LLMBackend):
    name = "fake"
    label = "Fake"

    def __init__(self, tokens=None, error: str | None = None, delay: float = 0.0):
        self._tokens = tokens or ["Hello", " ", "world"]
        self._error = error
        self._delay = delay
        self.seen_messages: list[list[Message]] = []

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def stream_chat(self, messages, *, model, temperature=0.7, max_tokens=None):
        self.seen_messages.append(list(messages))
        if self._error:
            raise LLMBackendError(self._error)
        for token in self._tokens:
            if self._delay:
                time.sleep(self._delay)
            yield token


@dataclass
class _FakeSettings:
    llm_provider: str = "fake"
    llm_model: str = "fake-model"
    llm_ollama_base_url: str = "http://localhost:11434"
    llm_openai_base_url: str = ""
    llm_codex_executable: str = ""
    llm_claude_executable: str = ""
    llm_private_project_data: dict = field(default_factory=dict)

    def llm_project_data_private(self, provider: str) -> bool:
        return bool(self.llm_private_project_data.get(provider, provider != "ollama"))

    def set_llm_project_data_private(self, provider: str, private: bool) -> None:
        self.llm_private_project_data = {
            **self.llm_private_project_data,
            provider: private,
        }


def _make_widget(
    qtbot: QtBot,
    monkeypatch,
    backend_factory,
    *,
    settings: _FakeSettings | None = None,
) -> ChatWidget:
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.build_static_prompt", lambda: "STATIC"
    )
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.GuiSettings",
        lambda: settings or _FakeSettings(),
    )
    monkeypatch.setattr(ChatWidget, "_make_backend", backend_factory)
    widget = ChatWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget.isVisible(), timeout=WAIT_TIMEOUT_MS)
    return widget


def _make_project() -> Project:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stimulus",
            StimType.Excitation,
            np.zeros((10, 3), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    project.update_limits()
    return project


def test_send_appends_user_and_assistant_to_history(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend(tokens=["Hi!"])
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )

    assert widget._history == [
        Message(Role.USER, "hello"),
        Message(Role.ASSISTANT, "Hi!"),
    ]
    assert "hello" in widget.history_view.toPlainText()
    assert "Hi!" in widget.history_view.toPlainText()


@pytest.mark.parametrize("key", [QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter])
def test_enter_sends_message(qtbot: QtBot, monkeypatch, key) -> None:
    backend = FakeBackend(tokens=["Hi!"])
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)
    widget.input_edit.setPlainText("hello")

    qtbot.keyClick(widget.input_edit, key)
    qtbot.waitUntil(lambda: widget._worker is None)

    assert widget._history == [
        Message(Role.USER, "hello"),
        Message(Role.ASSISTANT, "Hi!"),
    ]
    assert widget.input_edit.toPlainText() == ""


def test_shift_enter_inserts_newline(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend()
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)
    qtbot.keyClicks(widget.input_edit, "first")
    qtbot.keyClick(
        widget.input_edit,
        QtCore.Qt.Key.Key_Return,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    qtbot.keyClicks(widget.input_edit, "second")

    assert widget.input_edit.toPlainText() == "first\nsecond"
    assert widget._history == []
    assert backend.seen_messages == []


def test_enter_does_not_send_blank_message(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend()
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)
    widget.input_edit.setPlainText("   ")
    qtbot.keyClick(widget.input_edit, QtCore.Qt.Key.Key_Return)

    assert widget._history == []
    assert backend.seen_messages == []
    assert widget.input_edit.toPlainText() == "   "


def test_send_disabled_while_streaming(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend(tokens=["A", "B"], delay=0.2)
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()

    qtbot.waitUntil(lambda: widget.stop_button.isEnabled(), timeout=WAIT_TIMEOUT_MS)
    assert not widget.send_button.isEnabled()
    assert widget.stop_button.isEnabled()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )
    assert widget.send_button.isEnabled()
    assert not widget.stop_button.isEnabled()


def test_error_surfaces_in_status_line_and_does_not_append_to_history(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    backend = FakeBackend(error="boom")
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text().startswith("Error:"), timeout=WAIT_TIMEOUT_MS
    )

    assert widget._history == [Message(Role.USER, "hello")]
    assert "Error: boom" in widget.status_label.text()


def test_clear_chat_resets_history_and_view(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend(tokens=["Hi!"])
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )
    widget.clear_button.click()

    assert widget._history == []
    assert widget.history_view.toPlainText() == ""


def test_set_project_updates_snapshot_used_in_next_request(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    backend = FakeBackend(tokens=["ok"])
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("first")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )

    widget.set_project(_make_project())
    widget.input_edit.setPlainText("second")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )

    assert "no project loaded" in backend.seen_messages[0][0].content
    assert "Neurons: 3" in backend.seen_messages[1][0].content


def test_reload_backend_after_settings_change(qtbot: QtBot, monkeypatch) -> None:
    first_backend = FakeBackend(tokens=["one"])
    second_backend = FakeBackend(tokens=["two"])
    settings = _FakeSettings()
    active_backend = {"backend": first_backend}

    widget = _make_widget(
        qtbot,
        monkeypatch,
        lambda self, current_settings: active_backend["backend"],
        settings=settings,
    )

    settings.llm_provider = "openai"
    settings.llm_model = "gpt-test"
    active_backend["backend"] = second_backend
    widget.reload_backend()
    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )

    assert "Openai · gpt-test" in widget.badge_label.text()
    assert second_backend.seen_messages


def test_stop_button_interrupts_stream(qtbot: QtBot, monkeypatch) -> None:
    backend = FakeBackend(tokens=["a", "b", "c", "d"], delay=0.15)
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: "a" in widget.history_view.toPlainText(), timeout=WAIT_TIMEOUT_MS
    )
    widget.stop_button.click()
    qtbot.waitUntil(
        lambda: widget.status_label.text() == "Idle", timeout=WAIT_TIMEOUT_MS
    )

    assert len(widget._history) == 2
    assert widget._history[1].role == Role.ASSISTANT
    assert 0 < len(widget._history[1].content) < 4
    assert widget.send_button.isEnabled()


class _SilentBackend(FakeBackend):
    """Ends the stream before any token, as Stop before the first token does."""

    def stream_chat(self, messages, *, model, temperature=0.7, max_tokens=None):
        self.seen_messages.append(list(messages))
        yield from ()


def test_stream_without_tokens_adds_no_empty_assistant_message(
    qtbot: QtBot, monkeypatch
) -> None:
    backend = _SilentBackend()
    widget = _make_widget(qtbot, monkeypatch, lambda self, settings: backend)

    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)

    assert [message.role for message in widget._history] == [Role.USER]
    assert all(message.content for message in widget._history)


def test_private_data_setting_controls_snapshot_and_badge(
    qtbot: QtBot, monkeypatch
) -> None:
    backend = FakeBackend(tokens=["ok"])
    settings = _FakeSettings()
    widget = _make_widget(
        qtbot, monkeypatch, lambda self, current: backend, settings=settings
    )
    widget.set_project(_make_project())
    assert "data not shared" in widget.badge_label.text()

    def send() -> str:
        count = len(backend.seen_messages)
        widget.input_edit.setPlainText("hello")
        widget.send_button.click()
        qtbot.waitUntil(
            lambda: (
                len(backend.seen_messages) > count
                and widget.status_label.text() == "Idle"
            ),
            timeout=WAIT_TIMEOUT_MS,
        )
        return backend.seen_messages[-1][0].content

    private_prompt = send()
    assert "'stimulus'" not in private_prompt
    assert widget._read_access is not None and widget._read_access.private_data

    settings.set_llm_project_data_private("fake", False)
    widget.reload_backend()
    assert "data not shared" not in widget.badge_label.text()
    shared_prompt = send()
    assert "'stimulus'" in shared_prompt
    assert not widget._read_access.private_data


def test_send_rereads_settings_changed_outside_this_panel(
    qtbot: QtBot, monkeypatch
) -> None:
    # Detached panels and other tabs never get the settings dialog's signal.
    backends = {"fake": FakeBackend(tokens=["one"]), "openai": FakeBackend(["two"])}
    settings = _FakeSettings()
    settings.set_llm_project_data_private("fake", False)
    settings.set_llm_project_data_private("openai", True)
    widget = _make_widget(
        qtbot,
        monkeypatch,
        lambda self, current: backends[current.llm_provider],
        settings=settings,
    )
    widget.set_project(_make_project())

    settings.llm_provider = "openai"
    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)

    assert not backends["fake"].seen_messages
    assert "'stimulus'" not in backends["openai"].seen_messages[0][0].content
    assert widget._read_access is not None and widget._read_access.private_data
    assert "data not shared" in widget.badge_label.text()


def _send_and_wait(qtbot: QtBot, widget: ChatWidget, text: str = "hello") -> None:
    widget.input_edit.setPlainText(text)
    widget.send_button.click()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)


def test_provider_change_clears_history_with_a_note(qtbot: QtBot, monkeypatch) -> None:
    backends = {"fake": FakeBackend(["one"]), "openai": FakeBackend(["two"])}
    settings = _FakeSettings()
    widget = _make_widget(
        qtbot,
        monkeypatch,
        lambda self, current: backends[current.llm_provider],
        settings=settings,
    )
    _send_and_wait(qtbot, widget, "first")
    settings.llm_model = "other-model"
    widget.reload_backend()
    assert len(widget._history) == 2

    settings.llm_provider = "openai"
    widget.reload_backend()
    assert widget._history == []
    assert "provider changed" in widget.history_view.toPlainText()

    _send_and_wait(qtbot, widget, "second")
    sent = backends["openai"].seen_messages[0][1:]
    assert sent == [Message(Role.USER, "second")]
    widget.clear_button.click()
    assert "provider changed" not in widget.history_view.toPlainText()


def test_turning_privacy_on_clears_history_but_turning_it_off_keeps_it(
    qtbot: QtBot, monkeypatch
) -> None:
    settings = _FakeSettings()
    widget = _make_widget(
        qtbot, monkeypatch, lambda self, current: FakeBackend(), settings=settings
    )
    _send_and_wait(qtbot, widget)
    settings.set_llm_project_data_private("fake", False)
    widget.reload_backend()
    assert len(widget._history) == 2

    settings.set_llm_project_data_private("fake", True)
    widget.reload_backend()
    assert widget._history == []
    assert "no longer shared" in widget.history_view.toPlainText()


def test_settings_change_during_stream_applies_after_reply(
    qtbot: QtBot, monkeypatch
) -> None:
    settings = _FakeSettings()
    settings.set_llm_project_data_private("fake", False)
    widget = _make_widget(
        qtbot,
        monkeypatch,
        lambda self, current: FakeBackend(["a", "b"], delay=0.1),
        settings=settings,
    )
    widget.input_edit.setPlainText("hello")
    widget.send_button.click()
    settings.set_llm_project_data_private("fake", True)
    widget.reload_backend()
    assert widget._worker is not None and not widget._private_data
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)

    assert widget._history == []
    assert widget._private_data
    assert "no longer shared" in widget.history_view.toPlainText()


def test_copied_conversation_is_cleared_when_provider_changed_since(
    qtbot: QtBot, monkeypatch
) -> None:
    settings = _FakeSettings()
    source = _make_widget(
        qtbot, monkeypatch, lambda self, current: FakeBackend(), settings=settings
    )
    _send_and_wait(qtbot, source)

    same = ChatWidget()
    qtbot.addWidget(same)
    same.copy_conversation_from(source)
    assert same._history == source._history
    assert "hello" in same.history_view.toPlainText()

    settings.llm_provider = "openai"
    detached = ChatWidget()
    qtbot.addWidget(detached)
    detached.copy_conversation_from(source)
    assert detached._history == []
    assert len(source._history) == 2
