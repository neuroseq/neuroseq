from __future__ import annotations

import time
from dataclasses import dataclass

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot

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
