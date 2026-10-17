from __future__ import annotations

import html
import logging

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.gui_settings import GuiSettings
from neuroseq.llm import Message
from neuroseq.llm import Role
from neuroseq.llm.anthropic_backend import AnthropicBackend
from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.ollama import OllamaBackend
from neuroseq.llm.openai_backend import OpenAIBackend
from neuroseq.llm.system_prompt import build_context_snapshot
from neuroseq.llm.system_prompt import build_static_prompt
from neuroseq.project import Project


class _ChatWorker(QtCore.QThread):
    token = QtCore.Signal(str)
    completed = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(
        self,
        backend: LLMBackend,
        messages: list[Message],
        model: str,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._messages = messages
        self._model = model
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            for token in self._backend.stream_chat(self._messages, model=self._model):
                if self._stop_requested:
                    break
                self.token.emit(token)
            self.completed.emit()
        except LLMBackendError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Internal error: {e!r}")


class ChatWidget(QtWidgets.QWidget):
    request_settings_dialog = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._project: Project | None = None
        self._history: list[Message] = []
        self._worker: _ChatWorker | None = None
        self._backend: LLMBackend | None = None
        self._backend_name = ""
        self._backend_model = ""
        self._current_assistant_buffer = ""
        self._transient_assistant_html: str | None = None
        self._static_prompt = build_static_prompt()
        self._build_ui()
        self.reload_backend()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        self.badge_label = QtWidgets.QLabel("LLM", self)
        self.clear_button = QtWidgets.QPushButton("Clear chat", self)
        self.settings_button = QtWidgets.QPushButton("Settings...", self)
        header.addWidget(self.badge_label)
        header.addStretch(1)
        header.addWidget(self.clear_button)
        header.addWidget(self.settings_button)
        layout.addLayout(header)

        self.history_view = QtWidgets.QTextBrowser(self)
        self.history_view.setOpenExternalLinks(True)
        layout.addWidget(self.history_view, stretch=1)

        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QPlainTextEdit(self)
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.setEnabled(False)
        input_row.addWidget(self.input_edit, stretch=1)
        input_button_col = QtWidgets.QVBoxLayout()
        input_button_col.addWidget(self.send_button)
        input_button_col.addWidget(self.stop_button)
        input_button_col.addStretch(1)
        input_row.addLayout(input_button_col)
        layout.addLayout(input_row)

        self.status_label = QtWidgets.QLabel("Idle", self)
        layout.addWidget(self.status_label)

        self.send_button.clicked.connect(self._on_send_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.settings_button.clicked.connect(self.request_settings_dialog.emit)

    def set_project(self, project: Project | None) -> None:
        self._project = project

    def reload_backend(self) -> None:
        settings = GuiSettings()
        self._backend_name = settings.llm_provider
        self._backend_model = settings.llm_model
        try:
            self._backend = self._make_backend(settings)
            ok, reason = self._backend.is_available()
            if ok:
                self._set_status("Idle")
            else:
                self._set_status(f"Backend unavailable: {reason}")
        except LLMBackendError as e:
            self._backend = None
            self._set_status(f"Backend init failed: {e}")
        self._update_badge()
        self._update_input_enabled()

    def _make_backend(self, settings: GuiSettings) -> LLMBackend:
        if settings.llm_provider == "ollama":
            return OllamaBackend(base_url=settings.llm_ollama_base_url)
        if settings.llm_provider == "openai":
            return OpenAIBackend(base_url=settings.llm_openai_base_url or None)
        if settings.llm_provider == "anthropic":
            return AnthropicBackend()
        raise LLMBackendError(f"Unknown provider '{settings.llm_provider}'")

    def _update_badge(self) -> None:
        label = self._backend_name.capitalize() if self._backend_name else "LLM"
        if self._backend_name == "ollama":
            label = "Ollama"
        self.badge_label.setText(f"{label} · {self._backend_model}")

    def _update_input_enabled(self) -> None:
        enabled = self._backend is not None and self._worker is None
        self.input_edit.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.stop_button.setEnabled(self._worker is not None)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_send_clicked(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text or self._worker is not None or self._backend is None:
            return
        self.input_edit.clear()
        self._history.append(Message(Role.USER, text))
        self._current_assistant_buffer = ""
        self._transient_assistant_html = ""
        self._render_history()
        self._spawn_worker()

    def _spawn_worker(self) -> None:
        if self._backend is None:
            return
        system_content = (
            self._static_prompt + "\n" + build_context_snapshot(self._project)
        )
        messages = [Message(Role.SYSTEM, system_content), *self._history]
        self._worker = _ChatWorker(
            backend=self._backend,
            messages=messages,
            model=self._backend_model,
        )
        self._worker.token.connect(self._on_token)
        self._worker.completed.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_status("Streaming...")
        self._update_input_enabled()

    def _on_token(self, token: str) -> None:
        self._current_assistant_buffer += token
        self._transient_assistant_html = self._render_message_html(
            Role.ASSISTANT,
            self._current_assistant_buffer,
        )
        self._render_history()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None

    def _on_worker_finished(self) -> None:
        self._history.append(Message(Role.ASSISTANT, self._current_assistant_buffer))
        self._transient_assistant_html = None
        self._current_assistant_buffer = ""
        self._cleanup_worker()
        self._set_status("Idle")
        self._render_history()
        self._update_input_enabled()

    def _on_worker_error(self, message: str) -> None:
        self._transient_assistant_html = (
            "<div class='msg assistant error'><pre>"
            f"{html.escape(f'Error: {message}')}"
            "</pre></div>"
        )
        self._current_assistant_buffer = ""
        self._cleanup_worker()
        self._set_status(f"Error: {message}")
        self._render_history()
        self._update_input_enabled()

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _on_clear_clicked(self) -> None:
        self._history.clear()
        self._transient_assistant_html = None
        self.history_view.clear()
        self._set_status("Idle")

    def _render_message_html(self, role: Role, content: str) -> str:
        css_role = "user" if role == Role.USER else "assistant"
        title = "User" if role == Role.USER else "Assistant"
        return (
            f"<div class='msg {css_role}'><div class='title'>{title}</div><pre>"
            f"{html.escape(content)}"
            "</pre></div>"
        )

    def _render_history(self) -> None:
        parts = [
            """
            <style>
            .msg { margin: 8px 0; padding: 8px 10px; border-radius: 8px; }
            .user { background: #e8f1ff; }
            .assistant { background: #f3f3f3; }
            .error { color: #b42318; }
            .title { font-weight: 600; margin-bottom: 4px; }
            pre { white-space: pre-wrap; margin: 0; font-family: inherit; }
            </style>
            """
        ]
        for message in self._history:
            parts.append(self._render_message_html(message.role, message.content))
        if self._transient_assistant_html is not None:
            parts.append(self._transient_assistant_html)
        self.history_view.setHtml("".join(parts))
        scrollbar = self.history_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
