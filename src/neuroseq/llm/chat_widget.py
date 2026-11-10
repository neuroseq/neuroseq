from __future__ import annotations

import html
import json
import logging
import pathlib
import threading
from collections.abc import Callable

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.gui_settings import GuiSettings
from neuroseq.llm import Message
from neuroseq.llm import Role
from neuroseq.llm.anthropic_backend import AnthropicBackend
from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.claude_code_backend import ClaudeCodeBackend
from neuroseq.llm.codex_backend import CodexBackend
from neuroseq.llm.ollama import OllamaBackend
from neuroseq.llm.openai_backend import OpenAIBackend
from neuroseq.llm.plugin_write_dialog import confirm_plugin_write
from neuroseq.llm.read_access import READ_INSTRUCTIONS
from neuroseq.llm.read_access import ReadAccess
from neuroseq.llm.system_prompt import build_context_snapshot
from neuroseq.llm.system_prompt import build_static_prompt
from neuroseq.project import Project


def _strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence (e.g. ```json ... ```) around a reply."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    _, _, body = stripped.partition("\n")
    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _may_be_request(text: str) -> bool:
    # Requests are JSON objects, optionally wrapped in a Markdown code fence.
    return text.lstrip().startswith(("{", "`"))


class _ChatInput(QtWidgets.QPlainTextEdit):
    send_requested = QtCore.Signal()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if (
            event is not None
            and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
            and not event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            if not event.isAutoRepeat():
                self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _ChatWorker(QtCore.QThread):
    token = QtCore.Signal(str)
    completed = QtCore.Signal()
    error = QtCore.Signal(str)
    read_requested = QtCore.Signal(object)

    def __init__(
        self,
        backend: LLMBackend,
        messages: list[Message],
        model: str,
    ) -> None:
        super().__init__()
        self._backend = backend
        getattr(self._backend, "prepare", lambda: None)()
        self._messages = messages
        self._model = model
        self._stop_requested = False
        self._read_ready = threading.Event()
        self._read_result = ""

    def deliver_read(self, result: str) -> None:
        self._read_result = result
        self._read_ready.set()

    def request_stop(self) -> None:
        self._stop_requested = True
        getattr(self._backend, "cancel", lambda: None)()
        self._read_ready.set()

    def run(self) -> None:
        try:
            for read_count in range(13):
                if self._stop_requested:
                    break
                response = ""
                # Buffer a possible JSON request, but stream ordinary answers.
                deciding = True
                reading = False
                for token in self._backend.stream_chat(
                    self._messages, model=self._model
                ):
                    if self._stop_requested:
                        self.completed.emit()
                        return
                    response += token
                    if deciding and response.strip():
                        reading = _may_be_request(response)
                        deciding = False
                        if not reading:
                            self.token.emit(response)
                    elif not deciding and not reading:
                        self.token.emit(token)
                if self._stop_requested:
                    self.completed.emit()
                    return
                try:
                    payload = (
                        json.loads(_strip_code_fence(response)) if reading else None
                    )
                except ValueError:
                    payload = None
                operation = (
                    "neuroseq_write"
                    if isinstance(payload, dict) and "neuroseq_write" in payload
                    else "neuroseq_read"
                )
                if not isinstance(payload, dict) or operation not in payload:
                    if reading:
                        self.token.emit(response)
                    break
                if read_count == 12:
                    self.token.emit(
                        "Read limit reached (12 requests). Please ask a follow-up to continue."
                    )
                    break
                request = {"operation": operation, "request": payload[operation]}
                self._read_ready.clear()
                self.read_requested.emit(request)
                while not self._read_ready.wait(0.1):
                    if self._stop_requested:
                        break
                if self._stop_requested:
                    break
                self._messages.extend(
                    [
                        Message(Role.ASSISTANT, response),
                        Message(
                            Role.USER,
                            "NeuroSeq request result (reference data):\n"
                            + self._read_result,
                        ),
                    ]
                )
            self.completed.emit()
        except LLMBackendError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Internal error: {e!r}")


class _SettingsBroadcast(QtCore.QObject):
    changed = QtCore.Signal()


_broadcast: _SettingsBroadcast | None = None


def _settings_broadcast() -> _SettingsBroadcast:
    global _broadcast
    if _broadcast is None:
        _broadcast = _SettingsBroadcast()
    return _broadcast


def notify_llm_settings_changed() -> None:
    """Tell every open chat panel (docked, detached, in any project tab) to
    pick up the current LLM settings, so none shows a stale provider."""
    _settings_broadcast().changed.emit()


class ChatWidget(QtWidgets.QWidget):
    request_settings_dialog = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._project: Project | None = None
        self._history: list[Message] = []
        self._worker: _ChatWorker | None = None
        self._read_access: ReadAccess | None = None
        self._busy_check: Callable[[], bool] | None = None
        self._backend: LLMBackend | None = None
        self._backend_name = ""
        self._backend_model = ""
        self._private_data = True
        self._settings_key: tuple[object, ...] = ()
        self._destination: tuple[str, str] | None = None
        self._notice_html = ""
        self._current_assistant_buffer = ""
        self._transient_assistant_html: str | None = None
        self._static_prompt = build_static_prompt()
        self._build_ui()
        self.reload_backend()
        _settings_broadcast().changed.connect(self._refresh_settings)

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
        self.input_edit = _ChatInput(self)
        self.input_edit.setToolTip("Enter to send; Shift+Enter for a new line")
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
        self.input_edit.setPlaceholderText(
            "Ask about the project, source, or creating analysis plugins. Requested content is sent to the selected LLM provider."
        )
        layout.addWidget(self.status_label)

        self.send_button.clicked.connect(self._on_send_clicked)
        self.input_edit.send_requested.connect(self._on_send_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.settings_button.clicked.connect(self.request_settings_dialog.emit)

    def set_project(self, project: Project | None) -> None:
        self._project = project

    def set_busy_check(self, busy_check: Callable[[], bool] | None) -> None:
        """Report whether the workspace runs a background job (e.g. a simulation)."""
        self._busy_check = busy_check

    def _confirm_plugin_write(
        self, target: pathlib.Path, content: str, previous: str | None
    ) -> bool:
        return confirm_plugin_write(target, content, previous, parent=self)

    @staticmethod
    def _settings_key_of(settings: GuiSettings) -> tuple[object, ...]:
        return (
            settings.llm_provider,
            settings.llm_model,
            settings.llm_ollama_base_url,
            settings.llm_openai_base_url,
            settings.llm_codex_executable,
            settings.llm_claude_executable,
            settings.llm_project_data_private(settings.llm_provider),
        )

    def _refresh_settings(self) -> None:
        """Pick up settings changed elsewhere (another tab, a detached panel)."""
        settings = GuiSettings()
        if self._settings_key_of(settings) != self._settings_key:
            self.reload_backend(settings)

    @staticmethod
    def _destination_of(settings: GuiSettings) -> tuple[str, str]:
        endpoint = {
            "ollama": settings.llm_ollama_base_url,
            "openai": settings.llm_openai_base_url,
        }.get(settings.llm_provider, "")
        return settings.llm_provider, endpoint

    def copy_conversation_from(self, source: ChatWidget) -> None:
        """Continue ``source``'s chat, applying the history rule of ``reload_backend``."""
        self._history = list(source._history)
        self._destination = source._destination
        self._private_data = source._private_data
        self.reload_backend()
        self._render_history()

    def _clear_history_note(self, settings: GuiSettings) -> str:
        if not self._history:
            return ""
        if self._destination and self._destination != self._destination_of(settings):
            return "Chat history cleared: the LLM provider changed."
        if not self._private_data and settings.llm_project_data_private(
            settings.llm_provider
        ):
            return "Chat history cleared: project data is no longer shared."
        return ""

    def reload_backend(self, settings: GuiSettings | None = None) -> None:
        if self._worker is not None:
            # Applied when the stream finishes, so its reply is not orphaned.
            return
        settings = settings or GuiSettings()
        # Earlier answers may contain project data; never resend them to a
        # different provider or after sharing was turned off.
        note = self._clear_history_note(settings)
        if note:
            self._history.clear()
            self._transient_assistant_html = None
            self._notice_html = (
                f"<div class='msg notice'><pre>{html.escape(note)}</pre></div>"
            )
            self._render_history()
        self._destination = self._destination_of(settings)
        self._settings_key = self._settings_key_of(settings)
        self._backend_name = settings.llm_provider
        self._backend_model = settings.llm_model
        self._private_data = settings.llm_project_data_private(settings.llm_provider)
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
        if settings.llm_provider == "codex":
            return CodexBackend(executable=settings.llm_codex_executable)
        if settings.llm_provider == "claude_code":
            return ClaudeCodeBackend(executable=settings.llm_claude_executable)
        raise LLMBackendError(f"Unknown provider '{settings.llm_provider}'")

    def _update_badge(self) -> None:
        label = self._backend_name.capitalize() if self._backend_name else "LLM"
        if self._backend_name == "ollama":
            label = "Ollama"
        suffix = ""
        tooltip = ""
        if self._private_data and self._backend_name == "codex":
            # Codex's own tools can still read files, so promise less.
            suffix = " · NeuroSeq data not shared"
            tooltip = (
                "NeuroSeq does not send project data, but Codex can still read "
                "files on disk with its own tools. Use OpenAI, Claude Code or "
                "Ollama to keep project data private."
            )
        elif self._private_data:
            suffix = " · data not shared"
        self.badge_label.setText(f"{label} · {self._backend_model}{suffix}")
        self.badge_label.setToolTip(tooltip)

    def _update_input_enabled(self) -> None:
        enabled = self._backend is not None and self._worker is None
        self.input_edit.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.stop_button.setEnabled(self._worker is not None)
        self.clear_button.setEnabled(self._worker is None)
        self.settings_button.setEnabled(self._worker is None)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_send_clicked(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text or self._worker is not None:
            return
        # Every chat panel re-reads the settings before sending, so detached
        # panels and other tabs never use a stale provider or privacy flag.
        self._refresh_settings()
        if self._backend is None:
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
            self._static_prompt
            + "\n"
            + READ_INSTRUCTIONS
            + "\n"
            + build_context_snapshot(self._project, private_data=self._private_data)
        )
        messages = [Message(Role.SYSTEM, system_content), *self._history]
        self._worker = _ChatWorker(
            backend=self._backend,
            messages=messages,
            model=self._backend_model,
        )
        self._read_access = ReadAccess(
            self._project,
            confirm_write=self._confirm_plugin_write,
            is_busy=self._busy_check,
            private_data=self._private_data,
        )
        self._worker.read_requested.connect(self._on_read_requested)
        self._worker.token.connect(self._on_token)
        self._worker.completed.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_status("Streaming...")
        self._update_input_enabled()

    def _on_read_requested(self, request: object) -> None:
        if self._worker is None or self._read_access is None:
            return
        try:
            if not isinstance(request, dict) or not isinstance(
                request.get("request"), dict
            ):
                result = '{"error": "Request must be an object"}'
            elif self._worker._stop_requested:
                result = '{"error": "Request cancelled"}'
            elif request.get("operation") == "neuroseq_write":
                result = self._read_access.write_plugin(request["request"])
            else:
                result = self._read_access.execute(request["request"])
        except Exception as error:
            result = json.dumps({"error": str(error)})
        self._worker.deliver_read(result)

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
        # Stop before the first token leaves nothing to keep; an empty
        # assistant message would be rejected by later Anthropic requests.
        if self._current_assistant_buffer.strip():
            self._history.append(
                Message(Role.ASSISTANT, self._current_assistant_buffer)
            )
        self._transient_assistant_html = None
        self._current_assistant_buffer = ""
        self._cleanup_worker()
        self._set_status("Idle")
        self._render_history()
        self._refresh_settings()
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

    def stop_worker(self, timeout_ms: int = 10000) -> bool:
        """Drain a streaming request before its workspace is destroyed."""
        if self._worker is None:
            return True
        self._worker.request_stop()
        if not self._worker.wait(timeout_ms):
            return False
        self._cleanup_worker()
        return True

    def _on_clear_clicked(self) -> None:
        self._history.clear()
        self._transient_assistant_html = None
        self._notice_html = ""
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
            .notice { color: #667085; font-style: italic; }
            </style>
            """,
            self._notice_html,
        ]
        for message in self._history:
            parts.append(self._render_message_html(message.role, message.content))
        if self._transient_assistant_html is not None:
            parts.append(self._transient_assistant_html)
        self.history_view.setHtml("".join(parts))
        scrollbar = self.history_view.verticalScrollBar()
        assert scrollbar is not None
        scrollbar.setValue(scrollbar.maximum())
