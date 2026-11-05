"""Claude Code CLI transport for the existing project-aware chat panel."""

from __future__ import annotations

import contextlib
import json
import pathlib
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any

from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.backend import register

# Aliases resolved by Claude Code itself; access depends on the signed-in plan.
MODEL_ALIASES = ["default", "opus", "sonnet", "haiku"]

_NOT_FOUND = (
    "Claude Code CLI not found. Install Claude Code, run 'claude auth login', and "
    "restart NeuroSeq, or set its executable path in LLM assistant settings."
)
_SIGN_IN = "Sign in to Claude Code first by running 'claude auth login'."


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _PrintSession:
    """One ``claude -p`` process; stdout is read outside the Qt thread."""

    def __init__(
        self,
        args: list[str],
        prompt: str,
        cwd: str,
        cancelled: threading.Event,
    ) -> None:
        self.cancelled = cancelled
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                cwd=cwd,
                creationflags=_creationflags(),
            )
        except OSError as error:
            raise LLMBackendError(f"Could not start Claude Code: {error}") from error
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        # The transcript goes through stdin: it can exceed command-line limits.
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(prompt)
            self.process.stdin.close()
        except OSError as error:
            self.close()
            raise LLMBackendError(
                "Claude Code connection closed unexpectedly."
            ) from error

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        self.events.put(event)
                except ValueError:
                    continue
        finally:
            self.events.put(None)

    def receive(self, timeout: float = 120) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while not self.cancelled.is_set():
            if time.monotonic() >= deadline:
                raise LLMBackendError(
                    "Claude Code timed out. Check its sign-in and connection."
                )
            try:
                event = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is None:
                raise LLMBackendError(
                    "Claude Code exited unexpectedly. Check the Claude Code "
                    "installation and sign-in."
                )
            return event
        raise InterruptedError

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.reader.join(timeout=2)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


@register
class ClaudeCodeBackend(LLMBackend):
    name = "claude_code"
    label = "Claude Code (local sign-in)"

    def __init__(self, executable: str = "") -> None:
        self.executable = executable or "claude"
        self._cancelled = threading.Event()

    def prepare(self) -> None:
        self._cancelled.clear()

    def cancel(self) -> None:
        self._cancelled.set()

    def _resolve(self) -> str:
        executable = shutil.which(self.executable)
        if not executable:
            raise LLMBackendError(_NOT_FOUND)
        return executable

    def is_available(self) -> tuple[bool, str]:
        # Keep startup local and quick. Authentication is checked by the first
        # request; do not perform model inference when opening a workspace.
        if shutil.which(self.executable):
            return True, ""
        return (
            False,
            "Claude Code CLI not found. Set its executable path and run "
            "'claude auth login'.",
        )

    def list_models(self) -> list[str]:
        self._resolve()
        return list(MODEL_ALIASES)

    def check_connection(self) -> tuple[bool, str]:
        """Check the CLI and sign-in without sending a model request."""
        executable = self._resolve()
        try:
            completed = subprocess.run(
                [executable, "auth", "status", "--json"],
                capture_output=True,
                encoding="utf-8",
                timeout=20,
                creationflags=_creationflags(),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LLMBackendError(f"Could not run Claude Code: {error}") from error
        try:
            status = json.loads(completed.stdout)
        except ValueError:
            return False, "Could not read Claude Code sign-in status."
        if not isinstance(status, dict) or not status.get("loggedIn"):
            return False, _SIGN_IN
        return True, ""

    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        executable = self._resolve()
        messages = list(messages)
        instructions = "\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        # The panel owns history and its project-read loop. A fresh print
        # session receives that history so clearing chat really starts over.
        transcript = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        prompt = (
            "Continue this NeuroSeq chat transcript. Answer the last user message.\n"
            + json.dumps(transcript, ensure_ascii=False)
        )
        # An empty working directory keeps a checkout's CLAUDE.md out of context.
        with tempfile.TemporaryDirectory(prefix="neuroseq-claude-") as workdir:
            system_file = pathlib.Path(workdir) / "system_prompt.md"
            system_file.write_text(instructions, encoding="utf-8")
            args = [
                executable,
                "--print",
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--verbose",
                # NeuroSeq's own read protocol is the only project access; no
                # native tools, MCP servers, settings hooks or saved sessions.
                "--tools",
                "",
                "--strict-mcp-config",
                "--setting-sources",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--system-prompt-file",
                str(system_file),
            ]
            if model and model != "default":
                args += ["--model", model]
            session = _PrintSession(args, prompt, workdir, self._cancelled)
            try:
                while True:
                    event = session.receive()
                    kind = event.get("type")
                    if kind == "stream_event":
                        delta = event.get("event", {}).get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta["text"]
                    elif kind == "result":
                        if event.get("is_error"):
                            raise LLMBackendError(
                                str(
                                    event.get("result") or "Claude Code request failed."
                                )
                            )
                        return
            except InterruptedError:
                return
            finally:
                session.close()
