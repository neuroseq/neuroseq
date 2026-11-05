"""Codex app-server transport for the existing project-aware chat panel."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any

from neuroseq import __version__
from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.backend import register


class _AppServer:
    """One bounded stdio connection; stdout is read outside the Qt thread."""

    def __init__(self, executable: str, cancelled: threading.Event) -> None:
        self.cancelled = cancelled
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.next_id = 0
        self.pending: deque[dict[str, Any]] = deque()
        try:
            self.process = subprocess.Popen(
                [executable, "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            raise LLMBackendError(f"Could not start Codex: {error}") from error
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

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

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except OSError as error:
            raise LLMBackendError("Codex connection closed unexpectedly.") from error

    def receive(self, timeout: float = 120) -> dict[str, Any]:
        if self.pending:
            return self.pending.popleft()
        return self._receive(timeout)

    def _receive(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while not self.cancelled.is_set():
            if time.monotonic() >= deadline:
                raise LLMBackendError(
                    "Codex timed out. Check its sign-in and connection."
                )
            try:
                event = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is None:
                raise LLMBackendError(
                    "Codex exited unexpectedly. Check the Codex CLI installation."
                )
            if "method" in event and "id" in event:
                # This help panel does not grant native tool approvals or answer
                # interactive tool prompts. Never leave the server waiting.
                self.send(
                    {
                        "id": event["id"],
                        "error": {
                            "code": -32601,
                            "message": "Interactive tools are unavailable in NeuroSeq chat.",
                        },
                    }
                )
                continue
            if "error" in event:
                raise LLMBackendError(
                    str(event["error"].get("message", event["error"]))
                )
            return event
        raise InterruptedError

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + 20
        while True:
            event = self._receive(timeout=max(0, deadline - time.monotonic()))
            if event.get("id") == request_id:
                return event["result"]
            self.pending.append(event)

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "neuroseq",
                    "title": "NeuroSeq",
                    "version": __version__,
                }
            },
        )
        self.send({"method": "initialized", "params": {}})

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
class CodexBackend(LLMBackend):
    name = "codex"
    label = "Codex (local sign-in)"

    def __init__(self, executable: str = "") -> None:
        self.executable = executable or "codex"
        self._cancelled = threading.Event()

    def prepare(self) -> None:
        self._cancelled.clear()

    def cancel(self) -> None:
        self._cancelled.set()

    def _connect(self) -> _AppServer:
        executable = shutil.which(self.executable)
        if not executable:
            raise LLMBackendError(
                "Codex CLI not found. Install Codex, run 'codex login', and restart "
                "NeuroSeq, or set its executable path in LLM assistant settings."
            )
        return _AppServer(executable, self._cancelled)

    def is_available(self) -> tuple[bool, str]:
        # Keep startup local and quick. Authentication is checked by the first
        # request; do not perform model inference when opening a workspace.
        if shutil.which(self.executable):
            return True, ""
        return (
            False,
            "Codex CLI not found. Set its executable path and run 'codex login'.",
        )

    def list_models(self) -> list[str]:
        server = self._connect()
        try:
            server.initialize()
            result = server.request("model/list", {"limit": 100})
            return ["default", *[item["model"] for item in result["data"]]]
        finally:
            server.close()

    def check_connection(self) -> tuple[bool, str]:
        """Check the protocol and sign-in without sending a model request."""
        server = self._connect()
        try:
            server.initialize()
            account = server.request("account/read", {"refreshToken": False})
            if account.get("requiresOpenaiAuth") and not account.get("account"):
                return False, "Sign in to Codex first by running 'codex login'."
            return True, ""
        finally:
            server.close()

    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        server = self._connect()
        try:
            server.initialize()
            account = server.request("account/read", {"refreshToken": False})
            if account.get("requiresOpenaiAuth") and not account.get("account"):
                raise LLMBackendError(
                    "Sign in to Codex first by running 'codex login'."
                )
            messages = list(messages)
            instructions = "\n".join(
                m.content for m in messages if m.role == Role.SYSTEM
            )
            params: dict[str, Any] = {
                "cwd": str(pathlib.Path(__file__).resolve().parents[1]),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "developerInstructions": instructions,
            }
            if model and model != "default":
                params["model"] = model
            thread = server.request("thread/start", params)
            # The panel owns history and its project-read loop. A fresh Codex
            # thread receives that history so clearing chat really starts over.
            transcript = [
                {"role": m.role.value, "content": m.content}
                for m in messages
                if m.role != Role.SYSTEM
            ]
            server.request(
                "turn/start",
                {
                    "threadId": thread["thread"]["id"],
                    "input": [
                        {
                            "type": "text",
                            "text": "Continue this NeuroSeq chat transcript. Answer the last user message.\n"
                            + json.dumps(transcript, ensure_ascii=False),
                        }
                    ],
                },
            )
            while True:
                event = server.receive()
                method = event.get("method")
                data = event.get("params", {})
                if method == "item/agentMessage/delta":
                    yield data["delta"]
                elif method == "turn/completed":
                    turn = data["turn"]
                    if turn["status"] != "completed":
                        raise LLMBackendError(
                            str(turn.get("error") or "Codex turn interrupted.")
                        )
                    return
                elif method == "error" and not data.get("willRetry", False):
                    raise LLMBackendError(
                        str(data.get("error", "Codex request failed."))
                    )
        except InterruptedError:
            return
        finally:
            server.close()
