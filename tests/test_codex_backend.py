from __future__ import annotations

import json
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest

from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.chat_widget import ChatWidget
from neuroseq.llm.codex_backend import CodexBackend
from neuroseq.llm.codex_backend import _AppServer

# A real child process exercises stdio framing, early events, cancellation,
# failures and cleanup without authentication, network traffic or model calls.
_SERVER = r"""
import json
import sys
import time

mode, log = sys.argv[1:]
def emit(value):
    print(json.dumps(value), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    with open(log, "a", encoding="utf-8") as output:
        output.write(json.dumps(request) + "\n")
    method = request.get("method")
    if "id" not in request:
        continue
    if method == "initialize":
        if mode == "rpc_error":
            emit({"id": request["id"], "error": {"message": "Upgrade Codex"}})
            continue
        result = {}
    elif method == "account/read":
        result = {"account": None if mode == "signed_out" else {"type": "chatgpt"},
                  "requiresOpenaiAuth": True}
    elif method == "model/list":
        result = {"data": [{"model": "available-model"}], "nextCursor": None}
    elif method == "thread/start":
        result = {"thread": {"id": "thread-1"}}
    elif method == "turn/start":
        result = {"turn": {"id": "turn-1"}}
        if mode == "exit":
            sys.exit(1)
        if mode == "read" and "NeuroSeq request result" not in request["params"]["input"][0]["text"]:
            emit({"id": request["id"], "result": result})
            emit({"method": "item/agentMessage/delta", "params": {"delta": json.dumps(
                {"neuroseq_read": {"kind": "source", "path": "llm/backend.py", "limit": 10}})}})
            emit({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
            continue
        if mode != "silent":
            # Events may precede the turn/start response.
            emit({"method": "item/agentMessage/delta", "params": {"delta": "Hello "}})
        emit({"id": request["id"], "result": result})
        if mode == "silent":
            time.sleep(60)
        elif mode == "request":
            emit({"id": 1000, "method": "item/commandExecution/requestApproval", "params": {}})
        else:
            emit({"method": "item/agentMessage/delta", "params": {"delta": "NeuroSeq"}})
            emit({"method": "turn/completed", "params": {"turn": {
                "status": "failed" if mode == "failed" else "completed",
                "error": {"message": "quota exceeded"} if mode == "failed" else None}}})
        continue
    else:
        # NeuroSeq must reject server tool requests rather than hanging.
        emit({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
        continue
    emit({"id": request["id"], "result": result})
"""


@pytest.fixture
def fake_codex(monkeypatch, tmp_path):
    popen = subprocess.Popen
    log = tmp_path / "requests.jsonl"
    processes = []
    modes = {"mode": "ok"}

    def start(args, **kwargs):
        assert args == ["fake-codex", "app-server"]
        process = popen(
            [sys.executable, "-u", "-c", _SERVER, modes["mode"], str(log)], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(
        "neuroseq.llm.codex_backend.shutil.which", lambda _: "fake-codex"
    )
    monkeypatch.setattr("neuroseq.llm.codex_backend.subprocess.Popen", start)
    yield modes, log, processes
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_stream_and_context_over_stdio(fake_codex):
    _, log, processes = fake_codex
    messages = [
        Message(Role.SYSTEM, "NeuroSeq docs and project snapshot"),
        Message(Role.USER, "Explain the model"),
        Message(Role.ASSISTANT, "Earlier answer"),
        Message(Role.USER, "And the source?"),
    ]
    assert (
        "".join(CodexBackend().stream_chat(messages, model="chosen-model"))
        == "Hello NeuroSeq"
    )
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "account/read",
        "thread/start",
        "turn/start",
    ]
    params = requests[3]["params"]
    assert params["sandbox"] == "read-only"
    assert params["approvalPolicy"] == "never"
    assert params["ephemeral"] is True
    assert params["model"] == "chosen-model"
    assert params["developerInstructions"] == messages[0].content
    assert "Earlier answer" in requests[4]["params"]["input"][0]["text"]
    assert all(process.poll() is not None for process in processes)


def test_default_model_uses_codex_configuration(fake_codex):
    _, log, _ = fake_codex
    list(CodexBackend().stream_chat([Message(Role.USER, "Hi")], model="default"))
    request = next(
        json.loads(line)
        for line in log.read_text().splitlines()
        if json.loads(line).get("method") == "thread/start"
    )
    assert "model" not in request["params"]


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("signed_out", "codex login"),
        ("failed", "quota exceeded"),
        ("rpc_error", "Upgrade Codex"),
        ("exit", "exited unexpectedly"),
    ],
)
def test_failures_are_visible_and_process_is_closed(fake_codex, mode, reason):
    modes, _, processes = fake_codex
    modes["mode"] = mode
    with pytest.raises(LLMBackendError, match=reason):
        list(CodexBackend().stream_chat([Message(Role.USER, "Hi")], model="default"))
    assert all(process.poll() is not None for process in processes)


def test_cancel_unblocks_silent_codex(fake_codex):
    modes, _, processes = fake_codex
    modes["mode"] = "silent"
    backend = CodexBackend()
    timer = threading.Timer(0.5, backend.cancel)
    timer.start()
    try:
        assert (
            list(backend.stream_chat([Message(Role.USER, "Hi")], model="default")) == []
        )
    finally:
        timer.cancel()
    assert all(process.poll() is not None for process in processes)
    backend.prepare()
    modes["mode"] = "ok"
    assert list(backend.stream_chat([], model="default"))


def test_interactive_tool_request_is_rejected(fake_codex):
    modes, log, _ = fake_codex
    modes["mode"] = "request"
    list(CodexBackend().stream_chat([], model="default"))
    replies = [json.loads(line) for line in log.read_text().splitlines()]
    assert replies[-1]["id"] == 1000
    assert replies[-1]["error"]["code"] == -32601


def test_model_listing(fake_codex):
    assert CodexBackend().list_models() == ["default", "available-model"]


@pytest.mark.parametrize("signed_out", [False, True])
def test_connection_checks_sign_in_without_inference(fake_codex, signed_out):
    modes, log, _ = fake_codex
    modes["mode"] = "signed_out" if signed_out else "ok"
    ok, reason = CodexBackend().check_connection()
    assert ok is not signed_out
    assert not signed_out or "codex login" in reason
    assert "turn/start" not in log.read_text()


def test_missing_cli(monkeypatch):
    monkeypatch.setattr("neuroseq.llm.codex_backend.shutil.which", lambda _: None)
    backend = CodexBackend()
    assert backend.is_available()[0] is False
    with pytest.raises(LLMBackendError, match="CLI not found"):
        list(backend.stream_chat([], model="default"))


def test_local_availability_does_not_start_process(monkeypatch):
    monkeypatch.setattr("neuroseq.llm.codex_backend.shutil.which", lambda _: "codex")
    start = Mock(
        side_effect=AssertionError("Must not start Codex during widget construction")
    )
    monkeypatch.setattr("neuroseq.llm.codex_backend.subprocess.Popen", start)
    assert CodexBackend().is_available() == (True, "")
    start.assert_not_called()


def test_transport_timeout_and_start_failure(fake_codex, monkeypatch):
    backend = CodexBackend()
    server = backend._connect()
    try:
        with pytest.raises(LLMBackendError, match="timed out"):
            server.receive(timeout=0)
    finally:
        server.close()
    monkeypatch.setattr(
        "neuroseq.llm.codex_backend.subprocess.Popen",
        Mock(side_effect=OSError("missing")),
    )
    with pytest.raises(LLMBackendError, match="Could not start Codex"):
        _AppServer("fake-codex", threading.Event())


def test_codex_widget_source_read_round_trip(qtbot, fake_codex, monkeypatch):
    modes, log, processes = fake_codex
    modes["mode"] = "read"
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.build_static_prompt", lambda: "NeuroSeq docs"
    )
    monkeypatch.setattr(
        ChatWidget, "_make_backend", lambda self, settings: CodexBackend()
    )
    widget = ChatWidget()
    qtbot.addWidget(widget)
    widget.input_edit.setPlainText("Explain NeuroSeq's source code")
    widget.send_button.click()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=5000)
    assert widget.status_label.text() == "Idle"
    assert widget._history[-1].content == "Hello NeuroSeq"
    assert len(processes) == 2
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    turns = [request for request in requests if request.get("method") == "turn/start"]
    assert "NeuroSeq request result" in turns[-1]["params"]["input"][0]["text"]
    assert (
        "from __future__ import annotations" in turns[-1]["params"]["input"][0]["text"]
    )


def test_codex_widget_stop_drains_process(qtbot, fake_codex, monkeypatch):
    modes, log, processes = fake_codex
    modes["mode"] = "silent"
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.build_static_prompt", lambda: "NeuroSeq docs"
    )
    monkeypatch.setattr(
        ChatWidget, "_make_backend", lambda self, settings: CodexBackend()
    )
    widget = ChatWidget()
    qtbot.addWidget(widget)
    widget.input_edit.setPlainText("Hi")
    widget.send_button.click()
    qtbot.waitUntil(
        lambda: log.exists() and "turn/start" in log.read_text(), timeout=5000
    )
    assert not widget.clear_button.isEnabled()
    assert not widget.settings_button.isEnabled()
    assert widget.stop_worker(timeout_ms=3000)
    assert all(process.poll() is not None for process in processes)
    qtbot.waitUntil(lambda: widget.send_button.isEnabled())
