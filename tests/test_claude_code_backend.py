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
from neuroseq.llm.claude_code_backend import MODEL_ALIASES
from neuroseq.llm.claude_code_backend import ClaudeCodeBackend

# A real child process exercises stdin/stdout framing, cancellation, failures
# and cleanup without authentication, network traffic or model calls.
_CLI = r"""
import json
import os
import sys
import time

mode, log = sys.argv[1:3]
args = sys.argv[3:]
prompt = sys.stdin.read()
system = open(args[args.index("--system-prompt-file") + 1], encoding="utf-8").read()
with open(log, "a", encoding="utf-8") as output:
    output.write(json.dumps({"args": args, "prompt": prompt, "system": system,
                             "cwd": os.getcwd()}) + "\n")

def emit(value):
    print(json.dumps(value), flush=True)

def text(chunk):
    emit({"type": "stream_event", "event": {"type": "content_block_delta",
          "index": 0, "delta": {"type": "text_delta", "text": chunk}}})

emit({"type": "system", "subtype": "init"})
if mode == "silent":
    time.sleep(60)
if mode == "exit":
    sys.exit(1)
if mode == "read" and "NeuroSeq request result" not in prompt:
    text(json.dumps({"neuroseq_read": {"kind": "source", "path": "llm/backend.py",
                                       "limit": 10}}))
    emit({"type": "result", "is_error": False})
    sys.exit(0)
text("Hello ")
emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 1,
      "delta": {"type": "thinking_delta", "thinking": "hidden"}}})
text("NeuroSeq")
emit({"type": "assistant", "message": {"content": [{"type": "text",
      "text": "Hello NeuroSeq"}]}})
if mode == "failed":
    emit({"type": "result", "is_error": True, "result": "Not logged in"})
else:
    emit({"type": "result", "is_error": False, "result": "Hello NeuroSeq"})
"""


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    popen = subprocess.Popen
    log = tmp_path / "calls.jsonl"
    processes = []
    modes = {"mode": "ok"}

    def start(args, **kwargs):
        assert args[0] == "fake-claude"
        process = popen(
            [sys.executable, "-u", "-c", _CLI, modes["mode"], str(log), *args[1:]],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.shutil.which", lambda _: "fake-claude"
    )
    monkeypatch.setattr("neuroseq.llm.claude_code_backend.subprocess.Popen", start)
    yield modes, log, processes
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_stream_and_context_over_stdio(fake_claude):
    _, log, processes = fake_claude
    messages = [
        Message(Role.SYSTEM, "NeuroSeq docs and project snapshot"),
        Message(Role.USER, "Explain the model"),
        Message(Role.ASSISTANT, "Earlier answer"),
        Message(Role.USER, "And the source?"),
    ]
    assert (
        "".join(ClaudeCodeBackend().stream_chat(messages, model="sonnet"))
        == "Hello NeuroSeq"
    )
    (call,) = _calls(log)
    args = call["args"]
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--no-session-persistence" in args
    assert "--strict-mcp-config" in args
    assert call["system"] == messages[0].content
    assert "Earlier answer" in call["prompt"]
    assert "NeuroSeq docs" not in call["prompt"]
    assert all(process.poll() is not None for process in processes)


def test_default_model_uses_claude_configuration(fake_claude):
    _, log, _ = fake_claude
    list(ClaudeCodeBackend().stream_chat([Message(Role.USER, "Hi")], model="default"))
    assert "--model" not in _calls(log)[0]["args"]


@pytest.mark.parametrize(
    "mode,reason",
    [("failed", "Not logged in"), ("exit", "exited unexpectedly")],
)
def test_failures_are_visible_and_process_is_closed(fake_claude, mode, reason):
    modes, _, processes = fake_claude
    modes["mode"] = mode
    with pytest.raises(LLMBackendError, match=reason):
        list(
            ClaudeCodeBackend().stream_chat([Message(Role.USER, "Hi")], model="default")
        )
    assert all(process.poll() is not None for process in processes)


def test_cancel_unblocks_silent_claude(fake_claude):
    modes, _, processes = fake_claude
    modes["mode"] = "silent"
    backend = ClaudeCodeBackend()
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


def test_model_listing(fake_claude):
    assert ClaudeCodeBackend().list_models() == MODEL_ALIASES


@pytest.mark.parametrize(
    "stdout,ok",
    [
        (json.dumps({"loggedIn": True, "authMethod": "claude.ai"}), True),
        (json.dumps({"loggedIn": False}), False),
        ("not json", False),
    ],
)
def test_connection_checks_sign_in_without_inference(monkeypatch, stdout, ok):
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.shutil.which", lambda _: "fake-claude"
    )
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=stdout))
    monkeypatch.setattr("neuroseq.llm.claude_code_backend.subprocess.run", run)
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.subprocess.Popen",
        Mock(side_effect=AssertionError("Must not start a model request")),
    )
    result, reason = ClaudeCodeBackend().check_connection()
    assert result is ok
    assert ok or reason
    assert run.call_args.args[0] == ["fake-claude", "auth", "status", "--json"]


def test_connection_check_start_failure(monkeypatch):
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.shutil.which", lambda _: "fake-claude"
    )
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.subprocess.run",
        Mock(side_effect=OSError("missing")),
    )
    with pytest.raises(LLMBackendError, match="Could not run Claude Code"):
        ClaudeCodeBackend().check_connection()


def test_missing_cli(monkeypatch):
    monkeypatch.setattr("neuroseq.llm.claude_code_backend.shutil.which", lambda _: None)
    backend = ClaudeCodeBackend()
    assert backend.is_available()[0] is False
    with pytest.raises(LLMBackendError, match="CLI not found"):
        list(backend.stream_chat([], model="default"))
    with pytest.raises(LLMBackendError, match="CLI not found"):
        backend.list_models()


def test_local_availability_does_not_start_process(monkeypatch):
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.shutil.which", lambda _: "claude"
    )
    start = Mock(
        side_effect=AssertionError("Must not start Claude during widget construction")
    )
    monkeypatch.setattr("neuroseq.llm.claude_code_backend.subprocess.Popen", start)
    assert ClaudeCodeBackend().is_available() == (True, "")
    start.assert_not_called()


def test_start_failure(fake_claude, monkeypatch):
    monkeypatch.setattr(
        "neuroseq.llm.claude_code_backend.subprocess.Popen",
        Mock(side_effect=OSError("missing")),
    )
    with pytest.raises(LLMBackendError, match="Could not start Claude Code"):
        list(ClaudeCodeBackend().stream_chat([], model="default"))


def test_claude_widget_source_read_round_trip(qtbot, fake_claude, monkeypatch):
    modes, log, processes = fake_claude
    modes["mode"] = "read"
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.build_static_prompt", lambda: "NeuroSeq docs"
    )
    monkeypatch.setattr(
        ChatWidget, "_make_backend", lambda self, settings: ClaudeCodeBackend()
    )
    widget = ChatWidget()
    qtbot.addWidget(widget)
    widget.input_edit.setPlainText("Explain NeuroSeq's source code")
    widget.send_button.click()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=5000)
    assert widget.status_label.text() == "Idle"
    assert widget._history[-1].content == "Hello NeuroSeq"
    assert len(processes) == 2
    last_prompt = _calls(log)[-1]["prompt"]
    assert "NeuroSeq request result" in last_prompt
    assert "from __future__ import annotations" in last_prompt


def test_claude_widget_stop_drains_process(qtbot, fake_claude, monkeypatch):
    modes, log, processes = fake_claude
    modes["mode"] = "silent"
    monkeypatch.setattr(
        "neuroseq.llm.chat_widget.build_static_prompt", lambda: "NeuroSeq docs"
    )
    monkeypatch.setattr(
        ChatWidget, "_make_backend", lambda self, settings: ClaudeCodeBackend()
    )
    widget = ChatWidget()
    qtbot.addWidget(widget)
    widget.input_edit.setPlainText("Hi")
    widget.send_button.click()
    qtbot.waitUntil(lambda: log.exists(), timeout=5000)
    assert widget.stop_worker(timeout_ms=3000)
    assert all(process.poll() is not None for process in processes)
    qtbot.waitUntil(lambda: widget.send_button.isEnabled())
