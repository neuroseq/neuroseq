from __future__ import annotations

import builtins
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

from neuroseq.llm import Message
from neuroseq.llm import Role
from neuroseq.llm import available_backends
from neuroseq.llm.anthropic_backend import AnthropicBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.ollama import OllamaBackend
from neuroseq.llm.openai_backend import OpenAIBackend


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code=200,
        lines=None,
        text="",
        raise_on_text: bool = False,
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self._raise_on_text = raise_on_text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self):
        yield from self._lines

    @property
    def text(self):
        if self._raise_on_text:
            raise RuntimeError("Attempted to access streaming response content")
        return self._text

    def read(self):
        return self._text.encode("utf-8")


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "httpx", module)


def test_ollama_missing_httpx_error_includes_active_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("missing httpx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "httpx", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(LLMBackendError) as excinfo:
        OllamaBackend()

    message = str(excinfo.value)
    assert "Current interpreter:" in message
    assert sys.executable in message
    assert "uv pip install httpx" in message


def test_ollama_is_available_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(200, {"models": []})
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse()
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    assert backend.is_available() == (True, "")


def test_ollama_is_available_false_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))

    class _ConnectError(RuntimeError):
        pass

    def _raise(*args, **kwargs):
        raise _ConnectError("no route")

    fake_httpx.get = _raise
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse()
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()
    ok, message = backend.is_available()

    assert ok is False
    assert "http://localhost:11434" in message


def test_ollama_list_models_returns_sorted_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(
        200,
        {"models": [{"name": "qwen:7b"}, {"name": "llama3:8b"}]},
    )
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse()
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    assert backend.list_models() == ["llama3:8b", "qwen:7b"]


def test_ollama_stream_chat_yields_tokens_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(200, {"models": []})
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse(
        lines=[
            '{"message": {"content": "Hel"}, "done": false}',
            '{"message": {"content": "lo"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]
    )
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    result = "".join(
        backend.stream_chat([Message(Role.USER, "hi")], model="qwen2.5:7b")
    )

    assert result == "Hello"


def test_ollama_stream_chat_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(200, {"models": []})
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse(
        status_code=500,
        text="model not found",
    )
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    with pytest.raises(LLMBackendError, match="500"):
        list(backend.stream_chat([Message(Role.USER, "hi")], model="missing"))


def test_ollama_stream_chat_reads_body_before_reporting_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(200, {"models": []})
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse(
        status_code=404,
        text="model 'qwen3:1.7b' not found",
        raise_on_text=True,
    )
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    with pytest.raises(LLMBackendError, match="model 'qwen3:1.7b' not found"):
        list(backend.stream_chat([Message(Role.USER, "hi")], model="qwen3:1.7b"))


def test_ollama_stream_chat_raises_on_error_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx = cast(Any, ModuleType("httpx"))
    fake_httpx.get = lambda url, timeout=0: _FakeResponse(200, {"models": []})
    fake_httpx.stream = lambda *args, **kwargs: _FakeStreamResponse(
        lines=['{"error": "out of memory"}']
    )
    fake_httpx.Timeout = lambda **kwargs: kwargs
    fake_httpx.HTTPError = RuntimeError
    _install_fake_httpx(monkeypatch, fake_httpx)

    backend = OllamaBackend()

    with pytest.raises(LLMBackendError, match="out of memory"):
        list(backend.stream_chat([Message(Role.USER, "hi")], model="qwen2.5:7b"))


def test_openai_requires_key_or_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_openai = cast(Any, ModuleType("openai"))
    fake_openai.OpenAI = lambda **kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "neuroseq.llm.openai_backend.get_api_key",
        lambda provider: None,
    )

    with pytest.raises(LLMBackendError, match="OpenAI API key"):
        OpenAIBackend()


def test_openai_uses_env_var_when_no_keyring_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, str | None] = {}

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            recorded.update(kwargs)

    fake_openai = cast(Any, ModuleType("openai"))
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "neuroseq.llm.openai_backend.get_api_key",
        lambda provider: None,
    )

    OpenAIBackend()

    assert recorded["api_key"] == "sk-test"


def test_anthropic_strips_system_messages_into_top_level_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class _FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        @property
        def text_stream(self):
            yield "hello"

    class _Messages:
        def stream(self, **kwargs):
            recorded.update(kwargs)
            return _FakeStream()

    class _FakeAnthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake_anthropic = cast(Any, ModuleType("anthropic"))
    fake_anthropic.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setattr(
        "neuroseq.llm.anthropic_backend.get_api_key",
        lambda provider: "anthropic-key",
    )

    backend = AnthropicBackend()
    list(
        backend.stream_chat(
            [
                Message(Role.SYSTEM, "be helpful"),
                Message(Role.USER, "hi"),
            ],
            model="claude",
        )
    )

    assert recorded["system"] == "be helpful"
    assert recorded["messages"] == [{"role": "user", "content": "hi"}]


def test_registry_exposes_backends() -> None:
    names = {name for name, _label in available_backends()}
    assert names == {"anthropic", "ollama", "openai", "codex"}
