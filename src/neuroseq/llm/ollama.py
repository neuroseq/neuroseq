from __future__ import annotations

import json
from collections.abc import Iterable

from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import missing_dependency_message
from neuroseq.llm.backend import register


@register
class OllamaBackend(LLMBackend):
    name = "ollama"
    label = "Ollama (local)"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        try:
            import httpx
        except ImportError as e:
            raise LLMBackendError(
                missing_dependency_message("httpx", "uv pip install httpx")
            ) from e
        self._httpx = httpx

    def is_available(self) -> tuple[bool, str]:
        try:
            response = self._httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                return True, ""
            return False, f"Ollama responded with HTTP {response.status_code}"
        except Exception as e:
            return False, (
                f"Could not reach Ollama at {self.base_url}. Install from "
                f"https://ollama.com and run `ollama serve`. "
                f"({e.__class__.__name__})"
            )

    def list_models(self) -> list[str]:
        try:
            response = self._httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return sorted(model["name"] for model in data.get("models", []))
        except Exception as e:
            raise LLMBackendError(f"Failed to list Ollama models: {e}") from e

    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterable[str]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"] = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        try:
            with self._httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self._httpx.Timeout(
                    connect=5.0,
                    read=120.0,
                    write=10.0,
                    pool=5.0,
                ),
            ) as response:
                if response.status_code != 200:
                    response_body = ""
                    try:
                        response_body = response.read().decode(
                            "utf-8", errors="replace"
                        )
                    except Exception:
                        response_body = "<unable to read response body>"
                    raise LLMBackendError(
                        f"Ollama returned HTTP {response.status_code}: "
                        f"{response_body[:200]}"
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise LLMBackendError(f"Ollama error: {chunk['error']}")
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        return
        except LLMBackendError:
            raise
        except self._httpx.HTTPError as e:
            raise LLMBackendError(f"Ollama network error: {e}") from e
