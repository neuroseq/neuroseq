from __future__ import annotations

import os
from collections.abc import Iterable

from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import missing_dependency_message
from neuroseq.llm.backend import register
from neuroseq.llm.secrets import get_api_key


@register
class OpenAIBackend(LLMBackend):
    name = "openai"
    label = "OpenAI"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMBackendError(
                missing_dependency_message("openai", "uv pip install openai")
            ) from e
        key = api_key or get_api_key("openai") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMBackendError(
                "No OpenAI API key found. Set one in Settings → LLM, or export "
                "OPENAI_API_KEY."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def list_models(self) -> list[str]:
        try:
            models = self._client.models.list()
            return sorted(
                model.id
                for model in models.data
                if not any(
                    excluded in model.id
                    for excluded in ["embedding", "whisper", "tts", "transcribe"]
                )
            )
        except Exception as e:
            raise LLMBackendError(f"Failed to list OpenAI models: {e}") from e

    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterable[str]:
        try:
            stream = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": message.role.value, "content": message.content}
                    for message in messages
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise LLMBackendError(f"OpenAI error: {e}") from e
