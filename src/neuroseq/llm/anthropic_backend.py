from __future__ import annotations

import os
from collections.abc import Iterable

from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.backend import missing_dependency_message
from neuroseq.llm.backend import register
from neuroseq.llm.secrets import get_api_key


@register
class AnthropicBackend(LLMBackend):
    name = "anthropic"
    label = "Anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMBackendError(
                missing_dependency_message("anthropic", "uv pip install anthropic")
            ) from e
        key = api_key or get_api_key("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMBackendError(
                "No Anthropic API key found. Set one in Settings → LLM, or export "
                "ANTHROPIC_API_KEY."
            )
        self._client = Anthropic(api_key=key)

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def list_models(self) -> list[str]:
        try:
            response = self._client.models.list()
            return sorted(model.id for model in response.data)
        except Exception as e:
            raise LLMBackendError(f"Failed to list Anthropic models: {e}") from e

    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterable[str]:
        all_messages: list[Message] = list(messages)
        system_messages = [
            message.content for message in all_messages if message.role == Role.SYSTEM
        ]
        non_system_messages = [
            {"role": message.role.value, "content": message.content}
            for message in all_messages
            if message.role != Role.SYSTEM
        ]
        system_prompt = "\n\n".join(system_messages) or None
        try:
            with self._client.messages.stream(
                model=model,
                system=system_prompt,
                messages=non_system_messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
            ) as stream:
                yield from stream.text_stream
        except Exception as e:
            raise LLMBackendError(f"Anthropic error: {e}") from e
