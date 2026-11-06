from __future__ import annotations

import abc
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):  # noqa: UP042
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class LLMBackendError(Exception):
    """Raised for any user-visible backend failure."""


def missing_dependency_message(package: str, install_hint: str) -> str:
    return (
        f"The '{package}' package is required in the active Python environment. "
        f"Current interpreter: {sys.executable}. "
        f"Install it there, for example with: {install_hint}"
    )


class LLMBackend(abc.ABC):
    name: str = ""
    label: str = ""

    def prepare(self) -> None:
        """Prepare for a new worker before it starts."""
        return None

    def cancel(self) -> None:
        """Interrupt a pending request, when supported by the transport."""
        return None

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: Iterable[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterable[str]: ...

    @abc.abstractmethod
    def is_available(self) -> tuple[bool, str]: ...

    @abc.abstractmethod
    def list_models(self) -> list[str]: ...


_REGISTRY: dict[str, type[LLMBackend]] = {}


def register(cls: type[LLMBackend]) -> type[LLMBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend_class(name: str) -> type[LLMBackend]:
    if name not in _REGISTRY:
        raise LLMBackendError(
            f"Unknown backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def get_backend(name: str) -> LLMBackend:
    return get_backend_class(name)()


def available_backends() -> list[tuple[str, str]]:
    return [(name, _REGISTRY[name].label) for name in sorted(_REGISTRY)]
