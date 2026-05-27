from __future__ import annotations

from contextlib import suppress

SERVICE = "neuroseq-llm"


def _keyring():
    try:
        import keyring

        return keyring
    except ImportError:
        return None


def get_api_key(provider: str) -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, provider)
    except Exception:
        return None


def set_api_key(provider: str, key: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(SERVICE, provider, key)
        return True
    except Exception:
        return False


def delete_api_key(provider: str) -> None:
    kr = _keyring()
    if kr is None:
        return
    with suppress(Exception):
        kr.delete_password(SERVICE, provider)
