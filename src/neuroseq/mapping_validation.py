"""
AI-Generated
"""


def ensure_unique_ids(values: list[int], error_message: str) -> list[int]:
    seen: set[int] = set()
    for value in values:
        if value in seen:
            raise ValueError(error_message.format(value=value))
        seen.add(value)
    return values
