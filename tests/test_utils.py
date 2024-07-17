from __future__ import annotations

from typing import Any

import brian2 as b2
import pytest
from formozov_project import utils


def test_quantity_parser() -> None:
    assert utils.parse_string_to_quantity("3")[0] == "3"

    assert utils.parse_string_to_quantity("3 * ms")[0] == 3 * b2.ms

    assert utils.parse_string_to_quantity("3 ms")[0] == 3 * b2.ms

    assert utils.parse_string_to_quantity("1/(10) * ms")[0] == 1 / 10 * b2.ms

    with pytest.raises(RuntimeError, match="Error when interpreting unit string"):
        utils.parse_string_to_quantity("1/(10) * Mx")


def test_update_nested_dict() -> None:
    d: dict[Any, Any] = {
        "h": {
            "l": 5,
            3: "hello",
            "m": 66,
        },
        "r": {"d": {"x": 12, "y": "z"}, "e": "test"},
        "a": 3.14,
        "b": 42,
    }

    utils.update_nested_dict(
        d["h"],
        {
            "l": 55,
        },
    )

    assert d["h"]["l"] == 55
    # rest is as was
    assert d["r"] == {"d": {"x": 12, "y": "z"}, "e": "test"}

    utils.update_nested_dict(
        d,
        {
            "r": {
                55: "other",
            }
        },
    )

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }

    assert d["r"] == {55: "other", "d": {"x": 12, "y": "z"}, "e": "test"}

    utils.update_nested_dict(d, {"a": 6.28})

    assert d["a"] == 6.28
    assert d["r"] == {55: "other", "d": {"x": 12, "y": "z"}, "e": "test"}

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }
