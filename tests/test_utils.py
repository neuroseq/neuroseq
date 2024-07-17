from __future__ import annotations

from typing import Any

from formozov_project import utils


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
    print(d)

    assert d["h"] == {
        "l": 55,
        3: "hello",
        "m": 66,
    }

    assert d["r"] == {
        55: "other",
    }

    # utils.update_nested_dict(d, {"a": 6.28})

    # assert d["a"] == 6.28
    # assert d["r"] == {
    #     55: "other",
    # }

    # assert d["h"] == {
    #     "l": 55,
    #     3: "hello",
    #     "m": 66,
    # }
