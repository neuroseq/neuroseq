from pathlib import Path

import numpy as np
import yaml
from brian2 import TimedArray
from brian2 import get_dimensions
from brian2 import metre
from brian2 import ms
from brian2 import nS
from brian2 import second
from formozov_project import yaml_extensions  # noqa: F401


def test_quantity_yml(tmp_path: Path) -> None:
    q = np.array([[1, 2, 3], [4, 5, 6]]) * (metre / second)

    d = {"a": q, "b": 3.13 * 1 / second}

    with open(tmp_path / "test.yml", "w") as file:
        yaml.safe_dump(d, file)

    with open(tmp_path / "test.yml") as file:
        read = yaml.safe_load(file)

    assert list(d.keys()) == list(read.keys())
    assert d["b"] == read["b"]
    assert d["a"].shape == read["a"].shape
    assert (
        get_dimensions(d["a"])._str_representation()
        == get_dimensions(read["a"])._str_representation()
    )
    np.testing.assert_allclose(np.array(d["a"]), np.array(read["a"]))


def test_timed_array_yml(tmp_path: Path) -> None:
    dt = 1 * ms
    arr = np.array([[1, 2, 3], [4, 5, 6]]) * nS
    t = TimedArray(arr, dt=dt)

    d = {
        "t": t,
    }

    with open(tmp_path / "test.yml", "w") as file:
        yaml.safe_dump(d, file)

    with open(tmp_path / "test.yml") as file:
        read = yaml.safe_load(file)

    np.testing.assert_allclose(np.array(d["t"].values), np.array(read["t"].values))
    assert (
        get_dimensions(d["t"])._str_representation()
        == get_dimensions(read["t"])._str_representation()
    )


def test_numpy_yml(tmp_path: Path) -> None:
    q = np.array([1, 2, 3])
    d = {
        "a": q,
    }

    with open(tmp_path / "test.yml", "w") as file:
        yaml.safe_dump(d, file)

    with open(tmp_path / "test.yml") as file:
        read = yaml.safe_load(file)

    np.testing.assert_allclose(d["a"], read["a"])
