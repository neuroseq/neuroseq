import json
import zipfile
from pathlib import Path

import brian2.numpy_ as np
from brian2 import TimedArray
from brian2 import get_dimensions
from brian2 import metre
from brian2 import ms
from brian2 import nS
from brian2 import second
from brian2.tests.utils import assert_allclose

from neuroseq import b2json


def test_quantity_json(tmp_path: Path) -> None:
    q = np.array([[1, 2, 3], [4, 5, 6]]) * (metre / second)

    d = {"a": q, "b": 3.13 * 1 / second}

    tmp_file = tmp_path / "test.nseq"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        json_str = json.dumps(d, cls=b2json.b2json_encoder_factory(zf))
        read = json.loads(json_str, object_hook=b2json.b2_object_hook_factory(zf))

    assert list(d.keys()) == list(read.keys())
    assert d["b"] == read["b"]
    assert d["a"].shape == read["a"].shape
    assert (
        get_dimensions(d["a"])._str_representation()
        == get_dimensions(read["a"])._str_representation()
    )
    assert_allclose(np.array(d["a"]), np.array(read["a"]))


def test_timed_array_json(tmp_path: Path) -> None:
    dt = 1 * ms
    arr = np.array([[1, 2, 3], [4, 5, 6]]) * nS
    t = TimedArray(arr, dt=dt)

    d = {
        "t": t,
    }

    tmp_file = tmp_path / "test.nseq"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        json_str = json.dumps(d, cls=b2json.b2json_encoder_factory(zf))
        read = json.loads(json_str, object_hook=b2json.b2_object_hook_factory(zf))

    assert_allclose(np.array(d["t"].values), np.array(read["t"].values))
    assert (
        get_dimensions(d["t"])._str_representation()
        == get_dimensions(read["t"])._str_representation()
    )


def test_numpy_json(tmp_path: Path) -> None:
    q = np.array([1, 2, 3])
    d = {
        "a": q,
    }

    tmp_file = tmp_path / "test.nseq"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        json_str = json.dumps(d, cls=b2json.b2json_encoder_factory(zf))
        read = json.loads(json_str, object_hook=b2json.b2_object_hook_factory(zf))

    assert_allclose(d["a"], read["a"])
