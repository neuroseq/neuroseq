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
from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData


def _save_and_reload_json(data: dict, tmp_path: Path) -> dict:
    tmp_file = tmp_path / "test.nseq"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        json_str = json.dumps(data, cls=b2json.b2json_encoder_factory(zf))
        return json.loads(json_str, object_hook=b2json.b2_object_hook_factory(zf))


def test_quantity_json(tmp_path: Path) -> None:
    q = np.array([[1, 2, 3], [4, 5, 6]]) * (metre / second)

    d = {"a": q, "b": 3.13 * 1 / second}

    read = _save_and_reload_json(d, tmp_path)

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

    read = _save_and_reload_json(d, tmp_path)

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

    read = _save_and_reload_json(d, tmp_path)

    assert_allclose(d["a"], read["a"])


def test_trace_data_json(tmp_path: Path) -> None:
    d = {"s": TraceData("name", np.random.random_sample((10, 5)), "mV", 0.01)}

    read = _save_and_reload_json(d, tmp_path)

    assert d["s"].name == read["s"].name
    assert np.allclose(d["s"].traces, read["s"].traces)
    assert d["s"].units == read["s"].units
    assert d["s"].dt == read["s"].dt


def test_spike_data_json(tmp_path: Path) -> None:
    d = {
        "s": SpikeData(
            "name",
            np.random.random_integers(0, 9, (10, 10)),
            np.random.random_sample((10, 1)),
        )
    }

    read = _save_and_reload_json(d, tmp_path)

    assert d["s"].name == read["s"].name
    assert np.allclose(d["s"].spike_indices, read["s"].spike_indices)
    assert np.allclose(d["s"].spike_times, read["s"].spike_times)


def test_synapse_data_json(tmp_path: Path) -> None:
    d = {
        "s": SynapseData(
            "name",
            np.random.random_sample((6, 4, 7)),
            np.random.random_sample((7,)),
        )
    }

    read = _save_and_reload_json(d, tmp_path)

    assert d["s"].name == read["s"].name
    assert np.allclose(d["s"].values, read["s"].values)
    assert np.allclose(d["s"].timestamps, read["s"].timestamps)
