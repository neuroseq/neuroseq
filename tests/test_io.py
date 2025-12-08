import zipfile
from pathlib import Path

import brian2.numpy_ as np
import pytest
from brian2 import get_dimensions
from brian2 import metre
from brian2 import second
from brian2.tests.utils import assert_allclose

from neuroseq import io
from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def assert_equal(original: Data, loaded: Data) -> None:
    assert type(original) is type(loaded)
    if isinstance(original, TraceData):
        assert isinstance(loaded, TraceData)
        assert original.name == loaded.name
        assert np.allclose(original.traces, loaded.traces)
        assert np.allclose(original.indices, loaded.indices)
        assert original.units == loaded.units
        assert original.dt == loaded.dt
    elif isinstance(original, SpikeData):
        assert isinstance(loaded, SpikeData)
        assert original.name == loaded.name
        assert np.allclose(original.spike_indices, loaded.spike_indices)
        assert np.allclose(original.spike_times, loaded.spike_times)
    elif isinstance(original, SynapseData):
        assert isinstance(loaded, SynapseData)
        assert original.name == loaded.name
        assert np.allclose(original.values, loaded.values)
        assert np.allclose(original.timestamps, loaded.timestamps)
    else:
        raise ValueError(f"Unsupported data object type: {type(original)}")


def _save_and_reload_json(data: dict, tmp_path: Path) -> dict:
    tmp_file = tmp_path / "test.nseq"
    data_key = "data"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        io.append_to_zip(data, data_key, zf)
    with zipfile.ZipFile(tmp_file, "r", zipfile.ZIP_DEFLATED) as zf:
        return io.load_from_zip(data_key, zf)


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


def test_numpy_json(tmp_path: Path) -> None:
    q = np.array([1, 2, 3])
    d = {
        "a": q,
    }

    read = _save_and_reload_json(d, tmp_path)

    assert_allclose(d["a"], read["a"])


def test_stim_array_json(tmp_path: Path) -> None:
    d = {
        "s": StimArray(
            "name", StimType.External, np.random.random_sample((10, 5)), "mV", 0.01
        )
    }

    read = _save_and_reload_json(d, tmp_path)

    assert d["s"].name == read["s"].name
    assert d["s"].stim_type == read["s"].stim_type
    assert np.allclose(d["s"].values, read["s"].values)
    assert d["s"].units == read["s"].units
    assert d["s"].dt == read["s"].dt


def test_trace_data_json(tmp_path: Path) -> None:
    d = {
        "s": TraceData(
            "name",
            np.random.random_sample((10, 5)),
            np.arange(4, 14, dtype=np.uint32),
            "mV",
            0.01,
        )
    }

    read = _save_and_reload_json(d, tmp_path)

    assert d["s"].name == read["s"].name
    assert np.allclose(d["s"].traces, read["s"].traces)
    assert np.allclose(d["s"].indices, read["s"].indices)
    assert d["s"].units == read["s"].units
    assert d["s"].dt == read["s"].dt


def test_spike_data_json(tmp_path: Path) -> None:
    d = {
        "s": SpikeData(
            "name",
            np.random.randint(0, 10, (10, 10)),
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


def test_experiment_json(tmp_path: Path) -> None:
    d = {
        "e": Experiment(
            "name",
            10.0,
            [0.0, 2.5, 5.0],
            [
                SynapseData(
                    "syn",
                    np.random.random_sample((6, 4, 7)),
                    np.random.random_sample((7,)),
                ),
                SpikeData(
                    "spike",
                    np.random.randint(0, 10, (10, 10)),
                    np.random.random_sample((10, 1)),
                ),
                TraceData(
                    "tr1",
                    np.random.random_sample((10, 5)),
                    np.arange(4, 14, dtype=np.uint32),
                    "mV",
                    0.01,
                ),
                TraceData(
                    "tr2",
                    np.random.random_sample((10, 5)),
                    np.arange(4, 14, dtype=np.uint32),
                    "mV",
                    0.01,
                ),
            ],
        )
    }

    read = _save_and_reload_json(d, tmp_path)

    assert d["e"].name == read["e"].name
    assert d["e"].duration == pytest.approx(read["e"].duration)
    assert np.allclose(d["e"].timepoints, read["e"].timepoints)
    assert len(d["e"].data) == len(read["e"].data)
    for original, loaded in zip(d["e"].data, read["e"].data, strict=True):
        assert_equal(original, loaded)
