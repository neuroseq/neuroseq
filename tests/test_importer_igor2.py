from __future__ import annotations

import importlib
from pathlib import Path

import brian2.numpy_ as np
import pytest

from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import create_stim_array_from_igor_wave
from neuroseq.importer_igor2 import import_igor_wave_into_stim_array
from neuroseq.importer_igor2 import is_pxp_file
from neuroseq.importer_igor2 import load_igor_csv_file
from neuroseq.importer_igor2 import load_igor_ibw_file
from neuroseq.importer_igor2 import load_igor_pxp_file
from neuroseq.importer_igor2 import prepare_igor_wave_for_neuroseq
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def test_load_igor_csv_file_parses_simple_wave(tmp_path: Path) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("example_wave\r\n1.0\r\n2.5\r\n-3.0\r\n", encoding="utf-8")

    wave = load_igor_csv_file(csv_path)

    assert wave.name == "example_wave"
    assert wave.source_dt is None
    assert wave.units is None
    np.testing.assert_allclose(wave.values, np.array([1.0, 2.5, -3.0]))


def test_prepare_igor_wave_for_neuroseq_resamples_to_1ms() -> None:
    wave = IgorWave(
        name="test_wave",
        values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        source_dt=0.0005,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )

    prepared = prepare_igor_wave_for_neuroseq(wave)

    assert prepared.resampled is True
    assert prepared.target_dt == pytest.approx(0.001)
    np.testing.assert_allclose(prepared.values, np.array([1.0, 3.0]))


def test_create_stim_array_from_igor_wave_uses_single_neuron_column() -> None:
    wave = IgorWave(
        name="test_wave",
        values=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )

    prepared = prepare_igor_wave_for_neuroseq(wave)
    stim_array = create_stim_array_from_igor_wave(
        prepared,
        stim_name="imported_stim",
        stim_type=StimType.External,
    )

    assert stim_array.name == "imported_stim"
    assert stim_array.values.shape == (3, 1)
    assert stim_array.units == "mV"
    assert stim_array.dt == pytest.approx(0.001)


def test_import_igor_wave_into_stim_array_replaces_target_column_and_extends() -> None:
    target_stim = StimArray(
        name="stimulus",
        stim_type=StimType.External,
        values=np.array([[9.0, 1.0], [9.0, 1.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave)

    report = import_igor_wave_into_stim_array(
        target_stim,
        prepared,
        neuron_index=1,
        replace_existing=True,
        extend_to_fit=True,
    )

    assert report.imported_samples == 3
    assert target_stim.values.shape == (3, 2)
    np.testing.assert_allclose(target_stim.values[:, 0], np.array([9.0, 9.0, 0.0]))
    np.testing.assert_allclose(target_stim.values[:, 1], np.array([5.0, 6.0, 7.0]))


def test_load_igor_ibw_file_extracts_name_dt_and_units(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ibw_path = tmp_path / "sample.ibw"
    ibw_path.write_bytes(b"igor")

    fake_wave_data = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"Single_actionPotential\x00",
                "sfA": [0.0005, 0.0, 0.0, 0.0],
                "dataUnits": b"pA\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "note": b"example note\x00",
        },
    }

    class FakeBinaryWaveModule:
        @staticmethod
        def load(path: Path) -> dict[str, object]:
            assert path == ibw_path
            return fake_wave_data

    original_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "igor2.binarywave":
            return FakeBinaryWaveModule()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    wave = load_igor_ibw_file(ibw_path)

    assert wave.name == "Single_actionPotential"
    assert wave.source_dt == pytest.approx(0.0005)
    assert wave.units == "pA"
    assert wave.axis_units == "s"
    assert wave.note == "example note"


class _FakeWaveRecord:
    def __init__(self, wave: dict[str, object]) -> None:
        self.wave = wave


class _FakePackedModule:
    def __init__(self, filesystem: dict[object, object]) -> None:
        self._filesystem = filesystem

    def load(self, path: Path) -> tuple[list[object], dict[object, object]]:
        return [], self._filesystem

    def walk(
        self,
        fs: dict[object, object],
        callback,
        dirpath: list[object] | None = None,
    ) -> None:
        if dirpath is None:
            dirpath = []
        for key, value in sorted(fs.items(), key=lambda item: str(item[0])):
            callback(dirpath, key, value)
            if isinstance(value, dict):
                self.walk(value, callback, dirpath + [key])


def _make_fake_packed_module(filesystem: dict[object, object]) -> _FakePackedModule:
    return _FakePackedModule(filesystem)


def _make_fake_wave_record(wave_dict: dict[str, object]) -> _FakeWaveRecord:
    return _FakeWaveRecord(wave_dict)


def test_load_igor_pxp_file_returns_all_1d_waves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake pxp")

    wave_dict = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"MyWave\x00",
                "sfA": [0.001, 0.0, 0.0, 0.0],
                "dataUnits": b"pA\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "note": b"",
        },
    }

    fake_packed = _make_fake_packed_module(
        {b"root": {b"MyWave": _make_fake_wave_record(wave_dict)}}
    )

    class FakeWaveRecordModule:
        WaveRecord = _FakeWaveRecord

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "igor2.packed":
            return fake_packed
        if name == "igor2.record.wave":
            return FakeWaveRecordModule()
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    waves = load_igor_pxp_file(pxp_path)

    assert waves.keys() == {"root/MyWave"}
    wave = waves["root/MyWave"]
    assert wave.name == "MyWave"
    assert wave.source_dt == pytest.approx(0.001)
    assert wave.units == "pA"
    np.testing.assert_allclose(wave.values, np.array([1.0, 2.0, 3.0]))


def test_load_igor_pxp_file_skips_multidimensional_waves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")

    wave_2d_dict = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"TwoDWave\x00",
                "sfA": [0.001, 0.0, 0.0, 0.0],
                "dataUnits": b"\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.ones((3, 4), dtype=np.float32),
            "note": b"",
        },
    }
    wave_1d_dict = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"OneDWave\x00",
                "sfA": [0.001, 0.0, 0.0, 0.0],
                "dataUnits": b"mV\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.array([0.5, 1.5], dtype=np.float32),
            "note": b"",
        },
    }

    fake_packed = _make_fake_packed_module(
        {
            b"TwoDWave": _make_fake_wave_record(wave_2d_dict),
            b"OneDWave": _make_fake_wave_record(wave_1d_dict),
        }
    )

    class FakeWaveRecordModule:
        WaveRecord = _FakeWaveRecord

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "igor2.packed":
            return fake_packed
        if name == "igor2.record.wave":
            return FakeWaveRecordModule()
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    waves = load_igor_pxp_file(pxp_path)

    assert waves.keys() == {"OneDWave"}
    np.testing.assert_allclose(waves["OneDWave"].values, np.array([0.5, 1.5]))


def test_load_igor_pxp_file_raises_if_no_waves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pxp_path = tmp_path / "empty.pxp"
    pxp_path.write_bytes(b"fake")

    fake_packed = _make_fake_packed_module({})

    class FakeWaveRecordModule:
        WaveRecord = _FakeWaveRecord

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "igor2.packed":
            return fake_packed
        if name == "igor2.record.wave":
            return FakeWaveRecordModule()
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ValueError, match="No importable"):
        load_igor_pxp_file(pxp_path)


def test_load_igor_pxp_file_raises_if_igor2_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in ("igor2.packed", "igor2.record.wave"):
            raise ImportError("no module named igor2")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="igor2"):
        load_igor_pxp_file(pxp_path)


def test_is_pxp_file_detects_extension() -> None:
    assert is_pxp_file("foo.pxp") is True
    assert is_pxp_file("foo.PXP") is True
    assert is_pxp_file("foo.ibw") is False
    assert is_pxp_file("foo.csv") is False
