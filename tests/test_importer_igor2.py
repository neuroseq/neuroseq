from __future__ import annotations

import importlib
from pathlib import Path

import brian2.numpy_ as np
import pytest

from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import create_stim_array_from_igor_wave
from neuroseq.importer_igor2 import import_igor_wave_into_stim_array
from neuroseq.importer_igor2 import load_igor_csv_file
from neuroseq.importer_igor2 import load_igor_ibw_file
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
