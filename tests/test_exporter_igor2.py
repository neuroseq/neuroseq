from __future__ import annotations

import importlib
import struct
from pathlib import Path

import brian2.numpy_ as np
import pytest

from neuroseq.exporter_igor2 import _sanitize_igor_wave_name
from neuroseq.exporter_igor2 import _write_ibw_v5
from neuroseq.exporter_igor2 import export_igor2_patchclamp_csv
from neuroseq.exporter_igor2 import export_igor2_patchclamp_ibw
from neuroseq.exporter_igor2 import export_igor2_pxp
from neuroseq.exporter_igor2 import generate_igor2_patchclamp_csv_text
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def test_generate_igor2_patchclamp_csv_text_exports_single_neuron_column() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array(
            [
                [0.0, 10.0],
                [1.5, 20.0],
                [0.0, 30.0],
            ],
            dtype=np.float32,
        ),
        units="pA",
        dt=0.001,
    )

    csv_text = generate_igor2_patchclamp_csv_text(
        stim_array,
        experiment_name="one_neuron_stim_example",
        neuron_index=1,
    )

    assert csv_text == "one_neuron_stim_example\n10.0\n20.0\n30.0\n"


def test_export_igor2_patchclamp_csv_writes_file(tmp_path: Path) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[0.0], [2.0], [4.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "export.csv"

    report = export_igor2_patchclamp_csv(
        output_path,
        stim_array,
        experiment_name="example_export",
        neuron_index=0,
    )

    assert output_path.read_text(encoding="utf-8") == "example_export\n0.0\n2.0\n4.0\n"
    assert report.experiment_name == "example_export"
    assert report.neuron_index == 0
    assert report.sample_count == 3


def test_sanitize_igor_wave_name_replaces_invalid_chars() -> None:
    assert _sanitize_igor_wave_name("my stim-array!") == "my_stim_array_"


def test_sanitize_igor_wave_name_prepends_w_for_leading_digit() -> None:
    assert _sanitize_igor_wave_name("1_stim") == "w1_stim"


def test_sanitize_igor_wave_name_truncates_to_31_chars() -> None:
    long_name = "A" * 50
    assert len(_sanitize_igor_wave_name(long_name)) == 31


def test_sanitize_igor_wave_name_preserves_valid_name() -> None:
    assert _sanitize_igor_wave_name("stim_neuron_0") == "stim_neuron_0"


def test_write_ibw_v5_produces_correct_byte_structure() -> None:
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _write_ibw_v5(data, wave_name="myWave", dt=0.001, data_units="pA")

    (version,) = struct.unpack_from("<h", result, offset=0)
    (wh_version,) = struct.unpack_from("<h", result, offset=90)
    (wfm_size,) = struct.unpack_from("<l", result, offset=4)

    assert version == 5
    assert wh_version == 1
    assert wfm_size == 320 + 3 * 4
    assert len(result) == 2 + 62 + wfm_size


def test_write_ibw_v5_round_trips_via_binarywave_load(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")

    data = np.array([0.0, 1.5, 3.0, 0.0], dtype=np.float32)
    ibw_path = tmp_path / "roundtrip.ibw"
    ibw_path.write_bytes(
        _write_ibw_v5(
            data,
            wave_name="RoundTrip",
            dt=0.0005,
            data_units="pA",
            x_units="s",
        )
    )

    loaded = binarywave.load(ibw_path)
    wave = loaded["wave"]
    header = wave["wave_header"]

    np.testing.assert_array_almost_equal(
        wave["wData"].astype(np.float32),
        data,
        decimal=5,
    )
    assert abs(header["sfA"][0] - 0.0005) < 1e-10
    assert header["bname"].rstrip(b"\x00").decode("ascii") == "RoundTrip"
    assert (
        b"".join(header["dataUnits"].tolist()).rstrip(b"\x00").decode("ascii") == "pA"
    )


def test_export_igor2_patchclamp_ibw_writes_file(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[0.0, 10.0], [1.5, 20.0], [0.0, 30.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "stim_neuron_1.ibw"

    report = export_igor2_patchclamp_ibw(
        output_path,
        stim_array,
        wave_name="stim_neuron_1",
        neuron_index=1,
    )

    assert output_path.exists()
    assert report.neuron_index == 1
    assert report.sample_count == 3
    assert report.experiment_name == "stim_neuron_1"

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        decimal=5,
    )
    assert abs(loaded["wave"]["wave_header"]["sfA"][0] - 0.001) < 1e-10


def test_export_igor2_patchclamp_ibw_rejects_invalid_neuron() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    with pytest.raises(ValueError, match="out of range"):
        export_igor2_patchclamp_ibw(
            "/dev/null",
            stim_array,
            wave_name="w",
            neuron_index=5,
        )


def test_sanitize_igor_wave_name_is_applied_on_export(tmp_path: Path) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.ones((3, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    report = export_igor2_patchclamp_ibw(
        tmp_path / "out.ibw",
        stim_array,
        wave_name="my wave name!",
        neuron_index=0,
    )

    assert " " not in report.experiment_name
    assert "!" not in report.experiment_name


def test_generate_igor2_patchclamp_csv_text_rejects_invalid_neuron() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    with pytest.raises(ValueError, match="out of range"):
        generate_igor2_patchclamp_csv_text(
            stim_array,
            experiment_name="example_export",
            neuron_index=3,
        )


def test_export_igor2_pxp_roundtrip_single_array(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Exc",
        stim_type=StimType.Excitation,
        values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "test.pxp"

    report = export_igor2_pxp(output_path, {"Exc": stim})

    assert output_path.exists()
    assert report.wave_names == ["Exc_n0", "Exc_n1"]
    records, _ = packed.load(output_path)
    assert len(records) == 2
    names = [
        record.wave["wave"]["wave_header"]["bname"].rstrip(b"\x00").decode("ascii")
        for record in records
        if hasattr(record, "wave")
    ]
    assert names == ["Exc_n0", "Exc_n1"]


def test_export_igor2_pxp_inhibitory_arrays_are_negated(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Inh",
        stim_type=StimType.Inhibition,
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        units="1",
        dt=0.001,
    )
    output_path = tmp_path / "inh.pxp"

    export_igor2_pxp(output_path, {"Inh": stim})

    records, _ = packed.load(output_path)
    wave_data = records[0].wave["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        decimal=5,
    )


def test_export_igor2_pxp_writes_one_wave_per_neuron_per_array(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    exc = StimArray(
        name="Exc",
        stim_type=StimType.Excitation,
        values=np.ones((10, 3), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    inh = StimArray(
        name="Inh",
        stim_type=StimType.Inhibition,
        values=np.ones((10, 2), dtype=np.float32),
        units="1",
        dt=0.001,
    )
    output_path = tmp_path / "multi.pxp"

    report = export_igor2_pxp(output_path, {"Exc": exc, "Inh": inh})

    records, _ = packed.load(output_path)
    assert len(records) == 5
    assert len(report.wave_names) == 5


def test_export_igor2_pxp_external_arrays_are_not_negated(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Ext",
        stim_type=StimType.External,
        values=np.array([[-1.0], [2.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "ext.pxp"

    export_igor2_pxp(output_path, {"Ext": stim})

    records, _ = packed.load(output_path)
    wave_data = records[0].wave["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([-1.0, 2.0], dtype=np.float32),
        decimal=5,
    )


def test_export_igor2_pxp_deduplicates_long_wave_names(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    base = "A" * 31
    first = StimArray(
        name=base,
        stim_type=StimType.Excitation,
        values=np.ones((2, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    second = StimArray(
        name=f"{base}!",
        stim_type=StimType.Excitation,
        values=np.ones((2, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "dedup.pxp"

    report = export_igor2_pxp(output_path, {first.name: first, second.name: second})

    records, _ = packed.load(output_path)
    names = [
        record.wave["wave"]["wave_header"]["bname"].rstrip(b"\x00").decode("ascii")
        for record in records
        if hasattr(record, "wave")
    ]
    assert len(names) == 2
    assert len(set(names)) == 2
    assert len(report.wave_names) == 2
