from __future__ import annotations

import importlib
import struct
from pathlib import Path

import brian2.numpy_ as np
import pytest

from neuroseq.export.exporter_igor2 import _sanitize_igor_wave_name
from neuroseq.export.exporter_igor2 import _write_ibw_v5
from neuroseq.export.exporter_igor2 import export_igor2_csv
from neuroseq.export.exporter_igor2 import export_igor2_ibw
from neuroseq.export.exporter_igor2 import export_igor2_pxp
from neuroseq.export.exporter_igor2 import generate_igor2_csv_text
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def test_generate_igor2_csv_text_exports_single_neuron_column() -> None:
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

    csv_text = generate_igor2_csv_text(
        stim_array,
        experiment_name="one_neuron_stim_example",
        neuron_index=1,
    )

    assert csv_text == "one_neuron_stim_example\n10.0\n20.0\n30.0\n"


def test_export_igor2_csv_writes_file(tmp_path: Path) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[0.0], [2.0], [4.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "export.csv"

    report = export_igor2_csv(
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


def test_write_ibw_v5_two_d_emits_nDim_correctly(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    ibw_path = tmp_path / "two_d.ibw"
    ibw_path.write_bytes(_write_ibw_v5(data, wave_name="TwoD", dt=0.001))

    loaded = binarywave.load(ibw_path)
    header = loaded["wave"]["wave_header"]
    assert tuple(int(v) for v in header["nDim"][:4]) == (5, 3, 0, 0)
    assert int(header["npnts"]) == 15


def test_write_ibw_v5_two_d_data_is_column_major() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    result = _write_ibw_v5(data, wave_name="TwoD", dt=0.001)
    data_offset = 2 + 62 + 320
    raw = np.frombuffer(result[data_offset:], dtype="<f4", count=6)
    np.testing.assert_array_equal(raw, np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]))


def test_write_ibw_v5_round_trips_two_d(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    ibw_path = tmp_path / "roundtrip_two_d.ibw"
    ibw_path.write_bytes(_write_ibw_v5(data, wave_name="TwoD", dt=0.001))

    loaded = binarywave.load(ibw_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.shape == (5, 3)
    np.testing.assert_array_equal(wave_data, data)


def test_write_ibw_v5_rejects_three_d() -> None:
    data = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _write_ibw_v5(data, wave_name="bad", dt=0.001)


def test_export_igor2_ibw_writes_file(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[0.0, 10.0], [1.5, 20.0], [0.0, 30.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "stim_neuron_1.ibw"

    report = export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="stim_neuron_1",
        neuron_index=1,
    )

    assert output_path.exists()
    assert report.neuron_index == 1
    assert report.sample_count == 3
    assert report.experiment_name == "stim_neuron_1"
    assert report.n_sweeps == 1

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        decimal=5,
    )
    assert abs(loaded["wave"]["wave_header"]["sfA"][0] - 0.001) < 1e-10


def test_export_igor2_ibw_rejects_invalid_neuron() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    with pytest.raises(ValueError, match="out of range"):
        export_igor2_ibw(
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

    report = export_igor2_ibw(
        tmp_path / "out.ibw",
        stim_array,
        wave_name="my wave name!",
        neuron_index=0,
    )

    assert " " not in report.experiment_name
    assert "!" not in report.experiment_name


def test_export_ibw_as_sweeps_writes_2d_when_period_detected(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.tile(np.array([[1.0], [2.0], [3.0]], dtype=np.float32), (5, 1)),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "sweeps.ibw"

    report = export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="sweeps",
        neuron_index=0,
        as_sweeps=True,
    )

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.shape == (3, 5)
    assert report.n_sweeps == 5


def test_export_ibw_as_sweeps_falls_back_to_1d_when_no_period(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.arange(7, dtype=np.float32).reshape(7, 1),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "aperiodic.ibw"

    report = export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="aperiodic",
        neuron_index=0,
        as_sweeps=True,
    )

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.ndim == 1
    assert report.n_sweeps == 1


def test_export_ibw_as_sweeps_falls_back_when_single_sweep(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "single.ibw"

    report = export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="single",
        neuron_index=0,
        as_sweeps=True,
    )

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.ndim == 1
    assert report.n_sweeps == 1


def test_export_ibw_as_sweeps_applies_scaling_before_detection(tmp_path: Path) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    base = np.tile(np.array([[1.0], [2.0], [3.0]], dtype=np.float32), (4, 1))
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=(base * 0.5).astype(np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "scaled_sweeps.ibw"

    report = export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="scaled",
        neuron_index=0,
        scaling_factor=2.0,
        as_sweeps=True,
    )

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.shape == (3, 4)
    np.testing.assert_array_equal(
        wave_data[:, 0],
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert report.n_sweeps == 4


def test_generate_igor2_csv_text_rejects_invalid_neuron() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    with pytest.raises(ValueError, match="out of range"):
        generate_igor2_csv_text(
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


def test_export_igor2_pxp_keeps_the_neuron_suffix_of_long_array_names(
    tmp_path: Path,
) -> None:
    stim = StimArray(
        name="ChrimsonR_power_sweep_1140nm_cells",
        stim_type=StimType.Excitation,
        values=np.zeros((5, 3), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    report = export_igor2_pxp(tmp_path / "long.pxp", {stim.name: stim})

    assert report.wave_names == [
        "ChrimsonR_power_sweep_1140nm_n0",
        "ChrimsonR_power_sweep_1140nm_n1",
        "ChrimsonR_power_sweep_1140nm_n2",
    ]


def test_export_igor2_pxp_unit_override_replaces_per_array_units(
    tmp_path: Path,
) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Light",
        stim_type=StimType.Excitation,
        values=np.ones((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "unit.pxp"

    export_igor2_pxp(
        output_path,
        {"Light": stim},
        unit_override="nA",
    )

    records, _ = packed.load(output_path)
    data_units = (
        bytes(records[0].wave["wave"]["wave_header"]["dataUnits"])
        .rstrip(b"\x00")
        .decode("ascii")
    )
    assert data_units == "nA"  # overridden, not the stim's original "pA"


def test_export_igor2_pxp_empty_unit_override_keeps_array_units(
    tmp_path: Path,
) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Light",
        stim_type=StimType.Excitation,
        values=np.ones((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "keep_unit.pxp"

    export_igor2_pxp(output_path, {"Light": stim}, unit_override=None)

    records, _ = packed.load(output_path)
    data_units = (
        bytes(records[0].wave["wave"]["wave_header"]["dataUnits"])
        .rstrip(b"\x00")
        .decode("ascii")
    )
    assert data_units == "pA"


def test_export_igor2_pxp_scaling_factor_scales_values(tmp_path: Path) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Light",
        stim_type=StimType.Excitation,
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "scale.pxp"

    export_igor2_pxp(output_path, {"Light": stim}, scaling_factor=2.5)

    records, _ = packed.load(output_path)
    wave_data = records[0].wave["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([2.5, 5.0, 7.5], dtype=np.float32),
        decimal=5,
    )


def test_export_igor2_pxp_scaling_factor_applies_after_inhibition_negate(
    tmp_path: Path,
) -> None:
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Inh",
        stim_type=StimType.Inhibition,
        values=np.array([[1.0], [2.0]], dtype=np.float32),
        units="1",
        dt=0.001,
    )
    output_path = tmp_path / "neg_scale.pxp"

    export_igor2_pxp(output_path, {"Inh": stim}, scaling_factor=2.0)

    records, _ = packed.load(output_path)
    wave_data = records[0].wave["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([-2.0, -4.0], dtype=np.float32),
        decimal=5,
    )


def test_export_igor2_pxp_rejects_non_finite_scaling(tmp_path: Path) -> None:
    stim = StimArray(
        name="Light",
        stim_type=StimType.Excitation,
        values=np.ones((2, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    with pytest.raises(ValueError, match="scaling_factor"):
        export_igor2_pxp(
            output_path=tmp_path / "bad.pxp",
            stim_arrays={"Light": stim},
            scaling_factor=float("inf"),
        )


def test_write_ibw_v5_emits_extended_units_for_long_strings(tmp_path: Path) -> None:
    """Units > 3 ASCII chars must round-trip through the IBW v5 extended block."""
    binarywave = importlib.import_module("igor2.binarywave")
    data = np.array([0.0, 1.0], dtype=np.float32)
    ibw_path = tmp_path / "long_units.ibw"
    ibw_path.write_bytes(
        _write_ibw_v5(
            data,
            wave_name="LongUnits",
            dt=0.001,
            data_units="mW/mm^2",
        )
    )
    loaded = binarywave.load(ibw_path)
    extended = bytes(loaded["wave"]["data_units"]).decode("ascii")
    assert extended == "mW/mm^2"


def test_write_ibw_v5_short_units_have_empty_extended_block(tmp_path: Path) -> None:
    """3-char-or-shorter units stay in the short field; no extended block written."""
    binarywave = importlib.import_module("igor2.binarywave")
    data = np.array([0.0, 1.0], dtype=np.float32)
    ibw_path = tmp_path / "short_units.ibw"
    ibw_path.write_bytes(
        _write_ibw_v5(data, wave_name="ShortU", dt=0.001, data_units="pA")
    )
    loaded = binarywave.load(ibw_path)
    short = (
        b"".join(loaded["wave"]["wave_header"]["dataUnits"].tolist())
        .rstrip(b"\x00")
        .decode("ascii")
    )
    assert short == "pA"
    assert bytes(loaded["wave"]["data_units"]) == b""


def test_export_igor2_pxp_long_units_round_trip(tmp_path: Path) -> None:
    """Multi-array PXP export with long unit override writes extended units."""
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Light",
        stim_type=StimType.Excitation,
        values=np.ones((3, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "long.pxp"

    export_igor2_pxp(output_path, {"Light": stim}, unit_override="mW/mm^2")

    records, _ = packed.load(output_path)
    extended = bytes(records[0].wave["wave"]["data_units"]).decode("ascii")
    assert extended == "mW/mm^2"


def test_export_igor2_ibw_applies_unit_override_and_scaling(
    tmp_path: Path,
) -> None:
    binarywave = importlib.import_module("igor2.binarywave")
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "ibw_override.ibw"

    export_igor2_ibw(
        output_path,
        stim_array,
        wave_name="ibw_override",
        neuron_index=0,
        unit_override="mW/mm^2",
        scaling_factor=4.0,
    )

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([4.0, 8.0, 12.0], dtype=np.float32),
        decimal=5,
    )
    extended = bytes(loaded["wave"]["data_units"]).decode("ascii")
    assert extended == "mW/mm^2"


def test_export_igor2_ibw_rejects_non_finite_scaling(
    tmp_path: Path,
) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.ones((2, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    with pytest.raises(ValueError, match="scaling_factor"):
        export_igor2_ibw(
            tmp_path / "bad.ibw",
            stim_array,
            wave_name="bad",
            neuron_index=0,
            scaling_factor=float("nan"),
        )


def test_export_igor2_csv_applies_scaling(tmp_path: Path) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "scaled.csv"

    export_igor2_csv(
        output_path,
        stim_array,
        experiment_name="scaled_example",
        neuron_index=0,
        scaling_factor=2.5,
    )

    assert output_path.read_text(encoding="utf-8") == "scaled_example\n2.5\n5.0\n7.5\n"


def test_all_igor_formats_negate_inhibitory_arrays_alike(tmp_path: Path) -> None:
    """.pxp negated Inhibition arrays but .ibw/.csv did not, so the same array
    reached the rig with opposite sign depending on the format."""
    binarywave = importlib.import_module("igor2.binarywave")
    packed = importlib.import_module("igor2.packed")
    stim = StimArray(
        name="Inh",
        stim_type=StimType.Inhibition,
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    expected = np.array([-2.0, -4.0, -6.0])

    csv_text = generate_igor2_csv_text(
        stim, experiment_name="e", neuron_index=0, scaling_factor=2.0
    )
    np.testing.assert_allclose([float(x) for x in csv_text.split()[1:]], expected)

    export_igor2_ibw(
        tmp_path / "inh.ibw", stim, wave_name="inh", neuron_index=0, scaling_factor=2.0
    )
    ibw = binarywave.load(tmp_path / "inh.ibw")["wave"]["wData"]
    np.testing.assert_allclose(ibw, expected)

    export_igor2_pxp(tmp_path / "inh.pxp", {"Inh": stim}, scaling_factor=2.0)
    records, _ = packed.load(tmp_path / "inh.pxp")
    np.testing.assert_allclose(records[0].wave["wave"]["wData"], expected)
