from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import brian2.numpy_ as np
import pytest

from neuroseq.data import TraceData
from neuroseq.importers.importer_patch_clamp_data import IgorPxpLoadReport
from neuroseq.importers.importer_patch_clamp_data import IgorWave
from neuroseq.importers.importer_patch_clamp_data import (
    create_stim_array_from_igor_wave,
)
from neuroseq.importers.importer_patch_clamp_data import (
    create_trace_data_from_igor_wave,
)
from neuroseq.importers.importer_patch_clamp_data import (
    import_igor_wave_into_stim_array,
)
from neuroseq.importers.importer_patch_clamp_data import (
    import_igor_wave_into_trace_data,
)
from neuroseq.importers.importer_patch_clamp_data import is_pxp_file
from neuroseq.importers.importer_patch_clamp_data import load_csv_trace_table
from neuroseq.importers.importer_patch_clamp_data import load_igor_csv_file
from neuroseq.importers.importer_patch_clamp_data import load_igor_ibw_file
from neuroseq.importers.importer_patch_clamp_data import load_igor_pxp_file
from neuroseq.importers.importer_patch_clamp_data import load_igor_wave_file
from neuroseq.importers.importer_patch_clamp_data import parse_igor_binary_wave_data
from neuroseq.importers.importer_patch_clamp_data import prepare_igor_wave_for_neuroseq
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType

REAL_BROKEN_PXP = Path(
    "/mnt/live3/svintila/code_scraps/igor2_files/test_files/"
    "pxp_cristina/drive-download-20260416T122331Z-3-001/251118_0.pxp"
)
REAL_WORKING_PXP = Path(
    "/mnt/live3/svintila/code_scraps/igor2_files/test_files/260316_0.pxp"
)
REAL_SUTTERPATCH_IBW = Path(
    "/mnt/live3/svintila/random/igor2_files/test_files/R5_S1_IVstep.ibw"
)


def _fake_ibw_payload(
    wdata: np.ndarray,
    *,
    name: str = "wave",
    dt: float = 0.001,
    units: str = "pA",
    axis_units: str = "s",
) -> dict[str, object]:
    return {
        "wave": {
            "wData": wdata,
            "note": b"",
            "wave_header": {
                "bname": name.encode("ascii") + b"\x00",
                "sfA": np.array([dt], dtype=np.float64),
                "dataUnits": units.encode("ascii") + b"\x00",
                "dimUnits": [axis_units.encode("ascii") + b"\x00"],
            },
        }
    }


def test_load_igor_csv_file_parses_simple_wave(tmp_path: Path) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("example_wave\r\n1.0\r\n2.5\r\n-3.0\r\n", encoding="utf-8")

    wave = load_igor_csv_file(csv_path)

    assert wave.name == "example_wave"
    assert wave.source_dt is None
    assert wave.units is None
    np.testing.assert_allclose(wave.values, np.array([1.0, 2.5, -3.0]))


@pytest.mark.parametrize("delimiter", [",", " ", "   ", "\t"])
@pytest.mark.parametrize("header", ["", "cell_a,cell_b\n"])
def test_csv_trace_table_columns(tmp_path: Path, header: str, delimiter: str) -> None:
    path = tmp_path / "recording.csv"
    path.write_text(
        (header + "1,10\n2,20\n3,30\n").replace(",", delimiter), encoding="utf-8"
    )
    waves = load_csv_trace_table(path)
    assert [wave.name for wave in waves] == (
        ["cell_a", "cell_b"] if header else ["recording_1", "recording_2"]
    )
    np.testing.assert_array_equal(waves[0].values, [1, 2, 3])
    np.testing.assert_array_equal(waves[1].values, [10, 20, 30])


@pytest.mark.parametrize("padding", ["", "             ", "\t\t", " \t "])
def test_csv_trace_table_space_separated_voltage_samples(
    tmp_path: Path, padding: str
) -> None:
    path = tmp_path / "voltage.csv"
    samples = "-65.737501 -66.487499 -66.343747 -66.837497 -66.725001 -68.60625 -69.231249 -68.850003 -67.225002 -68.000004 -69.212496 -67.924999 -68.081252"
    path.write_text(samples + padding + "\n" + samples + "\n", encoding="utf-8")
    waves = load_csv_trace_table(path)
    assert len(waves) == 13
    for wave, sample in zip(waves, samples.split(), strict=True):
        np.testing.assert_allclose(wave.values, [float(sample), float(sample)])


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "a,b\n",
        "a,a\n1,2\n",
        "a,\n1,2\n",
        "a,b\n1\n",
        "a,b\n1,no\n",
        "a,b\n1,nan\n",
        "a,b\n1,inf\n",
    ],
)
def test_csv_trace_table_rejects_invalid_data(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv_trace_table(path)


def test_parse_binary_wave_accepts_two_d_and_concatenates() -> None:
    raw = np.array(
        [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]],
        dtype=np.float32,
    )
    wave = parse_igor_binary_wave_data(_fake_ibw_payload(raw))

    np.testing.assert_allclose(
        wave.values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )
    assert wave.n_sweeps == 3
    assert wave.samples_per_sweep == 4


def test_parse_binary_wave_rejects_three_d() -> None:
    raw = np.zeros((4, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="3 dimensions"):
        parse_igor_binary_wave_data(_fake_ibw_payload(raw))


def test_parse_binary_wave_keeps_one_d_unchanged() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wave = parse_igor_binary_wave_data(_fake_ibw_payload(raw))

    np.testing.assert_allclose(wave.values, raw)
    assert wave.n_sweeps == 1
    assert wave.samples_per_sweep == 4


@pytest.mark.parametrize(
    ("axis_units", "expected_dt"),
    [("s", 0.5), ("ms", 0.5e-3), ("us", 0.5e-6), ("\u00b5s", 0.5e-6), ("", 0.5)],
)
def test_parse_binary_wave_scales_dt_by_x_units(
    axis_units: str, expected_dt: float
) -> None:
    payload = _fake_ibw_payload(np.zeros(4, dtype=np.float32), dt=0.5)
    wave_header = payload["wave"]["wave_header"]  # type: ignore[index]
    # Igor writes units in a single-byte encoding (micro sign = 0xB5).
    wave_header["dimUnits"] = [axis_units.encode("latin-1") + b"\x00"]

    wave = parse_igor_binary_wave_data(payload)

    assert wave.source_dt == pytest.approx(expected_dt)


def test_parse_binary_wave_reads_x_units_from_igor2_dim_units_array() -> None:
    """igor2 returns v5 ``dimUnits`` as a (4 x 4) bytes array, not a list."""
    payload = _fake_ibw_payload(np.zeros(4, dtype=np.float32), dt=2.0)
    dim_units = np.zeros((4, 4), dtype="|S1")
    dim_units[0, :2] = [b"m", b"s"]
    payload["wave"]["wave_header"]["dimUnits"] = dim_units  # type: ignore[index]

    wave = parse_igor_binary_wave_data(payload)

    assert wave.axis_units == "ms"
    assert wave.source_dt == pytest.approx(2e-3)


@pytest.mark.skipif(not REAL_SUTTERPATCH_IBW.exists(), reason="real IBW not present")
def test_real_sutterpatch_file_imports() -> None:
    pytest.importorskip("igor2")
    wave = load_igor_ibw_file(REAL_SUTTERPATCH_IBW)

    assert wave.n_sweeps == 7
    assert wave.samples_per_sweep == 30000
    assert wave.source_dt == pytest.approx(5e-5)


def test_prepared_wave_carries_sweep_metadata() -> None:
    wave = IgorWave(
        name="test_wave",
        values=np.arange(12, dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
        n_sweeps=3,
        samples_per_sweep=4,
    )

    prepared = prepare_igor_wave_for_neuroseq(wave)

    assert prepared.n_sweeps == 3
    assert prepared.samples_per_sweep == 4


@pytest.mark.parametrize("source_dt", [0.0005, 0.002, 1 / 30000])
def test_prepare_igor_wave_for_neuroseq_preserves_source_timing(
    source_dt: float,
) -> None:
    wave = IgorWave(
        name="test_wave",
        values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        source_dt=source_dt,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )

    prepared = prepare_igor_wave_for_neuroseq(wave)

    assert prepared.resampled is False
    assert prepared.target_dt == source_dt
    np.testing.assert_array_equal(prepared.values, wave.values)


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


def test_create_trace_data_from_igor_wave_produces_single_neuron_row() -> None:
    wave = IgorWave(
        name="test_wave",
        values=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )

    prepared = prepare_igor_wave_for_neuroseq(wave)
    trace_data = create_trace_data_from_igor_wave(
        prepared,
        trace_name="imported_trace",
        neuron_id=7,
    )

    assert isinstance(trace_data, TraceData)
    assert trace_data.name == "imported_trace"
    assert trace_data.traces.shape == (1, 3)
    np.testing.assert_allclose(trace_data.traces[0], np.array([0.5, 1.5, 2.5]))
    np.testing.assert_array_equal(trace_data.indices, np.array([7], dtype=np.uint32))
    assert trace_data.units == "mV"
    assert trace_data.dt == pytest.approx(0.001)


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


def test_import_igor_wave_into_stim_array_allows_matching_nondefault_dt() -> None:
    target_stim = StimArray(
        name="stimulus",
        stim_type=StimType.External,
        values=np.zeros((2, 1), dtype=np.float32),
        units="pA",
        dt=0.0005,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        source_dt=0.0005,
        units="pA",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave, target_dt=0.0005)

    report = import_igor_wave_into_stim_array(
        target_stim,
        prepared,
        neuron_index=0,
        replace_existing=True,
        extend_to_fit=True,
    )

    assert report.target_dt == pytest.approx(0.0005)
    assert target_stim.dt == pytest.approx(0.0005)
    assert target_stim.values.shape == (3, 1)
    np.testing.assert_allclose(target_stim.values[:, 0], np.array([5.0, 6.0, 7.0]))


def test_import_igor_wave_into_trace_data_appends_new_neuron() -> None:
    trace_data = TraceData(
        name="trace_data",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([5.0, 6.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave)

    report = import_igor_wave_into_trace_data(
        trace_data,
        prepared,
        neuron_id=5,
        replace_existing=True,
        extend_to_fit=True,
    )

    assert report.target_trace_data_name == "trace_data"
    assert report.target_neuron_id == 5
    assert report.imported_samples == 2
    assert trace_data.traces.shape == (2, 2)
    np.testing.assert_array_equal(trace_data.indices, np.array([0, 5], dtype=np.uint32))
    np.testing.assert_allclose(trace_data.traces[1], np.array([5.0, 6.0]))


def test_import_igor_wave_into_trace_data_keeps_raw_cellpose_ids_aligned() -> None:
    """Appending a patch-clamp row to Cellpose-derived traces must not leave
    raw_cellpose_ids one row short (a later Cellpose remap would misalign)."""
    trace_data = TraceData(
        "s2p",
        np.zeros((3, 10), dtype=np.float32),
        np.arange(3, dtype=np.uint32),
        "u",
        0.1,
        np.arange(3, dtype=np.uint32),
    )
    prepared = prepare_igor_wave_for_neuroseq(
        IgorWave("w", np.ones(10, dtype=np.float32), 0.1, None, None, "csv")
    )

    import_igor_wave_into_trace_data(trace_data, prepared, neuron_id=7)

    np.testing.assert_array_equal(trace_data.indices, [0, 1, 2, 7])
    assert (
        trace_data.raw_cellpose_ids is None
        or trace_data.raw_cellpose_ids.shape[0] == trace_data.traces.shape[0]
    )


def test_import_igor_wave_into_trace_data_replaces_existing_neuron_when_allowed() -> (
    None
):
    trace_data = TraceData(
        name="trace_data",
        traces=np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32),
        indices=np.array([0, 5], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([7.0, 8.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave)

    report = import_igor_wave_into_trace_data(
        trace_data,
        prepared,
        neuron_id=5,
        replace_existing=True,
        extend_to_fit=True,
    )

    assert report.imported_samples == 2
    assert trace_data.traces.shape == (2, 2)
    np.testing.assert_allclose(trace_data.traces[1], np.array([7.0, 8.0]))


def test_import_igor_wave_into_trace_data_rejects_duplicate_neuron_without_replace() -> (
    None
):
    trace_data = TraceData(
        name="trace_data",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([5], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([7.0, 8.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave)

    with pytest.raises(ValueError, match="already exists"):
        import_igor_wave_into_trace_data(
            trace_data,
            prepared,
            neuron_id=5,
            replace_existing=False,
            extend_to_fit=True,
        )


def test_import_igor_wave_into_trace_data_extends_time_axis() -> None:
    trace_data = TraceData(
        name="trace_data",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave)

    report = import_igor_wave_into_trace_data(
        trace_data,
        prepared,
        neuron_id=3,
        replace_existing=True,
        extend_to_fit=True,
    )

    assert report.imported_samples == 4
    assert trace_data.traces.shape == (2, 4)
    np.testing.assert_allclose(trace_data.traces[0], np.array([1.0, 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(trace_data.traces[1], np.array([5.0, 6.0, 7.0, 8.0]))


def test_import_igor_wave_into_trace_data_rejects_mismatched_dt() -> None:
    trace_data = TraceData(
        name="trace_data",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    wave = IgorWave(
        name="test_wave",
        values=np.array([5.0, 6.0], dtype=np.float32),
        source_dt=0.0005,
        units="mV",
        axis_units="s",
        source_format="csv",
    )
    prepared = prepare_igor_wave_for_neuroseq(wave, target_dt=0.0005)

    with pytest.raises(ValueError, match="Target TraceData dt must match"):
        import_igor_wave_into_trace_data(
            trace_data,
            prepared,
            neuron_id=2,
            replace_existing=True,
            extend_to_fit=True,
        )


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


class _FakeRecord:
    def __init__(
        self,
        header: dict[str, object],
        data: bytes,
        byte_order: str | None = None,
    ) -> None:
        self.header = header
        self.data = bytes(data)
        self.byte_order = byte_order


class UnknownRecord(_FakeRecord):
    pass


class UnusedRecord(_FakeRecord):
    pass


class FolderStartRecord(_FakeRecord):
    def __init__(
        self,
        header: dict[str, object],
        data: bytes,
        byte_order: str | None = None,
    ) -> None:
        super().__init__(header, data, byte_order=byte_order)
        self.null_terminated_text = self.data.split(b"\x00", 1)[0]


class FolderEndRecord(_FakeRecord):
    pass


class _FakeField:
    def __init__(self, format: str, name: str) -> None:
        self.format = format
        self.name = name


class _FakeStructure:
    def __init__(
        self,
        name: str,
        fields: list[_FakeField],
        byte_order: str = "@",
    ) -> None:
        self.name = name
        self.fields = fields
        self.byte_order = byte_order
        self.setup()

    def setup(self) -> None:
        self._struct = struct.Struct(
            self.byte_order + "".join(field.format for field in self.fields)
        )
        self.size = self._struct.size

    def unpack_from(self, buffer: bytes, offset: int = 0) -> dict[str, object]:
        values = self._struct.unpack_from(buffer, offset)
        return {
            field.name: value for field, value in zip(self.fields, values, strict=False)
        }


def _fake_byte_order(need_to_reorder_bytes: bool) -> str:
    little_endian = sys.byteorder == "little"
    if need_to_reorder_bytes:
        little_endian = not little_endian
    return "<" if little_endian else ">"


def _fake_need_to_reorder_bytes(version: int) -> bool:
    return version & 0xFF == 0


class _FakePackedModule:
    def __init__(self, wave_record_class: type[object]) -> None:
        self._wave_record_class = wave_record_class

    def _build_filesystem(self, records: list[object]) -> dict[object, Any]:
        root: dict[object, Any] = {}
        filesystem: dict[object, Any] = {"root": root}
        dir_stack: list[tuple[object, dict[object, Any]]] = [("root", root)]
        for record in records:
            cwd = dir_stack[-1][1]
            if isinstance(record, FolderStartRecord):
                name = record.null_terminated_text
                child: dict[object, Any] = {}
                cwd[name] = child
                dir_stack.append((name, child))
            elif isinstance(record, FolderEndRecord):
                if len(dir_stack) > 1:
                    dir_stack.pop()
            elif isinstance(record, self._wave_record_class):
                wave_record = cast(Any, record)
                filename = wave_record.wave["wave"]["wave_header"]["bname"].split(
                    b"\x00", 1
                )[0]
                cwd[filename] = record
        return filesystem

    def walk(
        self,
        filesystem: dict[object, object],
        callback,
        dirpath: list[object] | None = None,
    ) -> None:
        if dirpath is None:
            dirpath = []
        for key, value in sorted(
            ((_as_bytes(key), value) for key, value in filesystem.items()),
            key=lambda item: item[0],
        ):
            callback(dirpath, key, value)
            if isinstance(value, dict):
                self.walk(value, callback, dirpath + [key])


def _make_fake_wave_record_class(
    payload_map: dict[bytes, dict[str, object] | Exception],
) -> type[object]:
    class WaveRecord(_FakeRecord):
        def __init__(
            self,
            header: dict[str, object],
            data: bytes,
            byte_order: str | None = None,
        ) -> None:
            super().__init__(header, data, byte_order=byte_order)
            payload = payload_map[bytes(self.data)]
            if isinstance(payload, Exception):
                raise payload
            self.wave = payload

    return WaveRecord


def _as_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode("utf-8", errors="ignore")


def _pack_fake_pxp_record(
    record_type: int,
    data: bytes,
    *,
    version: int = 1,
) -> bytes:
    return struct.pack("=Hhl", record_type, version, len(data)) + data


def _install_fake_pxp_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload_map: dict[bytes, dict[str, object] | Exception],
) -> None:
    wave_record_class = _make_fake_wave_record_class(payload_map)
    fake_packed = _FakePackedModule(wave_record_class)
    fake_modules = {
        "igor2.packed": fake_packed,
        "igor2.record.wave": SimpleNamespace(WaveRecord=wave_record_class),
        "igor2.record": SimpleNamespace(
            RECORD_TYPE={
                1: FolderStartRecord,
                2: FolderEndRecord,
                3: wave_record_class,
            }
        ),
        "igor2.record.base": SimpleNamespace(
            UnknownRecord=UnknownRecord,
            UnusedRecord=UnusedRecord,
        ),
        "igor2.record.folder": SimpleNamespace(
            FolderStartRecord=FolderStartRecord,
            FolderEndRecord=FolderEndRecord,
        ),
        "igor2.struct": SimpleNamespace(Structure=_FakeStructure, Field=_FakeField),
        "igor2.util": SimpleNamespace(
            byte_order=_fake_byte_order,
            need_to_reorder_bytes=_fake_need_to_reorder_bytes,
        ),
    }

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name in fake_modules:
            return fake_modules[name]
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)


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

    _install_fake_pxp_modules(monkeypatch, payload_map={b"wave-ok": wave_dict})
    pxp_path.write_bytes(_pack_fake_pxp_record(3, b"wave-ok"))

    report = load_igor_pxp_file(pxp_path)
    waves = report.waves

    assert report.skipped == []
    assert waves.keys() == {"root/MyWave"}
    wave = waves["root/MyWave"]
    assert wave.name == "MyWave"
    assert wave.source_dt == pytest.approx(0.001)
    assert wave.units == "pA"
    np.testing.assert_allclose(wave.values, np.array([1.0, 2.0, 3.0]))


def test_load_igor_pxp_file_imports_two_d_waves(
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
    wave_nested_dict = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"NestedWave\x00",
                "sfA": [0.001, 0.0, 0.0, 0.0],
                "dataUnits": b"pA\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.array([2.5, 3.5, 4.5], dtype=np.float32),
            "note": b"",
        },
    }

    _install_fake_pxp_modules(
        monkeypatch,
        payload_map={
            b"wave-2d": wave_2d_dict,
            b"wave-1d": wave_1d_dict,
            b"wave-nested": wave_nested_dict,
        },
    )
    pxp_path.write_bytes(
        _pack_fake_pxp_record(3, b"wave-2d")
        + _pack_fake_pxp_record(3, b"wave-1d")
        + _pack_fake_pxp_record(1, b"subfolder\x00")
        + _pack_fake_pxp_record(3, b"wave-nested")
        + _pack_fake_pxp_record(2, b"")
    )

    report = load_igor_pxp_file(pxp_path)
    waves = report.waves

    assert report.skipped == []
    assert waves.keys() == {
        "root/OneDWave",
        "root/TwoDWave",
        "root/subfolder/NestedWave",
    }
    np.testing.assert_allclose(waves["root/OneDWave"].values, np.array([0.5, 1.5]))
    two_d_wave = waves["root/TwoDWave"]
    assert two_d_wave.n_sweeps == 4
    assert two_d_wave.samples_per_sweep == 3
    np.testing.assert_allclose(two_d_wave.values, np.ones(12, dtype=np.float32))
    np.testing.assert_allclose(
        waves["root/subfolder/NestedWave"].values,
        np.array([2.5, 3.5, 4.5]),
    )


def test_load_igor_pxp_file_raises_if_no_waves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pxp_path = tmp_path / "empty.pxp"
    pxp_path.write_bytes(b"")
    _install_fake_pxp_modules(monkeypatch, payload_map={})

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
        if name.startswith("igor2."):
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


def test_load_igor_wave_file_rejects_pxp_dispatch() -> None:
    with pytest.raises(ValueError, match="load_igor_pxp_file"):
        load_igor_wave_file("packed_experiment.pxp")


def test_load_igor_pxp_file_skips_malformed_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single malformed record must not abort the whole load."""
    pxp_path = tmp_path / "partially_broken.pxp"

    good_wave_dict = {
        "version": 5,
        "wave": {
            "wave_header": {
                "bname": b"GoodWave\x00",
                "sfA": [0.001, 0.0, 0.0, 0.0],
                "dataUnits": b"pA\x00",
                "dimUnits": [b"s\x00", b"", b"", b""],
            },
            "wData": np.array([2.0, 4.0, 6.0], dtype=np.float32),
            "note": b"",
        },
    }
    _install_fake_pxp_modules(
        monkeypatch,
        payload_map={
            b"broken-wave": TypeError("buffer is too small for requested array"),
            b"good-wave": good_wave_dict,
        },
    )
    pxp_path.write_bytes(
        _pack_fake_pxp_record(1, b"session\x00")
        + _pack_fake_pxp_record(3, b"broken-wave")
        + _pack_fake_pxp_record(3, b"good-wave")
        + _pack_fake_pxp_record(2, b"")
    )

    report = load_igor_pxp_file(pxp_path)

    assert isinstance(report, IgorPxpLoadReport)
    assert len(report.waves) == 1
    assert "root/session/GoodWave" in report.waves
    assert len(report.skipped) == 1
    assert report.skipped[0].record_type == "WaveRecord"
    assert "buffer" in report.skipped[0].reason.lower()


@pytest.mark.skipif(
    not REAL_BROKEN_PXP.exists(),
    reason="Real-world broken .pxp test file not available on this host.",
)
def test_load_igor_pxp_file_recovers_from_real_world_broken_record() -> None:
    pytest.importorskip("igor2")

    report = load_igor_pxp_file(REAL_BROKEN_PXP)

    assert len(report.waves) > 50, (
        "Expected >50 usable waves after skipping the broken record; "
        f"got {len(report.waves)}. Skipped list: {report.skipped!r}"
    )
    assert any(s.record_type == "WaveRecord" for s in report.skipped), (
        f"Expected at least one skipped WaveRecord. skipped={report.skipped!r}"
    )


@pytest.mark.skipif(
    not REAL_WORKING_PXP.exists(),
    reason="Regression .pxp test file not available on this host.",
)
def test_load_igor_pxp_file_keeps_working_file_strictly_clean() -> None:
    pytest.importorskip("igor2")

    report = load_igor_pxp_file(REAL_WORKING_PXP)

    assert report.waves
    assert report.skipped == []


def test_wave_resampling_uses_exact_zero_order_hold_boundaries() -> None:
    from neuroseq.importers.importer_patch_clamp_data import (
        estimate_resampled_sample_count,
    )
    from neuroseq.importers.importer_patch_clamp_data import (
        resample_wave_zero_order_hold,
    )

    assert estimate_resampled_sample_count(950, 0.00002, target_dt=0.001) == 19
    values = np.arange(200, dtype=np.float32)
    resampled = resample_wave_zero_order_hold(values, source_dt=0.0001, target_dt=0.001)
    assert resampled.shape == (20,)
    assert resampled.dtype == np.float32
    assert resampled[11] == 110
