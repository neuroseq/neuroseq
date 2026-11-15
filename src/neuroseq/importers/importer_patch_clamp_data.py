from __future__ import annotations

import csv
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import brian2.numpy_ as np

from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_timing import resample_zero_order_hold
from neuroseq.stimulation.stim_timing import zero_order_hold_sample_count

if TYPE_CHECKING:
    from neuroseq.data import TraceData

logger = logging.getLogger(__name__)

NEUROSEQ_IGOR_TARGET_DT_SECONDS = 0.001
# File-dialog filter shared by the patch-clamp input and output importers; the
# combined first entry is the default, so every supported format is visible.
PATCH_CLAMP_FILE_FILTER = (
    "Patch-clamp files (*.pxp *.ibw *.csv *.txt);;Packed Experiment (*.pxp);;"
    "Binary Wave (*.ibw);;CSV files (*.csv);;Text files (*.txt);;All files (*)"
)
_DT_TOLERANCE_SECONDS = 1e-9


@dataclass
class IgorWave:
    name: str
    values: np.ndarray
    source_dt: float | None
    units: str | None
    axis_units: str | None
    source_format: str
    note: str | None = None
    n_sweeps: int = 1
    samples_per_sweep: int | None = None


@dataclass
class PreparedIgorWave:
    name: str
    values: np.ndarray
    source_dt: float
    target_dt: float
    units: str | None
    source_format: str
    resampled: bool
    original_sample_count: int
    n_sweeps: int = 1
    samples_per_sweep: int | None = None


@dataclass
class IgorImportReport:
    target_stim_name: str
    target_neuron_index: int
    source_samples: int
    imported_samples: int
    source_dt: float
    target_dt: float
    resampled: bool
    truncated_samples: int
    n_sweeps: int = 1
    samples_per_sweep: int | None = None


@dataclass
class IgorTraceImportReport:
    target_experiment_name: str
    target_trace_data_name: str
    target_neuron_id: int
    source_samples: int
    imported_samples: int
    source_dt: float
    target_dt: float
    resampled: bool
    truncated_samples: int
    n_sweeps: int = 1
    samples_per_sweep: int | None = None
    spikes_detected: int | None = None


@dataclass
class SkippedPxpRecord:
    """A record that could not be parsed while loading a .pxp file."""

    record_index: int
    record_type: str
    reason: str


@dataclass
class IgorPxpLoadReport:
    """Result of loading an Igor Packed Experiment file."""

    waves: dict[str, IgorWave]
    skipped: list[SkippedPxpRecord]


def _as_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", errors="ignore")
    if hasattr(value, "tobytes"):
        try:
            return value.tobytes()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        try:
            return bytes(int(item) for item in value)
        except Exception:
            parts = [_as_bytes(item) for item in value]
            nonempty_parts = [part for part in parts if part]
            if nonempty_parts:
                return b"".join(nonempty_parts)
            return None
    return str(value).encode("utf-8", errors="ignore")


def _decode_igor_text(value: object) -> str | None:
    raw_bytes = _as_bytes(value)
    if raw_bytes is None:
        return None
    raw_text = raw_bytes.split(b"\x00", 1)[0]
    try:
        text = raw_text.decode("utf-8").strip()
    except UnicodeDecodeError:
        # Igor stores text in a single-byte system encoding (Windows-1252 or
        # Mac Roman); both put the micro sign of "µs" at 0xB5, as Latin-1 does.
        text = raw_text.decode("latin-1").strip()
    return text or None


def _first_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _first_float(value[0])
    if hasattr(value, "shape"):
        array = np.asarray(value)
        if array.size == 0:
            return None
        return float(array.reshape(-1)[0])
    try:
        return float(str(value))
    except ValueError:
        return None


def _extract_wave_name(header: dict[str, object], fallback_name: str) -> str:
    return (
        _decode_igor_text(header.get("bname"))
        or _decode_igor_text(header.get("name"))
        or fallback_name
    )


# Seconds per x-axis unit for the time units Igor waves are commonly scaled in.
_TIME_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,  # micro sign
    "μs": 1e-6,  # Greek mu
    "ns": 1e-9,
}


def _extract_source_dt_seconds(
    header: dict[str, object], axis_units: str | None = None
) -> float | None:
    """The x-axis delta (``sfA``/``hsA``) converted to seconds.

    Igor stores the delta in the wave's own x units; a wave scaled in ms or µs
    would otherwise import 1000x/10^6x too slow. Unknown or missing units are
    taken as seconds.
    """
    if "sfA" in header:
        delta = _first_float(header.get("sfA"))
    elif "hsA" in header:
        delta = _first_float(header.get("hsA"))
    else:
        return None
    if delta is None:
        return None
    scale = _TIME_UNIT_SECONDS.get((axis_units or "").strip(), 1.0)
    return delta * scale


def _first_axis_units(dim_units: object) -> object | None:
    """Units of the first (x) dimension from a per-dimension units field."""
    if isinstance(dim_units, (list, tuple)):
        return dim_units[0] if len(dim_units) > 0 else None
    if isinstance(dim_units, np.ndarray) and dim_units.ndim >= 1:
        # igor2 returns v5 ``dimUnits`` as a (dims x chars) bytes array.
        return dim_units[0] if dim_units.shape[0] > 0 else None
    return None


def _extract_units(
    header: dict[str, object],
    wave: dict[str, object],
) -> tuple[str | None, str | None]:
    units = _decode_igor_text(wave.get("data_units"))
    if units is None:
        units = _decode_igor_text(header.get("dataUnits"))

    axis_units = _decode_igor_text(_first_axis_units(wave.get("dimension_units")))
    if axis_units is None:
        axis_units = _decode_igor_text(_first_axis_units(header.get("dimUnits")))
    if axis_units is None:
        axis_units = _decode_igor_text(header.get("xUnits"))
    return units, axis_units


def parse_igor_binary_wave_data(
    data: dict[str, object],
    *,
    fallback_name: str = "igor_wave",
) -> IgorWave:
    wave = data.get("wave", data)
    if not isinstance(wave, dict):
        raise ValueError("Invalid Igor Binary Wave data: missing wave payload")

    header = wave.get("wave_header")
    if not isinstance(header, dict):
        raise ValueError("Invalid Igor Binary Wave data: missing wave header")

    raw_values = np.asarray(wave.get("wData"), dtype=np.float32)
    squeezed = np.squeeze(raw_values)
    if squeezed.ndim == 1:
        values = squeezed.astype(np.float32, copy=False)
        n_sweeps = 1
        samples_per_sweep = int(values.size)
    elif squeezed.ndim == 2:
        # Igor stores multi-D waves column-major. Sweeps are along axis 1.
        # Concatenate sweep-by-sweep into a 1-D wave (sweep 0 all samples
        # first, then sweep 1, ...). The inter-sweep recording gaps from
        # the original SutterPatch protocol (see `note`'s SweepStartTimes
        # field) are NOT reconstructed - sweeps abut.
        # TODO: Parse the SutterPatch note text (ScaleFactor / ScaleOffset /
        # SweepStartTimes) and apply or surface it on import.
        samples_per_sweep = int(squeezed.shape[0])
        n_sweeps = int(squeezed.shape[1])
        values = squeezed.ravel(order="F").astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"Igor wave has {squeezed.ndim} dimensions after squeezing; "
            "only 1-D and 2-D (samples × sweeps) waves can be imported."
        )

    units, axis_units = _extract_units(header, wave)
    return IgorWave(
        name=_extract_wave_name(header, fallback_name=fallback_name),
        values=values.astype(np.float32, copy=False),
        source_dt=_extract_source_dt_seconds(header, axis_units),
        units=units,
        axis_units=axis_units,
        source_format="ibw",
        note=_decode_igor_text(wave.get("note")),
        n_sweeps=n_sweeps,
        samples_per_sweep=samples_per_sweep,
    )


def load_igor_ibw_file(path: str | Path) -> IgorWave:
    path = Path(path)
    try:
        binarywave = importlib.import_module("igor2.binarywave")
    except ImportError as exc:
        raise ImportError(
            "Reading Igor Binary Wave files requires the optional 'igor2' dependency."
        ) from exc

    wave_data = binarywave.load(path)
    if not isinstance(wave_data, dict):
        raise ValueError("Invalid Igor Binary Wave file contents")
    return parse_igor_binary_wave_data(wave_data, fallback_name=path.stem)


def is_pxp_file(path: str | Path) -> bool:
    """Return True if *path* has a .pxp extension (case-insensitive)."""
    return Path(path).suffix.lower() == ".pxp"


def load_igor_pxp_file(path: str | Path) -> IgorPxpLoadReport:
    """Load all 1-D waves from an Igor Packed Experiment (.pxp) file.

    Uses a tolerant record-by-record loader: a single malformed wave
    record is logged and skipped rather than aborting the whole load.

    Returns an :class:`IgorPxpLoadReport` whose ``waves`` dict maps
    slash-joined folder paths to :class:`IgorWave` objects, and whose
    ``skipped`` list enumerates records that could not be parsed.

    Raises ``ImportError`` if igor2 is not installed. Raises
    ``ValueError`` only if no importable 1-D waves survive.
    """
    path = Path(path)
    try:
        packed = importlib.import_module("igor2.packed")
        wave_record_module = importlib.import_module("igor2.record.wave")
        record_module = importlib.import_module("igor2.record")
        record_base = importlib.import_module("igor2.record.base")
        struct_module = importlib.import_module("igor2.struct")
        util_module = importlib.import_module("igor2.util")
    except ImportError as exc:
        raise ImportError(
            "Reading Igor Packed Experiment files requires the optional 'igor2' dependency."
        ) from exc

    WaveRecord = wave_record_module.WaveRecord
    RECORD_TYPE = record_module.RECORD_TYPE
    UnknownRecord = record_base.UnknownRecord
    Structure = struct_module.Structure
    Field = struct_module.Field
    byte_order = util_module.byte_order
    need_to_reorder_bytes = util_module.need_to_reorder_bytes

    def _make_header_struct(bo: str):
        header_struct = Structure(
            name="PackedFileRecordHeader",
            fields=[
                Field("H", "recordType"),
                Field("h", "version"),
                Field("l", "numDataBytes"),
            ],
            byte_order=bo,
        )
        header_struct.setup()
        return header_struct

    packed_record_type_mask = 0x7FFF

    records: list[object] = []
    skipped: list[SkippedPxpRecord] = []
    bo: str | None = None
    initial_bo = "="

    with path.open("rb") as fp:
        record_index = 0
        while True:
            header_struct = _make_header_struct(bo or initial_bo)
            header_bytes = fp.read(header_struct.size)
            if not header_bytes:
                break
            if len(header_bytes) < header_struct.size:
                logger.warning(
                    "Truncated record header in %s at byte offset %d; "
                    "stopping after %d records.",
                    path.name,
                    fp.tell() - len(header_bytes),
                    record_index,
                )
                break

            header = header_struct.unpack_from(header_bytes)
            if header["version"] and bo is None:
                reorder = need_to_reorder_bytes(header["version"])
                bo = initial_bo = byte_order(reorder)
                if reorder:
                    header_struct = _make_header_struct(bo)
                    header = header_struct.unpack_from(header_bytes)

            data = fp.read(header["numDataBytes"])
            if len(data) < header["numDataBytes"]:
                logger.warning(
                    "Truncated record #%d in %s (wanted %d bytes, got %d); "
                    "stopping the packed-record scan.",
                    record_index + 1,
                    path.name,
                    header["numDataBytes"],
                    len(data),
                )
                break

            record_index += 1
            record_type = RECORD_TYPE.get(
                header["recordType"] & packed_record_type_mask,
                UnknownRecord,
            )
            try:
                record_obj = record_type(header, data, byte_order=bo)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed Igor packed record #%d (%s) in %s: %s",
                    record_index,
                    record_type.__name__,
                    path.name,
                    exc,
                )
                skipped.append(
                    SkippedPxpRecord(
                        record_index=record_index,
                        record_type=record_type.__name__,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            records.append(record_obj)

    # Reuse igor2's filesystem builder; reimplementing its folder/name handling here
    # would duplicate library behavior and make upgrades more brittle.
    filesystem = packed._build_filesystem(records)

    waves: dict[str, IgorWave] = {}

    def _collect(dirpath: list[object], key: object, value: object) -> None:
        if not isinstance(value, WaveRecord):
            return
        parts = [
            seg.decode("utf-8", errors="ignore")
            if isinstance(seg, (bytes, bytearray))
            else str(seg)
            for seg in dirpath
        ]
        wave_name_str = (
            key.decode("utf-8", errors="ignore")
            if isinstance(key, (bytes, bytearray))
            else str(key)
        )
        parts.append(wave_name_str)
        path_key = "/".join(parts)

        try:
            igor_wave = parse_igor_binary_wave_data(
                value.wave,
                fallback_name=wave_name_str,
            )
        except ValueError as exc:
            logger.debug("Skipping wave %s: %s", path_key, exc)
            return

        waves[path_key] = igor_wave

    packed.walk(filesystem, _collect)

    if not waves:
        raise ValueError(
            f"No importable 1-D waves found in packed experiment file '{path.name}'"
        )

    return IgorPxpLoadReport(waves=waves, skipped=skipped)


def load_igor_csv_file(path: str | Path) -> IgorWave:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = [row for row in csv.reader(fp) if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError("File is empty")

    wave_name = rows[0][0].strip().strip('"')
    if not wave_name:
        raise ValueError("The first non-empty row must contain the Igor wave name")

    try:
        float(wave_name)
    except ValueError:
        sample_rows = rows[1:]
        first_sample_line = 2
    else:
        wave_name = path.stem
        sample_rows = rows
        first_sample_line = 1

    values: list[float] = []
    for line_number, row in enumerate(sample_rows, start=first_sample_line):
        if len(row) > 1 and any(cell.strip() for cell in row[1:]):
            raise ValueError(
                f"CSV row {line_number} contains multiple columns; expected one value per row"
            )
        try:
            values.append(float(row[0]))
        except ValueError as exc:
            raise ValueError(
                f"CSV row {line_number} does not contain a valid numeric sample"
            ) from exc

    if not values:
        raise ValueError("No numeric input samples found in CSV file")

    return IgorWave(
        name=wave_name,
        values=np.asarray(values, dtype=np.float32),
        source_dt=None,
        units=None,
        axis_units=None,
        source_format="csv",
        note=None,
    )


def load_csv_trace_table(path: str | Path) -> list[IgorWave]:
    """Load comma-, tab-, or space-separated output columns as traces."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        lines = [line.strip() for line in fp if line.strip()]
    if not lines:
        raise ValueError("File is empty")
    # Use a sample row so spaces in a single-column header remain intact.
    sample_line = lines[-1]
    if "," in sample_line:
        rows = list(csv.reader(lines))
    elif "\t" in sample_line:
        rows = list(csv.reader(lines, delimiter="\t"))
    elif len(sample_line.split()) > 1:
        rows = [line.split() for line in lines]
    else:
        rows = list(csv.reader(lines))
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError("Every CSV row must have the same number of columns")
    try:
        [float(cell) for cell in rows[0]]
    except ValueError:
        names = [cell.strip() for cell in rows.pop(0)]
    else:
        names = (
            [path.stem]
            if column_count == 1
            else [f"{path.stem}_{index + 1}" for index in range(column_count)]
        )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("CSV column names must be non-empty and unique")
    if not rows:
        raise ValueError("No numeric samples found in CSV file")
    try:
        values = np.asarray(rows, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("CSV samples must all be numeric") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("CSV samples must all be finite")
    return [
        IgorWave(
            name=name,
            values=values[:, index],
            source_dt=None,
            units=None,
            axis_units=None,
            source_format="csv",
            note=None,
        )
        for index, name in enumerate(names)
    ]


def load_igor_wave_file(path: str | Path) -> IgorWave:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return load_igor_csv_file(path)
    if suffix == ".ibw":
        return load_igor_ibw_file(path)
    if suffix == ".pxp":
        raise ValueError(
            "Igor Packed Experiment files (.pxp) contain multiple waves — "
            "use load_igor_pxp_file() instead."
        )
    raise ValueError(f"Unsupported Igor wave file format '{path.suffix}'")


def estimate_resampled_sample_count(
    sample_count: int,
    source_dt: float,
    *,
    target_dt: float = NEUROSEQ_IGOR_TARGET_DT_SECONDS,
) -> int:
    if sample_count <= 0:
        raise ValueError("Igor waves must contain at least one sample")
    if source_dt <= 0 or target_dt <= 0:
        raise ValueError("Sampling intervals must be positive")
    return max(
        1,
        zero_order_hold_sample_count(sample_count, float(source_dt), float(target_dt)),
    )


def resample_wave_zero_order_hold(
    values: np.ndarray,
    *,
    source_dt: float,
    target_dt: float = NEUROSEQ_IGOR_TARGET_DT_SECONDS,
) -> np.ndarray:
    source_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if source_values.size == 0:
        raise ValueError("Igor waves must contain at least one sample")
    if source_dt <= 0 or target_dt <= 0:
        raise ValueError("Sampling intervals must be positive")
    if abs(float(source_dt) - float(target_dt)) <= _DT_TOLERANCE_SECONDS:
        return source_values.copy()

    return resample_zero_order_hold(source_values, float(source_dt), float(target_dt))


def prepare_igor_wave_for_neuroseq(
    wave: IgorWave,
    *,
    source_dt_override: float | None = None,
    target_dt: float | None = None,
) -> PreparedIgorWave:
    """Prepare a wave, preserving its sample interval unless explicitly resampled."""
    source_dt = (
        float(source_dt_override) if source_dt_override is not None else wave.source_dt
    )
    if source_dt is None:
        raise ValueError("Source sample interval is required to import this Igor wave")
    if target_dt is None:
        target_dt = float(source_dt)

    resampled_values = resample_wave_zero_order_hold(
        wave.values,
        source_dt=float(source_dt),
        target_dt=float(target_dt),
    )
    return PreparedIgorWave(
        name=wave.name,
        values=resampled_values,
        source_dt=float(source_dt),
        target_dt=float(target_dt),
        units=wave.units,
        source_format=wave.source_format,
        resampled=abs(float(source_dt) - float(target_dt)) > _DT_TOLERANCE_SECONDS,
        original_sample_count=int(np.asarray(wave.values).size),
        n_sweeps=wave.n_sweeps,
        samples_per_sweep=wave.samples_per_sweep,
    )


def create_stim_array_from_igor_wave(
    prepared_wave: PreparedIgorWave,
    *,
    stim_name: str,
    stim_type: StimType = StimType.External,
    units: str | None = None,
) -> StimArray:
    return StimArray(
        name=stim_name,
        stim_type=stim_type,
        values=prepared_wave.values[:, np.newaxis].astype(np.float32, copy=False),
        units=(prepared_wave.units or "") if units is None else units,
        dt=prepared_wave.target_dt,
    )


def import_igor_wave_into_stim_array(
    stim_array: StimArray,
    prepared_wave: PreparedIgorWave,
    *,
    neuron_index: int = 0,
    replace_existing: bool = True,
    extend_to_fit: bool = True,
) -> IgorImportReport:
    if stim_array.values.ndim != 2:
        raise ValueError("Target Input Array must be 2-dimensional")
    if (
        abs(float(stim_array.dt) - float(prepared_wave.target_dt))
        > _DT_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "Target Input Array dt must match the imported wave target dt "
            f"({float(stim_array.dt):.9f}s != {float(prepared_wave.target_dt):.9f}s)"
        )

    n_existing_timepoints, n_existing_neurons = stim_array.values.shape
    if neuron_index < 0 or neuron_index >= n_existing_neurons:
        raise ValueError(
            f"Invalid neuron index {neuron_index} for {n_existing_neurons} neurons"
        )

    target_timepoints = n_existing_timepoints
    if extend_to_fit:
        target_timepoints = max(n_existing_timepoints, prepared_wave.values.shape[0])

    target_values = np.zeros(
        (target_timepoints, n_existing_neurons), dtype=stim_array.values.dtype
    )
    target_values[:n_existing_timepoints, :] = stim_array.values
    if replace_existing:
        target_values[:, neuron_index] = 0

    imported_values = prepared_wave.values.astype(target_values.dtype, copy=False)
    imported_samples = min(imported_values.shape[0], target_timepoints)
    if replace_existing:
        target_values[:imported_samples, neuron_index] = imported_values[
            :imported_samples
        ]
    else:
        target_values[:imported_samples, neuron_index] += imported_values[
            :imported_samples
        ]

    stim_array.values = target_values
    logger.info(
        "Imported Igor wave '%s' into '%s' neuron %d: %d -> %d samples",
        prepared_wave.name,
        stim_array.name,
        neuron_index,
        prepared_wave.original_sample_count,
        imported_samples,
    )
    return IgorImportReport(
        target_stim_name=stim_array.name,
        target_neuron_index=neuron_index,
        source_samples=prepared_wave.original_sample_count,
        imported_samples=imported_samples,
        source_dt=prepared_wave.source_dt,
        target_dt=prepared_wave.target_dt,
        resampled=prepared_wave.resampled,
        truncated_samples=max(0, prepared_wave.values.shape[0] - target_timepoints),
        n_sweeps=prepared_wave.n_sweeps,
        samples_per_sweep=prepared_wave.samples_per_sweep,
    )


def create_trace_data_from_igor_wave(
    prepared_wave: PreparedIgorWave,
    *,
    trace_name: str,
    neuron_id: int = 0,
    units: str | None = None,
) -> TraceData:
    """Build a single-neuron TraceData from a prepared Igor wave."""
    from neuroseq.data import TraceData

    trace_values = prepared_wave.values.reshape(1, -1).astype(np.float32, copy=False)
    return TraceData(
        name=trace_name,
        traces=trace_values,
        indices=np.array([int(neuron_id)], dtype=np.uint32),
        units=(prepared_wave.units or "") if units is None else units,
        dt=float(prepared_wave.target_dt),
    )


def import_igor_wave_into_trace_data(
    trace_data: TraceData,
    prepared_wave: PreparedIgorWave,
    *,
    neuron_id: int,
    replace_existing: bool = True,
    extend_to_fit: bool = True,
) -> IgorTraceImportReport:
    """Append or replace a single neuron row in an existing TraceData."""
    if trace_data.traces.ndim != 2:
        raise ValueError("Target TraceData must be 2-dimensional")
    if (
        abs(float(trace_data.dt) - float(prepared_wave.target_dt))
        > _DT_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "Target TraceData dt must match the imported wave target dt "
            f"({float(trace_data.dt):.9f}s != {float(prepared_wave.target_dt):.9f}s)"
        )

    imported_values = prepared_wave.values.astype(np.float32, copy=False)
    n_rows_existing, n_time_existing = trace_data.traces.shape
    n_time_imported = int(imported_values.size)

    indices_list = [int(x) for x in trace_data.indices.tolist()]
    row_index = (
        indices_list.index(int(neuron_id)) if int(neuron_id) in indices_list else -1
    )

    if row_index >= 0 and not replace_existing:
        raise ValueError(
            f"Neuron ID {neuron_id} already exists in trace data '{trace_data.name}'; "
            "enable 'Replace existing' to overwrite it."
        )

    if extend_to_fit:
        n_time_target = max(n_time_existing, n_time_imported)
    else:
        n_time_target = n_time_existing

    if n_time_target != n_time_existing:
        padded = np.zeros(
            (n_rows_existing, n_time_target), dtype=trace_data.traces.dtype
        )
        padded[:, :n_time_existing] = trace_data.traces
        trace_data.traces = padded

    truncated = max(0, n_time_imported - n_time_target)
    written_len = min(n_time_imported, n_time_target)

    new_row = np.zeros((1, n_time_target), dtype=trace_data.traces.dtype)
    new_row[0, :written_len] = imported_values[:written_len]

    if row_index >= 0:
        # Build a new buffer: the project's undo history may retain the old one.
        traces = trace_data.traces.copy()
        traces[row_index] = new_row[0]
        trace_data.traces = traces
    else:
        trace_data.traces = np.concatenate([trace_data.traces, new_row], axis=0)
        trace_data.indices = np.concatenate(
            [trace_data.indices, np.array([int(neuron_id)], dtype=np.uint32)]
        )
        if trace_data.raw_cellpose_ids is not None:
            # A patch-clamped neuron has no Cellpose ROI, so the per-row raw ids
            # can no longer cover every row; drop them (Project's Cellpose remap
            # then leaves this dataset alone) rather than keep a misaligned array.
            logger.info(
                "Trace data '%s' now mixes patch-clamp rows with Cellpose rows; "
                "it will no longer follow Cellpose mapping changes",
                trace_data.name,
            )
            trace_data.raw_cellpose_ids = None

    logger.info(
        "Imported Igor wave '%s' into TraceData '%s' neuron %d: %d -> %d samples",
        prepared_wave.name,
        trace_data.name,
        int(neuron_id),
        prepared_wave.original_sample_count,
        written_len,
    )
    return IgorTraceImportReport(
        target_experiment_name="",
        target_trace_data_name=trace_data.name,
        target_neuron_id=int(neuron_id),
        source_samples=prepared_wave.original_sample_count,
        imported_samples=written_len,
        source_dt=float(prepared_wave.source_dt),
        target_dt=float(prepared_wave.target_dt),
        resampled=prepared_wave.resampled,
        truncated_samples=truncated,
        n_sweeps=prepared_wave.n_sweeps,
        samples_per_sweep=prepared_wave.samples_per_sweep,
    )
