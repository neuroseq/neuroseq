from __future__ import annotations

import csv
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

import brian2.numpy_ as np

from neuroseq.stim import StimArray
from neuroseq.stim import StimType

logger = logging.getLogger(__name__)

NEUROSEQ_IGOR_TARGET_DT_SECONDS = 0.001
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
    text = raw_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
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


def _extract_source_dt_seconds(header: dict[str, object]) -> float | None:
    if "sfA" in header:
        return _first_float(header.get("sfA"))
    if "hsA" in header:
        return _first_float(header.get("hsA"))
    return None


def _extract_units(
    header: dict[str, object],
    wave: dict[str, object],
) -> tuple[str | None, str | None]:
    units = _decode_igor_text(wave.get("data_units"))
    if units is None:
        units = _decode_igor_text(header.get("dataUnits"))

    axis_units = None
    dimension_units = wave.get("dimension_units")
    if isinstance(dimension_units, (list, tuple)) and len(dimension_units) > 0:
        axis_units = _decode_igor_text(dimension_units[0])
    if axis_units is None:
        dim_units = header.get("dimUnits")
        if isinstance(dim_units, (list, tuple)) and len(dim_units) > 0:
            axis_units = _decode_igor_text(dim_units[0])
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
    values = np.squeeze(raw_values)
    if values.ndim != 1:
        raise ValueError(
            "Only 1-dimensional Igor Binary Waves can be imported into NeuroSeq"
        )

    units, axis_units = _extract_units(header, wave)
    return IgorWave(
        name=_extract_wave_name(header, fallback_name=fallback_name),
        values=values.astype(np.float32, copy=False),
        source_dt=_extract_source_dt_seconds(header),
        units=units,
        axis_units=axis_units,
        source_format="ibw",
        note=_decode_igor_text(wave.get("note")),
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


def load_igor_pxp_file(path: str | Path) -> dict[str, IgorWave]:
    """Load all 1-D waves from an Igor Packed Experiment (.pxp) file."""
    path = Path(path)
    try:
        packed = importlib.import_module("igor2.packed")
        wave_record_module = importlib.import_module("igor2.record.wave")
    except ImportError as exc:
        raise ImportError(
            "Reading Igor Packed Experiment files requires the optional 'igor2' dependency."
        ) from exc

    WaveRecord = wave_record_module.WaveRecord

    _records, filesystem = packed.load(path)

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

    return waves


def load_igor_csv_file(path: str | Path) -> IgorWave:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = [row for row in csv.reader(fp) if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError("File is empty")

    wave_name = rows[0][0].strip().strip('"')
    if not wave_name:
        raise ValueError("The first non-empty row must contain the Igor wave name")

    values: list[float] = []
    for line_number, row in enumerate(rows[1:], start=2):
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
        raise ValueError("No numeric stimulation samples found in CSV file")

    return IgorWave(
        name=wave_name,
        values=np.asarray(values, dtype=np.float32),
        source_dt=None,
        units=None,
        axis_units=None,
        source_format="csv",
        note=None,
    )


def load_igor_wave_file(path: str | Path) -> IgorWave:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
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
    total_duration = float(sample_count) * float(source_dt)
    return max(1, int(np.ceil(total_duration / float(target_dt))))


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

    target_sample_count = estimate_resampled_sample_count(
        int(source_values.size),
        float(source_dt),
        target_dt=float(target_dt),
    )
    target_times = np.arange(target_sample_count, dtype=np.float64) * float(target_dt)
    source_indices = np.floor(target_times / float(source_dt)).astype(np.int64)
    source_indices = np.clip(source_indices, 0, source_values.size - 1)
    return source_values[source_indices].astype(np.float32, copy=False)


def prepare_igor_wave_for_neuroseq(
    wave: IgorWave,
    *,
    source_dt_override: float | None = None,
    target_dt: float = NEUROSEQ_IGOR_TARGET_DT_SECONDS,
) -> PreparedIgorWave:
    source_dt = (
        float(source_dt_override) if source_dt_override is not None else wave.source_dt
    )
    if source_dt is None:
        raise ValueError("Source sample interval is required to import this Igor wave")

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
        raise ValueError("Target StimArray must be 2-dimensional")
    if (
        abs(float(stim_array.dt) - float(prepared_wave.target_dt))
        > _DT_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "Igor import into an existing stim array currently requires a 1 ms target dt"
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
    )
