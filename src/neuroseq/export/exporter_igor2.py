from __future__ import annotations

import csv
import io
import pathlib
import re
import struct
from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.utils import detect_period


@dataclass(frozen=True)
class Igor2SingleWaveExport:
    experiment_name: str
    neuron_index: int
    sample_count: int
    n_sweeps: int = 1


@dataclass(frozen=True)
class Igor2PxpExport:
    path: pathlib.Path
    wave_names: list[str]


_NT_FP32: int = 2
_PXP_WAVE_RECORD_TYPE: int = 3

_BIN_HEADER5_FMT: str = "<hllll4l4llll"
_PXP_RECORD_HEADER_FMT: str = "<HhI"

_WAVE_HEADER5_FMT: str = "<LLLlhh6sh32slL4l4d4d4s16shhddL4L4LL16lhhhBBLlhhLL"

assert struct.calcsize(_BIN_HEADER5_FMT) == 62, "BinHeader5 layout wrong"
assert struct.calcsize(_WAVE_HEADER5_FMT) == 320, "WaveHeader5 layout wrong"


def _sanitize_igor_wave_name(name: str) -> str:
    """Return a valid Igor wave name derived from *name*."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"w{sanitized}"
    return sanitized[:31]


def _deduplicate_igor_wave_name(base_name: str, used_names: set[str]) -> str:
    """Return a unique Igor wave name that respects the 31-character limit."""
    candidate = _sanitize_igor_wave_name(base_name)
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = 2
    while True:
        suffix_text = f"_{suffix}"
        candidate = f"{_sanitize_igor_wave_name(base_name)[: 31 - len(suffix_text)]}{suffix_text}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _write_ibw_v5(
    values: np.ndarray,
    *,
    wave_name: str,
    dt: float,
    data_units: str = "",
    x_units: str = "s",
) -> bytes:
    """Serialise a 1-D or 2-D float32 array to Igor Binary Wave v5 bytes."""
    data = np.asarray(values, dtype="<f4")
    if data.ndim not in (1, 2):
        raise ValueError("IBW export expects a 1-D or 2-D (samples × sweeps) array")
    if dt <= 0:
        raise ValueError("Sample interval must be positive")

    if data.ndim == 1:
        n_samples = int(data.size)
        n_sweeps = 0
        npnts = n_samples
        data_bytes = data.tobytes()
    else:
        n_samples = int(data.shape[0])
        n_sweeps = int(data.shape[1])
        npnts = n_samples * n_sweeps
        data_bytes = np.asarray(data, order="F").tobytes(order="F")
    wfm_size = struct.calcsize(_WAVE_HEADER5_FMT) + len(data_bytes)

    def _encode_units(text: str, width: int) -> bytes:
        encoded = text.encode("ascii", errors="replace")[: width - 1]
        return encoded.ljust(width, b"\x00")

    # Extended data units (IBW v5): if the requested units don't fit in the
    # 4-byte WaveHeader5.dataUnits field, write the full string into the
    # optional extended-units section after the wave data and set
    # BinHeader5.dataEUnitsSize accordingly. The short field still gets the
    # truncated first 3 ASCII chars so older readers see a best-effort label.
    data_units_ascii = data_units.encode("ascii", errors="replace")
    extended_data_units_bytes = data_units_ascii if len(data_units_ascii) > 3 else b""
    data_eunits_size = len(extended_data_units_bytes)

    # IBW files begin with a top-level int16 version selector (here: 5).
    # igor2's DynamicVersionField reads this before dispatching to Wave5.
    version_bytes = struct.pack("<h", 5)
    bname_bytes = wave_name.encode("ascii", errors="replace")[:31].ljust(32, b"\x00")
    data_units_bytes = _encode_units(data_units, 4)
    dim_units_bytes = _encode_units(x_units, 4) + (b"\x00" * 12)
    n_dim_values = (n_samples, n_sweeps, 0, 0)
    sfA_values = (float(dt), 1.0, 1.0, 1.0)
    sfB_values = (0.0, 0.0, 0.0, 0.0)

    wave_header_values = (
        0,
        0,
        0,
        npnts,
        _NT_FP32,
        0,
        b"\x00" * 6,
        1,
        bname_bytes,
        0,
        0,
        *n_dim_values,
        *sfA_values,
        *sfB_values,
        data_units_bytes,
        dim_units_bytes,
        0,
        0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        *([0] * 16),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    wave_header = struct.pack(_WAVE_HEADER5_FMT, *wave_header_values)

    # BinHeader5 fields (after checksum / wfmSize):
    #   formulaSize, noteSize, dataEUnitsSize,
    #   dimEUnitsSize[4], dimLabelsSize[4],
    #   sIndicesSize, optionsSize1, optionsSize2.
    bin_header_values = (
        0,
        wfm_size,
        0,
        0,
        data_eunits_size,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    bin_header = struct.pack(_BIN_HEADER5_FMT, *bin_header_values)

    checksum_buffer = version_bytes + bin_header + wave_header
    short_count = len(checksum_buffer) // 2
    short_sum = sum(struct.unpack(f"<{short_count}h", checksum_buffer))
    checksum = struct.unpack("<h", struct.pack("<H", (-short_sum) % 65536))[0]

    bin_header = struct.pack(
        _BIN_HEADER5_FMT,
        checksum,
        *bin_header_values[1:],
    )
    return (
        version_bytes
        + bin_header
        + wave_header
        + data_bytes
        + extended_data_units_bytes
    )


def _write_pxp_wave_record(ibw_bytes: bytes) -> bytes:
    """Wrap full IBW bytes (including version prefix) in a PXP WaveRecord."""
    header = struct.pack(
        _PXP_RECORD_HEADER_FMT,
        _PXP_WAVE_RECORD_TYPE,
        0,
        len(ibw_bytes),
    )
    return header + ibw_bytes


def generate_igor2_csv_text(
    stim_array: StimArray,
    *,
    experiment_name: str,
    neuron_index: int,
    scaling_factor: float = 1.0,
) -> str:
    """Serialize one neuron column from a stim array into the Igor2 CSV format."""
    normalized_experiment_name = experiment_name.strip()
    if not normalized_experiment_name:
        raise ValueError("Experiment name cannot be empty.")

    if stim_array.values.ndim != 2:
        raise ValueError(
            "Input export expects a 2D input array with shape (time, neuron)."
        )

    n_neurons = int(stim_array.values.shape[1])
    if n_neurons <= 0:
        raise ValueError("Input export requires at least one neuron.")
    if neuron_index < 0 or neuron_index >= n_neurons:
        raise ValueError(
            f"Neuron index {neuron_index} is out of range for {n_neurons} neurons."
        )

    values = np.asarray(stim_array.values[:, neuron_index], dtype=float)
    if scaling_factor != 1.0:
        values = values * scaling_factor
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow([normalized_experiment_name])
    for value in values:
        writer.writerow([float(value)])
    return csv_buffer.getvalue()


def export_igor2_csv(
    output_path: str | pathlib.Path,
    stim_array: StimArray,
    *,
    experiment_name: str,
    neuron_index: int,
    scaling_factor: float = 1.0,
) -> Igor2SingleWaveExport:
    """Export one neuron column to Igor2 CSV.

    ``scaling_factor`` multiplies each sample value. There is no unit field
    in the CSV format, so any unit override is ignored here.
    """
    if not np.isfinite(scaling_factor):
        raise ValueError(f"scaling_factor must be finite, got {scaling_factor!r}")
    csv_text = generate_igor2_csv_text(
        stim_array,
        experiment_name=experiment_name,
        neuron_index=neuron_index,
        scaling_factor=scaling_factor,
    )
    path = pathlib.Path(output_path)
    path.write_text(csv_text, encoding="utf-8", newline="")
    return Igor2SingleWaveExport(
        experiment_name=experiment_name.strip(),
        neuron_index=neuron_index,
        sample_count=int(stim_array.values.shape[0]),
    )


def export_igor2_ibw(
    output_path: str | pathlib.Path,
    stim_array: StimArray,
    *,
    wave_name: str,
    neuron_index: int,
    unit_override: str | None = None,
    scaling_factor: float = 1.0,
    as_sweeps: bool = False,
) -> Igor2SingleWaveExport:
    """Export one neuron column from *stim_array* as an Igor Binary Wave.

    ``unit_override`` replaces ``stim_array.units`` in the wave header when
    not ``None``/empty; units longer than 3 ASCII characters are written via
    the IBW v5 extended-units block. ``scaling_factor`` multiplies each
    sample before serialisation.
    """
    if stim_array.values.ndim != 2:
        raise ValueError(
            "Input export expects a 2D input array with shape (time, neuron)."
        )
    n_neurons = int(stim_array.values.shape[1])
    if n_neurons <= 0:
        raise ValueError("Input export requires at least one neuron.")
    if neuron_index < 0 or neuron_index >= n_neurons:
        raise ValueError(
            f"Neuron index {neuron_index} is out of range for {n_neurons} neurons."
        )
    if not np.isfinite(scaling_factor):
        raise ValueError(f"scaling_factor must be finite, got {scaling_factor!r}")

    sanitized_name = _sanitize_igor_wave_name(wave_name)
    values = np.asarray(stim_array.values[:, neuron_index], dtype=np.float32)
    if scaling_factor != 1.0:
        values = (values * scaling_factor).astype(np.float32)
    write_values: np.ndarray = values
    n_sweeps_out = 1
    if as_sweeps:
        period = detect_period(values)
        if period is not None and period > 0 and values.size % period == 0:
            sweep_count = values.size // period
            if sweep_count > 1:
                write_values = values.reshape(sweep_count, period).T
                write_values = np.ascontiguousarray(write_values)
                n_sweeps_out = sweep_count
    wave_units = unit_override if unit_override else str(stim_array.units)
    ibw_bytes = _write_ibw_v5(
        write_values,
        wave_name=sanitized_name,
        dt=float(stim_array.dt),
        data_units=wave_units,
        x_units="s",
    )

    path = pathlib.Path(output_path)
    path.write_bytes(ibw_bytes)
    return Igor2SingleWaveExport(
        experiment_name=sanitized_name,
        neuron_index=neuron_index,
        sample_count=int(stim_array.values.shape[0]),
        n_sweeps=n_sweeps_out,
    )


def export_igor2_pxp(
    output_path: str | pathlib.Path,
    stim_arrays: dict[str, StimArray],
    *,
    array_names: list[str] | None = None,
    unit_override: str | None = None,
    scaling_factor: float = 1.0,
) -> Igor2PxpExport:
    """Export selected stim arrays to an Igor2 Packed Experiment (.pxp) file.

    ``unit_override`` replaces each array's ``stim_array.units`` in the written
    wave headers when not ``None``/empty. ``scaling_factor`` multiplies all
    sample values before they are written; inhibition negation still applies
    on top, so an inhibitory array exported with ``scaling_factor=2`` produces
    values ``-2 * stim_array.values``.
    """
    if array_names is None:
        array_names = list(stim_arrays.keys())

    if not np.isfinite(scaling_factor):
        raise ValueError(f"scaling_factor must be finite, got {scaling_factor!r}")

    used_names: set[str] = set()
    exported_wave_names: list[str] = []
    pxp_data = bytearray()

    for name in array_names:
        stim_array = stim_arrays[name]
        if stim_array.values.ndim != 2:
            raise ValueError(
                "Input export expects a 2D input array with shape (time, neuron)."
            )

        values = stim_array.values
        if stim_array.stim_type == StimType.Inhibition:
            values = -values
        if scaling_factor != 1.0:
            values = values * scaling_factor

        wave_units = unit_override if unit_override else str(stim_array.units)

        base_wave_name = _sanitize_igor_wave_name(name)
        n_neurons = int(values.shape[1])
        for neuron_idx in range(n_neurons):
            neuron_wave_name = _deduplicate_igor_wave_name(
                f"{base_wave_name}_n{neuron_idx}",
                used_names,
            )
            exported_wave_names.append(neuron_wave_name)
            ibw_bytes = _write_ibw_v5(
                np.asarray(values[:, neuron_idx], dtype=np.float32),
                wave_name=neuron_wave_name,
                dt=float(stim_array.dt),
                data_units=wave_units,
                x_units="s",
            )
            pxp_data += _write_pxp_wave_record(ibw_bytes)

    path = pathlib.Path(output_path)
    path.write_bytes(bytes(pxp_data))
    return Igor2PxpExport(path=path, wave_names=exported_wave_names)
