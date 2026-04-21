from __future__ import annotations

import csv
import io
import pathlib
import re
import struct
from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.stim import StimArray


@dataclass(frozen=True)
class Igor2PatchClampExport:
    experiment_name: str
    neuron_index: int
    sample_count: int


_NT_FP32: int = 2

_BIN_HEADER5_FMT: str = "<hllll4l4llll"

_WAVE_HEADER5_FMT: str = "<LLLlhh6sh32slL4l4d4d4s16shhddL4L4LL16lhhhBBLlhhLL"

assert struct.calcsize(_BIN_HEADER5_FMT) == 62, "BinHeader5 layout wrong"
assert struct.calcsize(_WAVE_HEADER5_FMT) == 320, "WaveHeader5 layout wrong"


def _sanitize_igor_wave_name(name: str) -> str:
    """Return a valid Igor wave name derived from *name*."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"w{sanitized}"
    return sanitized[:31]


def _write_ibw_v5(
    values: np.ndarray,
    *,
    wave_name: str,
    dt: float,
    data_units: str = "",
    x_units: str = "s",
) -> bytes:
    """Serialise a 1-D float32 array to Igor Binary Wave v5 bytes."""
    data = np.asarray(values, dtype="<f4")
    if data.ndim != 1:
        raise ValueError("IBW export expects a 1-dimensional array")
    if dt <= 0:
        raise ValueError("Sample interval must be positive")

    n = int(data.size)
    data_bytes = data.tobytes()
    wfm_size = struct.calcsize(_WAVE_HEADER5_FMT) + len(data_bytes)

    def _encode_units(text: str, width: int) -> bytes:
        encoded = text.encode("ascii", errors="replace")[: width - 1]
        return encoded.ljust(width, b"\x00")

    # IBW files begin with a top-level int16 version selector (here: 5).
    # igor2's DynamicVersionField reads this before dispatching to Wave5.
    version_bytes = struct.pack("<h", 5)
    bname_bytes = wave_name.encode("ascii", errors="replace")[:31].ljust(32, b"\x00")
    data_units_bytes = _encode_units(data_units, 4)
    dim_units_bytes = _encode_units(x_units, 4).ljust(16, b"\x00")

    wave_header_values = (
        0,
        0,
        0,
        n,
        _NT_FP32,
        0,
        b"\x00" * 6,
        1,
        bname_bytes,
        0,
        0,
        n,
        0,
        0,
        0,
        float(dt),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
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

    bin_header_values = (
        0,
        wfm_size,
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
    return version_bytes + bin_header + wave_header + data_bytes


def generate_igor2_patchclamp_csv_text(
    stim_array: StimArray,
    *,
    experiment_name: str,
    neuron_index: int,
) -> str:
    """Serialize one neuron column from a stim array into the Igor2 CSV format."""
    normalized_experiment_name = experiment_name.strip()
    if not normalized_experiment_name:
        raise ValueError("Experiment name cannot be empty.")

    if stim_array.values.ndim != 2:
        raise ValueError(
            "Stimulus export expects a 2D stim array with shape (time, neuron)."
        )

    n_neurons = int(stim_array.values.shape[1])
    if n_neurons <= 0:
        raise ValueError("Stimulus export requires at least one neuron.")
    if neuron_index < 0 or neuron_index >= n_neurons:
        raise ValueError(
            f"Neuron index {neuron_index} is out of range for {n_neurons} neurons."
        )

    values = np.asarray(stim_array.values[:, neuron_index], dtype=float)
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow([normalized_experiment_name])
    for value in values:
        writer.writerow([float(value)])
    return csv_buffer.getvalue()


def export_igor2_patchclamp_csv(
    output_path: str | pathlib.Path,
    stim_array: StimArray,
    *,
    experiment_name: str,
    neuron_index: int,
) -> Igor2PatchClampExport:
    csv_text = generate_igor2_patchclamp_csv_text(
        stim_array,
        experiment_name=experiment_name,
        neuron_index=neuron_index,
    )
    path = pathlib.Path(output_path)
    path.write_text(csv_text, encoding="utf-8", newline="")
    return Igor2PatchClampExport(
        experiment_name=experiment_name.strip(),
        neuron_index=neuron_index,
        sample_count=int(stim_array.values.shape[0]),
    )


def export_igor2_patchclamp_ibw(
    output_path: str | pathlib.Path,
    stim_array: StimArray,
    *,
    wave_name: str,
    neuron_index: int,
) -> Igor2PatchClampExport:
    """Export one neuron column from *stim_array* as an Igor Binary Wave."""
    if stim_array.values.ndim != 2:
        raise ValueError(
            "Stimulus export expects a 2D stim array with shape (time, neuron)."
        )
    n_neurons = int(stim_array.values.shape[1])
    if n_neurons <= 0:
        raise ValueError("Stimulus export requires at least one neuron.")
    if neuron_index < 0 or neuron_index >= n_neurons:
        raise ValueError(
            f"Neuron index {neuron_index} is out of range for {n_neurons} neurons."
        )

    sanitized_name = _sanitize_igor_wave_name(wave_name)
    values = np.asarray(stim_array.values[:, neuron_index], dtype=np.float32)
    ibw_bytes = _write_ibw_v5(
        values,
        wave_name=sanitized_name,
        dt=float(stim_array.dt),
        data_units=str(stim_array.units),
        x_units="s",
    )

    path = pathlib.Path(output_path)
    path.write_bytes(ibw_bytes)
    return Igor2PatchClampExport(
        experiment_name=sanitized_name,
        neuron_index=neuron_index,
        sample_count=int(stim_array.values.shape[0]),
    )
