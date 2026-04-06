from __future__ import annotations

import csv
import io
import pathlib
from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.stim import StimArray


@dataclass(frozen=True)
class Igor2PatchClampExport:
    experiment_name: str
    neuron_index: int
    sample_count: int


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
