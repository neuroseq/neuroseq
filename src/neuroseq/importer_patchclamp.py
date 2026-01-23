from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import brian2.numpy_ as np
import numpy.typing as npt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spike_detection import detect_spikes_voltage_trace


@dataclass
class PeakDetectionParams:
    peak_height: float
    peak_prominence: float
    min_interval: float
    peak_distance: int
    peak_width: int


def validate_append_import(
    model: object,
    experiment_name: str,
    neuron_ids_to_import: Sequence[int],
) -> None:
    """Ensure append mode will not misalign timestamps."""
    if model is None:
        return

    existing_experiments = getattr(model, "experiments", {})
    existing_data = existing_experiments.get(experiment_name, [])

    existing_trace_data = None
    for data in existing_data:
        if (
            isinstance(data, TraceData)
            and getattr(data, "name", "") == "Patch Clamp Traces"
        ):
            existing_trace_data = data
            break

    if existing_trace_data is None:
        return

    existing_neuron_indices = set(existing_trace_data.indices)
    importing_existing = [
        nid for nid in neuron_ids_to_import if nid in existing_neuron_indices
    ]
    importing_new = [
        nid for nid in neuron_ids_to_import if nid not in existing_neuron_indices
    ]

    if importing_existing and importing_new:
        raise ValueError(
            f"Cannot extend data when importing both existing neurons {importing_existing} "
            f"and new neurons {importing_new} together. "
            f"This would cause timestamp misalignment. Use 'Replace' mode instead, "
            f"or import all neurons with the same timeline separately."
        )

    if importing_new and existing_neuron_indices:
        raise ValueError(
            f"Cannot extend data when importing new neurons {importing_new} - "
            f"existing neurons {sorted(existing_neuron_indices)} already have data. "
            f"This would cause timestamp misalignment. Use 'Replace' mode instead, "
            f"or import all neurons with the same timeline."
        )


def generate_stimulus_mapping(
    cellpose_import: CellposeImportResult | None,
    strategy: str,
    data_row_count: int,
    stim_ids: Sequence[int],
) -> dict[int, int]:
    """Create stimulus->cellpose mapping consistent with Cellpose data."""
    if cellpose_import is None:
        return {}

    existing_mapping = cellpose_import.stimulus_to_cellpose_mapping or {}

    if existing_mapping:
        return dict(existing_mapping)

    if cellpose_import.original_cellpose_ids is None:
        return {}

    original_ids = cellpose_import.original_cellpose_ids
    row_count = data_row_count or len(original_ids)

    stim_sequence = list(range(row_count))

    temp_mapping: dict[int, int] = {}
    for stim_id, cellpose_id in zip(stim_sequence, original_ids, strict=False):
        temp_mapping[int(stim_id)] = int(cellpose_id)

    return temp_mapping


def import_patchclamp_data(
    npy_data: npt.NDArray[np.float_],
    neuron_ids: Sequence[int],
    *,
    dt: float,
    detection_params: PeakDetectionParams,
    logger: logging.Logger | None = None,
) -> tuple[TraceData, list[SpikeData]]:
    """Convert raw npy traces into TraceData + SpikeData list."""
    if npy_data.ndim == 1:
        if len(neuron_ids) > 1:
            raise ValueError(
                f"Single trace data cannot be imported to multiple neurons {neuron_ids}"
            )
        trace_data_rows = [npy_data]
    elif npy_data.ndim == 2:
        if npy_data.shape[0] != len(neuron_ids):
            raise ValueError(
                f"Data has {npy_data.shape[0]} rows but {len(neuron_ids)} neuron IDs specified"
            )
        trace_data_rows = [npy_data[i] for i in range(npy_data.shape[0])]
    else:
        raise ValueError(f"Unsupported data dimensions: {npy_data.shape}")

    all_traces: list[npt.NDArray[np.float_]] = []
    all_neuron_indices: list[int] = []
    imported_spike_data: list[SpikeData] = []

    for i, neuron_id in enumerate(neuron_ids):
        trace_row = trace_data_rows[i]

        if logger:
            logger.debug(
                "Detecting spikes for patchclamp neuron %d: dt=%.6f height=%.2f prominence=%.2f min_interval=%.4f distance=%d width=%d",
                neuron_id,
                dt,
                detection_params.peak_height,
                detection_params.peak_prominence,
                detection_params.min_interval,
                detection_params.peak_distance,
                detection_params.peak_width,
            )

        spike_data = detect_spikes_voltage_trace(
            trace_row,
            neuron_id=neuron_id,
            dt=dt,
            detection_method="peak",
            peak_height_threshold=detection_params.peak_height,
            peak_prominence=detection_params.peak_prominence,
            min_spike_interval=detection_params.min_interval,
            peak_distance=detection_params.peak_distance,
            peak_width=detection_params.peak_width,
        )

        spike_data.name = "Patch Clamp Spikes"
        imported_spike_data.append(spike_data)
        all_traces.append(trace_row)
        all_neuron_indices.append(neuron_id)

        if logger:
            logger.debug(
                "Patchclamp import complete for neuron=%d spikes=%d trace_len=%d dt=%.6f",
                neuron_id,
                len(spike_data.spike_times),
                len(trace_row),
                dt,
            )

    trace_array = np.array(all_traces)
    neuron_indices_array = np.array(all_neuron_indices, dtype=np.uint32)
    trace_data = TraceData(
        name="Patch Clamp Traces",
        traces=trace_array,
        indices=neuron_indices_array,
        units="mV",
        dt=dt,
    )

    return trace_data, imported_spike_data
