from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import brian2.numpy_ as np
import numpy.typing as npt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
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
) -> str | None:
    """Validate patchclamp append behavior and return a disjoint warning if needed."""
    if model is None:
        return None

    existing_experiments = getattr(model, "experiments", {})
    existing_experiment = existing_experiments.get(experiment_name)
    if existing_experiment is None:
        return None
    existing_data = getattr(existing_experiment, "data", [])

    existing_trace_data = None
    for data in existing_data:
        if (
            isinstance(data, TraceData)
            and getattr(data, "name", "") == "Patch Clamp Traces"
        ):
            existing_trace_data = data
            break

    if existing_trace_data is None:
        return None

    existing_neuron_indices = set(map(int, existing_trace_data.indices))
    importing_existing = [
        nid for nid in neuron_ids_to_import if nid in existing_neuron_indices
    ]
    importing_new = [
        nid for nid in neuron_ids_to_import if nid not in existing_neuron_indices
    ]

    # allow no crossover with existing neuron ids
    if importing_existing and importing_new:
        raise ValueError(
            f"Cannot extend data when importing both existing neurons {importing_existing} "
            f"and new neurons {importing_new} together. "
            "Either replace the same set of neurons or import a fully disjoint set on the same timeline."
        )
    if importing_new and not importing_existing and existing_neuron_indices:
        return (
            f"Patchclamp import is disjoint for experiment '{experiment_name}' "
            f"(incoming={sorted(importing_new)}, existing={sorted(existing_neuron_indices)})."
        )
    return None


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

    ordered_pairs = sorted(
        zip(neuron_ids, trace_data_rows, strict=False), key=lambda pair: pair[0]
    )
    for neuron_id, trace_row in ordered_pairs:
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
