from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.stim import StimArray

_BLOCK_START_RE = re.compile(r"^\[(\d+)\]$")
_HEADER_HASH_RE = re.compile(r"^[0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){15}$")

_REQUIRED_HEADER_DEFAULTS: dict[str, str] = {
    "version": "3",
    "scanmode": "Accurate",
    "runs": "0",
    "DeviceSequenceID": "0",
    "invertedportcnt": "0",
    "AutoTriggerRulesEnabled": "False",
    "AutoTriggerRulesCount": "0",
    "timelinezoomfactor": "10",
    "timelineviewposx": "0",
    "timelineviewposy": "0",
}

_DEFAULT_LIGHTSOURCE_ID = "dummy_1_0"
_DEFAULT_LIGHTSOURCE_PORT = "dummy_1_0 RMI In"


@dataclass(frozen=True)
class _StimSegment:
    neuron_index: int
    start_idx: int
    duration_idx: int


@dataclass(frozen=True)
class _MappedSegment:
    neuron_index: int
    start_us: int
    duration_ticks: int
    center_x: float
    center_y: float


@dataclass
class SysconExportReport:
    total_neurons: int
    active_neurons: int
    mapped_active_neurons: int
    unmapped_active_neurons: list[int]
    total_segments: int
    exported_segments: int
    skipped_segments_unmapped: int
    entity_lanes: int
    generated_objects: int


def generate_syscon2_sequence_text(
    stim_array: StimArray,
    cellpose_data: CellposeImportResult | None,
    *,
    template_text: str,
    tick_resolution_us: int = 500,
    active_threshold: float = 0.0,
    binary_intensity: float = 1.0,
) -> tuple[str, SysconExportReport]:
    """Generate SysCon2 `.seq` content from a stim array + Cellpose segmentation.

    Export behavior:
    - Binary mode: values > active_threshold are considered ON.
    - Segment compression: contiguous ON windows per neuron become one segment.
    - Unmapped active neurons are skipped and reported.
    - Entity timelines use per-neuron lanes [0..N-1].
    - Matching laser TTLPulses are written on lanes [N..2N-1].
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")
    if binary_intensity < 0 or binary_intensity > 1:
        raise ValueError("binary_intensity must be within [0, 1]")
    if cellpose_data is None:
        raise ValueError("Cellpose data is required to export SysCon2 .seq")
    if cellpose_data.segmentation_array is None:
        raise ValueError("Cellpose segmentation is required to export SysCon2 .seq")

    header_lines = _build_header_lines(template_text)
    lightsource_id, lightsource_port = _detect_lightsource_info(
        template_text, header_lines
    )

    segments = _extract_active_segments(stim_array.values, active_threshold)
    active_neurons = sorted({segment.neuron_index for segment in segments})

    segmentation = cellpose_data.segmentation_array
    centroids = _compute_cell_centroids(segmentation)
    neuron_to_cellpose = _resolve_neuron_to_cellpose_mapping(
        cellpose_data,
        stim_array.values.shape[1],
        set(centroids.keys()),
    )

    sample_us = float(stim_array.dt * 1e6)
    mapped_segments: list[_MappedSegment] = []
    skipped_segments_unmapped = 0
    active_neurons_set = set(active_neurons)
    mapped_active_neurons: set[int] = set()
    unmapped_active_neurons: set[int] = set()

    for segment in segments:
        cellpose_id = neuron_to_cellpose.get(segment.neuron_index)
        if cellpose_id is None or cellpose_id not in centroids:
            skipped_segments_unmapped += 1
            if segment.neuron_index in active_neurons_set:
                unmapped_active_neurons.add(segment.neuron_index)
            continue

        mapped_active_neurons.add(segment.neuron_index)
        center_x, center_y = centroids[cellpose_id]
        start_ticks = int(
            np.rint((segment.start_idx * sample_us) / float(tick_resolution_us))
        )
        start_us = int(start_ticks * tick_resolution_us)
        duration_us = float(segment.duration_idx) * sample_us
        duration_ticks = max(1, int(np.ceil(duration_us / float(tick_resolution_us))))
        mapped_segments.append(
            _MappedSegment(
                neuron_index=segment.neuron_index,
                start_us=start_us,
                duration_ticks=duration_ticks,
                center_x=center_x,
                center_y=center_y,
            )
        )

    if len(mapped_segments) == 0:
        raise ValueError(
            "No mapped active stimulation segments to export. "
            "Check Cellpose mapping and stim activity."
        )

    lane_neurons = sorted({segment.neuron_index for segment in mapped_segments})
    lane_by_neuron = {neuron_idx: lane for lane, neuron_idx in enumerate(lane_neurons)}
    n_entity_lanes = len(lane_neurons)
    mapped_segments.sort(
        key=lambda segment: (
            segment.start_us,
            lane_by_neuron[segment.neuron_index],
            segment.neuron_index,
        )
    )

    object_blocks: list[list[str]] = []
    next_object_id = 1
    for entity_counter, mapped_segment in enumerate(mapped_segments):
        entity_lane = lane_by_neuron[mapped_segment.neuron_index]
        ttl_lane = entity_lane + n_entity_lanes

        object_blocks.append(
            _format_object_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                object_type="Entity",
                properties=[
                    ("TotalTimings", "1"),
                    ("TimelineInfo0_TimelineIndex", str(entity_lane)),
                    ("TimelineInfo0_starttime", str(mapped_segment.start_us)),
                    ("TimelineInfo0_description", ""),
                    ("TimelineInfo0_repeats", "0"),
                    ("TimelineInfo0_LightsourceCount", "1"),
                    ("TimelineInfo0_LightsourceID_0", lightsource_id),
                    ("TimelineInfo0_Intensity_0", _format_float(binary_intensity)),
                    ("TimelineInfo0_timelinegroupid", "-1"),
                    ("TimelineInfo0_Stepsize", "1"),
                    (f"Entity{entity_counter}_Type", "0"),
                    (f"Entity{entity_counter}_Filled", "False"),
                    (
                        f"Entity{entity_counter}_CenterX",
                        _format_float(mapped_segment.center_x),
                    ),
                    (
                        f"Entity{entity_counter}_CenterY",
                        _format_float(mapped_segment.center_y),
                    ),
                    (f"Entity{entity_counter}_CenterZ", "0"),
                    (f"Entity{entity_counter}_Rotation", "0"),
                    (f"Entity{entity_counter}_ScaleX", "1"),
                    (f"Entity{entity_counter}_ScaleY", "1"),
                    (f"Entity{entity_counter}_ScaleZ", "1"),
                    (f"Entity{entity_counter}_VertexCount", "0"),
                    (f"Entity{entity_counter}_reversed", "False"),
                ],
            )
        )
        next_object_id += 1

        ttl_repeats = max(0, mapped_segment.duration_ticks - 1)
        object_blocks.append(
            _format_object_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                object_type="TTLPulse",
                properties=[
                    ("TotalTimings", "1"),
                    ("TimelineInfo0_TimelineIndex", str(ttl_lane)),
                    ("TimelineInfo0_starttime", str(mapped_segment.start_us)),
                    ("TimelineInfo0_description", ""),
                    ("TimelineInfo0_repeats", str(ttl_repeats)),
                    ("TimelineInfo0_LightsourceCount", "1"),
                    ("TimelineInfo0_LightsourceID_0", lightsource_id),
                    ("TimelineInfo0_Intensity_0", "0"),
                    ("TimelineInfo0_timelinegroupid", "-1"),
                    ("TimelineInfo0_Stepsize", "100"),
                    ("port", lightsource_port),
                ],
            )
        )
        next_object_id += 1

    lines = list(header_lines)
    for block in object_blocks:
        lines.extend(block)
    sequence_text = "\n".join(lines).rstrip() + "\n"

    # Ensure generated sequence can be parsed by our SysCon2 parser.
    parse_syscon2_seq_text(sequence_text)

    report = SysconExportReport(
        total_neurons=stim_array.values.shape[1],
        active_neurons=len(active_neurons),
        mapped_active_neurons=len(mapped_active_neurons),
        unmapped_active_neurons=sorted(unmapped_active_neurons),
        total_segments=len(segments),
        exported_segments=len(mapped_segments),
        skipped_segments_unmapped=skipped_segments_unmapped,
        entity_lanes=n_entity_lanes,
        generated_objects=len(object_blocks),
    )
    return sequence_text, report


def _extract_active_segments(
    values: np.ndarray,
    threshold: float,
) -> list[_StimSegment]:
    segments: list[_StimSegment] = []
    n_neurons = values.shape[1]
    for neuron_idx in range(n_neurons):
        active = values[:, neuron_idx] > threshold
        if not np.any(active):
            continue
        padded = np.concatenate(
            (
                np.array([False], dtype=bool),
                active.astype(bool),
                np.array([False], dtype=bool),
            )
        )
        starts = np.where((~padded[:-1]) & padded[1:])[0]
        ends = np.where(padded[:-1] & (~padded[1:]))[0]
        for start_idx, end_idx in zip(starts, ends, strict=False):
            duration_idx = int(end_idx - start_idx)
            if duration_idx <= 0:
                continue
            segments.append(
                _StimSegment(
                    neuron_index=int(neuron_idx),
                    start_idx=int(start_idx),
                    duration_idx=duration_idx,
                )
            )
    return segments


def _compute_cell_centroids(segmentation: np.ndarray) -> dict[int, tuple[float, float]]:
    if segmentation.ndim != 2:
        raise ValueError("Cellpose segmentation must be a 2D array")

    centroids: dict[int, tuple[float, float]] = {}
    unique_ids = np.unique(segmentation)
    for cell_id in unique_ids:
        cell_int = int(cell_id)
        if cell_int < 0:
            continue
        y_coords, x_coords = np.where(segmentation == cell_int)
        if x_coords.size == 0:
            continue
        centroids[cell_int] = (float(np.mean(x_coords)), float(np.mean(y_coords)))
    return centroids


def _resolve_neuron_to_cellpose_mapping(
    cellpose_data: CellposeImportResult,
    n_neurons: int,
    available_cellpose_ids: set[int],
) -> dict[int, int]:
    explicit_mapping = cellpose_data.stimulus_to_cellpose_mapping or {}
    resolved: dict[int, int] = {}

    if explicit_mapping:
        for neuron_idx in range(n_neurons):
            raw_cellpose_id: int | None = None
            if neuron_idx in explicit_mapping:
                raw_cellpose_id = int(explicit_mapping[neuron_idx])
            elif (neuron_idx + 1) in explicit_mapping:
                # Accept 1-based mapping keys for backwards compatibility.
                raw_cellpose_id = int(explicit_mapping[neuron_idx + 1])

            if raw_cellpose_id is None:
                continue
            if raw_cellpose_id in available_cellpose_ids:
                resolved[neuron_idx] = raw_cellpose_id
        return resolved

    if cellpose_data.original_cellpose_ids is not None:
        candidate_ids = [int(value) for value in cellpose_data.original_cellpose_ids]
    else:
        candidate_ids = sorted(available_cellpose_ids)

    filtered_ids = [cid for cid in candidate_ids if cid in available_cellpose_ids]
    for neuron_idx, cellpose_id in zip(range(n_neurons), filtered_ids, strict=False):
        resolved[int(neuron_idx)] = int(cellpose_id)
    return resolved


def _strip_comment(line: str) -> str:
    if "//" not in line:
        return line
    return line.split("//", 1)[0]


def _build_header_lines(template_text: str) -> list[str]:
    raw_lines = template_text.splitlines()
    header_candidates: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if _BLOCK_START_RE.match(stripped):
            break
        cleaned = _strip_comment(line).strip()
        if cleaned != "":
            header_candidates.append(cleaned)

    first_line = "nohash"
    ordered_header = OrderedDict[str, str]()

    if header_candidates:
        candidate_first_line = header_candidates[0]
        if candidate_first_line == "nohash" or _HEADER_HASH_RE.match(
            candidate_first_line
        ):
            first_line = candidate_first_line
            kv_lines = header_candidates[1:]
        else:
            kv_lines = header_candidates
    else:
        kv_lines = []

    for line in kv_lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        ordered_header[key.strip()] = value.strip()

    for key, default_value in _REQUIRED_HEADER_DEFAULTS.items():
        if key not in ordered_header:
            ordered_header[key] = default_value

    lines = [first_line]
    lines.extend([f"{key}={value}" for key, value in ordered_header.items()])
    return lines


def _detect_lightsource_info(
    template_text: str,
    header_lines: list[str],
) -> tuple[str, str]:
    lightsource_id = ""
    lightsource_port = ""
    parsed_header: dict[str, str] = {}
    for line in header_lines[1:]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed_header[key.strip()] = value.strip()

    for line in template_text.splitlines():
        cleaned = _strip_comment(line).strip()
        if cleaned.startswith("TimelineInfo0_LightsourceID_0="):
            _, value = cleaned.split("=", 1)
            value = value.strip()
            if value:
                lightsource_id = value
                break

    if not lightsource_id:
        source_port = parsed_header.get("SourceConnectorPort0", "").strip()
        if source_port.endswith(" RMI In"):
            lightsource_id = source_port[: -len(" RMI In")]
        elif source_port:
            lightsource_id = source_port

    if not lightsource_id:
        lightsource_id = _DEFAULT_LIGHTSOURCE_ID

    for line in template_text.splitlines():
        cleaned = _strip_comment(line).strip()
        if not cleaned.startswith("port="):
            continue
        _, value = cleaned.split("=", 1)
        value = value.strip()
        if lightsource_id in value:
            lightsource_port = value
            break

    if not lightsource_port:
        source_port = parsed_header.get("SourceConnectorPort0", "").strip()
        if source_port:
            lightsource_port = source_port

    if not lightsource_port:
        lightsource_port = f"{lightsource_id} RMI In"

    if lightsource_port.strip() == "":
        lightsource_port = _DEFAULT_LIGHTSOURCE_PORT

    return lightsource_id, lightsource_port


def _format_object_block(
    *,
    object_id: int,
    block_index: int,
    object_type: str,
    properties: list[tuple[str, str]],
) -> list[str]:
    lines = [f"[{block_index}]", f"ID={object_id}", f"type={object_type}"]
    lines.extend([f"{key}={value}" for key, value in properties])
    lines.append(f"[/{block_index}]")
    return lines


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
