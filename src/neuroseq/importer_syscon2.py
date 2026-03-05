from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import brian2.numpy_ as np

from neuroseq.stim import StimArray

logger = logging.getLogger(__name__)

_BLOCK_START_RE = re.compile(r"^\[(\d+)\]$")
_BLOCK_END_RE = re.compile(r"^\[/(\d+)\]$")
_HEADER_HASH_RE = re.compile(r"^[0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){15}$")

_REQUIRED_HEADER_KEYS = (
    "version",
    "scanmode",
    "runs",
    "DeviceSequenceID",
    "invertedportcnt",
    "AutoTriggerRulesEnabled",
    "AutoTriggerRulesCount",
    "timelinezoomfactor",
    "timelineviewposx",
    "timelineviewposy",
)

_MAPPABLE_SHAPE_TYPES = {"shape", "entity"}
_ENTITY_DEFAULT_DURATION_SECONDS = 16.666e-3


@dataclass
class SysconSequenceObject:
    block_index: int
    object_id: int | None
    object_type: str
    properties: dict[str, str]
    start_time_us: int
    repeats: int
    step_size: int
    intensity: float


@dataclass
class SysconSequence:
    header: dict[str, str]
    objects: list[SysconSequenceObject]


@dataclass
class SysconImportResult:
    total_objects: int
    eligible_objects: int
    grouped_objects: int
    merged_objects: int
    mapped_objects: int
    dropped_objects: int
    generated_timepoints: int
    truncated_timepoints: int
    truncated_repeats: int


def _strip_comment(line: str) -> str:
    # The official .seq format does not support comments. We strip them so a
    # commented reference file can still be parsed for preview/import in NeuroSeq.
    if "//" in line:
        return line.split("//", 1)[0].rstrip()
    return line


def _is_shape_like_object_type(object_type: str) -> bool:
    return object_type.strip().lower() in _MAPPABLE_SHAPE_TYPES


def _safe_int(value: str | None, default: int = 0) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(float(value.strip()))
    except ValueError:
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def parse_syscon2_seq_text(text: str) -> SysconSequence:
    lines = text.splitlines()
    cleaned = [_strip_comment(line).strip() for line in lines]
    cleaned = [line for line in cleaned if line != ""]
    if not cleaned:
        raise ValueError("File is empty")

    # Handle optional UTF-8 BOM in first line.
    cleaned[0] = cleaned[0].lstrip("\ufeff")

    first_line = cleaned[0]
    if first_line == "nohash" or _HEADER_HASH_RE.match(first_line):
        first_line_is_valid = True
    else:
        raise ValueError(
            "Invalid SysCon2 header: expected first line 'nohash' or a hash ID line"
        )

    header: dict[str, str] = {}
    objects: list[SysconSequenceObject] = []

    i = 1 if first_line_is_valid else 0
    if _HEADER_HASH_RE.match(first_line):
        header["hash_id"] = first_line
    while i < len(cleaned):
        start_match = _BLOCK_START_RE.match(cleaned[i])
        if start_match:
            break
        if "=" in cleaned[i]:
            key, value = cleaned[i].split("=", 1)
            header[key.strip()] = value.strip()
        i += 1

    missing_header_keys = [k for k in _REQUIRED_HEADER_KEYS if k not in header]
    if missing_header_keys:
        raise ValueError(
            f"Invalid SysCon2 header, missing keys: {', '.join(missing_header_keys)}"
        )

    while i < len(cleaned):
        start_match = _BLOCK_START_RE.match(cleaned[i])
        if not start_match:
            raise ValueError(f"Expected block start marker at line '{cleaned[i]}'")
        block_index = int(start_match.group(1))
        i += 1

        props: dict[str, str] = {}
        while i < len(cleaned) and not _BLOCK_END_RE.match(cleaned[i]):
            line = cleaned[i]
            if "=" not in line:
                raise ValueError(
                    f"Invalid object property in block [{block_index}]: '{line}'"
                )
            key, value = line.split("=", 1)
            props[key.strip()] = value.strip()
            i += 1

        if i >= len(cleaned):
            raise ValueError(f"Missing closing marker for block [{block_index}]")

        end_match = _BLOCK_END_RE.match(cleaned[i])
        assert end_match is not None
        end_index = int(end_match.group(1))
        if end_index != block_index:
            raise ValueError(
                f"Mismatched block end marker [/{end_index}] for block [{block_index}]"
            )
        i += 1

        object_type = props.get("type", "")
        default_intensity = 1.0 if _is_shape_like_object_type(object_type) else 0.0
        obj = SysconSequenceObject(
            block_index=block_index,
            object_id=_safe_int(props.get("ID"), default=-1),
            object_type=object_type,
            properties=props,
            start_time_us=max(0, _safe_int(props.get("TimelineInfo0_starttime"), 0)),
            repeats=max(0, _safe_int(props.get("TimelineInfo0_repeats"), 0)),
            step_size=max(1, _safe_int(props.get("TimelineInfo0_Stepsize"), 1)),
            intensity=_safe_float(
                props.get("TimelineInfo0_Intensity_0"), default_intensity
            ),
        )
        if obj.object_id == -1:
            obj.object_id = None
        objects.append(obj)

    if not objects:
        raise ValueError("No sequence object blocks found")

    return SysconSequence(header=header, objects=objects)


def parse_syscon2_seq_file(path: str | Path) -> SysconSequence:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_syscon2_seq_text(text)


def get_timed_objects(
    sequence: SysconSequence, include_non_shape: bool = False
) -> list[SysconSequenceObject]:
    eligible: list[SysconSequenceObject] = []
    for obj in sequence.objects:
        has_timing = "TimelineInfo0_starttime" in obj.properties
        if not has_timing:
            continue
        if include_non_shape or _is_shape_like_object_type(obj.object_type):
            eligible.append(obj)
    return eligible


@dataclass
class _CircularObjectGeometry:
    center_x: float
    center_y: float
    radius: float


def _find_property_value_by_suffix(
    properties: dict[str, str],
    suffixes: tuple[str, ...],
) -> str | None:
    lower_keys = {key.lower(): value for key, value in properties.items()}
    for suffix in suffixes:
        for key, value in lower_keys.items():
            if key.endswith(suffix):
                return value
    return None


def _parse_shape_entity_geometry_type(obj: SysconSequenceObject) -> int | None:
    for key, value in obj.properties.items():
        lower_key = key.lower()
        if not (lower_key.startswith("shape") or lower_key.startswith("entity")):
            continue
        if not lower_key.endswith("_type"):
            continue
        return _safe_int(value, default=-1)
    return None


def _extract_circular_geometry(
    obj: SysconSequenceObject,
) -> _CircularObjectGeometry | None:
    # In SysCon2 files, circular ROIs are encoded as shape/entity type 2.
    # The radius is defined by the center and Vertex1 coordinates, with
    # vertex coordinates stored relative to the center.
    shape_type = _parse_shape_entity_geometry_type(obj)
    if shape_type != 2:
        return None

    center_x_raw = _find_property_value_by_suffix(
        obj.properties, ("_centerx", "centerx")
    )
    center_y_raw = _find_property_value_by_suffix(
        obj.properties, ("_centery", "centery")
    )
    vertex_x_raw = _find_property_value_by_suffix(
        obj.properties, ("_vertex1_x", "_vertex1x", "vertex1_x", "vertex1x")
    )
    vertex_y_raw = _find_property_value_by_suffix(
        obj.properties, ("_vertex1_y", "_vertex1y", "vertex1_y", "vertex1y")
    )
    # Compatibility fallback for sequences that only contain Vertex0.
    if vertex_x_raw is None or vertex_y_raw is None:
        vertex_x_raw = _find_property_value_by_suffix(
            obj.properties, ("_vertex0_x", "_vertex0x", "vertex0_x", "vertex0x")
        )
        vertex_y_raw = _find_property_value_by_suffix(
            obj.properties, ("_vertex0_y", "_vertex0y", "vertex0_y", "vertex0y")
        )
    if (
        center_x_raw is None
        or center_y_raw is None
        or vertex_x_raw is None
        or vertex_y_raw is None
    ):
        return None

    center_x = _safe_float(center_x_raw, default=0.0)
    center_y = _safe_float(center_y_raw, default=0.0)
    vertex_x = _safe_float(vertex_x_raw, default=0.0)
    vertex_y = _safe_float(vertex_y_raw, default=0.0)
    radius = float(np.hypot(vertex_x, vertex_y))
    if radius <= 0:
        return None
    return _CircularObjectGeometry(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )


def _event_duration_timepoints_for_object(
    obj: SysconSequenceObject,
    *,
    dt_seconds: float,
) -> int:
    object_type = obj.object_type.strip().lower()
    if object_type == "entity":
        return max(1, int(np.ceil(_ENTITY_DEFAULT_DURATION_SECONDS / dt_seconds)))
    return 1


def _group_eligible_objects_by_target_neuron(
    eligible_objects: list[SysconSequenceObject],
) -> list[list[SysconSequenceObject]]:
    if len(eligible_objects) == 0:
        return []

    circle_geometry_by_index: dict[int, _CircularObjectGeometry] = {}
    for idx, obj in enumerate(eligible_objects):
        geometry = _extract_circular_geometry(obj)
        if geometry is not None:
            circle_geometry_by_index[idx] = geometry

    parent = list(range(len(eligible_objects)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    circle_indices = sorted(circle_geometry_by_index.keys())
    for i, idx_a in enumerate(circle_indices):
        ga = circle_geometry_by_index[idx_a]
        for idx_b in circle_indices[i + 1 :]:
            gb = circle_geometry_by_index[idx_b]
            distance = float(
                np.hypot(ga.center_x - gb.center_x, ga.center_y - gb.center_y)
            )
            # If circles overlap or touch, treat as the same target neuron.
            if distance <= (ga.radius + gb.radius):
                union(idx_a, idx_b)

    groups: list[list[SysconSequenceObject]] = []
    root_to_group_index: dict[int, int] = {}
    for idx, obj in enumerate(eligible_objects):
        if idx in circle_geometry_by_index:
            root = find(idx)
            if root not in root_to_group_index:
                root_to_group_index[root] = len(groups)
                groups.append([])
            groups[root_to_group_index[root]].append(obj)
        else:
            groups.append([obj])
    return groups


def group_timed_objects_by_neuron_target(
    sequence: SysconSequence,
    *,
    include_non_shape: bool = False,
) -> list[list[SysconSequenceObject]]:
    """Group timed objects by target neuron.

    Circular shape/entity objects that overlap are grouped together and mapped
    to the same neuron channel.
    """
    eligible_objects = get_timed_objects(sequence, include_non_shape=include_non_shape)
    return _group_eligible_objects_by_target_neuron(eligible_objects)


def sequence_to_stim_values(
    sequence: SysconSequence,
    *,
    dt_seconds: float,
    n_neurons: int,
    tick_resolution_us: int = 500,
    include_non_shape: bool = False,
    max_events_per_object: int = 500_000,
) -> tuple[np.ndarray, SysconImportResult]:
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")

    grouped_objects = group_timed_objects_by_neuron_target(
        sequence, include_non_shape=include_non_shape
    )
    eligible_objects_count = sum(len(group) for group in grouped_objects)
    grouped_objects_count = len(grouped_objects)
    mapped_objects = min(n_neurons, grouped_objects_count)
    dropped_objects = grouped_objects_count - mapped_objects

    truncated_repeats = 0
    max_timepoints = 1
    for object_group in grouped_objects[:mapped_objects]:
        for obj in object_group:
            event_count = obj.repeats + 1
            if event_count > max_events_per_object:
                truncated_repeats += event_count - max_events_per_object
                event_count = max_events_per_object
            if event_count <= 0:
                continue
            last_time_us = obj.start_time_us + (
                (event_count - 1) * obj.step_size * tick_resolution_us
            )
            last_time_index = int(np.rint((last_time_us * 1e-6) / dt_seconds))
            duration_timepoints = _event_duration_timepoints_for_object(
                obj, dt_seconds=dt_seconds
            )
            max_timepoints = max(max_timepoints, last_time_index + duration_timepoints)

    n_timepoints = max_timepoints
    values = np.zeros((n_timepoints, n_neurons), dtype=np.float32)

    for neuron_idx, object_group in enumerate(grouped_objects[:mapped_objects]):
        for obj in object_group:
            event_count = obj.repeats + 1
            if event_count > max_events_per_object:
                event_count = max_events_per_object
            if event_count <= 0:
                continue
            times_us = obj.start_time_us + np.arange(event_count, dtype=np.int64) * (
                obj.step_size * tick_resolution_us
            )
            time_indices = np.rint((times_us * 1e-6) / dt_seconds).astype(np.int64)
            valid_time_indices = np.unique(
                time_indices[(time_indices >= 0) & (time_indices < n_timepoints)]
            )
            if valid_time_indices.size == 0:
                continue
            duration_timepoints = _event_duration_timepoints_for_object(
                obj, dt_seconds=dt_seconds
            )
            for time_index in valid_time_indices:
                start_idx = int(time_index)
                end_idx = min(n_timepoints, start_idx + duration_timepoints)
                values[start_idx:end_idx, neuron_idx] = np.maximum(
                    values[start_idx:end_idx, neuron_idx], obj.intensity
                )

    result = SysconImportResult(
        total_objects=len(sequence.objects),
        eligible_objects=eligible_objects_count,
        grouped_objects=grouped_objects_count,
        merged_objects=max(0, eligible_objects_count - grouped_objects_count),
        mapped_objects=mapped_objects,
        dropped_objects=dropped_objects,
        generated_timepoints=n_timepoints,
        truncated_timepoints=0,
        truncated_repeats=truncated_repeats,
    )
    return values, result


def import_sequence_into_stim_array(
    stim_array: StimArray,
    sequence: SysconSequence,
    *,
    tick_resolution_us: int = 500,
    include_non_shape: bool = False,
    replace_existing: bool = True,
    extend_to_fit: bool = True,
    neuron_offset: int = 0,
) -> SysconImportResult:
    if stim_array.values.ndim != 2:
        raise ValueError("Target StimArray must be 2-dimensional")

    n_existing_timepoints, n_existing_neurons = stim_array.values.shape
    if neuron_offset < 0 or neuron_offset >= n_existing_neurons:
        raise ValueError(
            f"Invalid neuron offset {neuron_offset} for {n_existing_neurons} neurons"
        )

    n_target_neurons = n_existing_neurons - neuron_offset
    imported_values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=stim_array.dt,
        n_neurons=n_target_neurons,
        tick_resolution_us=tick_resolution_us,
        include_non_shape=include_non_shape,
    )

    target_timepoints = n_existing_timepoints
    if extend_to_fit:
        target_timepoints = max(n_existing_timepoints, imported_values.shape[0])

    target_values = np.zeros(
        (target_timepoints, n_existing_neurons), dtype=stim_array.values.dtype
    )
    if not replace_existing:
        target_values[:n_existing_timepoints, :] = stim_array.values

    if imported_values.shape[0] > target_timepoints:
        result.truncated_timepoints = imported_values.shape[0] - target_timepoints
        imported_values = imported_values[:target_timepoints, :]

    destination = target_values[
        : imported_values.shape[0],
        neuron_offset : neuron_offset + imported_values.shape[1],
    ]
    if replace_existing:
        destination[:, :] = imported_values
    else:
        destination[:, :] = np.maximum(destination, imported_values)

    stim_array.values = target_values
    logger.info(
        "Imported SysCon2 sequence into '%s': mapped_channels=%d/%d eligible_objects (merged=%d), target shape=%s",
        stim_array.name,
        result.mapped_objects,
        result.eligible_objects,
        result.merged_objects,
        target_values.shape,
    )
    return result
