from __future__ import annotations

import hashlib
import logging
import math
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
    "invertedportcnt",
    "AutoTriggerRulesEnabled",
    "AutoTriggerRulesCount",
    "timelinezoomfactor",
    "timelineviewposx",
    "timelineviewposy",
)

_HEADER_COMPAT_DEFAULTS: dict[str, str] = {
    # Some SysCon2 exports omit this field although the file is valid.
    "DeviceSequenceID": "0",
}

_MAPPABLE_SHAPE_TYPES = {"shape", "entity"}
# Fallback duration for legacy Entity objects that lack an associated TTL pulse.
# Equals one SLM window (32 ticks × 520.8125 µs = 16 666 µs); rounded to 16.666 ms.
_ENTITY_DEFAULT_DURATION_SECONDS = 16.666e-3
_SAME_NEURON_MAX_CENTER_DISTANCE_UM = 1.0
_SAME_ROI_SPOT_MAX_DISTANCE_UM = 9.0
_HOLO4D_TTL_OUT_START_PULSE_MIN_US = 9_000
_SYSCON2_TTL_PULSE_QUANTUM_US = 520.8125
_SYSCON2_NEURON_OBJECT_QUANTUM_US = 32 * _SYSCON2_TTL_PULSE_QUANTUM_US
_SYSCON2_WINDOW_TICKS = 32
_NEUROSEQ_DT_TOLERANCE_SECONDS = 1e-9
_MAX_IMPORT_BYTES = 512 * 1024 * 1024


@dataclass
class SysconSequenceObject:
    block_index: int
    object_id: int | None
    object_type: str
    properties: dict[str, str]
    start_time_us: float
    repeats: int
    step_size: int
    intensity: float


@dataclass
class SysconSequence:
    header: dict[str, str]
    objects: list[SysconSequenceObject]
    # The stim array this file was exported from, restored from the sidecar
    # written next to the .seq (the file itself holds hardware-adjusted timings).
    original_values: np.ndarray | None = None
    original_dt: float | None = None


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


@dataclass
class _LaserTTLPulse:
    block_index: int
    start_time_us: float
    duration_us: float
    lightsource_ids: set[str]


def _strip_comment(line: str) -> str:
    # The official .seq format does not support comments. We strip them so a
    # commented reference file can still be parsed for preview/import in NeuroSeq.
    if "//" in line:
        return line.split("//", 1)[0].rstrip()
    return line


def _is_shape_like_object_type(object_type: str) -> bool:
    return object_type.strip().lower() in _MAPPABLE_SHAPE_TYPES


def _display_intensity(obj: SysconSequenceObject) -> float:
    """Intensity used when rendering a stimulation object into a stim array.

    SysCon2 Holo4D exports frequently leave the per-shape intensity at 0 because
    the laser power is controlled by the hardware rather than encoded in the
    ``.seq`` file. Rendering those events verbatim would produce an all-zero
    (invisible) stim array, so a shape-like object whose intensity is 0 falls
    back to a nominal full-scale value of 1.0. Objects with an explicit positive
    intensity keep it, and non-shape objects are left untouched.
    """
    if _is_shape_like_object_type(obj.object_type) and obj.intensity <= 0.0:
        return 1.0
    return obj.intensity


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


def _validate_object_properties(props: dict[str, str], block_index: int) -> None:
    integer_keys = {
        "ID",
        "TotalTimings",
        "TimelineInfo0_repeats",
        "TimelineInfo0_Stepsize",
        "TimelineInfo0_TimelineIndex",
        "TimelineInfo0_LightsourceCount",
    }
    for key, raw in props.items():
        numeric = (
            key in integer_keys
            or key == "TimelineInfo0_starttime"
            or "_Intensity_" in key
            or key.lower().endswith(("_centerx", "_centery", "_centerz"))
        )
        if not numeric:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Block [{block_index}]: invalid {key}={raw!r}") from exc
        if not math.isfinite(value) or (key in integer_keys and not value.is_integer()):
            raise ValueError(
                f"Block [{block_index}]: {key} must be a finite {'integer' if key in integer_keys else 'number'}"
            )
        if (
            key in {"TimelineInfo0_starttime", "TimelineInfo0_repeats"}
            or "_Intensity_" in key
        ) and value < 0:
            raise ValueError(f"Block [{block_index}]: {key} cannot be negative")
        if key == "TimelineInfo0_Stepsize" and value <= 0:
            raise ValueError(f"Block [{block_index}]: step size must be positive")
    if _safe_int(props.get("TotalTimings"), 1) > 1 or any(
        re.match(r"TimelineInfo[1-9]", key) for key in props
    ):
        raise ValueError(
            f"Block [{block_index}]: multiple timings per object are not supported"
        )
    if _safe_int(props.get("TimelineInfo0_LightsourceCount"), 1) > 1:
        raise ValueError(
            f"Block [{block_index}]: multiple lightsources per object are not supported"
        )


def _check_allocation(n_timepoints: int, n_neurons: int, itemsize: int = 4) -> None:
    if n_timepoints * n_neurons * itemsize > _MAX_IMPORT_BYTES:
        raise ValueError(
            "Sequence would require more than 512 MiB for its stimulus array. Reduce duration or target channel count before importing."
        )


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

    for key, default_value in _HEADER_COMPAT_DEFAULTS.items():
        header.setdefault(key, default_value)

    missing_header_keys = [k for k in _REQUIRED_HEADER_KEYS if k not in header]
    if missing_header_keys:
        raise ValueError(
            f"Invalid SysCon2 header, missing keys: {', '.join(missing_header_keys)}"
        )

    seen_blocks: set[int] = set()
    while i < len(cleaned):
        start_match = _BLOCK_START_RE.match(cleaned[i])
        if not start_match:
            raise ValueError(f"Expected block start marker at line '{cleaned[i]}'")
        block_index = int(start_match.group(1))
        if block_index in seen_blocks:
            raise ValueError(f"Duplicate block index [{block_index}]")
        seen_blocks.add(block_index)
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

        _validate_object_properties(props, block_index)
        object_type = props.get("type", "")
        default_intensity = 1.0 if _is_shape_like_object_type(object_type) else 0.0
        obj = SysconSequenceObject(
            block_index=block_index,
            object_id=_safe_int(props.get("ID"), default=-1),
            object_type=object_type,
            properties=props,
            start_time_us=max(
                0.0, _safe_float(props.get("TimelineInfo0_starttime"), 0.0)
            ),
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


ORIGINAL_STIM_SUFFIX = "_original_stim.npz"


def original_stim_sidecar_path(seq_path: str | Path) -> Path:
    seq_path = Path(seq_path)
    return seq_path.with_name(seq_path.stem + ORIGINAL_STIM_SUFFIX)


def write_original_stim_sidecar(
    path: str | Path, values: np.ndarray, dt: float, seq_text: str
) -> None:
    """Save the exported stim array so a re-import can restore its exact timings."""
    digest = hashlib.sha256(seq_text.encode("utf-8")).hexdigest()
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            values=np.asarray(values, dtype=np.float32),
            dt=np.float64(dt),
            seq_sha256=np.str_(digest),
        )


def _load_original_stim_sidecar(
    seq_path: Path, seq_bytes: bytes
) -> tuple[np.ndarray, float] | None:
    sidecar = original_stim_sidecar_path(seq_path)
    if not sidecar.is_file():
        return None
    try:
        with np.load(sidecar, allow_pickle=False) as data:
            if str(data["seq_sha256"]) != hashlib.sha256(seq_bytes).hexdigest():
                logger.info("Ignoring %s: it does not match %s", sidecar, seq_path)
                return None
            values = np.asarray(data["values"], dtype=np.float32)
            dt = float(data["dt"])
    except (OSError, ValueError, KeyError):
        logger.warning("Could not read %s", sidecar, exc_info=True)
        return None
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return None
    return values, dt


def parse_syscon2_seq_file(path: str | Path) -> SysconSequence:
    path = Path(path)
    raw = path.read_bytes()
    sequence = parse_syscon2_seq_text(raw.decode("utf-8"))
    original = _load_original_stim_sidecar(path, raw)
    if original is not None:
        sequence.original_values, sequence.original_dt = original
    return sequence


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


def _extract_circular_entity_center(
    obj: SysconSequenceObject,
) -> tuple[float, float] | None:
    # In SysCon2 files, circular entities are encoded as type 2.
    # Objects are grouped by center-to-center distance only.
    if obj.object_type.strip().lower() != "entity":
        return None

    shape_type = _parse_shape_entity_geometry_type(obj)
    if shape_type != 2:
        return None

    center_x_raw = _find_property_value_by_suffix(
        obj.properties, ("_centerx", "centerx")
    )
    center_y_raw = _find_property_value_by_suffix(
        obj.properties, ("_centery", "centery")
    )
    if center_x_raw is None or center_y_raw is None:
        return None

    center_x = _safe_float(center_x_raw, default=0.0)
    center_y = _safe_float(center_y_raw, default=0.0)
    return (center_x, center_y)


def _extract_lightsource_ids(obj: SysconSequenceObject) -> set[str]:
    lightsource_ids: set[str] = set()
    for key, value in obj.properties.items():
        if "_lightsourceid_" not in key.lower():
            continue
        stripped = value.strip()
        if stripped != "":
            lightsource_ids.add(stripped)
    return lightsource_ids


def _ttl_pulse_duration_us(
    obj: SysconSequenceObject,
    *,
    tick_resolution_us: int,
) -> float | None:
    # SysCon2 v3 TTLPulse duration is `(repeats + 1) * 520.8125us`.
    if obj.object_type.strip().lower() != "ttlpulse":
        return None
    if len(_extract_lightsource_ids(obj)) == 0:
        # Not a laser timeline pulse (e.g., main-controller TTL outputs).
        return None
    _ = tick_resolution_us  # TTLPulse duration is always in 520.8125 µs quanta
    duration_ticks = max(1, obj.repeats + 1)
    return float(duration_ticks) * _SYSCON2_TTL_PULSE_QUANTUM_US


def _time_us_to_tick(time_us: float) -> int:
    return int(np.rint(time_us / _SYSCON2_TTL_PULSE_QUANTUM_US))


def _shape_object_duration_us(
    obj: SysconSequenceObject,
    *,
    tick_resolution_us: int,
) -> float:
    # SysCon2 v3 Shape neuron objects use 16.666ms quanta, duration=(repeats+2)*q.
    object_type = obj.object_type.strip().lower()
    if object_type == "shape":
        return float(max(2, obj.repeats + 2)) * _SYSCON2_NEURON_OBJECT_QUANTUM_US
    if object_type == "entity":
        # Backwards compatibility for older Entity-based exports.
        duration_ticks = max(1, (obj.repeats + 1) * obj.step_size)
        return float(duration_ticks * tick_resolution_us)
    return float(max(2, obj.repeats + 2)) * _SYSCON2_NEURON_OBJECT_QUANTUM_US


def _entity_timing_from_laser_ttl(
    sequence: SysconSequence,
    *,
    dt_seconds: float,
    tick_resolution_us: int,
) -> dict[int, tuple[float, int]]:
    """Map entity block indices to (start_us, duration_timepoints) from laser TTLPulse objects."""
    entities = [
        obj for obj in sequence.objects if obj.object_type.strip().lower() == "entity"
    ]
    if len(entities) == 0:
        return {}

    ttl_pulses: list[_LaserTTLPulse] = []
    for obj in sequence.objects:
        duration_us = _ttl_pulse_duration_us(obj, tick_resolution_us=tick_resolution_us)
        if duration_us is None:
            continue
        ttl_pulses.append(
            _LaserTTLPulse(
                block_index=obj.block_index,
                start_time_us=obj.start_time_us,
                duration_us=duration_us,
                lightsource_ids=_extract_lightsource_ids(obj),
            )
        )
    if len(ttl_pulses) == 0:
        return {}

    entities_sorted = sorted(
        entities, key=lambda obj: (obj.start_time_us, obj.block_index)
    )
    ttl_pulses.sort(key=lambda pulse: (pulse.start_time_us, pulse.block_index))
    pulse_used = [False] * len(ttl_pulses)
    timing_by_entity_block: dict[int, tuple[float, int]] = {}

    for entity in entities_sorted:
        entity_lightsources = _extract_lightsource_ids(entity)
        best_pulse_idx: int | None = None
        best_key: tuple[int, float, float, int] | None = None

        for idx, pulse in enumerate(ttl_pulses):
            if pulse_used[idx]:
                continue
            pulse_lightsources = pulse.lightsource_ids
            if (
                len(entity_lightsources) > 0
                and len(pulse_lightsources) > 0
                and not (entity_lightsources & pulse_lightsources)
            ):
                continue

            pulse_start_us = pulse.start_time_us
            delta_us = pulse_start_us - entity.start_time_us
            # Prefer a TTL pulse that starts at/after the entity start; otherwise
            # fall back to nearest by absolute distance.
            is_before_entity = 1 if delta_us < 0 else 0
            abs_delta_us = abs(delta_us)
            candidate_key = (
                is_before_entity,
                abs_delta_us,
                pulse_start_us,
                pulse.block_index,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_pulse_idx = idx

        if best_pulse_idx is None:
            continue

        pulse_used[best_pulse_idx] = True
        matched_start_us = ttl_pulses[best_pulse_idx].start_time_us
        matched_duration_us = ttl_pulses[best_pulse_idx].duration_us
        duration_timepoints = max(
            1, int(np.rint((matched_duration_us * 1e-6) / dt_seconds))
        )
        timing_by_entity_block[entity.block_index] = (
            matched_start_us,
            duration_timepoints,
        )

    return timing_by_entity_block


def _event_start_time_us_for_object(
    obj: SysconSequenceObject,
    *,
    entity_timing_by_block: dict[int, tuple[float, int]] | None = None,
) -> float:
    object_type = obj.object_type.strip().lower()
    if (
        object_type == "entity"
        and entity_timing_by_block is not None
        and obj.block_index in entity_timing_by_block
    ):
        return entity_timing_by_block[obj.block_index][0]
    return obj.start_time_us


def _event_duration_timepoints_for_object(
    obj: SysconSequenceObject,
    *,
    dt_seconds: float,
    tick_resolution_us: int,
    entity_timing_by_block: dict[int, tuple[float, int]] | None = None,
) -> int:
    object_type = obj.object_type.strip().lower()
    if object_type == "entity":
        if (
            entity_timing_by_block is not None
            and obj.block_index in entity_timing_by_block
        ):
            return entity_timing_by_block[obj.block_index][1]
        return max(1, int(np.ceil(_ENTITY_DEFAULT_DURATION_SECONDS / dt_seconds)))
    if object_type == "shape":
        if (
            entity_timing_by_block is not None
            and obj.block_index in entity_timing_by_block
        ):
            return entity_timing_by_block[obj.block_index][1]
        duration_us = _shape_object_duration_us(
            obj, tick_resolution_us=tick_resolution_us
        )
        return max(1, int(np.rint((duration_us * 1e-6) / dt_seconds)))
    return 1


def _timeline_index_for_object(obj: SysconSequenceObject) -> int:
    return _safe_int(obj.properties.get("TimelineInfo0_TimelineIndex"), default=0)


def _extract_entity_center_any_shape(
    obj: SysconSequenceObject,
) -> tuple[float, float] | None:
    if not _is_shape_like_object_type(obj.object_type):
        return None
    center_x_raw = _find_property_value_by_suffix(
        obj.properties, ("_centerx", "centerx")
    )
    center_y_raw = _find_property_value_by_suffix(
        obj.properties, ("_centery", "centery")
    )
    if center_x_raw is None or center_y_raw is None:
        return None
    return (
        _safe_float(center_x_raw, default=0.0),
        _safe_float(center_y_raw, default=0.0),
    )


def _neuron_tag(obj: SysconSequenceObject) -> int | None:
    """Original neuron index written by the NeuroSeq exporter, if present."""
    raw = _find_property_value_by_suffix(obj.properties, ("_description",))
    match = re.fullmatch(r"\s*neuron=(\d+)\s*", raw or "")
    return int(match.group(1)) if match else None


def _group_entities_by_center_distance(
    entities: list[SysconSequenceObject],
) -> list[list[SysconSequenceObject]]:
    if len(entities) == 0:
        return []

    tags = [_neuron_tag(obj) for obj in entities]
    if all(tag is not None for tag in tags):
        # Exported by NeuroSeq: every spot carries its original neuron index,
        # so multi-spot ROIs collapse back to the original neuron.
        by_tag: dict[int, list[SysconSequenceObject]] = {}
        for obj, tag in zip(entities, tags, strict=True):
            by_tag.setdefault(int(tag), []).append(obj)  # type: ignore[arg-type]
        # Keep original rows: untouched neurons stay as empty placeholder groups.
        return [by_tag.get(row, []) for row in range(max(by_tag) + 1)]

    parent = list(range(len(entities)))

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

    centers_by_index: dict[int, tuple[float, float]] = {}
    for idx, obj in enumerate(entities):
        if obj.object_type.strip().lower() != "entity" and not (
            obj.object_type.strip().lower() == "shape"
            and _parse_shape_entity_geometry_type(obj) == 0
        ):
            continue
        center = _extract_entity_center_any_shape(obj)
        if center is not None:
            centers_by_index[idx] = center

    centered_indices = sorted(centers_by_index.keys())
    for i, idx_a in enumerate(centered_indices):
        center_a_x, center_a_y = centers_by_index[idx_a]
        for idx_b in centered_indices[i + 1 :]:
            center_b_x, center_b_y = centers_by_index[idx_b]
            distance = float(np.hypot(center_a_x - center_b_x, center_a_y - center_b_y))
            a, b = entities[idx_a], entities[idx_b]
            both_entities = a.object_type.lower() == b.object_type.lower() == "entity"
            same_point = (
                a.object_type.lower() == b.object_type.lower() == "shape"
                and distance == 0
                and _safe_float(
                    _find_property_value_by_suffix(a.properties, ("_centerz",))
                )
                == _safe_float(
                    _find_property_value_by_suffix(b.properties, ("_centerz",))
                )
            )
            # Multi-spot ROIs: spots of one neuron are written as simultaneous
            # shapes (same start and duration) a few micrometres apart.
            same_event_spots = (
                a.object_type.lower() == b.object_type.lower() == "shape"
                and distance <= _SAME_ROI_SPOT_MAX_DISTANCE_UM
                and a.start_time_us == b.start_time_us
                and a.repeats == b.repeats
            )
            if (
                (both_entities and distance <= _SAME_NEURON_MAX_CENTER_DISTANCE_UM)
                or same_point
                or same_event_spots
            ) and _extract_lightsource_ids(a) == _extract_lightsource_ids(b):
                union(idx_a, idx_b)

    grouped: list[list[SysconSequenceObject]] = []
    root_to_group_idx: dict[int, int] = {}
    for idx, obj in enumerate(entities):
        root = find(idx) if idx in centers_by_index else idx
        if root not in root_to_group_idx:
            root_to_group_idx[root] = len(grouped)
            grouped.append([])
        grouped[root_to_group_idx[root]].append(obj)
    return grouped


def _merge_time_intervals_us(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(intervals) == 0:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = []
    current_start, current_end = sorted_intervals[0]
    for start_us, end_us in sorted_intervals[1:]:
        if start_us <= current_end:
            current_end = max(current_end, end_us)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start_us, end_us
    merged.append((current_start, current_end))
    return merged


def _looks_like_holo4d_timeline_layout(
    sequence: SysconSequence,
    *,
    tick_resolution_us: int,
) -> bool:
    shape_like_objects = [
        obj for obj in sequence.objects if _is_shape_like_object_type(obj.object_type)
    ]
    if len(shape_like_objects) == 0:
        return False

    laser_pulses = [
        obj
        for obj in sequence.objects
        if _ttl_pulse_duration_us(obj, tick_resolution_us=tick_resolution_us)
        is not None
    ]
    if len(laser_pulses) == 0:
        return False

    ttl_out_like = [
        obj
        for obj in sequence.objects
        if obj.object_type.strip().lower() == "ttlpulse"
        and len(_extract_lightsource_ids(obj)) == 0
    ]
    if len(ttl_out_like) == 0:
        return False

    has_start_pulse = any(
        obj.start_time_us == 0
        and (max(1, obj.repeats + 1) * _SYSCON2_TTL_PULSE_QUANTUM_US)
        >= _HOLO4D_TTL_OUT_START_PULSE_MIN_US
        for obj in ttl_out_like
    )
    return has_start_pulse


def _has_laser_ttl_pulses(sequence: SysconSequence) -> bool:
    return any(
        obj.object_type.strip().lower() == "ttlpulse"
        and len(_extract_lightsource_ids(obj)) > 0
        for obj in sequence.objects
    )


def _sequence_to_stim_values_holo4d_layout(
    sequence: SysconSequence,
    *,
    dt_seconds: float,
    n_neurons: int,
    tick_resolution_us: int,
) -> tuple[np.ndarray, SysconImportResult]:
    shape_like_objects = get_timed_objects(sequence)
    grouped_entities = _group_entities_by_center_distance(
        sorted(shape_like_objects, key=lambda obj: (obj.start_time_us, obj.block_index))
    )
    grouped_objects_count = len(grouped_entities)
    mapped_objects = min(n_neurons, grouped_objects_count)
    dropped_objects = grouped_objects_count - mapped_objects

    laser_pulses = [
        obj
        for obj in sequence.objects
        if _ttl_pulse_duration_us(obj, tick_resolution_us=tick_resolution_us)
        is not None
    ]
    all_lightsources = set().union(
        *(_extract_lightsource_ids(pulse) for pulse in laser_pulses)
    )
    intervals_by_lightsource = {
        source: _merge_time_intervals_us(
            [
                (
                    pulse.start_time_us,
                    pulse.start_time_us
                    + float(
                        _ttl_pulse_duration_us(
                            pulse, tick_resolution_us=tick_resolution_us
                        )
                        or 0
                    ),
                )
                for pulse in laser_pulses
                if source in _extract_lightsource_ids(pulse)
            ]
        )
        for source in all_lightsources
    }

    # One SLM window (32 ticks × 520.8125 µs = 16 666 µs) is the hologram load-up
    # time during which no laser pulse may fire.
    entity_unusable_lead_us = int(_SYSCON2_NEURON_OBJECT_QUANTUM_US)
    intervals_by_neuron: list[list[tuple[int, int, float]]] = [
        [] for _ in range(mapped_objects)
    ]

    # A later point object on the same lane replaces earlier placement on that lane.
    next_start_by_block: dict[int, float] = {}
    shape_like_sorted_by_lane = sorted(
        shape_like_objects,
        key=lambda obj: (
            _timeline_index_for_object(obj),
            obj.start_time_us,
            obj.block_index,
        ),
    )
    lane_to_objects: dict[int, list[SysconSequenceObject]] = {}
    for obj in shape_like_sorted_by_lane:
        lane_to_objects.setdefault(_timeline_index_for_object(obj), []).append(obj)
    for lane_objects in lane_to_objects.values():
        for current, next_obj in zip(lane_objects, lane_objects[1:], strict=False):
            next_start_by_block[current.block_index] = next_obj.start_time_us

    max_timepoints = 1
    for neuron_idx, entity_group in enumerate(grouped_entities[:mapped_objects]):
        for entity in entity_group:
            sources = _extract_lightsource_ids(entity)
            if not sources:
                if len(all_lightsources) != 1:
                    raise ValueError(
                        "Cannot associate a shape without a lightsource ID with multiple laser timelines"
                    )
                sources = all_lightsources
            pulse_intervals = _merge_time_intervals_us(
                [
                    interval
                    for source in sources
                    for interval in intervals_by_lightsource.get(source, [])
                ]
            )
            entity_start_us = entity.start_time_us
            entity_end_us = entity_start_us + _shape_object_duration_us(
                entity,
                tick_resolution_us=tick_resolution_us,
            )
            if entity.block_index in next_start_by_block:
                entity_end_us = min(
                    entity_end_us, next_start_by_block[entity.block_index]
                )
            usable_start_us = entity_start_us + entity_unusable_lead_us
            if entity_end_us <= usable_start_us:
                continue
            for pulse_start_us, pulse_end_us in pulse_intervals:
                if pulse_end_us <= usable_start_us:
                    continue
                if pulse_start_us >= entity_end_us:
                    break
                effective_start_us = max(pulse_start_us, usable_start_us)
                effective_end_us = min(pulse_end_us, entity_end_us)
                if effective_end_us <= effective_start_us:
                    continue

                start_idx = int(np.rint((effective_start_us * 1e-6) / dt_seconds))
                duration_timepoints = max(
                    1,
                    int(
                        np.rint(
                            ((effective_end_us - effective_start_us) * 1e-6)
                            / dt_seconds
                        )
                    ),
                )
                end_idx = start_idx + duration_timepoints
                intervals_by_neuron[neuron_idx].append(
                    (start_idx, end_idx, _display_intensity(entity))
                )
                max_timepoints = max(max_timepoints, end_idx)

    _check_allocation(max_timepoints, n_neurons)
    values = np.zeros((max_timepoints, n_neurons), dtype=np.float32)
    for neuron_idx, intervals in enumerate(intervals_by_neuron):
        for start_idx, end_idx, intensity in intervals:
            start = max(0, start_idx)
            end = min(values.shape[0], end_idx)
            if end <= start:
                continue
            values[start:end, neuron_idx] = np.maximum(
                values[start:end, neuron_idx], intensity
            )

    result = SysconImportResult(
        total_objects=len(sequence.objects),
        eligible_objects=len(shape_like_objects),
        grouped_objects=grouped_objects_count,
        merged_objects=max(
            0, len(shape_like_objects) - sum(1 for g in grouped_entities if g)
        ),
        mapped_objects=mapped_objects,
        dropped_objects=dropped_objects,
        generated_timepoints=values.shape[0],
        truncated_timepoints=0,
        truncated_repeats=0,
    )
    return values, result


def _group_eligible_objects_by_target_neuron(
    eligible_objects: list[SysconSequenceObject],
) -> list[list[SysconSequenceObject]]:
    if len(eligible_objects) == 0:
        return []

    circle_center_by_index: dict[int, tuple[float, float]] = {}
    for idx, obj in enumerate(eligible_objects):
        center = _extract_circular_entity_center(obj)
        if center is not None:
            circle_center_by_index[idx] = center

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

    circle_indices = sorted(circle_center_by_index.keys())
    for i, idx_a in enumerate(circle_indices):
        center_a_x, center_a_y = circle_center_by_index[idx_a]
        for idx_b in circle_indices[i + 1 :]:
            center_b_x, center_b_y = circle_center_by_index[idx_b]
            distance = float(np.hypot(center_a_x - center_b_x, center_a_y - center_b_y))
            # If circle centers are within 1um, treat as the same target neuron.
            if distance <= _SAME_NEURON_MAX_CENTER_DISTANCE_UM:
                union(idx_a, idx_b)

    groups: list[list[SysconSequenceObject]] = []
    root_to_group_index: dict[int, int] = {}
    for idx, obj in enumerate(eligible_objects):
        if idx in circle_center_by_index:
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

    Circular entities (type 2) with centers within 1um are grouped together
    and mapped to the same neuron channel.
    """
    eligible_objects = get_timed_objects(sequence, include_non_shape=include_non_shape)
    if _has_laser_ttl_pulses(sequence):
        return _group_entities_by_center_distance(
            sorted(
                eligible_objects, key=lambda obj: (obj.start_time_us, obj.block_index)
            )
        )
    return _group_eligible_objects_by_target_neuron(eligible_objects)


def _original_stim_values(
    original: np.ndarray, n_neurons: int
) -> tuple[np.ndarray, SysconImportResult]:
    n_timepoints, n_original = original.shape
    mapped = min(n_neurons, n_original)
    _check_allocation(n_timepoints, n_neurons)
    values = np.zeros((n_timepoints, n_neurons), dtype=np.float32)
    values[:, :mapped] = original[:, :mapped]
    active = int(np.count_nonzero(original.any(axis=0)))
    return values, SysconImportResult(
        total_objects=0,
        eligible_objects=active,
        grouped_objects=n_original,
        merged_objects=0,
        mapped_objects=mapped,
        dropped_objects=n_original - mapped,
        generated_timepoints=n_timepoints,
        truncated_timepoints=0,
        truncated_repeats=0,
    )


def sequence_to_stim_values(
    sequence: SysconSequence,
    *,
    dt_seconds: float,
    n_neurons: int,
    tick_resolution_us: int = 500,
    include_non_shape: bool = False,
    max_events_per_object: int = 500_000,
) -> tuple[np.ndarray, SysconImportResult]:
    """Convert a parsed SysCon2 sequence to a (timepoints × neurons) stim array.

    Two reconstruction paths are used depending on the file format:

    - **Holo4D layout** (preferred): when laser ``TTLPulse`` objects are present,
      pulse intervals define the stimulation events and Shape/Entity objects define
      SLM availability windows.  This gives tick-accurate reconstruction.
    - **Legacy layout**: when no laser ``TTLPulse`` objects are present, timing is
      inferred from the Entity/Shape start times and durations.

    ``tick_resolution_us`` is used only for legacy ``Entity`` duration calculation
    and has no effect when the Holo4D path is active.
    """
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if not math.isfinite(tick_resolution_us) or tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")

    if max_events_per_object <= 0:
        raise ValueError("max_events_per_object must be positive")

    if (
        sequence.original_values is not None
        and sequence.original_dt is not None
        and math.isclose(sequence.original_dt, dt_seconds, rel_tol=1e-6)
    ):
        return _original_stim_values(sequence.original_values, n_neurons)

    # Prefer pulse-based reconstruction whenever laser TTLPulse events exist.
    # This avoids mode-flips caused by startup-TTL duration thresholds.
    if _has_laser_ttl_pulses(sequence):
        return _sequence_to_stim_values_holo4d_layout(
            sequence,
            dt_seconds=dt_seconds,
            n_neurons=n_neurons,
            tick_resolution_us=tick_resolution_us,
        )

    grouped_objects = group_timed_objects_by_neuron_target(
        sequence, include_non_shape=include_non_shape
    )
    entity_timing_by_block = _entity_timing_from_laser_ttl(
        sequence,
        dt_seconds=dt_seconds,
        tick_resolution_us=tick_resolution_us,
    )
    eligible_objects_count = sum(len(group) for group in grouped_objects)
    grouped_objects_count = len(grouped_objects)
    mapped_objects = min(n_neurons, grouped_objects_count)
    dropped_objects = grouped_objects_count - mapped_objects

    truncated_repeats = 0
    max_timepoints = 1
    for object_group in grouped_objects[:mapped_objects]:
        for obj in object_group:
            event_count = (
                1 if obj.object_type.strip().lower() == "shape" else obj.repeats + 1
            )
            if event_count > max_events_per_object:
                truncated_repeats += event_count - max_events_per_object
                event_count = max_events_per_object
            if event_count <= 0:
                continue
            event_start_time_us = _event_start_time_us_for_object(
                obj, entity_timing_by_block=entity_timing_by_block
            )
            last_time_us = event_start_time_us + (
                (event_count - 1) * obj.step_size * tick_resolution_us
            )
            last_time_index = int(np.rint((last_time_us * 1e-6) / dt_seconds))
            duration_timepoints = _event_duration_timepoints_for_object(
                obj,
                dt_seconds=dt_seconds,
                tick_resolution_us=tick_resolution_us,
                entity_timing_by_block=entity_timing_by_block,
            )
            max_timepoints = max(max_timepoints, last_time_index + duration_timepoints)

    n_timepoints = max_timepoints
    _check_allocation(n_timepoints, n_neurons)
    values = np.zeros((n_timepoints, n_neurons), dtype=np.float32)

    for neuron_idx, object_group in enumerate(grouped_objects[:mapped_objects]):
        for obj in object_group:
            event_count = (
                1 if obj.object_type.strip().lower() == "shape" else obj.repeats + 1
            )
            if event_count > max_events_per_object:
                event_count = max_events_per_object
            if event_count <= 0:
                continue
            event_start_time_us = _event_start_time_us_for_object(
                obj, entity_timing_by_block=entity_timing_by_block
            )
            times_us = event_start_time_us + np.arange(event_count, dtype=np.int64) * (
                obj.step_size * tick_resolution_us
            )
            time_indices = np.rint((times_us * 1e-6) / dt_seconds).astype(np.int64)
            valid_time_indices = np.unique(
                time_indices[(time_indices >= 0) & (time_indices < n_timepoints)]
            )
            if valid_time_indices.size == 0:
                continue
            duration_timepoints = _event_duration_timepoints_for_object(
                obj,
                dt_seconds=dt_seconds,
                tick_resolution_us=tick_resolution_us,
                entity_timing_by_block=entity_timing_by_block,
            )
            display_intensity = _display_intensity(obj)
            for time_index in valid_time_indices:
                start_idx = int(time_index)
                end_idx = min(n_timepoints, start_idx + duration_timepoints)
                values[start_idx:end_idx, neuron_idx] = np.maximum(
                    values[start_idx:end_idx, neuron_idx], display_intensity
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
    """Import a parsed SysCon2 sequence into an existing ``StimArray`` in place.

    ``dt_seconds`` is taken from ``stim_array.dt``; neurons are written starting
    at column ``neuron_offset``.  If ``extend_to_fit`` is ``True`` and the
    imported data is longer than the existing array, the array is extended.
    Excess timepoints are reported as truncated when ``extend_to_fit`` is ``False``.
    Replacement clears only mapped destination channels; other channels are preserved.
    """
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

    if result.mapped_objects == 0 or not np.any(imported_values):
        raise ValueError(
            "No stimulation events could be reconstructed; the target has not been changed."
        )
    _check_allocation(
        target_timepoints,
        n_existing_neurons,
        np.dtype(np.result_type(stim_array.values.dtype, np.float32)).itemsize,
    )
    target_values = np.zeros(
        (target_timepoints, n_existing_neurons),
        dtype=np.result_type(stim_array.values.dtype, np.float32),
    )
    target_values[:n_existing_timepoints, :] = stim_array.values
    # Replace only the mapped destination channels, preserving all other neurons.
    stop_neuron = neuron_offset + result.mapped_objects
    if replace_existing:
        target_values[:, neuron_offset:stop_neuron] = 0
    imported_values = imported_values[:, : result.mapped_objects]

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
