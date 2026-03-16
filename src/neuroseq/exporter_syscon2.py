from __future__ import annotations

import heapq
import re
from collections import OrderedDict
from collections import defaultdict
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
_DEFAULT_TTL_OUT_1_PORT = "Holo4D TTL Out 1"
_DEFAULT_TTL_OUT_2_PORT = "Holo4D TTL Out 2"

_SLM_LEAD_IN_US = 16_660
_ENTITY_MIN_DURATION_US = 33_320
_TTL_OUT_1_START_PULSE_US = 10_000

_SUPPORTED_LASER_WAVELENGTHS_NM = (940, 1040, 1140)
_DEFAULT_ACTIVE_LASER_WAVELENGTH_NM = 1140


@dataclass(frozen=True)
class _StimSegment:
    neuron_index: int
    start_idx: int
    duration_idx: int


@dataclass(frozen=True)
class _MappedSegment:
    neuron_index: int
    start_tick: int
    end_tick: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class _EntityWindow:
    neuron_index: int
    stimulation_start_tick: int
    stimulation_end_tick: int
    start_tick: int
    end_tick: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class _ScheduledEntityWindow:
    neuron_index: int
    stimulation_start_tick: int
    stimulation_end_tick: int
    start_tick: int
    end_tick: int
    center_x: float
    center_y: float
    lane: int


@dataclass(frozen=True)
class _ActiveSetRun:
    start_idx: int
    end_idx: int
    neuron_indices: tuple[int, ...]


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
    active_laser_wavelength_nm: int
    laser_timeline_index: int
    ttl_out1_timeline_index: int
    ttl_out2_timeline_index: int


@dataclass
class SysconInputSanityReport:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    export_report: SysconExportReport | None


@dataclass(frozen=True)
class SlmGapViolation:
    previous_start_idx: int
    previous_end_idx: int
    current_start_idx: int
    current_end_idx: int
    previous_neuron_indices: tuple[int, ...]
    current_neuron_indices: tuple[int, ...]
    gap_samples: int
    required_gap_samples: int


def generate_syscon2_sequence_text(
    stim_array: StimArray,
    cellpose_data: CellposeImportResult | None,
    *,
    template_text: str,
    tick_resolution_us: int = 500,
    active_threshold: float = 0.0,
    binary_intensity: float = 1.0,
    active_laser_wavelength_nm: int = _DEFAULT_ACTIVE_LASER_WAVELENGTH_NM,
) -> tuple[str, SysconExportReport]:
    """Generate SysCon2 `.seq` content from stim-array + Cellpose segmentation.

    Export model:
    - Binary mode: values > active_threshold are considered ON.
    - First timelines are entity lanes (N = max simultaneously stimulated neurons).
    - Next 3 timelines are fixed laser timelines (940, 1040, 1140 nm).
    - Next 2 timelines are TTL outputs:
      TTL Out 1 mirrors laser pulses + initial 10ms pulse, TTL Out 2 empty.
    - Entity objects include SLM lead-in and minimum placement duration constraints.
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")
    if binary_intensity < 0 or binary_intensity > 1:
        raise ValueError("binary_intensity must be within [0, 1]")
    if active_laser_wavelength_nm not in _SUPPORTED_LASER_WAVELENGTHS_NM:
        raise ValueError(
            "active_laser_wavelength_nm must be one of "
            f"{_SUPPORTED_LASER_WAVELENGTHS_NM}"
        )
    if cellpose_data is None:
        raise ValueError("Cellpose data is required to export SysCon2 .seq")
    if cellpose_data.segmentation_array is None:
        raise ValueError("Cellpose segmentation is required to export SysCon2 .seq")

    header_lines = _build_header_lines(template_text)
    (
        lightsource_id,
        lightsource_port,
        ttl_out_1_port,
        ttl_out_2_port,
    ) = _detect_port_info(template_text, header_lines)

    segments = _extract_active_segments(stim_array.values, active_threshold)
    active_neurons = sorted({segment.neuron_index for segment in segments})
    active_neurons_set = set(active_neurons)

    segmentation = cellpose_data.segmentation_array
    centroids = _compute_cell_centroids(segmentation)
    neuron_to_cellpose = _resolve_neuron_to_cellpose_mapping(
        cellpose_data,
        stim_array.values.shape[1],
        set(centroids.keys()),
    )

    sample_us = float(stim_array.dt * 1e6)
    lead_ticks = int(np.ceil(_SLM_LEAD_IN_US / float(tick_resolution_us)))
    min_entity_duration_ticks = int(
        np.ceil(_ENTITY_MIN_DURATION_US / float(tick_resolution_us))
    )

    mapped_segments: list[_MappedSegment] = []
    skipped_segments_unmapped = 0
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
        start_tick = (
            int(np.rint((segment.start_idx * sample_us) / float(tick_resolution_us)))
            + lead_ticks
        )
        duration_us = float(segment.duration_idx) * sample_us
        duration_ticks = max(1, int(np.ceil(duration_us / float(tick_resolution_us))))
        end_tick = start_tick + duration_ticks
        mapped_segments.append(
            _MappedSegment(
                neuron_index=segment.neuron_index,
                start_tick=start_tick,
                end_tick=end_tick,
                center_x=center_x,
                center_y=center_y,
            )
        )

    if len(mapped_segments) == 0:
        raise ValueError(
            "No mapped active stimulation segments to export. "
            "Check Cellpose mapping and stim activity."
        )

    pulse_intervals = _merge_tick_intervals(
        [(segment.start_tick, segment.end_tick) for segment in mapped_segments]
    )

    max_simultaneous = _max_simultaneous_segments(mapped_segments)
    entity_windows = _build_entity_windows(
        mapped_segments,
        pulse_intervals,
        lead_ticks=lead_ticks,
        min_duration_ticks=min_entity_duration_ticks,
    )
    scheduled_windows = _assign_entity_lanes(entity_windows, max_simultaneous)
    scheduled_windows.sort(
        key=lambda window: (window.start_tick, window.lane, window.neuron_index)
    )

    n_entity_lanes = max_simultaneous
    laser_timeline_base = n_entity_lanes
    laser_timeline_offsets = {940: 0, 1040: 1, 1140: 2}
    laser_timeline_index = (
        laser_timeline_base + laser_timeline_offsets[active_laser_wavelength_nm]
    )
    ttl_out1_timeline_index = laser_timeline_base + 3
    ttl_out2_timeline_index = laser_timeline_base + 4
    ttl_out_start_duration_ticks = max(
        1, int(np.ceil(_TTL_OUT_1_START_PULSE_US / float(tick_resolution_us)))
    )

    object_blocks: list[list[str]] = []
    next_object_id = 1

    for entity_counter, window in enumerate(scheduled_windows):
        duration_ticks = max(1, window.end_tick - window.start_tick)
        object_blocks.append(
            _format_object_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                object_type="Entity",
                properties=[
                    ("TotalTimings", "1"),
                    ("TimelineInfo0_TimelineIndex", str(window.lane)),
                    (
                        "TimelineInfo0_starttime",
                        str(window.start_tick * tick_resolution_us),
                    ),
                    ("TimelineInfo0_description", ""),
                    ("TimelineInfo0_repeats", str(duration_ticks - 1)),
                    ("TimelineInfo0_LightsourceCount", "1"),
                    ("TimelineInfo0_LightsourceID_0", lightsource_id),
                    ("TimelineInfo0_Intensity_0", _format_float(binary_intensity)),
                    ("TimelineInfo0_timelinegroupid", "-1"),
                    ("TimelineInfo0_Stepsize", "1"),
                    (f"Entity{entity_counter}_Type", "0"),
                    (f"Entity{entity_counter}_Filled", "False"),
                    (f"Entity{entity_counter}_CenterX", _format_float(window.center_x)),
                    (f"Entity{entity_counter}_CenterY", _format_float(window.center_y)),
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

    for start_tick, end_tick in pulse_intervals:
        duration_ticks = max(1, end_tick - start_tick)
        object_blocks.append(
            _format_ttl_pulse_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                timeline_index=laser_timeline_index,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                tick_resolution_us=tick_resolution_us,
                port=lightsource_port,
                lightsource_id=lightsource_id,
                include_lightsource=True,
            )
        )
        next_object_id += 1

    # TTL Out 1: fixed 10ms start pulse + mirrored laser pulses.
    object_blocks.append(
        _format_ttl_pulse_block(
            object_id=next_object_id,
            block_index=len(object_blocks),
            timeline_index=ttl_out1_timeline_index,
            start_tick=0,
            duration_ticks=ttl_out_start_duration_ticks,
            tick_resolution_us=tick_resolution_us,
            port=ttl_out_1_port,
            lightsource_id="",
            include_lightsource=False,
        )
    )
    next_object_id += 1
    for start_tick, end_tick in pulse_intervals:
        duration_ticks = max(1, end_tick - start_tick)
        object_blocks.append(
            _format_ttl_pulse_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                timeline_index=ttl_out1_timeline_index,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                tick_resolution_us=tick_resolution_us,
                port=ttl_out_1_port,
                lightsource_id="",
                include_lightsource=False,
            )
        )
        next_object_id += 1

    # TTL Out 2 intentionally remains empty.
    _ = ttl_out_2_port

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
        active_laser_wavelength_nm=active_laser_wavelength_nm,
        laser_timeline_index=laser_timeline_index,
        ttl_out1_timeline_index=ttl_out1_timeline_index,
        ttl_out2_timeline_index=ttl_out2_timeline_index,
    )
    return sequence_text, report


def check_syscon2_input_sanity(
    stim_array: StimArray,
    cellpose_data: CellposeImportResult | None,
    *,
    template_text: str,
    tick_resolution_us: int = 500,
    active_threshold: float = 0.0,
    binary_intensity: float = 1.0,
    active_laser_wavelength_nm: int = _DEFAULT_ACTIVE_LASER_WAVELENGTH_NM,
) -> SysconInputSanityReport:
    """Run a dry-run validation for SysCon2 export without writing files."""
    errors: list[str] = []
    warnings: list[str] = []

    values = stim_array.values
    if values.ndim != 2:
        errors.append("Stim array values must be 2D (timepoints x neurons).")
    elif values.shape[0] == 0 or values.shape[1] == 0:
        errors.append("Stim array must contain at least one timepoint and one neuron.")

    if stim_array.dt <= 0:
        errors.append("Stim array dt must be positive.")

    if values.size > 0 and not np.all(np.isfinite(values)):
        errors.append("Stim array contains non-finite values (NaN/Inf).")

    if values.size > 0:
        active_values = values[values > active_threshold]
        if active_values.size == 0:
            errors.append(
                "No active stimulation samples found above the active threshold."
            )
        if active_threshold == 0.0 and np.any(values < 0):
            warnings.append(
                "Negative stimulation values are exported as OFF "
                "(binary ON/OFF uses values > threshold)."
            )
        if active_values.size > 0 and not np.all(
            np.isclose(active_values, binary_intensity)
        ):
            warnings.append(
                "Binary export mode is enabled: all ON samples are exported "
                f"with fixed intensity {binary_intensity:g}."
            )

    if errors:
        return SysconInputSanityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            export_report=None,
        )

    try:
        _, export_report = generate_syscon2_sequence_text(
            stim_array,
            cellpose_data,
            template_text=template_text,
            tick_resolution_us=tick_resolution_us,
            active_threshold=active_threshold,
            binary_intensity=binary_intensity,
            active_laser_wavelength_nm=active_laser_wavelength_nm,
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return SysconInputSanityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            export_report=None,
        )

    if export_report.unmapped_active_neurons:
        warnings.append(
            "Some active neurons are unmapped and will be skipped: "
            + ", ".join(str(idx) for idx in export_report.unmapped_active_neurons)
        )

    return SysconInputSanityReport(
        is_valid=True,
        errors=[],
        warnings=warnings,
        export_report=export_report,
    )


def find_slm_gap_violations(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_gap_us: int = _SLM_LEAD_IN_US,
) -> list[SlmGapViolation]:
    """Find short-gap transitions between different active neuron sets.

    Consecutive stimulation runs with different active neuron sets must be at
    least `min_gap_us` apart to satisfy SLM hologram loading constraints.
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")
    if min_gap_us <= 0:
        raise ValueError("min_gap_us must be positive")

    required_gap_samples = int(np.ceil((min_gap_us * 1e-6) / float(stim_array.dt)))
    required_gap_samples = max(1, required_gap_samples)

    runs = _build_active_set_runs(stim_array.values, active_threshold)
    non_empty_runs = [run for run in runs if len(run.neuron_indices) > 0]

    violations: list[SlmGapViolation] = []
    for prev_run, current_run in zip(non_empty_runs, non_empty_runs[1:], strict=False):
        if prev_run.neuron_indices == current_run.neuron_indices:
            continue
        gap_samples = current_run.start_idx - prev_run.end_idx
        if gap_samples >= required_gap_samples:
            continue
        violations.append(
            SlmGapViolation(
                previous_start_idx=prev_run.start_idx,
                previous_end_idx=prev_run.end_idx,
                current_start_idx=current_run.start_idx,
                current_end_idx=current_run.end_idx,
                previous_neuron_indices=prev_run.neuron_indices,
                current_neuron_indices=current_run.neuron_indices,
                gap_samples=gap_samples,
                required_gap_samples=required_gap_samples,
            )
        )
    return violations


def space_out_stim_values_for_slm(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_gap_us: int = _SLM_LEAD_IN_US,
) -> tuple[np.ndarray, int]:
    """Right-shift tail segments to enforce SLM gap in-place (fixed length).

    Returns:
      (new_values, truncated_tail_samples)
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    values = np.array(stim_array.values, copy=True)
    n_timepoints = values.shape[0]
    truncated_tail_samples = 0

    max_iterations = max(1, n_timepoints * 2)
    for _ in range(max_iterations):
        tmp_stim_array = StimArray(
            name=stim_array.name,
            stim_type=stim_array.stim_type,
            values=values,
            units=stim_array.units,
            dt=stim_array.dt,
        )
        violations = find_slm_gap_violations(
            tmp_stim_array,
            active_threshold=active_threshold,
            min_gap_us=min_gap_us,
        )
        if len(violations) == 0:
            break

        violation = violations[0]
        shift_samples = violation.required_gap_samples - violation.gap_samples
        if shift_samples <= 0:
            break

        shift_start = violation.current_start_idx
        tail_before_shift = values[shift_start:, :].copy()
        values[shift_start:, :] = 0
        write_start = shift_start + shift_samples
        if write_start < n_timepoints:
            writable_rows = n_timepoints - write_start
            values[write_start:, :] = tail_before_shift[:writable_rows, :]
        truncated_tail_samples += min(shift_samples, tail_before_shift.shape[0])

    return values, truncated_tail_samples


def align_stim_values_left_for_slm(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_gap_us: int = _SLM_LEAD_IN_US,
) -> tuple[np.ndarray, int]:
    """Left-align violating runs to previous run start while preserving length.

    Returns:
      (new_values, moved_run_count)
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    values = np.array(stim_array.values, copy=True)
    n_timepoints = values.shape[0]
    moved_run_count = 0

    max_iterations = max(1, n_timepoints * 2)
    for _ in range(max_iterations):
        tmp_stim_array = StimArray(
            name=stim_array.name,
            stim_type=stim_array.stim_type,
            values=values,
            units=stim_array.units,
            dt=stim_array.dt,
        )
        violations = find_slm_gap_violations(
            tmp_stim_array,
            active_threshold=active_threshold,
            min_gap_us=min_gap_us,
        )
        if len(violations) == 0:
            break

        violation = violations[0]
        source_start = violation.current_start_idx
        source_end = violation.current_end_idx
        target_start = violation.previous_start_idx
        if source_end <= source_start or source_start <= target_start:
            break

        block = values[source_start:source_end, :].copy()
        if not np.any(block):
            values[source_start:source_end, :] = 0
            continue

        target_end = target_start + block.shape[0]
        target_existing = values[target_start:target_end, :].copy()

        values[source_start:source_end, :] = 0
        values[target_start:target_end, :] = np.maximum(target_existing, block)
        moved_run_count += 1

    return values, moved_run_count


def _build_active_set_runs(
    values: np.ndarray,
    active_threshold: float,
) -> list[_ActiveSetRun]:
    if values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    n_timepoints = values.shape[0]
    if n_timepoints == 0:
        return []

    active = values > active_threshold
    runs: list[_ActiveSetRun] = []
    run_start = 0
    run_neuron_indices = tuple(np.flatnonzero(active[0]).tolist())

    for idx in range(1, n_timepoints):
        current_indices = tuple(np.flatnonzero(active[idx]).tolist())
        if current_indices == run_neuron_indices:
            continue
        runs.append(
            _ActiveSetRun(
                start_idx=run_start,
                end_idx=idx,
                neuron_indices=run_neuron_indices,
            )
        )
        run_start = idx
        run_neuron_indices = current_indices

    runs.append(
        _ActiveSetRun(
            start_idx=run_start,
            end_idx=n_timepoints,
            neuron_indices=run_neuron_indices,
        )
    )
    return runs


def _extract_active_segments(
    values: np.ndarray, threshold: float
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


def _merge_tick_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(intervals) == 0:
        return []
    intervals_sorted = sorted(
        intervals, key=lambda interval: (interval[0], interval[1])
    )
    merged: list[tuple[int, int]] = []
    current_start, current_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _max_simultaneous_segments(segments: list[_MappedSegment]) -> int:
    timeline_events: dict[int, int] = defaultdict(int)
    for segment in segments:
        timeline_events[segment.start_tick] += 1
        timeline_events[segment.end_tick] -= 1
    concurrent = 0
    max_concurrent = 0
    for tick in sorted(timeline_events.keys()):
        concurrent += timeline_events[tick]
        max_concurrent = max(max_concurrent, concurrent)
    return max(1, max_concurrent)


def _build_entity_windows(
    mapped_segments: list[_MappedSegment],
    pulse_intervals: list[tuple[int, int]],
    *,
    lead_ticks: int,
    min_duration_ticks: int,
) -> list[_EntityWindow]:
    segments_by_neuron: dict[int, list[_MappedSegment]] = defaultdict(list)
    for segment in mapped_segments:
        segments_by_neuron[segment.neuron_index].append(segment)

    windows: list[_EntityWindow] = []
    for neuron_idx, neuron_segments in segments_by_neuron.items():
        sorted_segments = sorted(
            neuron_segments, key=lambda segment: (segment.start_tick, segment.end_tick)
        )
        current_start = sorted_segments[0].start_tick
        current_end = sorted_segments[0].end_tick
        center_x = sorted_segments[0].center_x
        center_y = sorted_segments[0].center_y

        for segment in sorted_segments[1:]:
            if _interval_overlaps_any(current_end, segment.start_tick, pulse_intervals):
                entity_start = max(0, current_start - lead_ticks)
                entity_end = max(current_end, entity_start + min_duration_ticks)
                windows.append(
                    _EntityWindow(
                        neuron_index=neuron_idx,
                        stimulation_start_tick=current_start,
                        stimulation_end_tick=current_end,
                        start_tick=entity_start,
                        end_tick=entity_end,
                        center_x=center_x,
                        center_y=center_y,
                    )
                )
                current_start = segment.start_tick
                current_end = segment.end_tick
                center_x = segment.center_x
                center_y = segment.center_y
                continue
            current_end = max(current_end, segment.end_tick)

        entity_start = max(0, current_start - lead_ticks)
        entity_end = max(current_end, entity_start + min_duration_ticks)
        windows.append(
            _EntityWindow(
                neuron_index=neuron_idx,
                stimulation_start_tick=current_start,
                stimulation_end_tick=current_end,
                start_tick=entity_start,
                end_tick=entity_end,
                center_x=center_x,
                center_y=center_y,
            )
        )
    return windows


def _interval_overlaps_any(
    interval_start: int,
    interval_end: int,
    sorted_intervals: list[tuple[int, int]],
) -> bool:
    if interval_end <= interval_start:
        return False
    for start, end in sorted_intervals:
        if end <= interval_start:
            continue
        return not start >= interval_end
    return False


def _assign_entity_lanes(
    windows: list[_EntityWindow],
    lane_count: int,
) -> list[_ScheduledEntityWindow]:
    if lane_count <= 0:
        raise ValueError("lane_count must be positive")
    windows_sorted = sorted(
        windows,
        key=lambda window: (
            window.stimulation_start_tick,
            window.stimulation_end_tick,
            window.neuron_index,
        ),
    )
    available_lanes = list(range(lane_count))
    heapq.heapify(available_lanes)
    active_lanes: list[tuple[int, int]] = []

    scheduled: list[_ScheduledEntityWindow] = []
    for window in windows_sorted:
        while (
            len(active_lanes) > 0
            and active_lanes[0][0] <= window.stimulation_start_tick
        ):
            _, released_lane = heapq.heappop(active_lanes)
            heapq.heappush(available_lanes, released_lane)

        if len(available_lanes) == 0:
            raise ValueError(
                "Could not schedule entity objects within the configured lane count. "
                "The stimulation pattern is too dense for the computed simultaneous-neuron count."
            )

        lane = heapq.heappop(available_lanes)
        heapq.heappush(active_lanes, (window.stimulation_end_tick, lane))
        scheduled.append(
            _ScheduledEntityWindow(
                neuron_index=window.neuron_index,
                stimulation_start_tick=window.stimulation_start_tick,
                stimulation_end_tick=window.stimulation_end_tick,
                start_tick=window.start_tick,
                end_tick=window.end_tick,
                center_x=window.center_x,
                center_y=window.center_y,
                lane=lane,
            )
        )
    return scheduled


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


def _detect_port_info(
    template_text: str,
    header_lines: list[str],
) -> tuple[str, str, str, str]:
    lightsource_id = ""
    lightsource_port = ""
    ttl_out_1_port = ""
    ttl_out_2_port = ""

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
        lowered = value.lower()
        if "ttl out 1" in lowered and not ttl_out_1_port:
            ttl_out_1_port = value
        if "ttl out 2" in lowered and not ttl_out_2_port:
            ttl_out_2_port = value
        if lightsource_id in value and not lightsource_port:
            lightsource_port = value

    if not lightsource_port:
        source_port = parsed_header.get("SourceConnectorPort0", "").strip()
        if source_port:
            lightsource_port = source_port
    if not lightsource_port:
        lightsource_port = f"{lightsource_id} RMI In"

    if not ttl_out_1_port:
        for key, value in parsed_header.items():
            if "TTL Out 1" in value and key.startswith("TargetConnectorPort"):
                ttl_out_1_port = value
                break
    if not ttl_out_2_port:
        for key, value in parsed_header.items():
            if "TTL Out 2" in value and key.startswith("TargetConnectorPort"):
                ttl_out_2_port = value
                break

    if ttl_out_1_port.strip() == "":
        ttl_out_1_port = _DEFAULT_TTL_OUT_1_PORT
    if ttl_out_2_port.strip() == "":
        ttl_out_2_port = _DEFAULT_TTL_OUT_2_PORT
    if lightsource_port.strip() == "":
        lightsource_port = _DEFAULT_LIGHTSOURCE_PORT

    return lightsource_id, lightsource_port, ttl_out_1_port, ttl_out_2_port


def _format_ttl_pulse_block(
    *,
    object_id: int,
    block_index: int,
    timeline_index: int,
    start_tick: int,
    duration_ticks: int,
    tick_resolution_us: int,
    port: str,
    lightsource_id: str,
    include_lightsource: bool,
) -> list[str]:
    properties: list[tuple[str, str]] = [
        ("TotalTimings", "1"),
        ("TimelineInfo0_TimelineIndex", str(timeline_index)),
        ("TimelineInfo0_starttime", str(start_tick * tick_resolution_us)),
        ("TimelineInfo0_description", ""),
        ("TimelineInfo0_repeats", str(max(0, duration_ticks - 1))),
        ("TimelineInfo0_LightsourceCount", "1" if include_lightsource else "0"),
    ]
    if include_lightsource:
        properties.append(("TimelineInfo0_LightsourceID_0", lightsource_id))
        properties.append(("TimelineInfo0_Intensity_0", "0"))
    properties.extend(
        [
            ("TimelineInfo0_timelinegroupid", "-1"),
            ("TimelineInfo0_Stepsize", "100"),
            ("port", port),
        ]
    )
    return _format_object_block(
        object_id=object_id,
        block_index=block_index,
        object_type="TTLPulse",
        properties=properties,
    )


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
