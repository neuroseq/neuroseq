from __future__ import annotations

import heapq
import pathlib
import re
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_syscon2 import SysconSequenceObject
from neuroseq.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.stim import StimArray

_BLOCK_START_RE = re.compile(r"^\[(\d+)\]$")
_HEADER_HASH_RE = re.compile(r"^[0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){15}$")

_REQUIRED_HEADER_DEFAULTS: dict[str, str] = {
    "version": "3",
    "scanmode": "Accurate",
    "runs": "1",
    "invertedportcnt": "1",
    "AutoTriggerRulesEnabled": "False",
    "AutoTriggerRulesCount": "1",
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
_SYSCON2_TTL_PULSE_QUANTUM_US = 520.8125
_SYSCON2_NEURON_OBJECT_QUANTUM_US = 16_666.0
_TTL_PULSE_BOUNDARY_GUARD_TICKS = 1
_SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS = int(
    round(_SYSCON2_NEURON_OBJECT_QUANTUM_US / _SYSCON2_TTL_PULSE_QUANTUM_US)
)

_SUPPORTED_LASER_WAVELENGTHS_NM = (940, 1030, 1140)
_LEGACY_LASER_WAVELENGTH_ALIASES_NM = {1040: 1030}
_DEFAULT_ACTIVE_LASER_WAVELENGTH_NM = 1140
_LASER_TIMELINE_OFFSETS = {940: 0, 1030: 1, 1140: 2}


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
    intensity: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class _EntityWindow:
    neuron_index: int
    stimulation_start_tick: int
    stimulation_end_tick: int
    start_tick: int
    end_tick: int
    intensity: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class _ScheduledEntityWindow:
    neuron_index: int
    stimulation_start_tick: int
    stimulation_end_tick: int
    start_tick: int
    end_tick: int
    intensity: float
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


@dataclass(frozen=True)
class SlmOffOnOverlapViolation:
    phase_start_idx: int
    phase_start_tick: int
    off_neuron_index: int
    on_neuron_index: int
    overlap_start_tick: int
    overlap_end_tick: int
    required_shift_samples: int
    required_shift_ticks: int


@dataclass(frozen=True)
class InitialSlmLoadViolation:
    first_active_idx: int
    required_start_idx: int


@dataclass(frozen=True)
class SimultaneousDurationViolation:
    start_idx: int
    neuron_indices: tuple[int, ...]
    durations_idx: tuple[int, ...]
    min_duration_idx: int


@dataclass(frozen=True)
class GlobalSlmWindowViolation:
    neuron_index: int
    start_idx: int
    duration_idx: int
    start_tick: int
    snapped_start_tick: int
    shift_ticks: int
    shift_samples: int


def _normalize_active_laser_wavelength_nm(wavelength_nm: int) -> int:
    normalized = int(wavelength_nm)
    return _LEGACY_LASER_WAVELENGTH_ALIASES_NM.get(normalized, normalized)


def generate_syscon2_sequence_text(
    stim_array: StimArray,
    cellpose_data: CellposeImportResult | None,
    *,
    template_text: str,
    tick_resolution_us: int = 500,
    active_threshold: float = 0.0,
    binary_intensity: float = 1.0,
    force_binary_intensity: bool = True,
    active_laser_wavelength_nm: int = _DEFAULT_ACTIVE_LASER_WAVELENGTH_NM,
) -> tuple[str, SysconExportReport]:
    """Generate SysCon2 `.seq` content from stim-array + Cellpose segmentation.

    Export model:
    - Binary mode: values > active_threshold are considered ON.
    - First timelines are neuron lanes (N = max simultaneously stimulated neurons).
    - Next 3 timelines are fixed laser timelines (940, 1030, 1140 nm).
    - Next 2 timelines are TTL outputs:
      TTL Out 1 mirrors laser pulses + initial 10ms pulse, TTL Out 2 empty.
    - Shape objects include SLM lead-in and 16.666ms-quantized placement constraints.
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")
    if force_binary_intensity and (binary_intensity < 0 or binary_intensity > 1):
        raise ValueError("binary_intensity must be within [0, 1]")
    active_laser_wavelength_nm = _normalize_active_laser_wavelength_nm(
        active_laser_wavelength_nm
    )
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
    _ = tick_resolution_us
    lead_ticks = int(np.ceil(_SLM_LEAD_IN_US / _SYSCON2_TTL_PULSE_QUANTUM_US))
    min_entity_duration_ticks = int(
        np.ceil(_ENTITY_MIN_DURATION_US / _SYSCON2_TTL_PULSE_QUANTUM_US)
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
        # Deround GUI timing (ms grid) back to SysCon2 pulse quantum units.
        start_tick = int(
            np.rint((segment.start_idx * sample_us) / _SYSCON2_TTL_PULSE_QUANTUM_US)
        )
        duration_us = float(segment.duration_idx) * sample_us
        duration_ticks = max(
            1, int(np.rint(duration_us / _SYSCON2_TTL_PULSE_QUANTUM_US))
        )
        end_tick = start_tick + duration_ticks
        if force_binary_intensity:
            export_intensity = binary_intensity
        else:
            segment_values = stim_array.values[
                segment.start_idx : segment.start_idx + segment.duration_idx,
                segment.neuron_index,
            ]
            export_intensity = float(np.max(segment_values))
            export_intensity = float(np.clip(export_intensity, 0.0, 1.0))
        mapped_segments.append(
            _MappedSegment(
                neuron_index=segment.neuron_index,
                start_tick=start_tick,
                end_tick=end_tick,
                intensity=export_intensity,
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
        lead_ticks=lead_ticks,
        min_duration_ticks=min_entity_duration_ticks,
        pulse_guard_ticks=_TTL_PULSE_BOUNDARY_GUARD_TICKS,
    )
    _validate_pulse_bounds_within_entity_windows(
        entity_windows,
        lead_ticks=lead_ticks,
        pulse_guard_ticks=_TTL_PULSE_BOUNDARY_GUARD_TICKS,
    )
    _validate_no_off_on_overlap(
        entity_windows,
        lead_ticks=lead_ticks,
        pulse_guard_ticks=_TTL_PULSE_BOUNDARY_GUARD_TICKS,
    )
    scheduled_windows = _assign_entity_lanes(entity_windows, max_simultaneous)
    scheduled_windows.sort(
        key=lambda window: (window.start_tick, window.lane, window.neuron_index)
    )
    _validate_laser_pulse_end_before_overlapping_shapes(
        pulse_intervals,
        scheduled_windows,
        pulse_guard_ticks=_TTL_PULSE_BOUNDARY_GUARD_TICKS,
    )

    n_entity_lanes = max_simultaneous
    laser_timeline_base = n_entity_lanes
    laser_timeline_index = (
        laser_timeline_base + _LASER_TIMELINE_OFFSETS[active_laser_wavelength_nm]
    )
    ttl_out1_timeline_index = laser_timeline_base + 3
    ttl_out2_timeline_index = laser_timeline_base + 4
    ttl_out_start_duration_ticks = max(
        1, int(np.rint(_TTL_OUT_1_START_PULSE_US / _SYSCON2_TTL_PULSE_QUANTUM_US))
    )

    object_blocks: list[list[str]] = []
    next_object_id = 1

    for shape_counter, window in enumerate(scheduled_windows):
        duration_ticks = max(1, window.end_tick - window.start_tick)
        duration_shape_quanta = max(
            2,
            int(
                np.ceil(
                    duration_ticks / float(_SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS)
                )
            ),
        )
        object_blocks.append(
            _format_object_block(
                object_id=next_object_id,
                block_index=len(object_blocks),
                object_type="Shape",
                properties=[
                    ("TotalTimings", "1"),
                    ("TimelineInfo0_TimelineIndex", str(window.lane)),
                    (
                        "TimelineInfo0_starttime",
                        _format_time_us(
                            window.start_tick * _SYSCON2_TTL_PULSE_QUANTUM_US
                        ),
                    ),
                    ("TimelineInfo0_description", ""),
                    (
                        "TimelineInfo0_repeats",
                        str(_shape_repeats_from_duration_quanta(duration_shape_quanta)),
                    ),
                    ("TimelineInfo0_LightsourceCount", "1"),
                    ("TimelineInfo0_LightsourceID_0", lightsource_id),
                    ("TimelineInfo0_Intensity_0", _format_float(window.intensity)),
                    ("TimelineInfo0_timelinegroupid", "-1"),
                    ("TimelineInfo0_Stepsize", "100"),
                    (f"Shape{shape_counter}_Type", "0"),
                    (f"Shape{shape_counter}_Filled", "False"),
                    (f"Shape{shape_counter}_CenterX", _format_float(window.center_x)),
                    (f"Shape{shape_counter}_CenterY", _format_float(window.center_y)),
                    (f"Shape{shape_counter}_CenterZ", "0"),
                    (f"Shape{shape_counter}_Rotation", "0"),
                    (f"Shape{shape_counter}_Scale", "1"),
                    (f"Shape{shape_counter}_VerticesCount", "0"),
                    (f"Shape{shape_counter}_VerticesToDrawCount", "0"),
                    (f"Shape{shape_counter}_TranslationVerticesCount", "0"),
                    (f"Shape{shape_counter}_Reversed", "False"),
                    (f"Shape{shape_counter}_GroupID", "-1"),
                    (f"Shape{shape_counter}_Linewidth", "1"),
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


def generate_syscon2_timeline_overview_png(
    sequence_text: str,
    output_path: str | pathlib.Path,
    *,
    tick_resolution_us: int = 500,
) -> pathlib.Path:
    """Render a visual timeline overview PNG for a SysCon2 `.seq` text."""
    if tick_resolution_us <= 0:
        raise ValueError("tick_resolution_us must be positive")

    sequence = parse_syscon2_seq_text(sequence_text)

    timeline_entries: list[tuple[int, float, float, str]] = []
    timeline_objects: dict[int, list[SysconSequenceObject]] = defaultdict(list)
    for obj in sequence.objects:
        timeline_index_raw = obj.properties.get("TimelineInfo0_TimelineIndex")
        if timeline_index_raw is None:
            continue
        try:
            timeline_index = int(float(timeline_index_raw))
        except ValueError:
            continue
        timeline_objects[timeline_index].append(obj)

        start_us = max(0.0, float(obj.start_time_us))
        duration_us = _preview_object_duration_us(
            obj,
            tick_resolution_us=tick_resolution_us,
        )
        timeline_entries.append(
            (
                timeline_index,
                start_us,
                max(1e-6, duration_us),
                obj.object_type.strip().lower(),
            )
        )

    if len(timeline_entries) == 0:
        raise ValueError("No timeline objects found to render overview PNG.")

    timeline_indices = sorted({entry[0] for entry in timeline_entries})
    timeline_labels = _build_timeline_axis_labels(
        timeline_indices,
        timeline_objects=timeline_objects,
    )
    y_by_timeline = {timeline: idx for idx, timeline in enumerate(timeline_indices)}
    max_end_us = max(
        start_us + duration_us for _, start_us, duration_us, _ in timeline_entries
    )
    max_end_ms = max(1.0, max_end_us / 1e3)
    fig_height = max(3.0, 1.4 + 0.45 * len(timeline_indices))
    fig_width = _compute_overview_figure_width_in(max_end_ms)
    render_dpi = _compute_overview_render_dpi(
        fig_width_inches=fig_width,
        fig_height_inches=fig_height,
    )
    min_visible_duration_ms = _compute_overview_min_visible_duration_ms(
        max_end_ms=max_end_ms,
        fig_width_inches=fig_width,
        dpi=render_dpi,
    )
    x_axis_scale, x_axis_label = _select_overview_time_axis(max_end_ms)
    render_entries: list[tuple[int, float, float, float, str]] = []
    max_display_end_ms = 1.0
    for timeline_idx, start_us, duration_us, object_type in timeline_entries:
        start_ms = start_us / 1e3
        duration_ms = duration_us / 1e3
        display_duration_ms = max(duration_ms, min_visible_duration_ms)
        render_entries.append(
            (timeline_idx, start_ms, duration_ms, display_duration_ms, object_type)
        )
        max_display_end_ms = max(max_display_end_ms, start_ms + display_duration_ms)
    x_max = max(1.0, max_display_end_ms * x_axis_scale)

    # Use Agg canvas directly for headless rendering in tests/servers.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import StrMethodFormatter

    fig = Figure(figsize=(fig_width, fig_height), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for timeline_idx in timeline_indices:
        y = y_by_timeline[timeline_idx]
        ax.hlines(y, 0.0, x_max, colors="#d0d0d0", linewidth=0.8, zorder=0)

    for (
        timeline_idx,
        start_ms,
        _duration_ms,
        display_duration_ms,
        object_type,
    ) in render_entries:
        y = y_by_timeline[timeline_idx]
        start_x = start_ms * x_axis_scale
        display_duration_x = display_duration_ms * x_axis_scale
        color = _preview_object_color(object_type)
        rect = Rectangle(
            (start_x, y - 0.32),
            display_duration_x,
            0.64,
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(rect)
        if object_type in {"shape", "entity"}:
            slm_lead_in_x = (_SLM_LEAD_IN_US / 1e3) * x_axis_scale
            lead_in_width_x = min(display_duration_x, slm_lead_in_x)
            if lead_in_width_x > 0:
                lead_in_rect = Rectangle(
                    (start_x, y - 0.32),
                    lead_in_width_x,
                    0.64,
                    facecolor="#bdbdbd",
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.95,
                    zorder=3,
                )
                ax.add_patch(lead_in_rect)

    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_ylim(-0.8, len(timeline_indices) - 0.2)
    ax.set_yticks([y_by_timeline[idx] for idx in timeline_indices])
    ax.set_yticklabels([timeline_labels[idx] for idx in timeline_indices])
    ax.invert_yaxis()
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Timeline")
    ax.set_title(
        f"SysCon2 Timeline Overview ({_format_overview_duration_label(max_end_ms)})"
    )
    y_tick_fontsize = float(
        np.clip(10.5 - 0.12 * max(0, len(timeline_indices) - 8), 7.0, 10.5)
    )
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)
    ax.tick_params(axis="x", labelsize=9.0)
    major_tick_count = int(np.clip(np.rint(fig_width * 0.9), 6, 14))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=major_tick_count, min_n_ticks=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    tick_decimals = _overview_tick_decimals(x_max)
    ax.xaxis.set_major_formatter(StrMethodFormatter(f"{{x:.{tick_decimals}f}}"))
    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.65, alpha=0.65)
    ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.35, alpha=0.35)
    if min_visible_duration_ms > 0.15:
        ax.text(
            0.995,
            -0.13,
            (
                "Short events are expanded to "
                f">= {min_visible_duration_ms:.2f} ms for readability."
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color="#545454",
        )

    legend_handles = [
        Patch(facecolor="#2c7fb8", edgecolor="#2c7fb8", label="Shape/Entity"),
        Patch(facecolor="#bdbdbd", edgecolor="#bdbdbd", label="SLM load-up (16.66ms)"),
        Patch(facecolor="#d95f0e", edgecolor="#d95f0e", label="TTLPulse"),
        Patch(facecolor="#636363", edgecolor="#636363", label="Other"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    output_path_obj = pathlib.Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=render_dpi)
    fig.clear()
    return output_path_obj


def check_syscon2_input_sanity(
    stim_array: StimArray,
    cellpose_data: CellposeImportResult | None,
    *,
    template_text: str,
    tick_resolution_us: int = 500,
    active_threshold: float = 0.0,
    binary_intensity: float = 1.0,
    force_binary_intensity: bool = True,
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
        initial_slm_load_violation = find_initial_slm_load_violation(
            stim_array,
            active_threshold=active_threshold,
            min_start_us=_SLM_LEAD_IN_US,
        )
        if initial_slm_load_violation is not None:
            first_start_ms = (
                initial_slm_load_violation.first_active_idx * stim_array.dt * 1e3
            )
            required_start_ms = (
                initial_slm_load_violation.required_start_idx * stim_array.dt * 1e3
            )
            errors.append(
                "First stimulation starts too early for SLM initialization: "
                f"{first_start_ms:.2f}ms (minimum {required_start_ms:.2f}ms)."
            )
        off_on_overlap_violations = find_slm_off_on_overlap_violations(
            stim_array,
            active_threshold=active_threshold,
            min_gap_us=_SLM_LEAD_IN_US,
        )
        if off_on_overlap_violations:
            first_overlap = off_on_overlap_violations[0]
            required_shift_ms = (
                first_overlap.required_shift_samples * stim_array.dt * 1e3
            )
            overlap_start_ms = (
                first_overlap.overlap_start_tick * _SYSCON2_TTL_PULSE_QUANTUM_US / 1e3
            )
            overlap_end_ms = (
                first_overlap.overlap_end_tick * _SYSCON2_TTL_PULSE_QUANTUM_US / 1e3
            )
            errors.append(
                "SLM off/on overlap detected: "
                f"off-neuron {first_overlap.off_neuron_index} overlaps ON interval of "
                f"neuron {first_overlap.on_neuron_index} in "
                f"[{overlap_start_ms:.2f}, {overlap_end_ms:.2f})ms. "
                f"Space input right by at least {required_shift_ms:.2f}ms."
            )
        slm_gap_violations = find_slm_gap_violations(
            stim_array,
            active_threshold=active_threshold,
            min_gap_us=_SLM_LEAD_IN_US,
        )
        if slm_gap_violations:
            first_violation = slm_gap_violations[0]
            gap_ms = first_violation.gap_samples * stim_array.dt * 1e3
            required_gap_ms = first_violation.required_gap_samples * stim_array.dt * 1e3
            errors.append(
                "SLM new-target gap violation detected: "
                f"{first_violation.previous_neuron_indices} -> "
                f"{first_violation.current_neuron_indices}, gap={gap_ms:.2f}ms "
                f"(required {required_gap_ms:.2f}ms)."
            )
        global_window_violations = find_global_slm_window_violations(
            stim_array,
            active_threshold=active_threshold,
        )
        if global_window_violations:
            first_global_window_violation = global_window_violations[0]
            shift_ms = (
                first_global_window_violation.shift_ticks
                * _SYSCON2_TTL_PULSE_QUANTUM_US
                / 1e3
            )
            warnings.append(
                "Global 16.66ms alignment warning: "
                f"{len(global_window_violations)} stimulation start(s) are off the "
                "global SLM window grid. "
                f"Example nearest correction: {shift_ms:+.3f}ms."
            )
        if force_binary_intensity:
            if active_values.size > 0 and not np.all(
                np.isclose(active_values, binary_intensity)
            ):
                warnings.append(
                    "Binary export mode is enabled: all ON samples are exported "
                    f"with fixed intensity {binary_intensity:g}."
                )
        else:
            if active_values.size > 0 and (
                np.any(active_values < 0.0) or np.any(active_values > 1.0)
            ):
                warnings.append(
                    "Non-binary export mode preserves ON amplitudes but clips values "
                    "outside [0, 1] to match SysCon2 intensity bounds."
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
            force_binary_intensity=force_binary_intensity,
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
    """Find short-gap transitions where a new neuron appears.

    Consecutive non-empty stimulation runs where the current run introduces at
    least one neuron not present in the previous run must be at least
    `min_gap_us` apart to satisfy SLM hologram loading constraints.
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
        previous_neurons = set(prev_run.neuron_indices)
        current_neurons = set(current_run.neuron_indices)
        if len(current_neurons - previous_neurons) == 0:
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


def find_slm_off_on_overlap_violations(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_gap_us: int = _SLM_LEAD_IN_US,
) -> list[SlmOffOnOverlapViolation]:
    """Find SLM off/on overlap violations implied by current timing.

    Whenever a new target phase starts, every active object in that phase has an
    OFF interval of `min_gap_us` before becoming usable. That OFF interval must
    not overlap the ON interval of any other neuron's object.
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")
    if min_gap_us <= 0:
        raise ValueError("min_gap_us must be positive")

    mapped_segments = _build_mapped_segments_for_timing(
        stim_array,
        active_threshold=active_threshold,
    )
    if len(mapped_segments) == 0:
        return []

    lead_ticks = int(np.ceil(min_gap_us / _SYSCON2_TTL_PULSE_QUANTUM_US))
    min_entity_duration_ticks = int(
        np.ceil(_ENTITY_MIN_DURATION_US / _SYSCON2_TTL_PULSE_QUANTUM_US)
    )
    windows = _build_entity_windows(
        mapped_segments,
        lead_ticks=lead_ticks,
        min_duration_ticks=min_entity_duration_ticks,
        pulse_guard_ticks=_TTL_PULSE_BOUNDARY_GUARD_TICKS,
    )
    if len(windows) <= 1:
        return []

    sample_us = float(stim_array.dt * 1e6)
    pulse_guard_ticks = _TTL_PULSE_BOUNDARY_GUARD_TICKS
    by_phase_start: dict[int, SlmOffOnOverlapViolation] = {}

    for off_window in windows:
        phase_start_tick = off_window.start_tick + lead_ticks + pulse_guard_ticks
        off_start_tick = off_window.start_tick
        off_end_tick = min(off_window.end_tick, phase_start_tick)
        if off_end_tick <= off_start_tick:
            continue

        for on_window in windows:
            if off_window is on_window:
                continue
            if off_window.neuron_index == on_window.neuron_index:
                continue

            on_start_tick = max(
                on_window.start_tick + lead_ticks + pulse_guard_ticks,
                on_window.start_tick,
            )
            on_end_tick = on_window.end_tick
            if on_end_tick <= on_start_tick:
                continue
            if off_end_tick <= on_start_tick or on_end_tick <= off_start_tick:
                continue

            overlap_start_tick = max(off_start_tick, on_start_tick)
            overlap_end_tick = min(off_end_tick, on_end_tick)
            required_phase_start_tick = on_end_tick + lead_ticks + pulse_guard_ticks
            required_shift_ticks = max(1, required_phase_start_tick - phase_start_tick)
            required_shift_samples = max(
                1,
                int(
                    np.ceil(
                        required_shift_ticks * _SYSCON2_TTL_PULSE_QUANTUM_US / sample_us
                    )
                ),
            )
            phase_start_idx = max(
                0,
                int(
                    np.rint(
                        phase_start_tick * _SYSCON2_TTL_PULSE_QUANTUM_US / sample_us
                    )
                ),
            )

            candidate = SlmOffOnOverlapViolation(
                phase_start_idx=phase_start_idx,
                phase_start_tick=phase_start_tick,
                off_neuron_index=off_window.neuron_index,
                on_neuron_index=on_window.neuron_index,
                overlap_start_tick=overlap_start_tick,
                overlap_end_tick=overlap_end_tick,
                required_shift_samples=required_shift_samples,
                required_shift_ticks=required_shift_ticks,
            )
            previous = by_phase_start.get(phase_start_tick)
            if (
                previous is None
                or candidate.required_shift_ticks > previous.required_shift_ticks
            ):
                by_phase_start[phase_start_tick] = candidate

    return sorted(
        by_phase_start.values(),
        key=lambda violation: (
            violation.phase_start_idx,
            violation.phase_start_tick,
            violation.off_neuron_index,
            violation.on_neuron_index,
        ),
    )


def find_initial_slm_load_violation(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_start_us: int = _SLM_LEAD_IN_US,
) -> InitialSlmLoadViolation | None:
    """Check whether first active stimulation starts before SLM load-up window."""
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")
    if min_start_us <= 0:
        raise ValueError("min_start_us must be positive")

    active_any = np.any(stim_array.values > active_threshold, axis=1)
    active_indices = np.flatnonzero(active_any)
    if active_indices.size == 0:
        return None

    first_active_idx = int(active_indices[0])
    required_start_idx = int(np.ceil((min_start_us * 1e-6) / float(stim_array.dt)))
    required_start_idx = max(1, required_start_idx)
    if first_active_idx >= required_start_idx:
        return None

    return InitialSlmLoadViolation(
        first_active_idx=first_active_idx,
        required_start_idx=required_start_idx,
    )


def shift_stim_values_right_for_initial_slm_load(
    stim_array: StimArray,
    *,
    min_start_us: int = _SLM_LEAD_IN_US,
) -> tuple[np.ndarray, int]:
    """Translate entire stim array to the right by the SLM load-up window.

    Returns:
      (new_values, truncated_tail_timepoints)
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")
    if min_start_us <= 0:
        raise ValueError("min_start_us must be positive")

    values = np.array(stim_array.values, copy=True)
    n_timepoints = values.shape[0]
    shift_samples = int(np.ceil((min_start_us * 1e-6) / float(stim_array.dt)))
    shift_samples = max(1, shift_samples)

    shifted = np.zeros_like(values)
    if shift_samples < n_timepoints:
        shifted[shift_samples:, :] = values[: n_timepoints - shift_samples, :]

    truncated_tail_timepoints = min(shift_samples, n_timepoints)
    return shifted, truncated_tail_timepoints


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


def space_out_stim_values_for_slm_off_on_overlap(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    min_gap_us: int = _SLM_LEAD_IN_US,
) -> tuple[np.ndarray, int]:
    """Right-shift tail segments to eliminate SLM off/on overlap violations.

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
        violations = find_slm_off_on_overlap_violations(
            tmp_stim_array,
            active_threshold=active_threshold,
            min_gap_us=min_gap_us,
        )
        if len(violations) == 0:
            break

        violation = violations[0]
        shift_samples = violation.required_shift_samples
        shift_start = violation.phase_start_idx
        if shift_samples <= 0 or shift_start >= n_timepoints:
            break

        tail_before_shift = values[shift_start:, :].copy()
        values[shift_start:, :] = 0
        write_start = shift_start + shift_samples
        if write_start < n_timepoints:
            writable_rows = n_timepoints - write_start
            values[write_start:, :] = tail_before_shift[:writable_rows, :]
        truncated_tail_samples += min(shift_samples, tail_before_shift.shape[0])

    return values, truncated_tail_samples


def find_simultaneous_duration_violations(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
) -> list[SimultaneousDurationViolation]:
    """Find simultaneous stimulation starts with unequal per-neuron durations."""
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")

    segments = _extract_active_segments(stim_array.values, active_threshold)
    segments_by_start: dict[int, list[_StimSegment]] = defaultdict(list)
    for segment in segments:
        segments_by_start[segment.start_idx].append(segment)

    violations: list[SimultaneousDurationViolation] = []
    for start_idx in sorted(segments_by_start.keys()):
        start_segments = sorted(
            segments_by_start[start_idx],
            key=lambda segment: segment.neuron_index,
        )
        if len(start_segments) < 2:
            continue
        durations_idx = tuple(int(segment.duration_idx) for segment in start_segments)
        min_duration_idx = min(durations_idx)
        if all(duration == min_duration_idx for duration in durations_idx):
            continue
        violations.append(
            SimultaneousDurationViolation(
                start_idx=int(start_idx),
                neuron_indices=tuple(
                    int(segment.neuron_index) for segment in start_segments
                ),
                durations_idx=durations_idx,
                min_duration_idx=int(min_duration_idx),
            )
        )

    return violations


def equalize_simultaneous_durations_to_min(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Trim simultaneous starts so all participating neurons share the minimum duration.

    Returns:
      (new_values, trimmed_timepoints)
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")

    values = np.array(stim_array.values, copy=True)
    violations = find_simultaneous_duration_violations(
        stim_array,
        active_threshold=active_threshold,
    )
    if len(violations) == 0:
        return values, 0

    trimmed_timepoints = 0
    n_timepoints = values.shape[0]
    for violation in violations:
        for neuron_idx, duration_idx in zip(
            violation.neuron_indices,
            violation.durations_idx,
            strict=False,
        ):
            if duration_idx <= violation.min_duration_idx:
                continue
            trim_start = violation.start_idx + violation.min_duration_idx
            trim_end = min(n_timepoints, violation.start_idx + duration_idx)
            if trim_end <= trim_start:
                continue
            values[trim_start:trim_end, neuron_idx] = 0
            trimmed_timepoints += trim_end - trim_start

    return values, trimmed_timepoints


def find_global_slm_window_violations(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    global_window_origin_tick: int = 0,
) -> list[GlobalSlmWindowViolation]:
    """Find pulse starts that do not align to global 16.666ms SLM windows.

    Alignment target:
      start_tick == global_window_origin_tick + lead_ticks + guard_ticks + 32*k
    where `k` is any integer.
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")

    sample_us = float(stim_array.dt * 1e6)
    lead_ticks = int(np.ceil(_SLM_LEAD_IN_US / _SYSCON2_TTL_PULSE_QUANTUM_US))
    pulse_start_offset_ticks = lead_ticks + _TTL_PULSE_BOUNDARY_GUARD_TICKS
    shape_quantum_ticks = _SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS

    violations: list[GlobalSlmWindowViolation] = []
    for segment in _extract_active_segments(stim_array.values, active_threshold):
        start_tick = int(
            np.rint((segment.start_idx * sample_us) / _SYSCON2_TTL_PULSE_QUANTUM_US)
        )
        snapped_start_tick = _nearest_global_slm_window_start_tick(
            start_tick,
            global_window_origin_tick=global_window_origin_tick,
            pulse_start_offset_ticks=pulse_start_offset_ticks,
            shape_quantum_ticks=shape_quantum_ticks,
        )
        if start_tick == snapped_start_tick:
            continue
        shift_ticks = snapped_start_tick - start_tick
        shift_samples = int(
            np.rint(shift_ticks * _SYSCON2_TTL_PULSE_QUANTUM_US / sample_us)
        )
        violations.append(
            GlobalSlmWindowViolation(
                neuron_index=int(segment.neuron_index),
                start_idx=int(segment.start_idx),
                duration_idx=int(segment.duration_idx),
                start_tick=start_tick,
                snapped_start_tick=snapped_start_tick,
                shift_ticks=shift_ticks,
                shift_samples=shift_samples,
            )
        )

    return sorted(
        violations,
        key=lambda violation: (
            abs(violation.shift_ticks),
            violation.start_tick,
            violation.neuron_index,
        ),
    )


def snap_stim_values_to_global_slm_windows(
    stim_array: StimArray,
    *,
    active_threshold: float = 0.0,
    global_window_origin_tick: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Snap active segment starts to nearest global 16.666ms SLM window.

    Returns:
      (new_values, moved_segments, truncated_tail_timepoints)
    """
    if stim_array.values.ndim != 2:
        raise ValueError("Stim array values must be 2D (timepoints x neurons)")
    if stim_array.dt <= 0:
        raise ValueError("Stim array dt must be positive")

    values = np.array(stim_array.values, copy=True)
    n_timepoints = values.shape[0]
    sample_us = float(stim_array.dt * 1e6)
    lead_ticks = int(np.ceil(_SLM_LEAD_IN_US / _SYSCON2_TTL_PULSE_QUANTUM_US))
    pulse_start_offset_ticks = lead_ticks + _TTL_PULSE_BOUNDARY_GUARD_TICKS
    shape_quantum_ticks = _SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS

    segments = _extract_active_segments(values, active_threshold)
    if len(segments) == 0:
        return values, 0, 0

    snapped_values = np.array(values, copy=True)
    for segment in segments:
        start_idx = int(segment.start_idx)
        end_idx = int(start_idx + segment.duration_idx)
        snapped_values[start_idx:end_idx, segment.neuron_index] = 0

    moved_segments = 0
    truncated_tail_timepoints = 0
    for segment in segments:
        start_idx = int(segment.start_idx)
        duration_idx = int(segment.duration_idx)
        end_idx = int(start_idx + duration_idx)
        neuron_index = int(segment.neuron_index)
        block = values[start_idx:end_idx, neuron_index].copy()
        if block.shape[0] == 0:
            continue

        start_tick = int(
            np.rint((start_idx * sample_us) / _SYSCON2_TTL_PULSE_QUANTUM_US)
        )
        snapped_start_tick = _nearest_global_slm_window_start_tick(
            start_tick,
            global_window_origin_tick=global_window_origin_tick,
            pulse_start_offset_ticks=pulse_start_offset_ticks,
            shape_quantum_ticks=shape_quantum_ticks,
        )
        snapped_start_idx = max(
            0,
            int(
                np.rint(snapped_start_tick * _SYSCON2_TTL_PULSE_QUANTUM_US / sample_us)
            ),
        )
        if snapped_start_idx != start_idx:
            moved_segments += 1

        target_end_idx = snapped_start_idx + block.shape[0]
        writable_end_idx = min(n_timepoints, target_end_idx)
        if writable_end_idx > snapped_start_idx:
            writable_rows = writable_end_idx - snapped_start_idx
            snapped_values[snapped_start_idx:writable_end_idx, neuron_index] = (
                np.maximum(
                    snapped_values[snapped_start_idx:writable_end_idx, neuron_index],
                    block[:writable_rows],
                )
            )
        if target_end_idx > n_timepoints:
            truncated_tail_timepoints += target_end_idx - n_timepoints

    return snapped_values, moved_segments, truncated_tail_timepoints


def _nearest_global_slm_window_start_tick(
    start_tick: int,
    *,
    global_window_origin_tick: int,
    pulse_start_offset_ticks: int,
    shape_quantum_ticks: int,
) -> int:
    base_tick = global_window_origin_tick + pulse_start_offset_ticks
    relative = start_tick - base_tick
    nearest_slot = int(np.rint(relative / float(shape_quantum_ticks)))
    return base_tick + nearest_slot * shape_quantum_ticks


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


def _build_mapped_segments_for_timing(
    stim_array: StimArray,
    *,
    active_threshold: float,
) -> list[_MappedSegment]:
    segments = _extract_active_segments(stim_array.values, active_threshold)
    sample_us = float(stim_array.dt * 1e6)
    mapped_segments: list[_MappedSegment] = []

    for segment in segments:
        start_tick = int(
            np.rint((segment.start_idx * sample_us) / _SYSCON2_TTL_PULSE_QUANTUM_US)
        )
        duration_us = float(segment.duration_idx) * sample_us
        duration_ticks = max(
            1,
            int(np.rint(duration_us / _SYSCON2_TTL_PULSE_QUANTUM_US)),
        )
        end_tick = start_tick + duration_ticks
        mapped_segments.append(
            _MappedSegment(
                neuron_index=segment.neuron_index,
                start_tick=start_tick,
                end_tick=end_tick,
                intensity=1.0,
                center_x=0.0,
                center_y=0.0,
            )
        )

    return mapped_segments


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
    *,
    lead_ticks: int,
    min_duration_ticks: int,
    pulse_guard_ticks: int = 0,
) -> list[_EntityWindow]:
    del min_duration_ticks  # duration is enforced in shape-quanta below.
    if pulse_guard_ticks < 0:
        raise ValueError("pulse_guard_ticks must be non-negative")
    if len(mapped_segments) == 0:
        return []

    segments_by_neuron: dict[int, list[_MappedSegment]] = defaultdict(list)
    for segment in mapped_segments:
        segments_by_neuron[segment.neuron_index].append(segment)
    for neuron_idx in list(segments_by_neuron.keys()):
        segments_by_neuron[neuron_idx].sort(
            key=lambda segment: (segment.start_tick, segment.end_tick)
        )

    boundary_ticks = _find_new_target_appearance_ticks(mapped_segments)
    first_stimulation_tick = min(segment.start_tick for segment in mapped_segments)
    phase_starts = [first_stimulation_tick]
    phase_starts.extend(
        tick for tick in boundary_ticks if tick > first_stimulation_tick
    )
    phase_starts = sorted(set(phase_starts))

    windows: list[_EntityWindow] = []
    for phase_idx, phase_start in enumerate(phase_starts):
        phase_end: int | None = None
        if phase_idx + 1 < len(phase_starts):
            phase_end = phase_starts[phase_idx + 1]

        for neuron_idx, neuron_segments in segments_by_neuron.items():
            stimulation_start_tick: int | None = None
            stimulation_end_tick: int | None = None
            max_intensity = 0.0
            center_x = neuron_segments[0].center_x
            center_y = neuron_segments[0].center_y

            for segment in neuron_segments:
                clipped_start = max(segment.start_tick, phase_start)
                clipped_end = segment.end_tick
                if phase_end is not None:
                    clipped_end = min(clipped_end, phase_end)
                if clipped_end <= clipped_start:
                    continue
                if stimulation_start_tick is None:
                    stimulation_start_tick = clipped_start
                    stimulation_end_tick = clipped_end
                else:
                    stimulation_start_tick = min(stimulation_start_tick, clipped_start)
                    assert stimulation_end_tick is not None
                    stimulation_end_tick = max(stimulation_end_tick, clipped_end)
                max_intensity = max(max_intensity, segment.intensity)
                center_x = segment.center_x
                center_y = segment.center_y

            if stimulation_start_tick is None or stimulation_end_tick is None:
                continue

            aligned_start_tick = max(
                0,
                stimulation_start_tick - lead_ticks - pulse_guard_ticks,
            )
            duration_ticks = max(
                1,
                stimulation_end_tick - aligned_start_tick + pulse_guard_ticks,
            )
            duration_shape_quanta = max(
                2,
                int(
                    np.ceil(
                        duration_ticks
                        / float(_SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS)
                    )
                ),
            )
            end_tick = (
                aligned_start_tick
                + duration_shape_quanta * _SYSCON2_NEURON_OBJECT_QUANTUM_IN_TTL_UNITS
            )

            windows.append(
                _EntityWindow(
                    neuron_index=neuron_idx,
                    stimulation_start_tick=stimulation_start_tick,
                    stimulation_end_tick=stimulation_end_tick,
                    start_tick=aligned_start_tick,
                    end_tick=end_tick,
                    intensity=max_intensity,
                    center_x=center_x,
                    center_y=center_y,
                )
            )
    return windows


def _find_new_target_appearance_ticks(
    mapped_segments: list[_MappedSegment],
) -> list[int]:
    start_events: dict[int, list[int]] = defaultdict(list)
    end_events: dict[int, list[int]] = defaultdict(list)
    for segment in mapped_segments:
        start_events[segment.start_tick].append(segment.neuron_index)
        end_events[segment.end_tick].append(segment.neuron_index)

    event_ticks = sorted(set(start_events.keys()) | set(end_events.keys()))
    first_stimulation_tick = min(segment.start_tick for segment in mapped_segments)
    active_counts: dict[int, int] = defaultdict(int)
    last_non_empty_set: set[int] | None = None
    boundaries: list[int] = []

    for tick in event_ticks:
        active_before_tick = {idx for idx, count in active_counts.items() if count > 0}
        starting_neurons = set(start_events.get(tick, []))
        introduces_new_target = False
        if len(active_before_tick) > 0:
            introduces_new_target = len(starting_neurons - active_before_tick) > 0
        elif last_non_empty_set is not None:
            introduces_new_target = len(starting_neurons - last_non_empty_set) > 0
        if introduces_new_target and tick > first_stimulation_tick:
            boundaries.append(tick)
        if len(active_before_tick) > 0:
            last_non_empty_set = set(active_before_tick)

        for neuron_idx in end_events.get(tick, []):
            next_count = active_counts[neuron_idx] - 1
            if next_count <= 0:
                active_counts.pop(neuron_idx, None)
            else:
                active_counts[neuron_idx] = next_count

        for neuron_idx in start_events.get(tick, []):
            active_counts[neuron_idx] = active_counts.get(neuron_idx, 0) + 1

    return sorted(set(boundaries))


def _validate_no_off_on_overlap(
    windows: list[_EntityWindow],
    *,
    lead_ticks: int,
    pulse_guard_ticks: int = 0,
) -> None:
    if lead_ticks <= 0:
        raise ValueError("lead_ticks must be positive")
    if pulse_guard_ticks < 0:
        raise ValueError("pulse_guard_ticks must be non-negative")
    if len(windows) <= 1:
        return

    for off_window in windows:
        off_start = off_window.start_tick
        off_end = min(
            off_window.end_tick,
            off_window.start_tick + lead_ticks + pulse_guard_ticks,
        )
        if off_end <= off_start:
            continue

        for on_window in windows:
            if off_window is on_window:
                continue
            if off_window.neuron_index == on_window.neuron_index:
                continue
            on_start = max(
                on_window.start_tick + lead_ticks + pulse_guard_ticks,
                on_window.start_tick,
            )
            on_end = on_window.end_tick
            if on_end <= on_start:
                continue
            if off_end <= on_start or on_end <= off_start:
                continue

            overlap_start = max(off_start, on_start)
            overlap_end = min(off_end, on_end)
            raise ValueError(
                "Invalid stimulation for SysCon2 export: an SLM off-state interval "
                "overlaps with another neuron's ON interval "
                f"(off neuron={off_window.neuron_index}, "
                f"on neuron={on_window.neuron_index}, "
                f"overlap_ticks=[{overlap_start}, {overlap_end}))."
            )


def _validate_pulse_bounds_within_entity_windows(
    windows: list[_EntityWindow],
    *,
    lead_ticks: int,
    pulse_guard_ticks: int = 0,
) -> None:
    if lead_ticks <= 0:
        raise ValueError("lead_ticks must be positive")
    if pulse_guard_ticks < 0:
        raise ValueError("pulse_guard_ticks must be non-negative")
    if len(windows) == 0:
        return

    for window in windows:
        expected_stimulation_start = window.start_tick + lead_ticks + pulse_guard_ticks
        if window.stimulation_start_tick != expected_stimulation_start:
            raise ValueError(
                "Invalid stimulation for SysCon2 export: first associated laser pulse "
                "must start exactly after lead-in + guard interval "
                f"(neuron={window.neuron_index}, "
                f"stimulation_start_tick={window.stimulation_start_tick}, "
                f"expected_start_tick={expected_stimulation_start})."
            )
        max_stimulation_end = window.end_tick - pulse_guard_ticks
        if window.stimulation_end_tick > max_stimulation_end:
            raise ValueError(
                "Invalid stimulation for SysCon2 export: laser pulse end is too "
                "close to the end of the associated shape object "
                f"(neuron={window.neuron_index}, "
                f"stimulation_end_tick={window.stimulation_end_tick}, "
                f"maximum_allowed_tick={max_stimulation_end})."
            )


def _validate_laser_pulse_end_before_overlapping_shapes(
    pulse_intervals: list[tuple[int, int]],
    scheduled_windows: list[_ScheduledEntityWindow],
    *,
    pulse_guard_ticks: int = 0,
) -> None:
    if pulse_guard_ticks < 0:
        raise ValueError("pulse_guard_ticks must be non-negative")
    if len(pulse_intervals) == 0:
        return
    if len(scheduled_windows) == 0:
        raise ValueError(
            "Invalid stimulation for SysCon2 export: laser pulses exist but no shape "
            "windows were scheduled."
        )

    for pulse_start_tick, pulse_end_tick in pulse_intervals:
        if pulse_end_tick <= pulse_start_tick:
            raise ValueError(
                "Invalid stimulation for SysCon2 export: encountered non-positive "
                f"laser pulse interval [{pulse_start_tick}, {pulse_end_tick})."
            )

        overlapping_windows = [
            window
            for window in scheduled_windows
            if not (
                pulse_end_tick <= window.start_tick
                or window.end_tick <= pulse_start_tick
            )
        ]
        if len(overlapping_windows) == 0:
            raise ValueError(
                "Invalid stimulation for SysCon2 export: laser pulse interval does not "
                "overlap any shape object "
                f"(pulse_ticks=[{pulse_start_tick}, {pulse_end_tick}))."
            )

        for window in overlapping_windows:
            latest_allowed_end_tick = window.end_tick - pulse_guard_ticks
            if pulse_end_tick > latest_allowed_end_tick:
                raise ValueError(
                    "Invalid stimulation for SysCon2 export: laser pulse end exceeds "
                    "the allowed end for an overlapping shape object "
                    f"(pulse_ticks=[{pulse_start_tick}, {pulse_end_tick}), "
                    f"shape_lane={window.lane}, neuron={window.neuron_index}, "
                    f"shape_ticks=[{window.start_tick}, {window.end_tick}), "
                    f"max_pulse_end_tick={latest_allowed_end_tick})."
                )


def _assign_entity_lanes(
    windows: list[_EntityWindow],
    lane_count: int,
) -> list[_ScheduledEntityWindow]:
    if lane_count <= 0:
        raise ValueError("lane_count must be positive")
    windows_sorted = sorted(
        windows,
        key=lambda window: (
            window.start_tick,
            window.end_tick,
            window.stimulation_start_tick,
            window.neuron_index,
        ),
    )
    available_lanes = list(range(lane_count))
    heapq.heapify(available_lanes)
    active_lanes: list[tuple[int, int]] = []

    scheduled: list[_ScheduledEntityWindow] = []
    for window in windows_sorted:
        while len(active_lanes) > 0 and active_lanes[0][0] <= window.start_tick:
            _, released_lane = heapq.heappop(active_lanes)
            heapq.heappush(available_lanes, released_lane)

        if len(available_lanes) == 0:
            raise ValueError(
                "Could not schedule entity objects within the configured lane count. "
                "The stimulation pattern is too dense for the computed simultaneous-neuron count."
            )

        lane = heapq.heappop(available_lanes)
        heapq.heappush(active_lanes, (window.end_tick, lane))
        scheduled.append(
            _ScheduledEntityWindow(
                neuron_index=window.neuron_index,
                stimulation_start_tick=window.stimulation_start_tick,
                stimulation_end_tick=window.stimulation_end_tick,
                start_tick=window.start_tick,
                end_tick=window.end_tick,
                intensity=window.intensity,
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

    # Normalize to current SysCon2 v3-style header and drop known legacy clutter.
    ordered_header["version"] = "3"
    ordered_header.pop("DeviceSequenceID", None)
    ordered_header.pop("TotalEditorGroups", None)

    numbered_key_re = re.compile(
        r"^(SourceConnectorPort|TargetConnectorPort|PreTiming_us|PostTiming_us|inverted)(\d+)$"
    )
    for key in list(ordered_header.keys()):
        match = numbered_key_re.match(key)
        if match is None:
            continue
        if int(match.group(2)) > 0:
            del ordered_header[key]

    ordered_header["invertedportcnt"] = "1"
    if "inverted0" not in ordered_header:
        ordered_header["inverted0"] = _DEFAULT_LIGHTSOURCE_PORT
    if "SourceConnectorPort0" not in ordered_header:
        ordered_header["SourceConnectorPort0"] = _DEFAULT_LIGHTSOURCE_PORT
    if "TargetConnectorPort0" not in ordered_header:
        ordered_header["TargetConnectorPort0"] = _DEFAULT_TTL_OUT_2_PORT
    if "PreTiming_us0" not in ordered_header:
        ordered_header["PreTiming_us0"] = "20000"
    if "PostTiming_us0" not in ordered_header:
        ordered_header["PostTiming_us0"] = "20000"

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
    port: str,
    lightsource_id: str,
    include_lightsource: bool,
) -> list[str]:
    properties: list[tuple[str, str]] = [
        ("TotalTimings", "1"),
        ("TimelineInfo0_TimelineIndex", str(timeline_index)),
        (
            "TimelineInfo0_starttime",
            _format_time_us(start_tick * _SYSCON2_TTL_PULSE_QUANTUM_US),
        ),
        ("TimelineInfo0_description", ""),
        (
            "TimelineInfo0_repeats",
            str(_ttl_repeats_from_duration_ticks(duration_ticks)),
        ),
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


def _format_time_us(value_us: float) -> str:
    rounded_int = int(np.rint(value_us))
    if np.isclose(value_us, float(rounded_int), atol=1e-9, rtol=0.0):
        return str(rounded_int)
    return f"{value_us:.4f}".rstrip("0").rstrip(".")


def _ttl_repeats_from_duration_ticks(duration_ticks: int) -> int:
    # TTLPulse duration follows SysCon2 rule:
    # duration = (repeats + 1) * TTL quantum.
    return max(0, int(duration_ticks) - 1)


def _shape_repeats_from_duration_quanta(duration_quanta: int) -> int:
    # Shape duration follows SysCon2 rule:
    # duration = (repeats + 2) * Shape quantum.
    return max(0, int(duration_quanta) - 2)


def _compute_overview_figure_width_in(max_end_ms: float) -> float:
    # Keep long timelines legible without exploding image size.
    target_ms_per_inch = 5000.0
    return float(np.clip(max_end_ms / target_ms_per_inch, 12.0, 34.0))


def _compute_overview_render_dpi(
    *,
    fig_width_inches: float,
    fig_height_inches: float,
) -> int:
    base_dpi = 180.0
    max_total_pixels = 28_000_000.0
    required_dpi = np.sqrt(
        max_total_pixels / max(1.0, fig_width_inches * fig_height_inches)
    )
    return int(np.clip(np.floor(required_dpi), 110.0, base_dpi))


def _compute_overview_min_visible_duration_ms(
    *,
    max_end_ms: float,
    fig_width_inches: float,
    dpi: int,
) -> float:
    ms_per_pixel = max_end_ms / max(1.0, fig_width_inches * float(dpi))
    return max(0.08, 1.25 * ms_per_pixel)


def _select_overview_time_axis(max_end_ms: float) -> tuple[float, str]:
    if max_end_ms >= 120_000.0:
        return 1.0 / 60_000.0, "Time (min)"
    if max_end_ms >= 8_000.0:
        return 1.0 / 1_000.0, "Time (s)"
    return 1.0, "Time (ms)"


def _overview_tick_decimals(x_max: float) -> int:
    if x_max >= 200.0:
        return 0
    if x_max >= 20.0:
        return 1
    if x_max >= 2.0:
        return 2
    return 3


def _format_overview_duration_label(duration_ms: float) -> str:
    if duration_ms >= 120_000.0:
        return f"{duration_ms / 60_000.0:.1f} min"
    if duration_ms >= 8_000.0:
        return f"{duration_ms / 1_000.0:.1f} s"
    return f"{duration_ms:.1f} ms"


def _preview_object_duration_us(
    obj: SysconSequenceObject,
    *,
    tick_resolution_us: int,
) -> float:
    object_type = obj.object_type.strip().lower()
    repeats = max(0, int(obj.repeats))
    step_size = max(1, int(obj.step_size))

    if object_type == "ttlpulse":
        return float(repeats + 1) * _SYSCON2_TTL_PULSE_QUANTUM_US
    if object_type == "shape":
        return float(max(2, repeats + 2)) * _SYSCON2_NEURON_OBJECT_QUANTUM_US
    if object_type == "entity":
        duration_ticks = max(1, (repeats + 1) * step_size)
        return float(duration_ticks * tick_resolution_us)

    duration_ticks = max(1, (repeats + 1) * step_size)
    return float(duration_ticks * tick_resolution_us)


def _preview_object_color(object_type: str) -> str:
    if object_type in {"shape", "entity"}:
        return "#2c7fb8"
    if object_type == "ttlpulse":
        return "#d95f0e"
    return "#636363"


def _extract_lightsource_ids_from_object(obj: SysconSequenceObject) -> set[str]:
    ids: set[str] = set()
    for key, value in obj.properties.items():
        if "_lightsourceid_" not in key.lower():
            continue
        stripped = value.strip()
        if stripped:
            ids.add(stripped)
    return ids


def _build_timeline_axis_labels(
    timeline_indices: list[int],
    *,
    timeline_objects: dict[int, list[SysconSequenceObject]],
) -> dict[int, str]:
    shape_timelines = sorted(
        idx
        for idx in timeline_indices
        if any(
            obj.object_type.strip().lower() in {"shape", "entity"}
            for obj in timeline_objects.get(idx, [])
        )
    )
    shape_lane_by_timeline = {
        timeline_idx: lane_idx for lane_idx, timeline_idx in enumerate(shape_timelines)
    }
    inferred_holo4d_base = _infer_holo4d_timeline_base(shape_timelines)
    explicit_holo4d_labels: dict[int, str] = {}
    if inferred_holo4d_base is not None:
        for wavelength_nm, offset in _LASER_TIMELINE_OFFSETS.items():
            explicit_holo4d_labels[inferred_holo4d_base + offset] = (
                f"Laser {wavelength_nm} nm"
            )
        explicit_holo4d_labels[inferred_holo4d_base + 3] = "TTL Out 1"
        explicit_holo4d_labels[inferred_holo4d_base + 4] = "TTL Out 2"

    labels: dict[int, str] = {}
    for timeline_idx in timeline_indices:
        objects = timeline_objects.get(timeline_idx, [])
        lower_types = {obj.object_type.strip().lower() for obj in objects}

        if timeline_idx in shape_lane_by_timeline:
            lane_idx = shape_lane_by_timeline[timeline_idx]
            labels[timeline_idx] = f"TL {timeline_idx} - Neuron lane {lane_idx}"
            continue

        explicit_name = explicit_holo4d_labels.get(timeline_idx)
        ttl_objects = [
            obj for obj in objects if obj.object_type.strip().lower() == "ttlpulse"
        ]
        if ttl_objects:
            lightsource_ids = sorted(
                {
                    lightsource_id
                    for obj in ttl_objects
                    for lightsource_id in _extract_lightsource_ids_from_object(obj)
                }
            )
            if lightsource_ids:
                if explicit_name is not None:
                    if len(lightsource_ids) == 1:
                        labels[timeline_idx] = (
                            f"TL {timeline_idx} - {explicit_name} ({lightsource_ids[0]})"
                        )
                    else:
                        labels[timeline_idx] = f"TL {timeline_idx} - {explicit_name}"
                elif len(lightsource_ids) == 1:
                    labels[timeline_idx] = (
                        f"TL {timeline_idx} - Laser ({lightsource_ids[0]})"
                    )
                else:
                    labels[timeline_idx] = f"TL {timeline_idx} - Laser"
                continue

            ports = sorted(
                {
                    obj.properties.get("port", "").strip()
                    for obj in ttl_objects
                    if obj.properties.get("port", "").strip()
                }
            )
            if len(ports) == 1:
                port_label = ports[0]
                if explicit_name is not None:
                    labels[timeline_idx] = (
                        f"TL {timeline_idx} - {explicit_name} ({port_label})"
                    )
                else:
                    labels[timeline_idx] = f"TL {timeline_idx} - {port_label}"
            else:
                if explicit_name is not None:
                    labels[timeline_idx] = f"TL {timeline_idx} - {explicit_name}"
                else:
                    labels[timeline_idx] = f"TL {timeline_idx} - TTL"
            continue

        if explicit_name is not None:
            labels[timeline_idx] = f"TL {timeline_idx} - {explicit_name}"
            continue

        if len(lower_types) == 1:
            object_type = next(iter(lower_types))
            labels[timeline_idx] = f"TL {timeline_idx} - {object_type}"
        elif len(lower_types) > 1:
            labels[timeline_idx] = f"TL {timeline_idx} - mixed"
        else:
            labels[timeline_idx] = f"TL {timeline_idx}"

    return labels


def _infer_holo4d_timeline_base(shape_timelines: list[int]) -> int | None:
    if len(shape_timelines) == 0:
        return None
    expected = list(range(shape_timelines[-1] + 1))
    if shape_timelines != expected:
        return None
    return shape_timelines[-1] + 1
