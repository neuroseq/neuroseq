from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from PIL import Image

from neuroseq.export.exporter_syscon2 import PulseOverrunViolation
from neuroseq.export.exporter_syscon2 import _compute_cell_centroids
from neuroseq.export.exporter_syscon2 import _compute_cell_stimulation_spots
from neuroseq.export.exporter_syscon2 import _find_pulse_overrun_violations
from neuroseq.export.exporter_syscon2 import _resolve_neuron_to_cellpose_mapping
from neuroseq.export.exporter_syscon2 import _ScheduledEntityWindow
from neuroseq.export.exporter_syscon2 import align_overlapping_pulses_to_shortest
from neuroseq.export.exporter_syscon2 import align_stim_values_left_for_slm
from neuroseq.export.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.export.exporter_syscon2 import equalize_simultaneous_durations_to_min
from neuroseq.export.exporter_syscon2 import find_initial_slm_load_violation
from neuroseq.export.exporter_syscon2 import find_overlapping_pulse_group_violations
from neuroseq.export.exporter_syscon2 import find_simultaneous_duration_violations
from neuroseq.export.exporter_syscon2 import find_slm_gap_violations
from neuroseq.export.exporter_syscon2 import find_slm_off_on_overlap_violations
from neuroseq.export.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.export.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.export.exporter_syscon2 import resample_stim_values_to_dt
from neuroseq.export.exporter_syscon2 import (
    shift_stim_values_right_for_initial_slm_load,
)
from neuroseq.export.exporter_syscon2 import space_out_stim_values_for_slm
from neuroseq.export.exporter_syscon2 import (
    space_out_stim_values_for_slm_off_on_overlap,
)
from neuroseq.export.power_lut import PowerLut
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.importers.importer_syscon2 import sequence_to_stim_values
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType

_TTL_QUANTUM_US = 520.8125
_DT_S = 0.0005208125  # exact literal; avoids float rounding from _TTL_QUANTUM_US * 1e-6


def _linear_power_lut(max_mw: float = 50.0) -> PowerLut:
    return PowerLut(
        description="",
        tuples=((0.0, 0.0), (100.0, max_mw)),
    )


def _template_header() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=5
DeviceSequenceID=1
invertedportcnt=1
inverted0=UGA-42 TTL Out 1
AutoTriggerRulesEnabled=False
AutoTriggerRulesCount=0
timelinezoomfactor=10
timelineviewposx=0
timelineviewposy=0
SourceConnectorPort0=dummy_1_0 RMI In
"""


def _cellpose_result(
    *,
    mapping: dict[int, int] | None = None,
) -> CellposeImportResult:
    segmentation = np.array(
        [
            [0, 0, -1, -1],
            [1, 1, -1, -1],
        ],
        dtype=np.int32,
    )
    overlay = np.zeros((2, 4, 3), dtype=np.uint8)
    original_ids = np.array([0, 1], dtype=np.int32)
    return CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=segmentation,
        overlay_image=overlay,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=original_ids,
        stimulus_to_cellpose_mapping=mapping,
    )


def _cellpose_result_many(
    *,
    n_neurons: int,
    mapping: dict[int, int] | None = None,
) -> CellposeImportResult:
    # Space synthetic centroids apart so importer center-distance merging does
    # not collapse distinct neurons in roundtrip pulse-logic tests.
    spacing_px = 3
    width = max(1, n_neurons * spacing_px)
    segmentation = np.full((1, width), -1, dtype=np.int32)
    for neuron_idx in range(n_neurons):
        segmentation[0, neuron_idx * spacing_px] = neuron_idx
    overlay = np.zeros((1, width, 3), dtype=np.uint8)
    original_ids = np.arange(n_neurons, dtype=np.int32)
    return CellposeImportResult(
        image_path=Path("dummy_many.png"),
        segmentation_array=segmentation,
        overlay_image=overlay,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=original_ids,
        stimulus_to_cellpose_mapping=mapping,
    )


def _timeline_index(obj) -> int:
    return int(obj.properties["TimelineInfo0_TimelineIndex"])


def _count_active_segments(values: np.ndarray) -> int:
    active = values > 0
    if not np.any(active):
        return 0
    padded = np.concatenate(
        (
            np.array([False], dtype=bool),
            active.astype(bool),
            np.array([False], dtype=bool),
        )
    )
    starts = np.where((~padded[:-1]) & padded[1:])[0]
    return int(starts.size)


def _quantized_us(sample_idx: int, dt_seconds: float) -> float:
    return float(
        int(np.rint((sample_idx * dt_seconds * 1e6) / _TTL_QUANTUM_US))
        * _TTL_QUANTUM_US
    )


def test_export_seq_uses_fixed_laser_and_ttl_out_timeline_layout() -> None:
    values = np.zeros((40, 2), dtype=np.float32)
    values[20:22, 0] = 0.2
    values[20:22, 1] = 0.9
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.entity_lanes == 2
    assert report.laser_timeline_index == 1
    assert report.ttl_out1_timeline_index == 4
    assert report.ttl_out2_timeline_index == 5
    assert report.unmapped_active_neurons == []

    assert len(entities) == 2
    entity_lanes = sorted(_timeline_index(obj) for obj in entities)
    assert entity_lanes == [0, 6]
    assert all(obj.repeats >= 0 for obj in entities)
    assert all(obj.step_size == 100 for obj in entities)

    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 1]
    ttl_out_1_pulses = [obj for obj in pulses if _timeline_index(obj) == 4]
    ttl_out_2_pulses = [obj for obj in pulses if _timeline_index(obj) == 5]

    assert len(laser_pulses) == 1
    # Initial acquisition-trigger pulse at tick 0 plus one pulse per laser pulse.
    assert len(ttl_out_1_pulses) == 2
    assert len(ttl_out_2_pulses) == 0
    # TTL pulse comes from the adjusted raw segment position (tick 20 shifted by
    # one preblank window to tick 52 → 52 × 520.8125µs = 27082.25µs).
    assert ttl_out_1_pulses[0].start_time_us == 0  # initial trigger pulse
    assert ttl_out_1_pulses[1].start_time_us == pytest.approx(27082.25)


def test_export_seq_creates_ten_spots_inside_each_cellpose_roi() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    segmentation = np.full((8, 8), -1, dtype=np.int32)
    segmentation[1:7, 1:7] = 0
    cellpose_data = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((8, 8, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0},
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        cellpose_data,
        template_text=_template_header(),
        spiralling=False,
        stimulation_spots_per_roi=10,
    )

    shapes = [
        obj
        for obj in parse_syscon2_seq_text(seq_text).objects
        if obj.object_type == "Shape"
    ]
    assert len(shapes) == 10
    assert report.entity_lanes == 10
    coordinates = []
    for shape in shapes:
        center_x = next(
            float(value)
            for key, value in shape.properties.items()
            if key.endswith("_CenterX")
        )
        center_y = next(
            float(value)
            for key, value in shape.properties.items()
            if key.endswith("_CenterY")
        )
        coordinates.append((center_x, center_y))
        assert segmentation[int(center_y), int(center_x)] == 0
    assert len(set(coordinates)) == 10


def test_edge_balanced_spots_keep_clear_of_roi_border() -> None:
    segmentation = np.full((24, 24), -1, dtype=np.int32)
    segmentation[2:22, 2:22] = 0

    legacy = _compute_cell_stimulation_spots(segmentation, 10)[0]
    balanced = _compute_cell_stimulation_spots(
        segmentation, 10, algorithm="edge_balanced"
    )[0]

    def edge_clearance(point: tuple[float, float]) -> float:
        x, y = point
        return min(x - 2, 21 - x, y - 2, 21 - y)

    assert min(map(edge_clearance, legacy)) == 0
    assert min(map(edge_clearance, balanced)) > 0
    assert len(set(balanced)) == 10


def test_export_seq_coalesces_nearby_segments_into_one_sanitized_window() -> None:
    values = np.zeros((80, 1), dtype=np.float32)
    values[40:42, 0] = 1.0
    values[45:47, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    # Shape object still coalesces both segments into one window (window 1) with
    # a blank load window before it.  TTL pulses, however, now come from the
    # exact drawn tick positions (no window-quantisation): ticks 40-42 and 45-47.
    assert len(entities) == 1
    assert [pulse.start_time_us for pulse in laser_pulses] == pytest.approx(
        [40 * _TTL_QUANTUM_US, 45 * _TTL_QUANTUM_US]
    )

    imported_values, _ = sequence_to_stim_values(
        sequence,
        dt_seconds=stim_array.dt,
        n_neurons=1,
        tick_resolution_us=500,
        include_non_shape=False,
    )
    # Shape intensity is exported as 0 to match SysCon v3 reference files, but
    # the laser TTL pulses carry the ON positions, so the importer reconstructs
    # the drawn ON samples at nominal full-scale intensity (1.0).
    active_indices = np.flatnonzero(imported_values[:, 0] > 0)
    assert active_indices.tolist() == [40, 41, 45, 46]
    assert np.all(imported_values[active_indices, 0] == 1.0)


def test_export_seq_routes_ttl_pulses_to_selected_1030_laser_timeline() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1030,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.active_laser_wavelength_nm == 1030
    assert report.entity_lanes == 1
    assert report.laser_timeline_index == 3

    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 3]
    lower_laser_timeline = [obj for obj in pulses if _timeline_index(obj) == 2]
    other_laser_timeline = [obj for obj in pulses if _timeline_index(obj) == 1]
    ttl_out_1_pulses = [obj for obj in pulses if _timeline_index(obj) == 4]

    assert len(laser_pulses) == 1
    assert len(lower_laser_timeline) == 0
    assert len(other_laser_timeline) == 0
    assert len(ttl_out_1_pulses) == 2


def test_export_seq_starts_with_ttl_out_1_trigger_pulse() -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1.0
    stim_array = StimArray(
        name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=_DT_S
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    ttl_out_1 = sorted(
        (
            obj
            for obj in sequence.objects
            if obj.object_type == "TTLPulse"
            and _timeline_index(obj) == report.ttl_out1_timeline_index
        ),
        key=lambda obj: obj.start_time_us,
    )
    start_pulse, first_pulse = ttl_out_1[0], ttl_out_1[1]
    assert start_pulse.start_time_us == 0
    assert (start_pulse.repeats + 1) * _TTL_QUANTUM_US >= 9_000
    # The trigger pulse ends before the next TTL Out 1 pulse begins.
    assert (
        start_pulse.start_time_us + (start_pulse.repeats + 1) * _TTL_QUANTUM_US
        <= first_pulse.start_time_us
    )
    # An exported file is recognised as a Holo4D layout when re-imported.
    from neuroseq.importers.importer_syscon2 import _looks_like_holo4d_timeline_layout

    assert _looks_like_holo4d_timeline_layout(sequence, tick_resolution_us=500)


def test_export_seq_requires_syscon2_tick_rate() -> None:
    values = np.zeros((16, 1), dtype=np.float32)
    values[4:7, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,  # 1ms — no longer valid
    )

    with pytest.raises(ValueError, match="0.0005208125"):
        generate_syscon2_sequence_text(
            stim_array,
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            tick_resolution_us=500,
            active_threshold=0.0,
            binary_intensity=1.0,
            active_laser_wavelength_nm=1100,
            spiralling=False,
        )


def test_check_syscon2_input_sanity_rejects_wrong_dt() -> None:
    values = np.zeros((96, 1), dtype=np.float32)
    values[32:65, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,  # 1ms — no longer valid
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert not sanity.is_valid
    assert sanity.export_report is None
    assert any("0.0005208125" in error for error in sanity.errors)


def test_generate_syscon2_timeline_overview_png_writes_file(tmp_path: Path) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    png_path = tmp_path / "overview.png"
    returned = generate_syscon2_timeline_overview_png(seq_text, png_path)

    assert returned == png_path
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_generate_syscon2_timeline_overview_png_scales_width_for_long_duration(
    tmp_path: Path,
) -> None:
    short_values = np.zeros((2_000, 1), dtype=np.float32)
    short_values[1_500:1_502, 0] = 1.0
    short_stim = StimArray(
        name="short",
        stim_type=StimType.Excitation,
        values=short_values,
        units="1",
        dt=_DT_S,
    )
    short_seq_text, _ = generate_syscon2_sequence_text(
        short_stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    long_values = np.zeros((200_000, 1), dtype=np.float32)
    long_values[180_000:180_002, 0] = 1.0
    long_stim = StimArray(
        name="long",
        stim_type=StimType.Excitation,
        values=long_values,
        units="1",
        dt=_DT_S,
    )
    long_seq_text, _ = generate_syscon2_sequence_text(
        long_stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    short_png_path = tmp_path / "overview_short.png"
    long_png_path = tmp_path / "overview_long.png"
    generate_syscon2_timeline_overview_png(short_seq_text, short_png_path)
    generate_syscon2_timeline_overview_png(long_seq_text, long_png_path)

    with Image.open(short_png_path) as short_image:
        short_width_px, short_height_px = short_image.size
    with Image.open(long_png_path) as long_image:
        long_width_px, long_height_px = long_image.size

    assert short_height_px > 0
    assert long_height_px > 0
    assert long_width_px > short_width_px
    assert long_width_px >= 3000


def test_export_seq_accepts_legacy_1040_alias_as_1030_timeline() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1040,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]
    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 3]

    assert report.active_laser_wavelength_nm == 1030
    assert report.laser_timeline_index == 3
    assert len(laser_pulses) == 1


def test_export_seq_accepts_legacy_1140_alias_as_1100_timeline() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]
    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 1]

    assert report.active_laser_wavelength_nm == 1100
    assert report.laser_timeline_index == 1
    assert len(laser_pulses) == 1


def test_export_seq_spiralling_shifts_laser_pulse_one_tick_earlier() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text_on, report_on = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=True,
    )
    seq_text_off, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    seq_on = parse_syscon2_seq_text(seq_text_on)
    seq_off = parse_syscon2_seq_text(seq_text_off)

    laser_on = [
        obj
        for obj in seq_on.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report_on.laser_timeline_index
    ]
    laser_off = [
        obj
        for obj in seq_off.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report_on.laser_timeline_index
    ]
    assert len(laser_on) == 1
    assert len(laser_off) == 1

    assert laser_off[0].start_time_us - laser_on[0].start_time_us == pytest.approx(
        _TTL_QUANTUM_US
    )
    assert laser_on[0].repeats == laser_off[0].repeats

    ttl_out_1_on = [
        obj
        for obj in seq_on.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report_on.ttl_out1_timeline_index
        and obj.start_time_us > 0  # skip the initial trigger pulse
    ]
    ttl_out_1_off = [
        obj
        for obj in seq_off.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report_on.ttl_out1_timeline_index
        and obj.start_time_us > 0  # skip the initial trigger pulse
    ]
    assert len(ttl_out_1_on) == 1
    assert len(ttl_out_1_off) == 1
    assert ttl_out_1_off[0].start_time_us - ttl_out_1_on[
        0
    ].start_time_us == pytest.approx(_TTL_QUANTUM_US)


@pytest.mark.parametrize(
    ("wavelength_nm", "expected_timeline_index", "expected_lightsource_id"),
    [
        (1100, 1, "dummy_2_0"),
        (940, 2, "dummy_1_0"),
        (1030, 3, "dummy_0_0"),
    ],
)
def test_export_seq_lightsource_id_matches_syscon_v3_template(
    wavelength_nm: int,
    expected_timeline_index: int,
    expected_lightsource_id: str,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=wavelength_nm,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == expected_timeline_index
    ]
    assert len(laser_pulses) >= 1
    for pulse in laser_pulses:
        assert (
            pulse.properties["TimelineInfo0_LightsourceID_0"] == expected_lightsource_id
        )
        assert pulse.properties["port"] == f"{expected_lightsource_id} RMI In"

    shape_objs = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    assert len(shape_objs) >= 1
    for shape in shape_objs:
        # A spot's Shape block must claim the same laser as its TTLPulse
        # block - RAPP reported SysCon2 needing the laser picked manually
        # after import whenever these two disagreed (issue #535 follow-up).
        assert (
            shape.properties["TimelineInfo0_LightsourceID_0"] == expected_lightsource_id
        )
        assert shape.properties["TimelineInfo0_Intensity_0"] == "0"


def test_export_seq_never_inverts_pulses() -> None:
    """Inverted pulses are never wanted, regardless of what the template
    says: invertedportcnt is always forced to 0 and inverted0 is dropped
    entirely, even though the template used here (_template_header()) has
    invertedportcnt=1/inverted0=... of its own."""
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
    )

    header_lines = seq_text.splitlines()
    assert "invertedportcnt=0" in header_lines
    assert not any(line.startswith("inverted0=") for line in header_lines)


def test_export_seq_roundtrip_preserves_burst_timing_at_exact_tick_positions() -> None:
    """Each drawn burst becomes a separate TTL pulse at its exact tick position.

    Shape objects still cover the full burst range (window-level coalescing for
    SLM hardware), but laser TTL pulses are no longer window-quantised.  The
    roundtrip (export → import) reconstructs each burst at the original tick.
    """
    values = np.zeros((300, 6), dtype=np.float32)
    first_group_starts = [100, 107, 114, 121, 128, 135, 142, 149]
    second_group_starts = [200, 207, 214, 221]

    for start_idx in first_group_starts:
        values[start_idx : start_idx + 2, 0:3] = 1.0
    for start_idx in second_group_starts:
        values[start_idx : start_idx + 2, 3:6] = 1.0

    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result_many(
            n_neurons=6,
            mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        ),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )
    sequence = parse_syscon2_seq_text(seq_text)
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    # 8 bursts in group 1 + 4 bursts in group 2 = 12 separate laser pulses
    assert len(laser_pulses) == 12
    expected_starts = [
        s * _TTL_QUANTUM_US for s in first_group_starts + second_group_starts
    ]
    assert [pulse.start_time_us for pulse in laser_pulses] == pytest.approx(
        expected_starts
    )

    imported_values, _ = sequence_to_stim_values(
        sequence,
        dt_seconds=stim_array.dt,
        n_neurons=6,
        tick_resolution_us=500,
        include_non_shape=False,
    )
    # Shape intensity is exported as 0, but the per-burst laser TTL pulses let
    # the importer reconstruct each drawn burst (at nominal full-scale
    # intensity): 8 bursts for the group-1 neurons, 4 for the group-2 neurons.
    segment_counts = [
        _count_active_segments(imported_values[:, idx]) for idx in range(6)
    ]
    assert segment_counts == [8, 8, 8, 4, 4, 4]


def test_export_seq_shape_intensity_remains_zero_when_non_binary_input_is_requested() -> (
    None
):
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 0.37
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        force_binary_intensity=False,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]

    assert len(shapes) == 1
    assert np.isclose(shapes[0].intensity, 0.0)
    assert shapes[0].properties["TimelineInfo0_description"] == "neuron=0"


def test_export_seq_writes_lut_inverted_intensity_when_power_requested() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        target_power_mw=25.0,
        power_lut=_linear_power_lut(),
    )

    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    assert shapes[0].properties["TimelineInfo0_Intensity_0"] == "0.5"
    assert laser_pulses[0].properties["TimelineInfo0_Intensity_0"] == "0.5"
    assert report.power_saturation_warnings == []
    assert report.target_power_mw == 25.0
    assert report.power_lut_description == ""
    assert (
        shapes[0].properties["TimelineInfo0_description"]
        == "neuron=0 power_mW=25 target_mW=25"
    )
    assert (
        laser_pulses[0].properties["TimelineInfo0_description"]
        == "power_mW=25 target_mW=25"
    )


def test_export_seq_clamps_and_warns_when_target_power_exceeds_lut_max() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        target_power_mw=200.0,
        power_lut=_linear_power_lut(),
    )

    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    assert shapes[0].properties["TimelineInfo0_Intensity_0"] == "1"
    assert laser_pulses[0].properties["TimelineInfo0_Intensity_0"] == "1"
    assert len(report.power_saturation_warnings) >= 1
    assert any("LUT max" in warning for warning in report.power_saturation_warnings)


def test_export_seq_uses_window_intensity_as_multiplier_when_non_binary() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 0.4
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        force_binary_intensity=False,
        active_laser_wavelength_nm=1100,
        target_power_mw=50.0,
        power_lut=_linear_power_lut(),
    )

    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    laser_pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert float(shapes[0].properties["TimelineInfo0_Intensity_0"]) == pytest.approx(
        0.4
    )
    assert (
        shapes[0].properties["TimelineInfo0_description"]
        == "neuron=0 power_mW=20 target_mW=50"
    )
    assert any(
        float(pulse.properties["TimelineInfo0_Intensity_0"]) == pytest.approx(0.4)
        for pulse in laser_pulses
        if "TimelineInfo0_Intensity_0" in pulse.properties
    )


def test_export_seq_skips_unmapped_active_neurons_and_reports_them() -> None:
    values = np.zeros((40, 3), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[25:27, 2] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        spiralling=False,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.active_neurons == 2
    assert report.mapped_active_neurons == 1
    assert report.unmapped_active_neurons == [2]
    assert report.total_segments == 2
    assert report.exported_segments == 1
    assert report.skipped_segments_unmapped == 1
    assert report.entity_lanes == 1
    assert report.generated_objects == 4
    assert len(entities) == 1
    assert len(pulses) == 3

    assert _timeline_index(entities[0]) == 0


def test_export_seq_accepts_one_based_mapping_keys() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={1: 0}),
        template_text=_template_header(),
        spiralling=False,
    )
    sequence = parse_syscon2_seq_text(seq_text)

    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    assert len(entities) == 1
    assert report.exported_segments == 1
    assert report.unmapped_active_neurons == []


def test_sparse_custom_mapping_never_targets_one_cell_twice() -> None:
    """A sparse 0-based Custom mapping must not borrow key i+1 for an unmapped
    neuron i: {1: 12, 2: 3} used to give neuron 0 cell 12 as well."""
    cellpose = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((1, 1), dtype=np.int32),
        overlay_image=np.zeros((1, 1, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=None,
        stimulus_to_cellpose_mapping={1: 12, 2: 3},
        mapping_strategy="Custom",
    )

    resolved = _resolve_neuron_to_cellpose_mapping(cellpose, 4, set(range(20)))

    assert resolved == {1: 12, 2: 3}


@pytest.mark.parametrize(
    ("mapping", "n_neurons", "expected"),
    [
        ({1: 5, 2: 6, 3: 7}, 3, {0: 5, 1: 6, 2: 7}),  # whole key set 1-based
        ({0: 5, 1: 6, 3: 7}, 3, {0: 5, 1: 6}),  # 0 key: 0-based, key 3 ignored
        ({2: 6, 3: 7}, 3, {1: 6, 2: 7}),  # key n_neurons, all keys in 1..n
        ({1: 6}, 3, {1: 6}),  # sparse 0-based
    ],
)
def test_one_based_mapping_detection_uses_whole_key_set(
    mapping: dict[int, int], n_neurons: int, expected: dict[int, int]
) -> None:
    cellpose = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((1, 1), dtype=np.int32),
        overlay_image=np.zeros((1, 1, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=None,
        stimulus_to_cellpose_mapping=mapping,
        mapping_strategy="Custom",
    )
    resolved = _resolve_neuron_to_cellpose_mapping(cellpose, n_neurons, set(range(20)))
    assert resolved == expected
    assert len(set(resolved.values())) == len(resolved)


def test_export_seq_applies_the_linear_mapping_offset() -> None:
    """The readout resolves a Linear mapping's offset; the export must stimulate
    the same cell (stim 0 -> Cellpose ID 1 with offset 1)."""
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    def shape_centers(cellpose: CellposeImportResult) -> list[tuple[str, str]]:
        seq_text, _ = generate_syscon2_sequence_text(
            stim_array, cellpose, template_text=_template_header(), spiralling=False
        )
        return [
            (obj.properties["Shape0_CenterX"], obj.properties["Shape0_CenterY"])
            for obj in parse_syscon2_seq_text(seq_text).objects
            if obj.object_type == "Shape"
        ]

    linear = _cellpose_result()
    linear.mapping_strategy = "Linear"
    linear.linear_offset = 1
    assert shape_centers(linear) == shape_centers(_cellpose_result(mapping={0: 1}))
    assert shape_centers(linear) != shape_centers(_cellpose_result(mapping={0: 0}))


def test_check_syscon2_input_sanity_passes_for_valid_inputs() -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[40:42, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.errors == []
    assert sanity.export_report is not None
    assert sanity.export_report.exported_segments == 1


def test_check_syscon2_input_sanity_warns_for_unmapped_and_non_binary() -> None:
    values = np.zeros((64, 2), dtype=np.float32)
    values[40:42, 0] = 0.2
    values[40:42, 1] = 0.8
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.export_report is not None
    assert sanity.export_report.unmapped_active_neurons == [1]
    assert len(sanity.warnings) >= 2
    assert any("fixed intensity" in warning.lower() for warning in sanity.warnings)
    assert any("unmapped" in warning.lower() for warning in sanity.warnings)


def test_check_syscon2_input_sanity_auto_fixes_when_first_stim_is_too_early() -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[1:3, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.export_report is not None
    assert sanity.sanitized_values is not None
    assert sanity.export_report.exported_segments == 1
    # sanitized_values reflects the preblank shift only: the original 2-tick pulse
    # at [1:3] is shifted by one window (32 ticks) to [33:35], duration preserved.
    assert np.array_equal(np.flatnonzero(sanity.sanitized_values[:, 0] > 0), [33, 34])
    assert any("Inserted blank window" in warning for warning in sanity.warnings)


def test_check_syscon2_input_sanity_fails_without_active_samples() -> None:
    values = np.zeros((4, 2), dtype=np.float32)
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
    )

    assert not sanity.is_valid
    assert sanity.export_report is None
    assert any(
        "no active input samples found" in error.lower() for error in sanity.errors
    )


def test_check_syscon2_input_sanity_auto_fixes_overlap_into_exportable_windows() -> (
    None
):
    values = np.zeros((120, 2), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[40:42, 0] = 1.0
    values[40:42, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.export_report is not None
    assert sanity.export_report.exported_segments == 3
    assert any("Inserted blank window" in warning for warning in sanity.warnings)
    assert sanity.sanitized_values is not None


def test_find_slm_gap_violations_detects_short_gap_between_different_neurons() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:11, 1] = 1.0
    values[48:51, 0] = 1.0  # gap of 37 samples from end of run 2 (> 32 required)
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_slm_gap_violations(stim_array, active_threshold=0.0)

    assert len(violations) == 1
    assert violations[0].gap_samples == 5
    assert violations[0].required_gap_samples == 32


def test_find_slm_gap_violations_ignores_transitions_without_new_neurons() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[20:23, 0] = 1.0
    values[20:23, 1] = 1.0
    values[23:26, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_slm_gap_violations(stim_array, active_threshold=0.0)

    assert len(violations) == 0


def test_find_slm_off_on_overlap_violations_detects_overlap() -> None:
    values = np.zeros((120, 2), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[40:42, 0] = 1.0
    values[40:42, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_slm_off_on_overlap_violations(
        stim_array,
        active_threshold=0.0,
    )

    assert len(violations) >= 1
    assert violations[0].required_shift_samples > 0


def test_find_initial_slm_load_violation_detects_too_early_first_stimulus() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violation = find_initial_slm_load_violation(stim_array, active_threshold=0.0)

    assert violation is not None
    assert violation.first_active_idx == 0
    assert violation.required_start_idx == 32


def test_resample_stim_values_to_dt_downsamples_via_nearest_neighbor() -> None:
    # old_dt=0.001s, new_dt=0.002s (exactly double): each new tick lands exactly on an
    # even-indexed old sample (0*0.001, 2*0.001, 4*0.001, ...), so the expected mapping is
    # unambiguous -- no rounding-parity edge cases to worry about.
    values = np.zeros((10, 1), dtype=np.float32)
    values[2:6, 0] = 1.0  # active at old indices 2, 3, 4, 5
    stim_array = StimArray(
        name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=0.001
    )

    new_values, n_new = resample_stim_values_to_dt(stim_array, new_dt=0.002)

    assert n_new == 5
    assert new_values.shape == (5, 1)
    # new sample i maps to old index 2*i -> old[0,2,4,6,8] -> [0, 1, 1, 0, 0]
    assert list(new_values[:, 0]) == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_resample_stim_values_to_dt_preserves_total_duration_when_upsampling() -> None:
    values = np.zeros((5, 1), dtype=np.float32)
    values[2, 0] = 1.0
    stim_array = StimArray(
        name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=0.002
    )

    new_values, n_new = resample_stim_values_to_dt(stim_array, new_dt=0.001)

    # duration = 5 * 0.002s = 0.01s -> round(0.01 / 0.001) = 10 new samples
    assert n_new == 10
    assert new_values.shape == (10, 1)
    # the original pulse at old index 2 (t=0.004s) should still be "on" somewhere near
    # the corresponding new index (t=0.004s / 0.001s = 4).
    assert new_values[4, 0] == 1.0


def test_resample_stim_values_to_dt_breaks_ties_consistently() -> None:
    # An exact 2x upsample puts every other tick halfway between two samples;
    # round-half-to-even used to alternate the tie direction and jitter pulse widths.
    values = np.zeros((6, 1), dtype=np.float32)
    values[1, 0] = 1.0
    values[3, 0] = 1.0
    stim_array = StimArray("stim", StimType.Excitation, values, "1", 0.002)

    new_values, n_new = resample_stim_values_to_dt(stim_array, new_dt=0.001)

    assert n_new == 12
    assert new_values[:, 0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]


def test_resample_stim_values_to_dt_tolerates_float_error_at_sample_boundaries() -> (
    None
):
    # 1 ms -> 100 us: tick 110 sits exactly on old sample 11 (110 * 1e-4 / 1e-3 is
    # 11.000000000000002 in floats); 100 us -> 1 ms: 950 * 2e-5 / 1e-3 rounds to 19.
    values = np.arange(20, dtype=np.float32)[:, np.newaxis]
    upsampled, _ = resample_stim_values_to_dt(
        StimArray("stim", StimType.Excitation, values, "1", 0.001), new_dt=0.0001
    )
    assert upsampled[110, 0] == 11
    _, n_new = resample_stim_values_to_dt(
        StimArray("s", StimType.Excitation, np.zeros((950, 1)), "1", 0.00002),
        new_dt=0.001,
    )
    assert n_new == 19


def test_resample_stim_values_to_dt_rejects_non_2d_input() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((10, 1), dtype=np.float32),
        units="1",
        dt=0.001,
    )
    stim_array.values = np.zeros(
        10, dtype=np.float32
    )  # set after construction: 1D, invalid

    with pytest.raises(ValueError, match="2D"):
        resample_stim_values_to_dt(stim_array, new_dt=0.002)


def test_resample_stim_values_to_dt_handles_zero_timepoints() -> None:
    """Regression: n_old=0 used to force n_new=max(1, round(0/new_dt))=1, then
    index the empty array with np.clip(idx, 0, -1) -> IndexError, before the
    caller's own "invalid input" sanity check ever got a chance to run."""
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((10, 2), dtype=np.float32),
        units="1",
        dt=0.001,
    )
    stim_array.values = np.zeros((0, 2), dtype=np.float32)  # set after construction

    new_values, n_new = resample_stim_values_to_dt(stim_array, new_dt=0.002)

    assert n_new == 0
    assert new_values.shape == (0, 2)


def test_shift_stim_values_right_for_initial_slm_load_moves_all_inputs() -> None:
    values = np.zeros((72, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[10:12, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, truncated_tail_samples = shift_stim_values_right_for_initial_slm_load(
        stim_array
    )
    shifted = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=stim_array.dt,
    )

    assert truncated_tail_samples == 32
    assert np.all(new_values[0:32, :] == 0.0)
    assert np.all(new_values[32:35, 0] == 1.0)
    assert np.all(new_values[42:44, 1] == 1.0)
    assert find_initial_slm_load_violation(shifted, active_threshold=0.0) is None


def test_space_out_stim_values_for_slm_resolves_violations() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:11, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, truncated_tail_samples = space_out_stim_values_for_slm(
        stim_array,
        active_threshold=0.0,
    )
    resolved = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=_DT_S,
    )

    assert len(find_slm_gap_violations(resolved, active_threshold=0.0)) == 0
    assert truncated_tail_samples > 0


def test_align_stim_values_left_for_slm_resolves_violations() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:11, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, moved_runs = align_stim_values_left_for_slm(
        stim_array,
        active_threshold=0.0,
    )
    resolved = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=_DT_S,
    )

    assert moved_runs > 0
    assert len(find_slm_gap_violations(resolved, active_threshold=0.0)) == 0


def test_space_out_stim_values_for_slm_off_on_overlap_resolves_violations() -> None:
    values = np.zeros((140, 2), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[40:42, 0] = 1.0
    values[40:42, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, truncated_tail_samples = space_out_stim_values_for_slm_off_on_overlap(
        stim_array,
        active_threshold=0.0,
    )
    resolved = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=_DT_S,
    )

    assert truncated_tail_samples > 0
    assert len(find_slm_off_on_overlap_violations(resolved, active_threshold=0.0)) == 0


def test_export_seq_restarts_and_aligns_objects_when_new_target_appears() -> None:
    values = np.zeros((140, 2), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[60:62, 0] = 1.0
    values[60:62, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
    )
    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    shape_starts = sorted(float(obj.start_time_us) for obj in shapes)

    assert len(shapes) == 3
    assert shape_starts[0] < shape_starts[1]
    assert np.isclose(shape_starts[1], shape_starts[2])


def test_find_simultaneous_duration_violations_detects_unequal_durations() -> None:
    values = np.zeros((30, 3), dtype=np.float32)
    values[2:10, 0] = 1.0
    values[2:6, 1] = 1.0
    values[2:10, 2] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_simultaneous_duration_violations(
        stim_array,
        active_threshold=0.0,
    )

    assert len(violations) == 1
    violation = violations[0]
    assert violation.start_idx == 2
    assert violation.neuron_indices == (0, 1, 2)
    assert violation.durations_idx == (8, 4, 8)
    assert violation.min_duration_idx == 4


def test_equalize_simultaneous_durations_to_min_trims_longer_segments() -> None:
    values = np.zeros((30, 3), dtype=np.float32)
    values[2:10, 0] = 1.0
    values[2:6, 1] = 1.0
    values[2:10, 2] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, trimmed_timepoints = equalize_simultaneous_durations_to_min(
        stim_array,
        active_threshold=0.0,
    )
    resolved = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=_DT_S,
    )

    assert trimmed_timepoints == 8
    assert np.all(new_values[2:6, 0] == 1.0)
    assert np.all(new_values[6:10, 0] == 0.0)
    assert np.all(new_values[2:6, 2] == 1.0)
    assert np.all(new_values[6:10, 2] == 0.0)
    assert (
        len(find_simultaneous_duration_violations(resolved, active_threshold=0.0)) == 0
    )


def test_find_overlapping_pulse_group_violations_detects_staggered_overlap() -> None:
    # Neuron 0: [40:48], neuron 1: [44:52] — overlap at [44:48], different starts and durations
    values = np.zeros((80, 2), dtype=np.float32)
    values[40:48, 0] = 1.0
    values[44:52, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_overlapping_pulse_group_violations(
        stim_array, active_threshold=0.0
    )

    assert len(violations) == 1
    v = violations[0]
    assert set(v.segment_neuron_indices) == {0, 1}
    assert v.aligned_start_idx == 40  # leftmost start
    assert (
        v.aligned_duration_idx == 8
    )  # both have duration 8 — different starts trigger violation


def test_find_overlapping_pulse_group_violations_no_violation_for_non_overlapping() -> (
    None
):
    values = np.zeros((80, 2), dtype=np.float32)
    values[10:20, 0] = 1.0
    values[30:40, 1] = 1.0  # gap between them — no overlap
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_overlapping_pulse_group_violations(
        stim_array, active_threshold=0.0
    )
    assert violations == []


def test_find_overlapping_pulse_group_violations_no_violation_when_already_aligned() -> (
    None
):
    # Both neurons exactly the same start and duration — no violation
    values = np.zeros((80, 2), dtype=np.float32)
    values[40:46, 0] = 1.0
    values[40:46, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_overlapping_pulse_group_violations(
        stim_array, active_threshold=0.0
    )
    assert violations == []


def test_align_overlapping_pulses_to_shortest_fixes_staggered_overlap() -> None:
    # Neuron 0: [40:45] (dur 5), neuron 1: [43:51] (dur 8) — overlap, different starts
    values = np.zeros((80, 2), dtype=np.float32)
    values[40:45, 0] = 1.0
    values[43:51, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    new_values, modified = align_overlapping_pulses_to_shortest(
        stim_array, active_threshold=0.0
    )

    # Both should be at aligned_start=40, aligned_duration=5 → [40:45]
    assert modified > 0
    assert np.all(new_values[40:45, 0] == 1.0)
    assert np.all(new_values[45:, 0] == 0.0)
    assert np.all(new_values[40:45, 1] == 1.0)
    assert np.all(new_values[45:, 1] == 0.0)
    assert np.all(new_values[:40, :] == 0.0)
    # No more violations after fix
    fixed = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=new_values,
        units="1",
        dt=_DT_S,
    )
    assert find_overlapping_pulse_group_violations(fixed, active_threshold=0.0) == []


def test_align_overlapping_pulses_to_shortest_transitive_grouping() -> None:
    # Neuron 0: [100:110], neuron 1: [105:120], neuron 2: [115:125]
    # 0 overlaps 1, 1 overlaps 2 → all three in one group
    values = np.zeros((200, 3), dtype=np.float32)
    values[100:110, 0] = 1.0
    values[105:120, 1] = 1.0
    values[115:125, 2] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    violations = find_overlapping_pulse_group_violations(
        stim_array, active_threshold=0.0
    )
    assert len(violations) == 1
    assert set(violations[0].segment_neuron_indices) == {0, 1, 2}
    assert violations[0].aligned_start_idx == 100
    assert violations[0].aligned_duration_idx == 10  # neuron 0 has shortest (10)

    new_values, _ = align_overlapping_pulses_to_shortest(
        stim_array, active_threshold=0.0
    )
    assert np.all(new_values[100:110, 0] == 1.0)
    assert np.all(new_values[100:110, 1] == 1.0)
    assert np.all(new_values[100:110, 2] == 1.0)
    assert np.all(new_values[110:, :] == 0.0)


def test_check_syscon2_input_sanity_auto_fixes_overlapping_pulses() -> None:
    # Neuron 0: [64:70] (dur 6), neuron 1: [67:75] (dur 8) — staggered overlap
    values = np.zeros((128, 2), dtype=np.float32)
    values[64:70, 0] = 1.0
    values[67:75, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.sanitized_values is not None
    assert len(sanity.fix_messages) >= 1
    assert any(
        "overlapping" in m.lower() or "aligned" in m.lower()
        for m in sanity.fix_messages
    )
    # After fix: both neurons at [64:70]
    assert np.all(sanity.sanitized_values[64:70, 0] == 1.0)
    assert np.all(sanity.sanitized_values[64:70, 1] == 1.0)
    assert np.all(sanity.sanitized_values[70:, :] == 0.0)


def test_export_clips_pulse_to_sanitized_window_extent_when_tail_is_truncated() -> None:
    # Regression test for: "laser pulse end exceeds the allowed end for an overlapping
    # shape object (pulse_ticks=[125, 131), shape_ticks=[64, 128))".
    #
    # Geometry (32-tick windows, 4 windows = 128 ticks):
    #   Neuron 0: ticks [35:38] → window 1 → hologram A (window 0 is blank preload ✓)
    #   Neuron 1: ticks [93:99] → straddles window 2 (64–95) and window 3 (96–127)
    #             → new hologram B (neuron 1 never seen); first active window = 2,
    #               but window 1 is occupied → sanitizer inserts preblank at window 2,
    #               shifting all hologram-B content right by 32 ticks.
    #   After insertion: neuron-1 segment → [125:131].
    #   Shape for hologram B covers windows 2–3 → shape end = 4 * 32 = 128.
    #   Without the clip fix: pulse [125:131] overruns shape end 128 → validation error.
    #   With the clip fix: pulse clipped to [125:128] → export succeeds.
    values = np.zeros((128, 2), dtype=np.float32)
    values[35:38, 0] = 1.0  # neuron 0: window 1
    values[93:99, 1] = 1.0  # neuron 1: straddles windows 2 and 3
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    # Must succeed without raising a pulse-overrun validation error.
    seq_text, _ = generate_syscon2_sequence_text(
        stim,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        spiralling=False,
    )
    assert "TTLPulse" in seq_text


# ---------------------------------------------------------------------------
# generate_syscon2_sequence_text — input-validation error branches
# ---------------------------------------------------------------------------


def _stim_1d() -> StimArray:
    return StimArray(
        name="s",
        stim_type=StimType.Excitation,
        values=np.zeros(10, dtype=np.float32),
        units="1",
        dt=_DT_S,
    )


def _stim_active() -> StimArray:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    return StimArray(
        name="s",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )


def test_generate_syscon2_sequence_text_raises_for_1d_values() -> None:
    with pytest.raises(ValueError, match="2D"):
        generate_syscon2_sequence_text(
            _stim_1d(),
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_for_non_positive_tick_resolution() -> (
    None
):
    with pytest.raises(ValueError, match="tick_resolution_us"):
        generate_syscon2_sequence_text(
            _stim_active(),
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            tick_resolution_us=-1,
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_for_out_of_range_binary_intensity() -> (
    None
):
    with pytest.raises(ValueError, match="binary_intensity"):
        generate_syscon2_sequence_text(
            _stim_active(),
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            binary_intensity=1.5,
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_when_cellpose_data_is_none() -> None:
    with pytest.raises(ValueError, match="Cellpose data is required"):
        generate_syscon2_sequence_text(
            _stim_active(),
            None,
            template_text=_template_header(),
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_when_segmentation_array_is_none() -> (
    None
):
    import numpy as _np

    from neuroseq.importers.importer_cellpose import (
        CellposeImportResult as _CellposeImportResult,
    )

    no_seg = _CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=None,
        overlay_image=_np.zeros((2, 4, 3), dtype=_np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=_np.array([0, 1], dtype=_np.int32),
        stimulus_to_cellpose_mapping=None,
    )
    with pytest.raises(ValueError, match="Cellpose segmentation is required"):
        generate_syscon2_sequence_text(
            _stim_active(),
            no_seg,
            template_text=_template_header(),
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_for_unsupported_wavelength() -> None:
    with pytest.raises(ValueError, match="active_laser_wavelength_nm"):
        generate_syscon2_sequence_text(
            _stim_active(),
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            active_laser_wavelength_nm=999,
            spiralling=False,
        )


def test_generate_syscon2_sequence_text_raises_when_no_active_stimulation() -> None:
    values = np.zeros((40, 1), dtype=np.float32)  # no active samples
    stim = StimArray(
        name="s",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    with pytest.raises(ValueError, match="No active input"):
        generate_syscon2_sequence_text(
            stim,
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            spiralling=False,
        )


# ---------------------------------------------------------------------------
# _find_pulse_overrun_violations
# ---------------------------------------------------------------------------


def _make_scheduled_window(
    start_tick: int,
    end_tick: int,
    *,
    neuron_index: int = 0,
    lane: int = 0,
) -> _ScheduledEntityWindow:
    return _ScheduledEntityWindow(
        neuron_index=neuron_index,
        stimulation_start_tick=start_tick,
        stimulation_end_tick=end_tick,
        start_tick=start_tick,
        end_tick=end_tick,
        intensity=1.0,
        center_x=0.0,
        center_y=0.0,
        lane=lane,
    )


def test_find_pulse_overrun_violations_empty_when_pulse_within_window() -> None:
    windows = [_make_scheduled_window(0, 64)]
    violations = _find_pulse_overrun_violations([(0, 32)], windows)
    assert violations == []


def test_find_pulse_overrun_violations_skips_degenerate_intervals() -> None:
    windows = [_make_scheduled_window(0, 64)]
    # pulse_end <= pulse_start → degenerate, must be skipped
    violations = _find_pulse_overrun_violations([(10, 5)], windows)
    assert violations == []


def test_find_pulse_overrun_violations_detects_overrun() -> None:
    windows = [_make_scheduled_window(0, 32)]
    # pulse ends at 40, past window end 32
    violations = _find_pulse_overrun_violations([(0, 40)], windows)
    assert len(violations) == 1
    v = violations[0]
    assert v.pulse_start_tick == 0
    assert v.pulse_end_tick == 40
    assert v.shape_start_tick == 0
    assert v.shape_end_tick == 32
    assert v.max_pulse_end_tick == 32
    assert v.neuron_index == 0
    assert v.shape_lane == 0


# ---------------------------------------------------------------------------
# check_syscon2_input_sanity — pulse overrun fix values
# ---------------------------------------------------------------------------


def test_check_syscon2_input_sanity_computes_pulse_overrun_fix_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pulse_overrun_fix_values is only propagated on the is_valid=False path (when
    # generate_syscon2_sequence_text raises).  Patch both:
    #   1. _find_pulse_overrun_violations → return a fake violation so the fix is computed
    #   2. generate_syscon2_sequence_text → raise so the is_valid=False return is taken
    values = np.zeros((64, 1), dtype=np.float32)
    values[32:36, 0] = 1.0
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    fake_violation = PulseOverrunViolation(
        pulse_start_tick=32,
        pulse_end_tick=40,
        shape_start_tick=32,
        shape_end_tick=36,
        max_pulse_end_tick=36,
        shape_lane=0,
        neuron_index=0,
    )
    monkeypatch.setattr(
        "neuroseq.export.exporter_syscon2._find_pulse_overrun_violations",
        lambda *a, **kw: [fake_violation],
    )
    monkeypatch.setattr(
        "neuroseq.export.exporter_syscon2.generate_syscon2_sequence_text",
        lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("laser pulse end exceeds shape boundary")
        ),
    )

    report = check_syscon2_input_sanity(
        stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert not report.is_valid
    assert report.pulse_overrun_fix_values is not None
    # Fix zeros [max_pulse_end_tick, pulse_end_tick) = [36, 40)
    assert np.all(report.pulse_overrun_fix_values[36:40, :] == 0.0)


def test_compute_cell_centroids_uses_top_left_origin() -> None:
    """SysCon ``.seq`` uses a top-left origin, like the stored segmentation
    (#601): a cell in the top row has y = 0."""
    segmentation = np.full((4, 3), -1, dtype=np.int32)
    segmentation[0, 0:2] = 0

    centroids = _compute_cell_centroids(segmentation)

    assert 0 in centroids
    x, y = centroids[0]
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.0)


def test_compute_cell_centroids_bottom_cell_maps_to_high_y() -> None:
    """A cell in the bottom row (H - 1) has y = H - 1."""
    segmentation = np.full((4, 3), -1, dtype=np.int32)
    segmentation[3, 0:2] = 0

    centroids = _compute_cell_centroids(segmentation)

    x, y = centroids[0]
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(3.0)  # H - 1 with H = 4


def test_compute_cell_centroids_x_axis_is_unchanged() -> None:
    """X centroids are unaffected (the importer does not horizontally flip)."""
    segmentation = np.full((2, 6), -1, dtype=np.int32)
    segmentation[0, 4:6] = 0

    centroids = _compute_cell_centroids(segmentation)

    x, _ = centroids[0]
    assert x == pytest.approx(4.5)


def test_roundtrip_revisiting_point_target_preserves_neuron_channel():
    values = np.zeros((256, 2), dtype=np.float32)
    values[33:36, 0] = 1
    values[129:132, 1] = 1
    values[225:228, 0] = 1
    stim = StimArray("repeat", StimType.Excitation, values, "1", _DT_S)
    text, _ = generate_syscon2_sequence_text(
        stim,
        _cellpose_result_many(n_neurons=2, mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        spiralling=False,
    )
    imported, report = sequence_to_stim_values(
        parse_syscon2_seq_text(text),
        dt_seconds=_DT_S,
        n_neurons=2,
    )
    assert report.dropped_objects == 0
    assert report.merged_objects >= 1
    np.testing.assert_array_equal(imported, values[: imported.shape[0]])


@pytest.mark.parametrize("wavelength", [940, 1030, 1100])
def test_lut_export_writes_normalized_power_fraction(wavelength):
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1
    stim = StimArray("power", StimType.Excitation, values, "1", _DT_S)
    text, _ = generate_syscon2_sequence_text(
        stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        target_power_mw=5,
        power_lut=_linear_power_lut(85),
        active_laser_wavelength_nm=wavelength,
        spiralling=False,
    )
    objects = parse_syscon2_seq_text(text).objects
    intensities = [
        float(obj.properties["TimelineInfo0_Intensity_0"])
        for obj in objects
        if "TimelineInfo0_Intensity_0" in obj.properties
    ]
    nonzero = [value for value in intensities if value > 0]
    assert len(nonzero) >= 2  # both Shape and laser TTL
    assert all(value * 85 == pytest.approx(5, abs=0.0001) for value in nonzero)


def test_multi_spot_roundtrip_keeps_rows_and_timings() -> None:
    n_neurons = 6
    values = np.zeros((300, n_neurons), dtype=np.float32)
    starts = {1: [100, 107, 114], 4: [100, 107, 114], 5: [200, 207]}
    for neuron, burst_starts in starts.items():
        for start_idx in burst_starts:
            values[start_idx : start_idx + 2, neuron] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    # Each ROI is a 3x3 block so that several spots per ROI can be placed.
    segmentation = np.full((3, n_neurons * 10), -1, dtype=np.int32)
    for neuron in range(n_neurons):
        segmentation[:, neuron * 10 : neuron * 10 + 3] = neuron
    cellpose = CellposeImportResult(
        image_path=Path("dummy_spots.png"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((3, n_neurons * 10, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.arange(n_neurons, dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
    )
    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        cellpose,
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
        stimulation_spots_per_roi=3,
    )
    sequence = parse_syscon2_seq_text(seq_text)
    assert len([o for o in sequence.objects if o.object_type == "Shape"]) > 3
    imported, _ = sequence_to_stim_values(
        sequence,
        dt_seconds=_DT_S,
        n_neurons=n_neurons,
        tick_resolution_us=500,
    )
    assert imported.shape[1] == n_neurons
    assert np.flatnonzero(imported.any(axis=0)).tolist() == [1, 4, 5]
    for neuron in (1, 4, 5):
        got = imported[:, neuron] > 0
        assert (
            np.flatnonzero(got).tolist() == np.flatnonzero(values[:, neuron]).tolist()
        )


def _export_and_reimport(values: np.ndarray, *, exact: bool) -> np.ndarray:
    n_neurons = values.shape[1]
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result_many(n_neurons=n_neurons),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
        exact_slm_timing=exact,
    )
    imported, _ = sequence_to_stim_values(
        parse_syscon2_seq_text(seq_text),
        dt_seconds=_DT_S,
        n_neurons=n_neurons,
        tick_resolution_us=500,
    )
    return imported


def _starts(column: np.ndarray) -> list[int]:
    return np.flatnonzero(np.diff((column > 0).astype(int), prepend=0) == 1).tolist()


def test_exact_slm_timing_keeps_timings_when_gaps_leave_room() -> None:
    values = np.zeros((900, 4), dtype=np.float32)
    values[100:292, 0] = 1.0
    values[330:522, 1] = 1.0
    values[600:792, 2] = 1.0
    imported = _export_and_reimport(values, exact=True)
    for neuron in range(3):
        assert _starts(imported[:, neuron]) == _starts(values[:, neuron])
        assert imported[:, neuron].sum() == values[:, neuron].sum()


def test_exact_slm_timing_delays_only_as_much_as_the_load_window_needs() -> None:
    values = np.zeros((1200, 3), dtype=np.float32)
    values[0:192, 0] = 1.0  # needs the initial load window: delayed by 32
    values[200:392, 1] = 1.0  # 8-tick gap: needs a delay, but less than a window
    values[900:1092, 2] = 1.0
    imported = _export_and_reimport(values, exact=True)
    assert _starts(imported[:, 0]) == [32]
    delay_1 = _starts(imported[:, 1])[0] - 200
    assert 0 < delay_1 < 32 * 2 + 1
    assert _starts(imported[:, 2]) == [900]


# check this works with both LF (linux) and CRLF (windows) line endings
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_original_stim_sidecar_restores_exact_timings_and_rows(
    tmp_path: Path, newline: str
) -> None:
    from neuroseq.importers.importer_syscon2 import original_stim_sidecar_path
    from neuroseq.importers.importer_syscon2 import parse_syscon2_seq_file
    from neuroseq.importers.importer_syscon2 import write_original_stim_sidecar

    n_neurons = 5
    values = np.zeros((900, n_neurons), dtype=np.float32)
    values[0:193, 0] = 1.0  # needs delays on export
    values[227:419, 1] = 0.5
    values[454:646, 3] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result_many(n_neurons=n_neurons),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1100,
        spiralling=False,
        exact_slm_timing=True,
    )
    seq_path = tmp_path / "script.seq"
    seq_path.write_text(seq_text, encoding="utf-8", newline=newline)

    plain = parse_syscon2_seq_file(seq_path)
    assert plain.original_values is None
    drifted, _ = sequence_to_stim_values(
        plain, dt_seconds=_DT_S, n_neurons=n_neurons, tick_resolution_us=500
    )
    assert _starts(drifted[:, 1]) != [227]

    write_original_stim_sidecar(
        original_stim_sidecar_path(seq_path), values, _DT_S, seq_text
    )
    restored_seq = parse_syscon2_seq_file(seq_path)
    restored, result = sequence_to_stim_values(
        restored_seq, dt_seconds=_DT_S, n_neurons=n_neurons, tick_resolution_us=500
    )
    assert np.array_equal(restored, values)
    assert result.grouped_objects == n_neurons

    # A sidecar that does not belong to the file is ignored.
    seq_path.write_text(seq_text + "\n", encoding="utf-8", newline=newline)
    assert parse_syscon2_seq_file(seq_path).original_values is None


@pytest.mark.parametrize("exact_slm_timing", [True, False])
def test_sanity_fixes_extend_array_instead_of_dropping_shifted_pulses(
    exact_slm_timing: bool,
) -> None:
    """Preblank/SLM-load shifts push neuron 1's pulse past tick 100; the fixed
    array must grow to keep it rather than silently cutting it."""
    values = np.zeros((100, 2), dtype=np.float32)
    values[0:10, 0] = 1.0
    values[40:50, 1] = 1.0
    stim = StimArray(
        name="s", stim_type=StimType.External, values=values, units="", dt=_DT_S
    )

    report = check_syscon2_input_sanity(
        stim, None, template_text="", exact_slm_timing=exact_slm_timing
    )

    fixed = report.sanitized_values
    assert fixed is not None
    assert fixed.shape[0] > values.shape[0]
    np.testing.assert_array_equal((fixed > 0).sum(axis=0), [10, 10])
    assert any("extended from 100" in warning for warning in report.warnings)
    # No phantom "truncated" warning for windows that were empty anyway.
    assert not any("truncated" in warning for warning in report.warnings)


def test_preblank_insertion_grows_window_matrix_only_when_needed() -> None:
    from neuroseq.export.exporter_syscon2 import _enforce_preblank_windows

    # Trailing windows are blank: shifting loses nothing, nothing is appended.
    windows = np.zeros((4, 2), dtype=np.float32)
    windows[0, 0] = 1.0
    windows[1, 1] = 1.0
    adjusted, _, fixes, extended, _ = _enforce_preblank_windows(windows)
    assert extended == 0
    assert adjusted.shape == (4, 2)
    assert len(fixes) == 2

    # Last window active: the shift must append a window instead of dropping it.
    windows = np.zeros((3, 1), dtype=np.float32)
    windows[1:3, 0] = 1.0
    adjusted, _, _, extended, _ = _enforce_preblank_windows(windows)
    assert extended == 0  # window 0 is already a blank preload
    windows = np.zeros((2, 2), dtype=np.float32)
    windows[0, 0] = 1.0
    windows[1, 1] = 1.0
    adjusted, _, _, extended, _ = _enforce_preblank_windows(windows)
    assert extended == 2
    np.testing.assert_array_equal(adjusted.sum(axis=0), [1.0, 1.0])
