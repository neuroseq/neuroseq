from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from PIL import Image

from neuroseq.exporter_syscon2 import PulseOverrunViolation
from neuroseq.exporter_syscon2 import _compute_cell_centroids
from neuroseq.exporter_syscon2 import _find_pulse_overrun_violations
from neuroseq.exporter_syscon2 import _ScheduledEntityWindow
from neuroseq.exporter_syscon2 import align_overlapping_pulses_to_shortest
from neuroseq.exporter_syscon2 import align_stim_values_left_for_slm
from neuroseq.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.exporter_syscon2 import equalize_simultaneous_durations_to_min
from neuroseq.exporter_syscon2 import find_initial_slm_load_violation
from neuroseq.exporter_syscon2 import find_overlapping_pulse_group_violations
from neuroseq.exporter_syscon2 import find_simultaneous_duration_violations
from neuroseq.exporter_syscon2 import find_slm_gap_violations
from neuroseq.exporter_syscon2 import find_slm_off_on_overlap_violations
from neuroseq.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.exporter_syscon2 import shift_stim_values_right_for_initial_slm_load
from neuroseq.exporter_syscon2 import space_out_stim_values_for_slm
from neuroseq.exporter_syscon2 import space_out_stim_values_for_slm_off_on_overlap
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.importer_syscon2 import sequence_to_stim_values
from neuroseq.power_lut import PowerLut
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

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
    assert len(ttl_out_1_pulses) == 1
    assert len(ttl_out_2_pulses) == 0
    # TTL pulse comes from the adjusted raw segment position (tick 20 shifted by
    # one preblank window to tick 52 → 52 × 520.8125µs = 27082.25µs).
    assert ttl_out_1_pulses[0].start_time_us == pytest.approx(27082.25)


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
    # Shape intensity is now always exported as 0 to match SysCon v3 reference
    # files, so importer roundtrip via Shape windows no longer reconstructs the
    # original ON samples from exporter output alone.
    assert np.count_nonzero(imported_values[:, 0] > 0) == 0


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
    assert len(ttl_out_1_pulses) == 1


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
    ]
    ttl_out_1_off = [
        obj
        for obj in seq_off.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report_on.ttl_out1_timeline_index
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
        assert shape.properties["TimelineInfo0_LightsourceID_0"] == "dummy_2_0"
        assert shape.properties["TimelineInfo0_Intensity_0"] == "0"


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
    # Shape intensity is now always 0 in the exported sequence, so roundtrip via
    # Shape windows no longer reconstructs active samples even though the laser
    # TTLPulse timing itself is preserved above.
    segment_counts = [
        _count_active_segments(imported_values[:, idx]) for idx in range(6)
    ]
    assert segment_counts == [0, 0, 0, 0, 0, 0]


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

    assert shapes[0].properties["TimelineInfo0_Intensity_0"] == "50"
    assert laser_pulses[0].properties["TimelineInfo0_Intensity_0"] == "50"
    assert report.power_saturation_warnings == []
    assert report.target_power_mw == 25.0
    assert report.power_lut_description == ""


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

    assert shapes[0].properties["TimelineInfo0_Intensity_0"] == "100"
    assert laser_pulses[0].properties["TimelineInfo0_Intensity_0"] == "100"
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
        40.0
    )
    assert any(
        float(pulse.properties["TimelineInfo0_Intensity_0"]) == pytest.approx(40.0)
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
    assert report.generated_objects == 3
    assert len(entities) == 1
    assert len(pulses) == 2

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
        "no active stimulation samples found" in error.lower()
        for error in sanity.errors
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

    from neuroseq.importer_cellpose import CellposeImportResult as _CellposeImportResult

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
    with pytest.raises(ValueError, match="No active stimulation"):
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
        "neuroseq.exporter_syscon2._find_pulse_overrun_violations",
        lambda *a, **kw: [fake_violation],
    )
    monkeypatch.setattr(
        "neuroseq.exporter_syscon2.generate_syscon2_sequence_text",
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
    """SysCon ``.seq`` uses top-left origin; the cellpose importer applies
    ``np.flipud`` on load (Fiji convention), so the exporter must undo that
    flip when producing centroids — otherwise SysCon renders cells mirrored
    along the Y axis. See ``_compute_cell_centroids`` docstring."""
    # 4×3 image, single cell occupying the *top-left* corner of the original
    # (pre-flipud) image — i.e. row 0, cols 0-1. After importer_cellpose
    # applies np.flipud, the cell sits at the *bottom-left* of the stored
    # segmentation. The exporter must report its centroid using the original
    # top-left frame: x ≈ 0.5, y = 0.
    segmentation = np.full((4, 3), -1, dtype=np.int32)
    segmentation[3, 0:2] = 0  # post-flipud position of the top-row cell

    centroids = _compute_cell_centroids(segmentation)

    assert 0 in centroids
    x, y = centroids[0]
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.0)


def test_compute_cell_centroids_bottom_cell_maps_to_high_y() -> None:
    """The mirror case: a cell at the bottom of the original image (row H-1)
    has been ``flipud``-mapped to row 0 of the stored array. The exporter
    must report y = H - 1 for it."""
    segmentation = np.full((4, 3), -1, dtype=np.int32)
    segmentation[0, 0:2] = 0  # post-flipud position of the bottom-row cell

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
