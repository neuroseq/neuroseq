from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from PIL import Image

from neuroseq.exporter_syscon2 import align_stim_values_left_for_slm
from neuroseq.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.exporter_syscon2 import equalize_simultaneous_durations_to_min
from neuroseq.exporter_syscon2 import find_initial_slm_load_violation
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
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

_TTL_QUANTUM_US = 520.8125


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
        dt=0.001,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.entity_lanes == 2
    assert report.laser_timeline_index == 4
    assert report.ttl_out1_timeline_index == 5
    assert report.ttl_out2_timeline_index == 6
    assert report.unmapped_active_neurons == []

    assert len(entities) == 2
    entity_lanes = sorted(_timeline_index(obj) for obj in entities)
    assert entity_lanes == [0, 1]
    assert all(obj.repeats >= 0 for obj in entities)
    assert all(obj.step_size == 100 for obj in entities)

    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 4]
    ttl_out_1_pulses = [obj for obj in pulses if _timeline_index(obj) == 5]
    ttl_out_2_pulses = [obj for obj in pulses if _timeline_index(obj) == 6]

    assert len(laser_pulses) == 1
    assert len(ttl_out_1_pulses) == 1
    assert len(ttl_out_2_pulses) == 0
    assert ttl_out_1_pulses[0].start_time_us == 16_666.0


def test_export_seq_coalesces_nearby_segments_into_one_sanitized_window() -> None:
    values = np.zeros((80, 1), dtype=np.float32)
    values[40:42, 0] = 1.0
    values[45:47, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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
    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    assert len(entities) == 1
    assert [pulse.start_time_us for pulse in laser_pulses] == [33_332.0]

    imported_values, _ = sequence_to_stim_values(
        sequence,
        dt_seconds=stim_array.dt,
        n_neurons=1,
        tick_resolution_us=500,
        include_non_shape=False,
    )
    assert np.array_equal(np.flatnonzero(imported_values[:, 0] > 0), np.arange(34, 51))


def test_export_seq_routes_ttl_pulses_to_selected_1030_laser_timeline() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1030,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.active_laser_wavelength_nm == 1030
    assert report.entity_lanes == 1
    assert report.laser_timeline_index == 2

    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 2]
    lower_laser_timeline = [obj for obj in pulses if _timeline_index(obj) == 1]
    upper_laser_timeline = [obj for obj in pulses if _timeline_index(obj) == 3]

    assert len(laser_pulses) == 1
    assert len(lower_laser_timeline) == 0
    assert len(upper_laser_timeline) == 0


def test_export_seq_requires_1ms_neuroseq_tick() -> None:
    dt_seconds = _TTL_QUANTUM_US * 1e-6
    values = np.zeros((16, 1), dtype=np.float32)
    values[4:7, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=dt_seconds,
    )

    with pytest.raises(ValueError, match="1ms"):
        generate_syscon2_sequence_text(
            stim_array,
            _cellpose_result(mapping={0: 0}),
            template_text=_template_header(),
            tick_resolution_us=500,
            active_threshold=0.0,
            binary_intensity=1.0,
            active_laser_wavelength_nm=1140,
        )


def test_check_syscon2_input_sanity_rejects_non_1ms_neuroseq_tick() -> None:
    dt_seconds = _TTL_QUANTUM_US * 1e-6
    values = np.zeros((96, 1), dtype=np.float32)
    values[32:65, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=dt_seconds,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
    )

    assert not sanity.is_valid
    assert sanity.export_report is None
    assert any("1ms" in error.lower() or "0.001" in error for error in sanity.errors)


def test_generate_syscon2_timeline_overview_png_writes_file(tmp_path: Path) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
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
        dt=0.001,
    )
    short_seq_text, _ = generate_syscon2_sequence_text(
        short_stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
    )

    long_values = np.zeros((120_000, 1), dtype=np.float32)
    long_values[110_000:110_002, 0] = 1.0
    long_stim = StimArray(
        name="long",
        stim_type=StimType.Excitation,
        values=long_values,
        units="1",
        dt=0.001,
    )
    long_seq_text, _ = generate_syscon2_sequence_text(
        long_stim,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
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
        dt=0.001,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1040,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]
    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 2]

    assert report.active_laser_wavelength_nm == 1030
    assert report.laser_timeline_index == 2
    assert len(laser_pulses) == 1


def test_export_seq_roundtrip_coalesces_dense_bursts_into_group_windows() -> None:
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
        dt=0.001,
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
        active_laser_wavelength_nm=1140,
    )
    sequence = parse_syscon2_seq_text(seq_text)
    laser_pulses = [
        obj
        for obj in sequence.objects
        if obj.object_type == "TTLPulse"
        and _timeline_index(obj) == report.laser_timeline_index
    ]

    assert len(laser_pulses) == 2
    assert [pulse.start_time_us for pulse in laser_pulses] == [83_330.0, 183_326.0]

    imported_values, _ = sequence_to_stim_values(
        sequence,
        dt_seconds=stim_array.dt,
        n_neurons=6,
        tick_resolution_us=500,
        include_non_shape=False,
    )
    segment_counts = [
        _count_active_segments(imported_values[:, idx]) for idx in range(6)
    ]
    assert segment_counts == [1, 1, 1, 1, 1, 1]


def test_export_seq_preserves_non_binary_shape_intensity_when_requested() -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 0.37
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        force_binary_intensity=False,
        active_laser_wavelength_nm=1140,
    )

    sequence = parse_syscon2_seq_text(seq_text)
    shapes = [obj for obj in sequence.objects if obj.object_type == "Shape"]

    assert len(shapes) == 1
    assert np.isclose(shapes[0].intensity, 0.37)


def test_export_seq_skips_unmapped_active_neurons_and_reports_them() -> None:
    values = np.zeros((40, 3), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[25:27, 2] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0}),
        template_text=_template_header(),
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
        dt=0.001,
    )

    seq_text, report = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={1: 0}),
        template_text=_template_header(),
    )
    sequence = parse_syscon2_seq_text(seq_text)

    entities = [obj for obj in sequence.objects if obj.object_type == "Shape"]
    assert len(entities) == 1
    assert report.exported_segments == 1
    assert report.unmapped_active_neurons == []


def test_check_syscon2_input_sanity_passes_for_valid_inputs() -> None:
    values = np.zeros((30, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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
    values = np.zeros((30, 2), dtype=np.float32)
    values[20:22, 0] = 0.2
    values[20:22, 1] = 0.8
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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
    values = np.zeros((30, 1), dtype=np.float32)
    values[1:3, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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
    assert np.array_equal(
        np.flatnonzero(sanity.sanitized_values[:, 0] > 0), np.arange(17, 30)
    )
    assert any("Inserted 17ms blank window" in warning for warning in sanity.warnings)


def test_check_syscon2_input_sanity_fails_without_active_samples() -> None:
    values = np.zeros((4, 2), dtype=np.float32)
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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
        dt=0.001,
    )

    sanity = check_syscon2_input_sanity(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
    )

    assert sanity.is_valid
    assert sanity.export_report is not None
    assert sanity.export_report.exported_segments == 3
    assert any("Inserted 17ms blank window" in warning for warning in sanity.warnings)
    assert sanity.sanitized_values is not None


def test_find_slm_gap_violations_detects_short_gap_between_different_neurons() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:11, 1] = 1.0
    values[35:38, 0] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
    )

    violations = find_slm_gap_violations(stim_array, active_threshold=0.0)

    assert len(violations) == 1
    assert violations[0].gap_samples == 5
    assert violations[0].required_gap_samples == 17


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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
    )

    violation = find_initial_slm_load_violation(stim_array, active_threshold=0.0)

    assert violation is not None
    assert violation.first_active_idx == 0
    assert violation.required_start_idx == 17


def test_shift_stim_values_right_for_initial_slm_load_moves_all_inputs() -> None:
    values = np.zeros((40, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[10:12, 1] = 1.0
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,
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

    assert truncated_tail_samples == 17
    assert np.all(new_values[0:17, :] == 0.0)
    assert np.all(new_values[17:20, 0] == 1.0)
    assert np.all(new_values[27:29, 1] == 1.0)
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
    )

    seq_text, _ = generate_syscon2_sequence_text(
        stim_array,
        _cellpose_result(mapping={0: 0, 1: 1}),
        template_text=_template_header(),
        tick_resolution_us=500,
        active_threshold=0.0,
        binary_intensity=1.0,
        active_laser_wavelength_nm=1140,
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
        dt=0.001,
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
        dt=0.001,
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
        dt=0.001,
    )

    assert trimmed_timepoints == 8
    assert np.all(new_values[2:6, 0] == 1.0)
    assert np.all(new_values[6:10, 0] == 0.0)
    assert np.all(new_values[2:6, 2] == 1.0)
    assert np.all(new_values[6:10, 2] == 0.0)
    assert (
        len(find_simultaneous_duration_violations(resolved, active_threshold=0.0)) == 0
    )
