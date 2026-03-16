from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np

from neuroseq.exporter_syscon2 import align_stim_values_left_for_slm
from neuroseq.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.exporter_syscon2 import find_slm_gap_violations
from neuroseq.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.exporter_syscon2 import space_out_stim_values_for_slm
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


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


def _timeline_index(obj) -> int:
    return int(obj.properties["TimelineInfo0_TimelineIndex"])


def test_export_seq_uses_fixed_laser_and_ttl_out_timeline_layout() -> None:
    values = np.zeros((6, 2), dtype=np.float32)
    values[1:3, 0] = 0.2
    values[1:3, 1] = 0.9
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
    entities = [obj for obj in sequence.objects if obj.object_type == "Entity"]
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.entity_lanes == 2
    assert report.laser_timeline_index == 4
    assert report.ttl_out1_timeline_index == 5
    assert report.ttl_out2_timeline_index == 6
    assert report.unmapped_active_neurons == []

    assert len(entities) == 2
    entity_lanes = sorted(_timeline_index(obj) for obj in entities)
    assert entity_lanes == [0, 1]
    assert all(obj.repeats >= 66 for obj in entities)  # >= 33.32ms minimum at 500us.

    laser_pulses = [obj for obj in pulses if _timeline_index(obj) == 4]
    ttl_out_1_pulses = [obj for obj in pulses if _timeline_index(obj) == 5]
    ttl_out_2_pulses = [obj for obj in pulses if _timeline_index(obj) == 6]

    assert len(laser_pulses) == 1
    assert len(ttl_out_1_pulses) == 2  # startup 10ms pulse + mirrored laser pulse
    assert len(ttl_out_2_pulses) == 0
    assert ttl_out_1_pulses[0].start_time_us == 0


def test_export_seq_skips_unmapped_active_neurons_and_reports_them() -> None:
    values = np.zeros((6, 3), dtype=np.float32)
    values[1:3, 0] = 1.0
    values[2:4, 2] = 1.0
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
    entities = [obj for obj in sequence.objects if obj.object_type == "Entity"]
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
    assert len(pulses) == 3  # one laser pulse + 2 TTL Out 1 pulses

    assert _timeline_index(entities[0]) == 0


def test_export_seq_accepts_one_based_mapping_keys() -> None:
    values = np.zeros((4, 1), dtype=np.float32)
    values[1:3, 0] = 1.0
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

    entities = [obj for obj in sequence.objects if obj.object_type == "Entity"]
    assert len(entities) == 1
    assert report.exported_segments == 1
    assert report.unmapped_active_neurons == []


def test_check_syscon2_input_sanity_passes_for_valid_inputs() -> None:
    values = np.zeros((4, 1), dtype=np.float32)
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
    assert sanity.errors == []
    assert sanity.export_report is not None
    assert sanity.export_report.exported_segments == 1


def test_check_syscon2_input_sanity_warns_for_unmapped_and_non_binary() -> None:
    values = np.zeros((6, 2), dtype=np.float32)
    values[1:3, 0] = 0.2
    values[2:4, 1] = 0.8
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
