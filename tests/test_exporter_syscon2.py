from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np

from neuroseq.exporter_syscon2 import generate_syscon2_sequence_text
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


def test_export_seq_uses_segment_compression_and_separate_ttl_lanes() -> None:
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
    )

    sequence = parse_syscon2_seq_text(seq_text)
    entities = [obj for obj in sequence.objects if obj.object_type == "Entity"]
    pulses = [obj for obj in sequence.objects if obj.object_type == "TTLPulse"]

    assert report.total_segments == 2
    assert report.exported_segments == 2
    assert report.entity_lanes == 2
    assert report.unmapped_active_neurons == []
    assert len(entities) == 2
    assert len(pulses) == 2

    entity_lanes = sorted(
        int(obj.properties["TimelineInfo0_TimelineIndex"]) for obj in entities
    )
    pulse_lanes = sorted(
        int(obj.properties["TimelineInfo0_TimelineIndex"]) for obj in pulses
    )
    assert entity_lanes == [0, 1]
    # TTLPulse lanes are written as lane + n_entity_lanes.
    assert pulse_lanes == [2, 3]

    assert {obj.start_time_us for obj in entities} == {1000}
    assert all(obj.repeats == 3 for obj in pulses)
    assert all(obj.intensity == 1.0 for obj in entities)


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
    assert report.generated_objects == 2
    assert len(entities) == 1
    assert len(pulses) == 1

    assert int(entities[0].properties["TimelineInfo0_TimelineIndex"]) == 0
    assert int(pulses[0].properties["TimelineInfo0_TimelineIndex"]) == 1


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
