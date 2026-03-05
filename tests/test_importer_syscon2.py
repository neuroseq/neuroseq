from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np

from neuroseq.importer_syscon2 import import_sequence_into_stim_array
from neuroseq.importer_syscon2 import parse_syscon2_seq_file
from neuroseq.importer_syscon2 import sequence_to_stim_values
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.syscon2_import_dialog import Syscon2ImportDialog


def _sample_seq_text() -> str:
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
[0]
ID=1
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=2
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.3
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape0_Type=0
Shape0_Filled=False
Shape0_CenterX=0
Shape0_CenterY=0
Shape0_CenterZ=0
Shape0_Rotation=0
Shape0_Scale=1
Shape0_VerticesCount=0
Shape0_VerticesToDrawCount=0
Shape0_TranslationVerticesCount=0
Shape0_Reversed=False
Shape0_GroupID=-1
Shape0_Linewidth=1
[/0]
[1]
ID=2
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=6000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.8
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=2
Shape1_Type=0
Shape1_Filled=False
Shape1_CenterX=0
Shape1_CenterY=0
Shape1_CenterZ=0
Shape1_Rotation=0
Shape1_Scale=1
Shape1_VerticesCount=0
Shape1_VerticesToDrawCount=0
Shape1_TranslationVerticesCount=0
Shape1_Reversed=False
Shape1_GroupID=-1
Shape1_Linewidth=1
[/1]
[2]
ID=3
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=10000
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
port=UGA-42 TTL Out 1
[/2]
"""


def _sample_seq_text_v2_hashed() -> str:
    return """F8-ED-6A-D0-E5-24-AE-09-8F-FD-46-D4-5D-03-81-E3
version=2
scanmode=Accurate
runs=0
DeviceSequenceID=0
invertedportcnt=3
inverted0=dummy_1_0 RMI In
AutoTriggerRulesEnabled=True
AutoTriggerRulesCount=0
timelinezoomfactor=48
timelineviewposx=0
timelineviewposy=0
[0]
ID=3
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=1000
TimelineInfo0_description=
TimelineInfo0_repeats=2
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_2_0
TimelineInfo0_Intensity_0=0.495
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=2
Entity0_Type=0
Entity0_Filled=False
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
Entity0_Rotation=0
Entity0_ScaleX=1
Entity0_ScaleY=1
Entity0_ScaleZ=1
Entity0_VertexCount=0
Entity0_reversed=False
[/0]
"""


def _sample_seq_text_overlapping_circular_entities() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=UGA-42 TTL Out 1
AutoTriggerRulesEnabled=False
AutoTriggerRulesCount=0
timelinezoomfactor=10
timelineviewposx=0
timelineviewposy=0
[0]
ID=1
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.2
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=2
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
Entity0_VertexCount=2
Entity0_Vertex0_X=1
Entity0_Vertex0_Y=0
Entity0_Vertex0_Z=0
Entity0_Vertex1_X=10
Entity0_Vertex1_Y=0
Entity0_Vertex1_Z=0
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=1000
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.7
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=2
Entity1_CenterX=15
Entity1_CenterY=0
Entity1_CenterZ=0
Entity1_VertexCount=2
Entity1_Vertex0_X=1
Entity1_Vertex0_Y=0
Entity1_Vertex0_Z=0
Entity1_Vertex1_X=10
Entity1_Vertex1_Y=0
Entity1_Vertex1_Z=0
[/1]
[2]
ID=3
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=2000
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity2_Type=2
Entity2_CenterX=50
Entity2_CenterY=0
Entity2_CenterZ=0
Entity2_VertexCount=2
Entity2_Vertex0_X=1
Entity2_Vertex0_Y=0
Entity2_Vertex0_Z=0
Entity2_Vertex1_X=10
Entity2_Vertex1_Y=0
Entity2_Vertex1_Z=0
[/2]
"""


def _sample_seq_text_non_overlapping_circular_entities() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=UGA-42 TTL Out 1
AutoTriggerRulesEnabled=False
AutoTriggerRulesCount=0
timelinezoomfactor=10
timelineviewposx=0
timelineviewposy=0
[0]
ID=1
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.2
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=2
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
Entity0_VertexCount=2
Entity0_Vertex0_X=30
Entity0_Vertex0_Y=0
Entity0_Vertex0_Z=0
Entity0_Vertex1_X=10
Entity0_Vertex1_Y=0
Entity0_Vertex1_Z=0
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=1000
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.7
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=2
Entity1_CenterX=25
Entity1_CenterY=0
Entity1_CenterZ=0
Entity1_VertexCount=2
Entity1_Vertex0_X=30
Entity1_Vertex0_Y=0
Entity1_Vertex0_Z=0
Entity1_Vertex1_X=10
Entity1_Vertex1_Y=0
Entity1_Vertex1_Z=0
[/1]
"""


def _sample_seq_text_single_entity() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=UGA-42 TTL Out 1
AutoTriggerRulesEnabled=False
AutoTriggerRulesCount=0
timelinezoomfactor=10
timelineviewposx=0
timelineviewposy=0
[0]
ID=1
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=
TimelineInfo0_Intensity_0=0.42
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=2
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
Entity0_VertexCount=2
Entity0_Vertex0_X=1
Entity0_Vertex0_Y=0
Entity0_Vertex0_Z=0
Entity0_Vertex1_X=10
Entity0_Vertex1_Y=0
Entity0_Vertex1_Z=0
[/0]
"""


def test_parse_syscon2_seq_file(tmp_path: Path) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    assert sequence.header["version"] == "3"
    assert sequence.header["scanmode"] == "Accurate"
    assert len(sequence.objects) == 3
    assert sequence.objects[0].object_type == "Shape"
    assert sequence.objects[1].start_time_us == 6000
    assert sequence.objects[1].repeats == 1


def test_parse_syscon2_seq_file_v2_hash_header_and_entity(tmp_path: Path) -> None:
    seq_path = tmp_path / "sample_v2.seq"
    seq_path.write_text(_sample_seq_text_v2_hashed(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    assert sequence.header["version"] == "2"
    assert "hash_id" in sequence.header
    assert sequence.objects[0].object_type == "Entity"

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=2,
        tick_resolution_us=500,
        include_non_shape=False,
    )
    assert result.eligible_objects == 1
    assert result.mapped_objects == 1
    # start=1000us, repeats=2, step=2 ticks (1000us): indices 1, 2, 3
    assert values[1, 0] == 0.495
    assert values[2, 0] == 0.495
    assert values[3, 0] == 0.495


def test_sequence_to_stim_values_shape_only(tmp_path: Path) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,  # 1ms
        n_neurons=4,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.total_objects == 3
    assert result.eligible_objects == 2
    assert result.mapped_objects == 2
    assert values.shape[1] == 4

    # object 0: start=0us, repeats=2, step=1 tick (500us) -> indices 0 and 1 after dedupe
    assert values[0, 0] == 0.3
    assert values[1, 0] == 0.3
    # object 1: start=6000us, repeats=1, step=2 ticks (1000us) -> indices 6, 7
    assert values[6, 1] == 0.8
    assert values[7, 1] == 0.8
    # excluded TTLPulse in shape-only mode: only first 2 channels are mapped
    assert np.all(values[:, 2] == 0.0)
    assert np.all(values[:, 3] == 0.0)


def test_sequence_to_stim_values_include_non_shape(tmp_path: Path) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    _, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=4,
        tick_resolution_us=500,
        include_non_shape=True,
    )

    assert result.eligible_objects == 3
    assert result.mapped_objects == 3
    assert result.dropped_objects == 0


def test_sequence_to_stim_values_merges_overlapping_circular_entities(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "overlap_entities.seq"
    seq_path.write_text(
        _sample_seq_text_overlapping_circular_entities(),
        encoding="utf-8",
    )
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=8,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.eligible_objects == 3
    assert result.grouped_objects == 2
    assert result.merged_objects == 1
    assert result.mapped_objects == 2
    assert result.dropped_objects == 0

    # overlapping circles [0] and [1] map to the same neuron channel
    assert values[0, 0] == 0.2
    assert values[1, 0] == 0.7
    # non-overlapping circle [2] maps to the next channel
    assert values[2, 1] == 0.5
    assert np.all(values[:, 2:] == 0.0)


def test_sequence_to_stim_values_keeps_non_overlapping_circular_entities_separate(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "nonoverlap_entities.seq"
    seq_path.write_text(
        _sample_seq_text_non_overlapping_circular_entities(),
        encoding="utf-8",
    )
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=8,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.eligible_objects == 2
    assert result.grouped_objects == 2
    assert result.merged_objects == 0
    assert result.mapped_objects == 2
    assert values[0, 0] == 0.2
    assert values[1, 1] == 0.7


def test_sequence_to_stim_values_entity_has_default_duration_16_666ms(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "single_entity.seq"
    seq_path.write_text(_sample_seq_text_single_entity(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=2,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    # 16.666ms at dt=1ms -> ceil(16.666) = 17 samples
    assert result.mapped_objects == 1
    assert values.shape[0] >= 17
    assert np.all(values[:17, 0] == 0.42)
    assert np.all(values[17:, 0] == 0.0)
    assert np.all(values[:, 1] == 0.0)


def test_import_sequence_into_stim_array_extends_and_offsets(tmp_path: Path) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    stim_array = StimArray(
        name="target",
        stim_type=StimType.Excitation,
        values=np.zeros((5, 4), dtype=np.float32),
        units="1",
        dt=0.001,
    )

    result = import_sequence_into_stim_array(
        stim_array,
        sequence,
        tick_resolution_us=500,
        include_non_shape=False,
        replace_existing=True,
        extend_to_fit=True,
        neuron_offset=1,
    )

    assert result.mapped_objects == 2
    assert stim_array.values.shape[0] >= 8
    assert stim_array.values[0, 1] == 0.3
    assert stim_array.values[1, 1] == 0.3
    assert stim_array.values[6, 2] == 0.8
    assert stim_array.values[7, 2] == 0.8
    assert np.all(stim_array.values[:, 0] == 0)


def test_syscon2_import_dialog_imports_into_project(tmp_path: Path, qtbot) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")

    project = Project()
    project.add_stim_array(
        StimArray(
            name="stimulus",
            stim_type=StimType.Excitation,
            values=np.zeros((5, 3), dtype=np.float32),
            units="1",
            dt=0.001,
        )
    )

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(seq_path))
    dialog.replace_checkbox.setChecked(True)
    dialog.extend_checkbox.setChecked(True)
    dialog.tick_resolution_spin.setValue(500)
    dialog._import_file()

    stim_values = project.stim_arrays["stimulus"].values
    assert stim_values.shape[0] >= 8
    assert stim_values[0, 0] == 0.3
    assert stim_values[1, 0] == 0.3
    assert stim_values[6, 1] == 0.8
    assert stim_values[7, 1] == 0.8
