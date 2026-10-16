from __future__ import annotations

import math
from pathlib import Path

import brian2.numpy_ as np
import pytest
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.importer_syscon2 import import_sequence_into_stim_array
from neuroseq.importer_syscon2 import parse_syscon2_seq_file
from neuroseq.importer_syscon2 import parse_syscon2_seq_text
from neuroseq.importer_syscon2 import sequence_to_stim_values
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog
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


def _sample_seq_text_close_center_circular_entities() -> str:
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
Entity1_CenterX=0.8
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
Entity2_CenterX=2.5
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


def _sample_seq_text_far_center_circular_entities() -> str:
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
Entity1_CenterX=1.2
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


def _sample_seq_text_entities_with_laser_ttl_durations() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=dummy_1_0 RMI In
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
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.4
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=0
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=10000
TimelineInfo0_description=
TimelineInfo0_repeats=0
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.7
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=0
Entity1_CenterX=10
Entity1_CenterY=0
Entity1_CenterZ=0
[/1]
[2]
ID=3
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=1000
TimelineInfo0_description=
TimelineInfo0_repeats=3
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/2]
[3]
ID=4
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=2
TimelineInfo0_starttime=11000
TimelineInfo0_description=
TimelineInfo0_repeats=7
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/3]
"""


def _sample_holo4d_layout_seq_text() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=Holo4D TTL Out 1
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
TimelineInfo0_repeats=100
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.4
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=0
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=100
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.7
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=0
Entity1_CenterX=10
Entity1_CenterY=0
Entity1_CenterZ=0
[/1]
[2]
ID=3
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=10000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/2]
[3]
ID=4
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/3]
[4]
ID=5
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=19
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/4]
[5]
ID=6
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=10000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/5]
[6]
ID=7
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/6]
"""


def _sample_holo4d_layout_same_center_seq_text() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=Holo4D TTL Out 1
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
TimelineInfo0_repeats=100
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=0
Entity0_CenterX=5
Entity0_CenterY=5
Entity0_CenterZ=0
[/0]
[1]
ID=2
type=Entity
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=30000
TimelineInfo0_description=
TimelineInfo0_repeats=100
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity1_Type=0
Entity1_CenterX=5
Entity1_CenterY=5
Entity1_CenterZ=0
[/1]
[2]
ID=3
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=3
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/2]
[3]
ID=4
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=3
TimelineInfo0_starttime=50000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/3]
[4]
ID=5
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=19
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/4]
[5]
ID=6
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/5]
[6]
ID=7
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=50000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/6]
"""


def _sample_holo4d_layout_three_short_pulses_seq_text() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=Holo4D TTL Out 1
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
TimelineInfo0_repeats=100
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.6
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Entity0_Type=0
Entity0_CenterX=0
Entity0_CenterY=0
Entity0_CenterZ=0
[/0]
[1]
ID=2
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=3
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=3
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/1]
[2]
ID=3
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=3
TimelineInfo0_starttime=23000
TimelineInfo0_description=
TimelineInfo0_repeats=3
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/2]
[3]
ID=4
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=3
TimelineInfo0_starttime=26000
TimelineInfo0_description=
TimelineInfo0_repeats=3
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/3]
[4]
ID=5
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=4
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=19
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/4]
"""


def _sample_holo4d_shape_lane_replacement_seq_text() -> str:
    return """nohash
version=3
scanmode=Accurate
runs=1
DeviceSequenceID=1
invertedportcnt=1
inverted0=Holo4D TTL Out 1
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
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape0_Type=0
Shape0_CenterX=10
Shape0_CenterY=10
Shape0_CenterZ=0
[/0]
[1]
ID=2
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape1_Type=0
Shape1_CenterX=20
Shape1_CenterY=10
Shape1_CenterZ=0
[/1]
[2]
ID=3
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=2
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.5
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape2_Type=0
Shape2_CenterX=30
Shape2_CenterY=10
Shape2_CenterZ=0
[/2]
[3]
ID=4
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=0
TimelineInfo0_starttime=30000
TimelineInfo0_description=
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.8
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape3_Type=0
Shape3_CenterX=110
Shape3_CenterY=10
Shape3_CenterZ=0
[/3]
[4]
ID=5
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=1
TimelineInfo0_starttime=30000
TimelineInfo0_description=
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.8
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape4_Type=0
Shape4_CenterX=120
Shape4_CenterY=10
Shape4_CenterZ=0
[/4]
[5]
ID=6
type=Shape
TotalTimings=1
TimelineInfo0_TimelineIndex=2
TimelineInfo0_starttime=30000
TimelineInfo0_description=
TimelineInfo0_repeats=120
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0.8
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=1
Shape5_Type=0
Shape5_CenterX=130
Shape5_CenterY=10
Shape5_CenterZ=0
[/5]
[6]
ID=7
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=20000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/6]
[7]
ID=8
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=25000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/7]
[8]
ID=9
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=5
TimelineInfo0_starttime=50000
TimelineInfo0_description=
TimelineInfo0_repeats=1
TimelineInfo0_LightsourceCount=1
TimelineInfo0_LightsourceID_0=dummy_1_0
TimelineInfo0_Intensity_0=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=dummy_1_0 RMI In
[/8]
[9]
ID=10
type=TTLPulse
TotalTimings=1
TimelineInfo0_TimelineIndex=6
TimelineInfo0_starttime=0
TimelineInfo0_description=
TimelineInfo0_repeats=19
TimelineInfo0_LightsourceCount=0
TimelineInfo0_timelinegroupid=-1
TimelineInfo0_Stepsize=100
port=Holo4D TTL Out 1
[/9]
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


def test_parse_syscon2_seq_file_accepts_missing_device_sequence_id(
    tmp_path: Path,
) -> None:
    seq_text = _sample_seq_text().replace("DeviceSequenceID=1\n", "")
    seq_path = tmp_path / "sample_missing_device_sequence_id.seq"
    seq_path.write_text(seq_text, encoding="utf-8")

    sequence = parse_syscon2_seq_file(seq_path)

    assert sequence.header.get("DeviceSequenceID") == "0"
    assert len(sequence.objects) == 3


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


def test_sequence_to_stim_values_merges_close_center_circular_entities(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "overlap_entities.seq"
    seq_path.write_text(
        _sample_seq_text_close_center_circular_entities(),
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

    # circles with center distance <= 1um map to the same neuron channel
    assert values[0, 0] == 0.2
    assert values[1, 0] == 0.7
    # center distance > 1um maps to a different neuron channel
    assert values[2, 1] == 0.5
    assert np.all(values[:, 2:] == 0.0)


def test_sequence_to_stim_values_keeps_far_center_circular_entities_separate(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "nonoverlap_entities.seq"
    seq_path.write_text(
        _sample_seq_text_far_center_circular_entities(),
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


def test_sequence_to_stim_values_does_not_merge_close_circular_shapes(
    tmp_path: Path,
) -> None:
    seq_text = _sample_seq_text_close_center_circular_entities().replace(
        "Entity", "Shape"
    )
    seq_path = tmp_path / "close_shapes.seq"
    seq_path.write_text(seq_text, encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=8,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.eligible_objects == 3
    assert result.grouped_objects == 3
    assert result.merged_objects == 0
    assert result.mapped_objects == 3
    assert values[0, 0] == 0.2
    assert values[1, 1] == 0.7
    assert values[2, 2] == 0.5


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


def test_sequence_to_stim_values_entity_duration_from_laser_ttl_pulses(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "entities_with_ttl.seq"
    seq_path.write_text(
        _sample_seq_text_entities_with_laser_ttl_durations(),
        encoding="utf-8",
    )
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=4,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.mapped_objects == 2
    # Entity[0] onset/duration from TTLPulse: start=1000us (idx 1), repeats=3 -> 2ms -> 2 samples.
    assert np.all(values[1:3, 0] == 0.4)
    assert np.all(values[0:1, 0] == 0.0)
    assert np.all(values[3:11, 0] == 0.0)
    # Entity[1] onset/duration from TTLPulse: start=11000us (idx 11), repeats=7 -> 4ms -> 4 samples.
    assert np.all(values[11:15, 1] == 0.7)
    assert np.all(values[:11, 1] == 0.0)


def test_sequence_to_stim_values_holo4d_layout_uses_shared_laser_pulses(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "holo4d_layout.seq"
    seq_path.write_text(_sample_holo4d_layout_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=3,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.mapped_objects == 2
    # First laser pulse at 10ms is ignored by SLM lead time (16.66ms).
    assert np.all(values[10, :2] == 0.0)
    # Second pulse at 20ms stimulates all active entity objects.
    assert values[20, 0] == 0.4
    assert values[20, 1] == 0.7
    assert np.all(values[:, 2] == 0.0)


def test_sequence_to_stim_values_holo4d_zero_intensity_renders_visible(
    tmp_path: Path,
) -> None:
    # Real Holo4D exports often leave per-shape intensity at 0 (power is set by
    # the hardware). Such events must still be visible, at a nominal 1.0.
    seq_text = (
        _sample_holo4d_layout_seq_text()
        .replace("TimelineInfo0_Intensity_0=0.4", "TimelineInfo0_Intensity_0=0")
        .replace("TimelineInfo0_Intensity_0=0.7", "TimelineInfo0_Intensity_0=0")
    )
    seq_path = tmp_path / "holo4d_zero_intensity.seq"
    seq_path.write_text(seq_text, encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=3,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.mapped_objects == 2
    assert values[20, 0] == 1.0
    assert values[20, 1] == 1.0
    assert np.all(values[:, 2] == 0.0)


def test_sequence_to_stim_values_uses_pulse_path_even_with_shorter_startup_ttl(
    tmp_path: Path,
) -> None:
    # Startup pulse becomes <9ms for tick_resolution_us=450, which previously
    # forced a fallback to legacy mode; pulse-based mode should still be used.
    seq_text = _sample_holo4d_layout_seq_text().replace(
        "TimelineInfo0_repeats=19",
        "TimelineInfo0_repeats=18",
        1,
    )
    seq_path = tmp_path / "holo4d_layout_short_startup.seq"
    seq_path.write_text(seq_text, encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=3,
        tick_resolution_us=450,
        include_non_shape=False,
    )

    assert result.mapped_objects == 2
    assert np.all(values[10, :2] == 0.0)
    assert values[20, 0] == 0.4
    assert values[20, 1] == 0.7
    assert np.all(values[:, 2] == 0.0)


def test_sequence_to_stim_values_holo4d_layout_merges_same_center_entities(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "holo4d_same_center.seq"
    seq_path.write_text(_sample_holo4d_layout_same_center_seq_text(), encoding="utf-8")
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=2,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.eligible_objects == 2
    assert result.grouped_objects == 1
    assert result.merged_objects == 1
    assert result.mapped_objects == 1
    assert values[20, 0] == 0.5
    assert values[50, 0] == 0.5
    assert np.all(values[:, 1] == 0.0)


def test_sequence_to_stim_values_holo4d_layout_keeps_individual_laser_pulses(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "holo4d_three_pulses.seq"
    seq_path.write_text(
        _sample_holo4d_layout_three_short_pulses_seq_text(),
        encoding="utf-8",
    )
    sequence = parse_syscon2_seq_file(seq_path)

    values, result = sequence_to_stim_values(
        sequence,
        dt_seconds=0.001,
        n_neurons=2,
        tick_resolution_us=500,
        include_non_shape=False,
    )

    assert result.mapped_objects == 1
    assert np.all(values[20:22, 0] == 0.6)
    assert np.all(values[23:25, 0] == 0.6)
    assert np.all(values[26:28, 0] == 0.6)
    assert np.all(values[22:23, 0] == 0.0)
    assert np.all(values[25:26, 0] == 0.0)
    assert np.all(values[:, 1] == 0.0)


def test_sequence_to_stim_values_holo4d_shape_lanes_replace_previous_objects(
    tmp_path: Path,
) -> None:
    seq_path = tmp_path / "holo4d_shape_lane_replacement.seq"
    seq_path.write_text(
        _sample_holo4d_shape_lane_replacement_seq_text(),
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

    assert result.mapped_objects == 6

    # First 3 shape lanes are active before lane replacement at 30ms.
    assert values[20, 0] == 0.5
    assert values[20, 1] == 0.5
    assert values[20, 2] == 0.5
    assert values[25, 0] == 0.5
    assert values[25, 1] == 0.5
    assert values[25, 2] == 0.5

    # After replacement, only second 3 lanes respond.
    assert values[50, 3] == 0.8
    assert values[50, 4] == 0.8
    assert values[50, 5] == 0.8
    assert values[50, 0] == 0.0
    assert values[50, 1] == 0.0
    assert values[50, 2] == 0.0

    assert np.all(values[:, 6:] == 0.0)


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


def test_syscon2_import_dialog_imports_into_project(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
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
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    dialog._import_file()

    stim_values = project.stim_arrays["stimulus"].values
    assert math.isclose(project.dt, _SYSCON2_TTL_PULSE_QUANTUM_S, abs_tol=1e-12)
    assert math.isclose(
        project.stim_arrays["stimulus"].dt,
        _SYSCON2_TTL_PULSE_QUANTUM_S,
        abs_tol=1e-12,
    )
    assert stim_values.shape[0] >= 14
    assert stim_values[0, 0] == 0.3
    assert stim_values[1, 0] == 0.3
    assert stim_values[12, 1] == 0.8
    assert stim_values[13, 1] == 0.8


def test_syscon2_import_dialog_remaps_existing_trace_data(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
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
    trace_data = TraceData(
        name="trace",
        traces=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    project.experiments["exp"] = Experiment(
        name="exp",
        duration=0.003,
        timepoints=[0.003],
        data=[trace_data],
    )
    project.update_limits()

    question_calls = {"count": 0}

    def _question(*args, **kwargs):
        question_calls["count"] += 1
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _question)

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(seq_path))
    dialog.tick_resolution_spin.setValue(500)
    dialog._import_file()

    assert question_calls["count"] == 1
    assert math.isclose(project.dt, _SYSCON2_TTL_PULSE_QUANTUM_S, abs_tol=1e-12)
    assert math.isclose(trace_data.dt, _SYSCON2_TTL_PULSE_QUANTUM_S, abs_tol=1e-12)
    assert trace_data.traces.shape == (1, 6)
    np.testing.assert_allclose(
        trace_data.traces[0],
        np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32),
    )


def test_syscon2_import_dialog_skips_remap_prompt_when_dt_matches(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")

    project = Project()
    project.add_stim_array(
        StimArray(
            name="stimulus",
            stim_type=StimType.Excitation,
            values=np.zeros((5, 3), dtype=np.float32),
            units="1",
            dt=_SYSCON2_TTL_PULSE_QUANTUM_S,
        )
    )
    project.dt = _SYSCON2_TTL_PULSE_QUANTUM_S

    question_calls = {"count": 0}

    def _question(*args, **kwargs):
        question_calls["count"] += 1
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _question)

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(seq_path))
    dialog.tick_resolution_spin.setValue(500)
    dialog._import_file()

    assert question_calls["count"] == 0
    assert math.isclose(project.dt, _SYSCON2_TTL_PULSE_QUANTUM_S, abs_tol=1e-12)


def test_syscon2_import_dialog_aborts_when_user_declines_remap(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
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
    original_values = project.stim_arrays["stimulus"].values.copy()

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(seq_path))
    dialog.tick_resolution_spin.setValue(500)
    dialog._import_file()

    assert math.isclose(project.dt, 0.001, abs_tol=1e-12)
    np.testing.assert_array_equal(
        project.stim_arrays["stimulus"].values,
        original_values,
    )
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted


def test_syscon2_import_dialog_updates_generator_template_only_after_import(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_text = _sample_seq_text().replace(
        "timelineviewposy=0\n",
        "timelineviewposy=0\nSourceConnectorPort0=dummy_2_0 RMI In\n",
    )
    seq_path.write_text(seq_text, encoding="utf-8")

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

    import_dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(import_dialog)
    import_dialog._load_file(str(seq_path))

    assert project.syscon2_export_template_text is None
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    import_dialog._import_file()
    assert project.syscon2_export_template_text is not None
    assert (
        "SourceConnectorPort0=dummy_2_0 RMI In" in project.syscon2_export_template_text
    )

    generator_dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(generator_dialog)
    assert (
        "SourceConnectorPort0=dummy_2_0 RMI In"
        in generator_dialog.config_data_edit.toPlainText()
    )


def test_syscon2_import_dialog_creates_new_input_array_in_empty_project(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    """Issue #591: importing must not require a pre-existing input array."""
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    project = Project()
    assert not project.stim_arrays

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    assert dialog.create_new_radio.isChecked()
    assert not dialog.import_existing_radio.isEnabled()
    dialog._load_file(str(seq_path))
    assert dialog.import_button.isEnabled()

    dialog.new_name_edit.setText("holo_stim")
    dialog.new_units_edit.setText("mW")
    dialog.new_type_combo.setCurrentIndex(dialog.new_type_combo.findText("Excitatory"))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    dialog._import_file()

    stim = project.stim_arrays["holo_stim"]
    assert dialog.selected_stim_name == "holo_stim"
    assert stim.stim_type == StimType.Excitation
    assert stim.units == "mW"
    assert math.isclose(stim.dt, _SYSCON2_TTL_PULSE_QUANTUM_S, abs_tol=1e-12)
    assert stim.values.shape[1] == 2
    assert stim.values[0, 0] == pytest.approx(0.3)
    assert stim.values[12, 1] == pytest.approx(0.8)


def test_syscon2_import_dialog_new_array_rejects_duplicate_name(
    tmp_path: Path, qtbot, monkeypatch
) -> None:
    seq_path = tmp_path / "sample.seq"
    seq_path.write_text(_sample_seq_text(), encoding="utf-8")
    project = Project()
    project.dt = _SYSCON2_TTL_PULSE_QUANTUM_S
    project.add_stim_array(
        StimArray(
            "stimulus",
            StimType.Excitation,
            np.zeros((5, 3), dtype=np.float32),
            "1",
            _SYSCON2_TTL_PULSE_QUANTUM_S,
        )
    )

    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    # Existing arrays: default stays on the existing-array workflow.
    assert dialog.import_existing_radio.isChecked()
    dialog._load_file(str(seq_path))
    dialog.create_new_radio.setChecked(True)
    assert dialog.new_name_edit.text() not in project.stim_arrays
    dialog.new_name_edit.setText("stimulus")
    warned = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: warned.append(args)
    )

    dialog._import_file()

    assert len(warned) == 1
    assert list(project.stim_arrays) == ["stimulus"]
    assert not project.stim_arrays["stimulus"].values.any()


@pytest.mark.parametrize(
    "key,value",
    [
        ("TimelineInfo0_starttime", "nan"),
        ("TimelineInfo0_starttime", "-1"),
        ("TimelineInfo0_repeats", "inf"),
        ("TimelineInfo0_repeats", "2.5"),
        ("TimelineInfo0_Intensity_0", "not-a-number"),
        ("TimelineInfo0_Intensity_0", "-1"),
        ("TimelineInfo0_Stepsize", "0"),
        ("Shape0_CenterX", "nan"),
        ("TotalTimings", "2"),
        ("TimelineInfo0_LightsourceCount", "2"),
    ],
)
def test_reject_malformed_or_unsupported_object_fields(key, value):
    import re

    text = re.sub(
        rf"^{key}=.*$",
        f"{key}={value}",
        _sample_seq_text(),
        count=1,
        flags=re.MULTILINE,
    )
    with pytest.raises(ValueError):
        parse_syscon2_seq_text(text)


def test_reject_duplicate_block_indices():
    text = _sample_seq_text().replace("[1]", "[0]").replace("[/1]", "[/0]")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_syscon2_seq_text(text)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), 0, -1])
def test_reject_invalid_import_dt(dt):
    with pytest.raises(ValueError, match="dt_seconds"):
        sequence_to_stim_values(
            parse_syscon2_seq_text(_sample_seq_text()), dt_seconds=dt, n_neurons=1
        )


def test_reject_unbounded_allocation():
    sequence = parse_syscon2_seq_text(
        _sample_seq_text().replace(
            "TimelineInfo0_starttime=0", "TimelineInfo0_starttime=1000000000000", 1
        )
    )
    with pytest.raises(ValueError, match="512 MiB"):
        sequence_to_stim_values(sequence, dt_seconds=0.001, n_neurons=4)


def test_replace_preserves_unmapped_channels_and_fractional_values():
    stim = StimArray(
        "target", StimType.Excitation, np.full((100, 5), 7, dtype=np.int32), "1", 0.001
    )
    import_sequence_into_stim_array(
        stim, parse_syscon2_seq_text(_sample_seq_text()), neuron_offset=1
    )
    np.testing.assert_array_equal(stim.values[:, [0, 3, 4]], 7)
    assert stim.values[0, 1] == pytest.approx(0.3)
    assert stim.values[-1, 1] == 0


def test_empty_reconstruction_cannot_erase_target():
    sequence = parse_syscon2_seq_text(_sample_seq_text())
    sequence.objects = [
        obj for obj in sequence.objects if obj.object_type == "TTLPulse"
    ]
    stim = StimArray("target", StimType.Excitation, np.ones((100, 2)), "1", 0.001)
    with pytest.raises(ValueError, match="No stimulation events"):
        import_sequence_into_stim_array(stim, sequence)
    assert np.all(stim.values == 1)


def _import_dialog_with_project(tmp_path, qtbot):
    path = tmp_path / "test.seq"
    path.write_text(_sample_seq_text())
    project = Project()
    project.add_stim_array(
        StimArray("target", StimType.Excitation, np.ones((100, 3)), "1", 0.001)
    )
    project.syscon2_export_template_text = "previous template"
    project.dirty = False
    dialog = Syscon2ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    return dialog, project


def test_cancel_after_preview_does_not_mutate_project(tmp_path, qtbot):
    dialog, project = _import_dialog_with_project(tmp_path, qtbot)
    dialog.reject()
    assert project.syscon2_export_template_text == "previous template"
    assert not project.dirty
    assert project.dt == 0.001
    assert np.all(project.stim_arrays["target"].values == 1)


def test_conversion_failure_does_not_remap_project(tmp_path, qtbot, mocker):
    dialog, project = _import_dialog_with_project(tmp_path, qtbot)
    original = project.stim_arrays["target"].values
    mocker.patch.object(
        QtWidgets.QMessageBox,
        "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    mocker.patch(
        "neuroseq.syscon2_import_dialog.import_sequence_into_stim_array",
        side_effect=ValueError("bad sequence"),
    )
    dialog._import_file()
    assert project.stim_arrays["target"].values is original
    assert project.dt == 0.001
    assert project.stim_arrays["target"].dt == 0.001
    assert project.syscon2_export_template_text == "previous template"
    assert not project.dirty


def test_declining_dropped_channels_leaves_project_unchanged(tmp_path, qtbot, mocker):
    dialog, project = _import_dialog_with_project(tmp_path, qtbot)
    original = project.stim_arrays["target"].values
    dialog.neuron_offset_spin.setValue(2)
    question = mocker.patch.object(
        QtWidgets.QMessageBox,
        "question",
        side_effect=[
            QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No,
        ],
    )
    dialog._import_file()
    assert question.call_count == 2
    assert "dropped" in question.call_args.args[2]
    assert project.stim_arrays["target"].values is original
    assert not project.dirty
    assert project.dt == 0.001


def _point_sequence(specs):
    header = _sample_seq_text().split("[0]")[0]
    blocks = []
    for index, (kind, start_tick, lane, source, center) in enumerate(specs):
        blocks.append(
            f"[{index}]\ntype={kind}\nTotalTimings=1\nTimelineInfo0_starttime={start_tick * 520.8125}\nTimelineInfo0_TimelineIndex={lane}\nTimelineInfo0_repeats={1 if kind == 'TTLPulse' else 0}\nTimelineInfo0_LightsourceID_0={source}\nShape{index}_Type=0\nShape{index}_CenterX={center}\nShape{index}_CenterY=0\nShape{index}_CenterZ=0\n[/{index}]"
        )
    return parse_syscon2_seq_text(header + "\n".join(blocks))


def test_repeated_point_shapes_map_to_same_channel_and_preview(qtbot):
    sequence = _point_sequence(
        [
            ("Shape", 0, 0, "laser-a", 1),
            ("Shape", 128, 0, "laser-a", 1),
            ("TTLPulse", 33, 2, "laser-a", 0),
            ("TTLPulse", 161, 2, "laser-a", 0),
        ]
    )
    values, report = sequence_to_stim_values(
        sequence, dt_seconds=_SYSCON2_TTL_PULSE_QUANTUM_S, n_neurons=1
    )
    assert report.grouped_objects == 1
    assert report.dropped_objects == 0
    np.testing.assert_array_equal(np.flatnonzero(values[:, 0]), [33, 34, 161, 162])
    dialog = Syscon2ImportDialog(Project())
    qtbot.addWidget(dialog)
    assert "Estimated neuron channels (shape/entity)=1" in dialog._build_preview_text(
        sequence
    )


def test_laser_timelines_are_associated_with_their_own_targets():
    sequence = _point_sequence(
        [
            ("Shape", 0, 0, "laser-a", 1),
            ("Shape", 0, 6, "laser-b", 5),
            ("TTLPulse", 33, 2, "laser-a", 0),
            ("TTLPulse", 43, 3, "laser-b", 0),
        ]
    )
    values, report = sequence_to_stim_values(
        sequence, dt_seconds=_SYSCON2_TTL_PULSE_QUANTUM_S, n_neurons=2
    )
    assert report.mapped_objects == 2
    np.testing.assert_array_equal(np.flatnonzero(values[:, 0]), [33, 34])
    np.testing.assert_array_equal(np.flatnonzero(values[:, 1]), [43, 44])


def test_ambiguous_laser_association_fails_instead_of_guessing():
    sequence = _point_sequence(
        [
            ("Shape", 0, 0, "", 1),
            ("TTLPulse", 33, 2, "laser-a", 0),
            ("TTLPulse", 43, 3, "laser-b", 0),
        ]
    )
    with pytest.raises(ValueError, match="Cannot associate"):
        sequence_to_stim_values(
            sequence, dt_seconds=_SYSCON2_TTL_PULSE_QUANTUM_S, n_neurons=1
        )


def test_untimed_shape_does_not_create_import_channel():
    sequence = _point_sequence(
        [
            ("Shape", 0, 0, "laser-a", 1),
            ("TTLPulse", 33, 2, "laser-a", 0),
        ]
    )
    sequence.objects[0].properties.pop("TimelineInfo0_starttime")
    _, report = sequence_to_stim_values(
        sequence, dt_seconds=_SYSCON2_TTL_PULSE_QUANTUM_S, n_neurons=1
    )
    assert report.mapped_objects == 0


def test_shape_repeats_define_duration_not_extra_shifted_events():
    sequence = parse_syscon2_seq_text(_sample_seq_text())
    sequence.objects = sequence.objects[:1]
    values, _ = sequence_to_stim_values(
        sequence, dt_seconds=_SYSCON2_TTL_PULSE_QUANTUM_S, n_neurons=1
    )
    # Shape repeats=2 means four 32-tick windows, not 3 copies spaced by Stepsize.
    assert values.shape == (128, 1)
    assert np.all(values[:, 0] == np.float32(0.3))


def test_reject_invalid_event_cap():
    with pytest.raises(ValueError, match="max_events_per_object"):
        sequence_to_stim_values(
            parse_syscon2_seq_text(_sample_seq_text()),
            dt_seconds=0.001,
            n_neurons=1,
            max_events_per_object=0,
        )


def test_remap_failure_does_not_mutate_any_live_arrays(tmp_path, qtbot, mocker):
    dialog, project = _import_dialog_with_project(tmp_path, qtbot)
    original = project.stim_arrays["target"].values
    project.add_stim_array(
        StimArray("second", StimType.Excitation, np.ones((5, 1)), "1", 0.002)
    )
    project.dirty = False
    mocker.patch.object(
        QtWidgets.QMessageBox,
        "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Yes,
    )
    from neuroseq.stim_timing import remap_stim_values_to_dt

    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("failed second remap")
        return remap_stim_values_to_dt(*args, **kwargs)

    mocker.patch("neuroseq.project.remap_stim_values_to_dt", side_effect=fail_second)
    dialog._import_file()
    assert project.stim_arrays["target"].values is original
    assert project.stim_arrays["target"].dt == 0.001
    assert project.stim_arrays["second"].dt == 0.002
    assert project.dt == 0.001
    assert not project.dirty


def test_group_merges_simultaneous_multi_spot_shapes() -> None:
    from neuroseq.importer_syscon2 import SysconSequenceObject
    from neuroseq.importer_syscon2 import _group_entities_by_center_distance

    def shape(idx: int, start_us: float, x: float, y: float) -> SysconSequenceObject:
        return SysconSequenceObject(
            block_index=idx,
            object_id=idx,
            object_type="Shape",
            properties={
                "Shape0_Type": "0",
                "Shape0_CenterX": str(x),
                "Shape0_CenterY": str(y),
                "Shape0_CenterZ": "0",
            },
            start_time_us=start_us,
            repeats=6,
            step_size=100,
            intensity=1.0,
        )

    # Neuron A: three spots at t=0 and the same spots again later; neuron B far away.
    spots_a = [(10.0, 10.0), (14.0, 12.0), (8.0, 15.0)]
    objs = [shape(i, 0.0, x, y) for i, (x, y) in enumerate(spots_a)]
    objs.append(shape(3, 0.0, 100.0, 100.0))
    objs += [shape(4 + i, 500_000.0, x, y) for i, (x, y) in enumerate(spots_a)]
    groups = _group_entities_by_center_distance(objs)
    assert sorted(len(g) for g in groups) == [1, 6]


def test_group_uses_exported_neuron_tags() -> None:
    from neuroseq.importer_syscon2 import SysconSequenceObject
    from neuroseq.importer_syscon2 import _group_entities_by_center_distance

    objs = [
        SysconSequenceObject(
            block_index=i,
            object_id=i,
            object_type="Shape",
            properties={
                "TimelineInfo0_description": f"neuron={tag}",
                "Shape0_CenterX": str(50.0 * i),
                "Shape0_CenterY": "0",
            },
            start_time_us=0.0,
            repeats=6,
            step_size=100,
            intensity=1.0,
        )
        for i, tag in enumerate([4, 2, 4, 2, 4])
    ]
    groups = _group_entities_by_center_distance(objs)
    assert [len(g) for g in groups] == [0, 0, 2, 0, 3]
