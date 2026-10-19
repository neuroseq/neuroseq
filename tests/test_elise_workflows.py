import numpy as np

from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.elise_support import workflows as elise_workflows
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def test_replay_inputs_use_latest_firing_rate_trace():
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    stim = StimArray("opsin_exc", StimType.Excitation, np.ones((10, 2)), "1", 0.001)
    project.add_stim_array(stim)
    traces = [
        TraceData(
            name="firing_rate_r",
            traces=np.full((2, 10), value),
            indices=np.arange(2),
            units="1",
            dt=0.001,
        )
        for value in (1.0, 2.0)
    ]
    for i, trace in enumerate(traces):
        project.experiments[str(i)] = Experiment(str(i), 0.01, [], [trace])

    result = elise_workflows.replay_inputs(project)

    assert result is not None
    assert result[0] is traces[-1]
    np.testing.assert_array_equal(result[1], stim.values)
    assert result[2] == stim.dt
    project.external_simulate_path = None
    assert elise_workflows.replay_inputs(project) is None


def test_protocol_inputs_require_matching_nonempty_stimulation():
    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    assert elise_workflows.protocol_inputs(project) is None
    for name, kind in (
        ("opsin_exc", StimType.Excitation),
        ("opsin_inh", StimType.Inhibition),
    ):
        project.add_stim_array(StimArray(name, kind, np.ones((10, 2)), "1", 0.001))

    result = elise_workflows.protocol_inputs(project)
    assert result is not None
    assert result[0].shape == result[1].shape == (10, 2)
    assert result[2] == 0.001
    project.stim_arrays["opsin_inh"].values = np.ones((5, 2))
    assert elise_workflows.protocol_inputs(project) is None
