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


def _elise_replay_workspace(qtbot, monkeypatch, tmp_path):
    from neuroseq.project_workspace import ProjectWorkspace

    monkeypatch.setattr(
        "neuroseq.elise_support.workflows.getsavefilename",
        lambda *a, **k: (str(tmp_path / "replay.csv"), ""),
    )
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = gui.project
    project.external_simulate_path = "neuroseq:elise"
    cycle = np.zeros((10, 2))
    cycle[0:5, 0] = 1.0
    cycle[5:10, 1] = 1.0
    project.add_stim_array(
        StimArray("opsin_exc", StimType.Excitation, np.tile(cycle, (4, 1)), "1", 0.001)
    )
    trace = TraceData(
        name="firing_rate_r",
        traces=np.random.default_rng(0).random((2, 40)),
        indices=np.arange(2),
        units="1",
        dt=0.001,
    )
    project.experiments["e"] = Experiment("e", 0.04, [], [trace])
    return gui


def test_analyse_replay_opens_target_mse_window(qtbot, monkeypatch, tmp_path):
    from neuroseq.elise_support.plugin_keys import TARGET_ATTR
    from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow

    gui = _elise_replay_workspace(qtbot, monkeypatch, tmp_path)
    try:
        elise_workflows.analyse_replay(gui)
        windows = gui.findChildren(CustomAnalysisWindow)
        assert len(windows) == 1
        assert getattr(windows[0].analysis_widget, TARGET_ATTR).shape[0] == 2
        assert (tmp_path / "replay.csv").exists()
    finally:
        gui.close()


def test_analyse_replay_without_plugin_opens_no_window(
    qtbot, monkeypatch, tmp_path, qmessagebox_calls
):
    from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow

    monkeypatch.setattr(
        "neuroseq.elise_support.workflows.discover_analysis_plugins", lambda: []
    )
    gui = _elise_replay_workspace(qtbot, monkeypatch, tmp_path)
    try:
        elise_workflows.analyse_replay(gui)
        # The window used to be created (and left empty) before the lookup.
        assert gui.findChildren(CustomAnalysisWindow) == []
        assert [call["name"] for call in qmessagebox_calls] == ["warning"]
    finally:
        gui.close()
