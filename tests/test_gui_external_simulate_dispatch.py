"""GUI tests for the Run-button dispatch between Brian2 and external simulation.

When ``project.external_simulate_path`` is set, the main "Run simulation"
button should route to ``_do_external_simulation`` instead of the b2_model
path.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType

import numpy as np
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq
from neuroseq.data import Experiment
from neuroseq.gui import Gui


class _MockImportDialog:
    """Stub for `AddExperimentDataDialog` so the external sim flow doesn't block."""

    append_to_experiment = False

    def __init__(self, experiment, project, parent=None):
        self.experiment = experiment
        self.project = project

    def exec(self):
        return False


def _set_simulation_time(monkeypatch: MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: (f"{seconds}", True),
    )


def _wait_for_external_simulation(qtbot: QtBot, gui: Gui):
    # Poll until the worker has been fully torn down. Using waitUntil
    # rather than waitSignal here is robust to the race where a fast
    # simulator (e.g. an "old-style" sim with no progress callback) emits
    # `finished` before the test enters the wait — `waitSignal` only
    # catches future emissions, while `waitUntil` is true whether the
    # worker is still running or already gone.
    qtbot.waitUntil(lambda: gui._external_sim_thread is None, timeout=10000)


def test_run_dispatches_to_external_when_path_set(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    """Run simulation should call _do_external_simulation, not b2_model.run_simulation."""
    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project

    # External sim active; load the bundled simulate.py to have a real callable.
    sim_path = (
        Path(neuroseq.__file__).parent.parent.parent
        / "external_simulation_example"
        / "simulate.py"
    )
    project.external_simulate_path = str(sim_path)
    project.load_NN()
    assert project.external_simulate_function is not None

    calls: dict[str, list] = {"external": [], "b2": []}

    def fake_external(simulation_time: float):
        calls["external"].append(simulation_time)

    def fake_b2(*args, **kwargs):
        calls["b2"].append((args, kwargs))

    monkeypatch.setattr(gui, "_do_external_simulation", fake_external)
    monkeypatch.setattr(project, "run_simulation", fake_b2)
    _set_simulation_time(monkeypatch, 1.5)

    gui.btn_continue_simulation_clicked()

    assert calls["external"] == [1.5]
    assert calls["b2"] == []
    gui.close()


def test_run_dispatches_to_b2_when_path_unset(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    """With no external_simulate_path, Run should hit the b2_model path."""
    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project
    assert project.external_simulate_path is None

    calls: dict[str, list] = {"external": [], "b2": []}

    def fake_external(simulation_time: float):
        calls["external"].append(simulation_time)

    def fake_b2(simulation_time: float):
        calls["b2"].append(simulation_time)

    monkeypatch.setattr(gui, "_do_external_simulation", fake_external)
    monkeypatch.setattr(gui, "_do_brian2_simulation_threaded", fake_b2)
    _set_simulation_time(monkeypatch, 0.5)

    gui.btn_continue_simulation_clicked()

    assert calls["external"] == []
    assert calls["b2"] == [0.5]
    gui.close()


def test_external_simulation_passes_params_kwarg(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """_do_external_simulation must call the simulate function with `params=`."""
    sim_path = tmp_path / "param_inspecting_sim.py"
    sim_path.write_text(
        "from neuroseq.data import Experiment\n"
        "captured = {}\n"
        "def simulate(stims_dict, simulation_time, params=None):\n"
        "    captured['stims_keys'] = list(stims_dict.keys())\n"
        "    captured['simulation_time'] = simulation_time\n"
        "    captured['params'] = params\n"
        "    return Experiment('mock', simulation_time, [], [])\n"
    )

    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project
    project.external_simulate_path = str(sim_path)
    project.load_NN()
    project.external_simulate_params = {
        "k1": {
            "value": 42,
            "default": 0,
            "type": "int",
            "group": "G",
            "description": ".",
        },
        "k2": {
            "value": 1.5,
            "default": 0.0,
            "type": "float",
            "group": "G",
            "description": ".",
        },
    }

    monkeypatch.setattr(neuroseq.gui, "AddExperimentDataDialog", _MockImportDialog)

    gui._do_external_simulation(2.0)
    _wait_for_external_simulation(qtbot, gui)

    fn = project.external_simulate_function
    assert isinstance(fn, FunctionType)
    captured = fn.__globals__["captured"]
    assert captured["simulation_time"] == 2.0
    assert captured["params"] == {"k1": 42, "k2": 1.5}
    gui.close()


def test_run_dialog_default_is_suggested_simulation_time(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    def _external_sim(
        stims_dict: dict,
        simulation_time: float,
        params=None,
    ) -> Experiment:
        return Experiment("unused", simulation_time, [], [])

    gui = Gui()
    qtbot.addWidget(gui)
    gui.project.external_simulate_path = "neuroseq:elise"
    gui.project.external_simulate_function = _external_sim

    seen_default: dict[str, str] = {}

    def _get_text(*args, **kwargs):
        seen_default["text"] = args[4]
        return ("1.92", True)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", _get_text)
    monkeypatch.setattr(gui.project, "suggested_simulation_time_seconds", lambda: 1.92)
    monkeypatch.setattr(gui, "_do_external_simulation", lambda simulation_time: None)

    gui.btn_continue_simulation_clicked()

    assert seen_default["text"] == "1.92"
    gui.close()


def test_run_with_overshoot_tiles_stim_before_dispatch(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    sim_path = tmp_path / "shape_capture_sim.py"
    sim_path.write_text(
        "from neuroseq.data import Experiment\n"
        "captured = {}\n"
        "def simulate(stims_dict, simulation_time, params=None):\n"
        "    captured['shape'] = stims_dict['u_inp']['array'].shape\n"
        "    return Experiment('mock', simulation_time, [], [])\n"
    )

    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project
    project.external_simulate_path = str(sim_path)
    project.load_NN()
    monkeypatch.setattr(neuroseq.gui, "AddExperimentDataDialog", _MockImportDialog)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("0.25", True),
    )

    stim = np.arange(200, dtype=np.float32).reshape(100, 2)
    project.add_stim_array(
        neuroseq.stim.StimArray(
            "u_inp",
            neuroseq.stim.StimType.Excitation,
            stim,
            "mV",
            0.001,
        )
    )
    gui.stim_editor.set_project(project)

    gui.btn_continue_simulation_clicked()
    _wait_for_external_simulation(qtbot, gui)

    fn = project.external_simulate_function
    assert isinstance(fn, FunctionType)
    captured = fn.__globals__["captured"]
    assert captured["shape"] == (250, 2)
    gui.close()


def test_btn_edit_clicked_writes_external_params_back(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    class _Dialog:
        def __init__(self, *args, **kwargs):
            self.external_simulate_params = {
                "alpha": {
                    "value": 12,
                    "default": 1,
                    "type": "int",
                    "group": "G",
                    "description": ".",
                }
            }
            self.b2model = object()

        def exec(self) -> bool:
            return True

    gui = Gui()
    qtbot.addWidget(gui)
    monkeypatch.setattr(neuroseq.gui, "ModelEditorDialog", _Dialog)
    monkeypatch.setattr(gui.project, "update_limits", lambda: None)
    monkeypatch.setattr(gui, "update_project", lambda project: None)

    gui.btn_edit_clicked()

    assert gui.project.external_simulate_params["alpha"]["value"] == 12
    gui.close()


def test_btn_edit_clicked_writes_b2model_back(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    class _Replacement:
        # minimal stand-in for the edited B2Model; btn_edit_clicked queries the
        # neuron count to keep stim arrays in step
        def get_defined_neuron_count(self) -> int:
            return 0

    replacement = _Replacement()

    class _Dialog:
        def __init__(self, *args, **kwargs):
            self.external_simulate_params = None
            self.b2model = replacement

        def exec(self) -> bool:
            return True

    gui = Gui()
    qtbot.addWidget(gui)
    monkeypatch.setattr(neuroseq.gui, "ModelEditorDialog", _Dialog)
    monkeypatch.setattr(gui.project, "update_limits", lambda: None)
    monkeypatch.setattr(gui, "update_project", lambda project: None)

    gui.btn_edit_clicked()

    assert gui.project.b2_model is replacement
    gui.close()


def test_load_external_simulation_function_failure_clears_state(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    def _previous_external_sim(
        stims_dict: dict,
        simulation_time: float,
        params=None,
    ) -> Experiment:
        return Experiment("previous", simulation_time, [], [])

    gui = Gui()
    qtbot.addWidget(gui)
    gui.project.external_simulate_path = "previous"
    gui.project.external_simulator = object()
    gui.project.external_simulate_function = _previous_external_sim

    monkeypatch.setattr(
        gui.project,
        "load_NN",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    gui._load_external_simulation_function(str(tmp_path / "missing.py"))

    assert gui.project.external_simulate_path is None
    assert gui.project.external_simulator is None
    assert gui.project.external_simulate_function is None
    gui.close()


def test_external_simulator_accepts_progress_detection() -> None:
    """The signature-introspection helper opts in only when ``progress`` is named."""
    from neuroseq.gui import _external_simulator_accepts_progress

    def old_sim(stims_dict, simulation_time, params=None):
        pass

    def new_sim(stims_dict, simulation_time, params=None, progress=None):
        pass

    def kwargs_sim(stims_dict, simulation_time, **kwargs):
        # Despite **kwargs swallowing progress=, the callback wouldn't run,
        # so the helper must NOT report this as progress-capable.
        pass

    assert not _external_simulator_accepts_progress(old_sim)
    assert _external_simulator_accepts_progress(new_sim)
    assert not _external_simulator_accepts_progress(kwargs_sim)


def test_external_simulation_passes_progress_when_accepted(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """When the simulator advertises a ``progress`` kwarg, _do_external_simulation
    must wire it to the main window's progress dialog."""
    sim_path = tmp_path / "progress_aware_sim.py"
    sim_path.write_text(
        "from neuroseq.data import Experiment\n"
        "captured = {'fractions': []}\n"
        "def simulate(stims_dict, simulation_time, params=None, progress=None):\n"
        "    if progress is not None:\n"
        "        progress(0.0)\n"
        "        progress(0.5)\n"
        "        progress(1.0)\n"
        "    captured['progress_was_passed'] = progress is not None\n"
        "    return Experiment('mock', simulation_time, [], [])\n"
    )

    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project
    project.external_simulate_path = str(sim_path)
    project.load_NN()
    monkeypatch.setattr(neuroseq.gui, "AddExperimentDataDialog", _MockImportDialog)

    # Spy on setValue to observe the progress percentages routed into the
    # dialog. The dialog itself auto-resets to -1 once value reaches 100.
    set_value_calls: list[int] = []
    original_set_value = gui.progress_dialog.setValue

    def _spy_set_value(v: int) -> None:
        set_value_calls.append(int(v))
        original_set_value(int(v))

    monkeypatch.setattr(gui.progress_dialog, "setValue", _spy_set_value)

    gui._do_external_simulation(1.0)
    _wait_for_external_simulation(qtbot, gui)

    fn = project.external_simulate_function
    assert isinstance(fn, FunctionType)
    captured = fn.__globals__["captured"]
    assert captured["progress_was_passed"] is True
    # The simulator emitted 0.0, 0.5, 1.0, and the GUI also sets 0 then 100.
    assert 50 in set_value_calls
    assert 100 in set_value_calls
    gui.close()


def test_external_simulation_skips_progress_for_old_simulators(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """An old-contract simulator without ``progress`` must still run successfully
    and not be passed an unexpected kwarg."""
    sim_path = tmp_path / "old_sim.py"
    sim_path.write_text(
        "from neuroseq.data import Experiment\n"
        "captured = {}\n"
        "def simulate(stims_dict, simulation_time, params=None):\n"
        "    captured['ran'] = True\n"
        "    return Experiment('mock', simulation_time, [], [])\n"
    )

    gui = Gui()
    qtbot.addWidget(gui)
    project = gui.project
    project.external_simulate_path = str(sim_path)
    project.load_NN()
    monkeypatch.setattr(neuroseq.gui, "AddExperimentDataDialog", _MockImportDialog)

    gui._do_external_simulation(1.0)
    _wait_for_external_simulation(qtbot, gui)

    fn = project.external_simulate_function
    assert isinstance(fn, FunctionType)
    captured = fn.__globals__["captured"]
    assert captured["ran"] is True
    gui.close()
