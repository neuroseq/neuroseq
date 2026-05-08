"""GUI tests for the Run-button dispatch between Brian2 and external simulation.

When ``project.external_simulate_path`` is set, the main "Run simulation"
button should route to ``_do_external_simulation`` instead of the b2_model
path.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType

from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq
from neuroseq.gui import Gui


class _MockImportDialog:
    """Stub for `ImportExperimentDialog` so the external sim flow doesn't block."""

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
    project.load_external_simulator()
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

    def fake_b2(time_seconds, **kwargs):
        calls["b2"].append(time_seconds)

    monkeypatch.setattr(gui, "_do_external_simulation", fake_external)
    monkeypatch.setattr(project, "run_simulation", fake_b2)
    monkeypatch.setattr(gui, "_update_data", lambda **kw: None)
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
    project.load_external_simulator()
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

    monkeypatch.setattr(neuroseq.gui, "ImportExperimentDialog", _MockImportDialog)

    gui._do_external_simulation(2.0)

    fn = project.external_simulate_function
    assert isinstance(fn, FunctionType)
    captured = fn.__globals__["captured"]
    assert captured["simulation_time"] == 2.0
    assert captured["params"] == {"k1": 42, "k2": 1.5}
    gui.close()
