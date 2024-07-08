from __future__ import annotations

# python3.8 does not yet have importlib.resources
try:
    from importlib.resources import files
except Exception:
    from importlib_resources import files  # type: ignore

from pathlib import Path

from formozov_project.b2_model import B2Model
from formozov_project.gui import Gui
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets


def test_gui_click_exit(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_exit_clicked()
    qtbot.waitUntil(lambda: not gui.isVisible())


def test_gui_playback(
    monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    assert gui.heatmap_widget.currentIndex == 0
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.model = mfx_ornstein
    gui.btn_continue_simulation_clicked()
    n_timesteps = len(gui.model.network["M"].t)
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex > 0)
    # pause and then continue playback
    gui.btn_play_clicked()
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex == n_timesteps - 1)
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    gui.btn_skip_to_start_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert gui.heatmap_widget.currentIndex == 0
    gui.btn_skip_to_end_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    assert gui.heatmap_widget.currentIndex == n_timesteps - 1
    gui.btn_stop_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert gui.heatmap_widget.currentIndex == 0


def test_gui_new(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    monkeypatch.setattr("formozov_project.gui.B2ModelWidget.exec", lambda *args: True)
    gui.btn_new_clicked()
    assert gui.model.name == B2Model().name


def test_gui_open(
    monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    import formozov_project

    filename = str(files(formozov_project) / "default_model.yaml")

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (f"{filename}", True)
    )

    assert gui.model.parameters is None
    assert gui.model.neurongroups == {}
    assert gui.model.synapses == {}
    assert gui.model.monitors == {}

    loaded_model = B2Model.load_model(filename)

    gui.btn_open_clicked()

    assert list(gui.model.parameters.keys()) == list(loaded_model.parameters.keys())  # type: ignore
    assert list(gui.model.neurongroups.keys()) == list(loaded_model.neurongroups.keys())  # type: ignore
    assert list(gui.model.synapses.keys()) == list(loaded_model.synapses.keys())  # type: ignore
    assert list(gui.model.parameters.keys()) == list(loaded_model.parameters.keys())  # type: ignore


def test_gui_save(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.model = mfx_ornstein
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    filename = tmp_path / "test.h5"
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    gui.btn_save_clicked()
    assert filename.exists()


def test_gui_checkpoints(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    # mock information dialog box (with no checkpoints message) to do nothing
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    gui.btn_restore_checkpoint_clicked()
    # create a checkpoint - mock getItem to return the desired checkpoint name
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("test", True)
    )
    gui.model = mfx_ornstein
    gui.btn_store_checkpoint_clicked()
    assert "test" in gui.model.checkpoints
    # simulate
    simulation_time = 0.1
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    assert gui.model.network["M"].t[-1] == approx(simulation_time, rel=1e-2)
    # restore checkpoint
    gui.btn_restore_checkpoint_clicked()
    assert len(gui.model.network["M"].t) == 0
