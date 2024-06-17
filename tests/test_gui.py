from __future__ import annotations

from formozov_project.gui import Gui
from pytest import approx, MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets
from pathlib import Path


def test_gui_click_exit(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_exit_clicked()
    qtbot.waitUntil(lambda: not gui.isVisible())


def test_gui_playback(monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    assert gui.heatmap_widget.currentIndex == 0
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True))
    gui.btn_simulate_clicked()
    n_timesteps = len(gui.model.network["M"].t)
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex > 0)
    gui.btn_stop_clicked()
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex == n_timesteps - 1)
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    gui.btn_skip_to_start_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert gui.heatmap_widget.currentIndex == 0
    gui.btn_skip_to_end_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    assert gui.heatmap_widget.currentIndex == n_timesteps - 1


def test_gui_save(tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    filename = tmp_path / "test.h5"
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True))
    gui.btn_save_clicked()
    assert filename.exists()
