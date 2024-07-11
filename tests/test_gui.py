from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import cast

import brian2 as b2
import formozov_project
from brian2 import numpy_ as np
from brian2.tests.utils import assert_allclose
from formozov_project.b2_model import B2Model
from formozov_project.gui import Gui
from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


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
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.load_model(
        str(files(formozov_project) / "example_models" / "default_model.json")
    )
    gui.btn_continue_simulation_clicked()
    n_timesteps = gui.heatmap_widget.tVals.size
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

    filename = str(files(formozov_project) / "example_models" / "default_model.json")

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (f"{filename}", True)
    )

    assert gui.model.parameters is None
    assert gui.model.neurongroups == {}
    assert gui.model.synapses == {}
    assert gui.model.monitors == {}

    loaded_model = B2Model.load_model(filename)

    gui.btn_open_clicked()

    assert gui.model.parameters is not None
    assert list(gui.model.parameters.keys()) == list(loaded_model.parameters.keys())
    assert list(gui.model.neurongroups.keys()) == list(loaded_model.neurongroups.keys())
    assert list(gui.model.synapses.keys()) == list(loaded_model.synapses.keys())
    assert list(gui.model.parameters.keys()) == list(loaded_model.parameters.keys())


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
    gui.update_model()
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


def test_gui_edit_stim_with_console(qtbot: QtBot, mfx_ornstein: B2Model) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 10 * b2.ms
    stim = np.zeros((n_timestamps, n_neurons))
    assert mfx_ornstein.parameters is None
    mfx_ornstein.parameters = {"stim": b2.TimedArray(stim, dt=dt)}
    gui.model = mfx_ornstein
    gui.update_model()
    # bar plot x-values are the center of the bars, so shifted forwards by 0.5dt compared to timestamps
    expected_timestamps_array = dt * (np.arange(n_timestamps) + 0.5) / b2.second
    stim_editor = gui.stim_editor
    stim_console = gui.stim_console
    assert stim_editor.stims_dict is not None
    assert_allclose(stim_editor.stims_dict["stim"].values, stim)
    # set all stimuli to 1: StimWidget and model stimuli updated
    stim_console.input.setText("stims['stim'][:,:] = 1")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    stim[:, :] = 1
    assert_allclose(stim_editor.stims_dict["stim"].values, stim)
    assert gui.model.parameters is not None
    assert_allclose(gui.model.parameters["stim"].values, stim)
    assert len(stim_editor.stim_widget.neuron_plots) == n_neurons
    neuron_plot = cast(SelectableBarPlotItem, stim_editor.stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert_allclose(neuron_plot.getData()[0], expected_timestamps_array)
    # set stims to an invalid value: StimWidget and model not updated
    stim_console.input.setText("stims['stim'] = 5")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert len(stim_editor.stim_widget.neuron_plots) == n_neurons
    assert_allclose(stim_editor.stims_dict["stim"].values, stim)
    assert_allclose(gui.model.parameters["stim"].values, stim)
    # assign random valid stim array with different shape, StimWidget and model updated
    stim_console.input.setText("stims['stim'] = np.random.rand(127, 83)")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_editor.stims_dict["stim"].values.shape == (127, 83)
    assert gui.model.parameters["stim"].values.shape == (127, 83)
    assert len(stim_editor.stim_widget.neuron_plots) == 83
