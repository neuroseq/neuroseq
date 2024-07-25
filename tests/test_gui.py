from __future__ import annotations

from pathlib import Path
from typing import cast

import brian2 as b2
import pytest
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

# no obvious signal to wait for after the model is updated to know when the GUI is ready,
# so for now we just wait for a fixed time
delay_after_update_model_call_in_ms = 50


@pytest.mark.screenshot
def test_update_screenshot(monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.resize(1600, 1200)
    simulation_time = 2.0
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.btn_play_clicked()
    qtbot.waitUntil(
        lambda: gui.heatmap_widget.currentIndex == gui.heatmap_widget.tVals.size - 1
    )
    gui.grab().save("screenshot.png")


def test_gui_click_exit(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_exit_clicked()
    qtbot.waitUntil(lambda: not gui.isVisible())


def test_gui_simulate_and_playback(
    monkeypatch: MonkeyPatch, qtbot: QtBot, default_model_path: Path
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.currentIndex == 0
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.cmb_synapse.count() == 1
    assert gui.cmb_synapse.itemText(0) == "Ce / we"
    assert gui.cmb_neuron.count() == 1
    assert gui.cmb_neuron.itemText(0) == "P / v"
    assert gui.cmb_spike.count() == 1
    assert gui.cmb_spike.itemText(0) == "P"
    gui.btn_continue_simulation_clicked()
    n_timesteps = gui.heatmap_widget.tVals.size
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex > 0)
    # pause and then continue playback
    gui.btn_play_clicked()
    gui.btn_play_clicked()
    qtbot.waitUntil(lambda: gui.heatmap_widget.currentIndex == n_timesteps - 1)
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    assert gui.spike_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    gui.btn_skip_to_start_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert gui.spike_widget.timeline.getXPos() == approx(0)
    assert gui.heatmap_widget.currentIndex == 0
    gui.btn_skip_to_end_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    assert gui.spike_widget.timeline.getXPos() == approx(simulation_time, rel=1e-2)
    assert gui.heatmap_widget.currentIndex == n_timesteps - 1
    gui.btn_stop_clicked()
    assert gui.trace_widget.timeline.getXPos() == approx(0)
    assert gui.spike_widget.timeline.getXPos() == approx(0)
    assert gui.heatmap_widget.currentIndex == 0


def test_gui_new(tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    assert gui.model.name == B2Model().name
    monkeypatch.setattr("formozov_project.gui.B2ModelWidget.exec", lambda *args: True)
    gui.btn_new_clicked()
    assert gui.model.name == B2Model().name


def test_gui_open(
    monkeypatch: MonkeyPatch, qtbot: QtBot, default_model_path: Path
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: (f"{default_model_path}", True),
    )

    loaded_model = B2Model.load_model(default_model_path)
    assert loaded_model.parameters is not None

    gui.btn_open_clicked()

    assert gui.model.parameters is not None
    assert list(gui.model.parameters.keys()) == list(loaded_model.parameters.keys())
    assert list(gui.model.neurongroups.keys()) == list(loaded_model.neurongroups.keys())
    assert list(gui.model.synapses.keys()) == list(loaded_model.synapses.keys())
    assert list(gui.model.monitors.keys()) == list(loaded_model.monitors.keys())


def test_gui_save(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.update_model(mfx_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    filename = tmp_path / "test.json"
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    gui.btn_save_clicked()
    assert filename.exists()
    loaded_model = B2Model.load_model(filename)
    assert mfx_ornstein.neurongroups.keys() == loaded_model.neurongroups.keys()
    assert mfx_ornstein.synapses.keys() == loaded_model.synapses.keys()
    assert mfx_ornstein.monitors.keys() == loaded_model.monitors.keys()


def test_gui_export(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.update_model(mfx_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    filename = tmp_path / "test.h5"
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    gui.btn_export_clicked()
    assert filename.exists()


def test_gui_export_empty_model(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.update_model(B2Model())
    filename = tmp_path / "test.h5"
    assert not filename.exists()
    assert len(qmessagebox_calls) == 0
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    gui.btn_export_clicked()
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "export error" in qmessagebox_calls[0]["title"].lower()
    assert not filename.exists()


def test_gui_checkpoints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    mfx_ornstein: B2Model,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    assert len(qmessagebox_calls) == 0
    gui.btn_restore_checkpoint_clicked()
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "no simulation checkpoints" in qmessagebox_calls[0]["title"].lower()
    # create a checkpoint - mock getItem to return the desired checkpoint name
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("test", True)
    )
    gui.update_model(mfx_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.btn_store_checkpoint_clicked()
    assert "test" in gui.model.checkpoints
    # simulate
    simulation_time = 0.1
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.model.network["M"].t[-1] == approx(simulation_time, rel=1e-2)
    # restore checkpoint
    gui.btn_restore_checkpoint_clicked()
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
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
    gui.update_model(mfx_ornstein)
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


def test_gui_link_axes(
    qtbot: QtBot, monkeypatch: MonkeyPatch, default_model_path: Path
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    stim_widget_viewbox = gui.stim_editor.stim_widget.getPlotItem().getViewBox()
    spike_widget_viewbox = gui.spike_widget.getPlotItem().getViewBox()
    trace_widget_viewbox = gui.trace_widget.getPlotItem().getViewBox()
    # initially both X and Y axes are linked to the stim widget viewbox
    assert spike_widget_viewbox.state["linkedViews"][0]() == stim_widget_viewbox
    assert spike_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][0]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    gui.link_x_axes(False)
    assert spike_widget_viewbox.state["linkedViews"][0] is None
    assert spike_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][0] is None
    assert trace_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    gui.link_y_axes(False)
    assert spike_widget_viewbox.state["linkedViews"][0] is None
    assert spike_widget_viewbox.state["linkedViews"][1] is None
    assert trace_widget_viewbox.state["linkedViews"][0] is None
    assert trace_widget_viewbox.state["linkedViews"][1] is None
    gui.link_y_axes(True)
    assert spike_widget_viewbox.state["linkedViews"][0] is None
    assert spike_widget_viewbox.state["linkedViews"][1] is not None
    assert spike_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][0] is None
    assert trace_widget_viewbox.state["linkedViews"][1] is not None
    assert trace_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    gui.link_x_axes(True)
    assert spike_widget_viewbox.state["linkedViews"][0] is not None
    assert spike_widget_viewbox.state["linkedViews"][0]() == stim_widget_viewbox
    assert spike_widget_viewbox.state["linkedViews"][1] is not None
    assert spike_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][0] is not None
    assert trace_widget_viewbox.state["linkedViews"][0]() == stim_widget_viewbox
    assert trace_widget_viewbox.state["linkedViews"][1] is not None
    assert trace_widget_viewbox.state["linkedViews"][1]() == stim_widget_viewbox


def test_gui_synapse_ram_limit(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.currentIndex == 0
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 1000
    # 1 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1", True))
    gui.btn_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 334
    # 2 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("2", True))
    gui.btn_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 500
    # 0 MB RAM limit == unlimited
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0", True))
    gui.btn_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 1000
    # invalid string displays error message
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("imnotanumber!", True)
    )
    gui.btn_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 1000
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "invalid input" in qmessagebox_calls[0]["title"].lower()
