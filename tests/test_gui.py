from __future__ import annotations

import json
from pathlib import Path

import brian2 as b2
import pytest
from brian2 import numpy_ as np
from brian2.tests.utils import assert_allclose
from platformdirs import user_config_dir
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.b2_model import B2Model
from neuroseq.gui import Gui
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog

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
    histogram_action = next(
        action for action in gui.trace_widget.actions() if action.text() == "Histogram"
    )
    simulation_time = 2.0
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    histogram_action.trigger()
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
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    simulation_time_ms = simulation_time * 1000
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
    assert gui.trace_widget.timeline is not None
    assert gui.trace_widget.timeline.selection == approx(simulation_time_ms, rel=2)
    # todo: re-enable these spike_widget timeline tests when fastplotlib supports the timeline in scatter plots
    # assert gui.spike_widget.timeline.selection == approx(simulation_time_ms, rel=2)
    gui.btn_skip_to_start_clicked()
    assert gui.trace_widget.timeline.selection == approx(0)
    # assert gui.spike_widget.timeline.selection == approx(0)
    assert gui.heatmap_widget.currentIndex == 0
    gui.btn_skip_to_end_clicked()
    assert gui.trace_widget.timeline.selection == approx(simulation_time_ms, rel=2)
    # assert gui.spike_widget.timeline.selection == approx(simulation_time_ms, rel=2)
    assert gui.heatmap_widget.currentIndex == n_timesteps - 1
    gui.btn_stop_clicked()
    assert gui.trace_widget.timeline.selection == approx(0)
    # assert gui.spike_widget.timeline.selection == approx(0)
    assert gui.heatmap_widget.currentIndex == 0


def test_gui_new(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, default_model_path: Path
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    default_model = B2Model.load_model(default_model_path)
    # GUI loads default model if no model file is provided:
    assert gui.model.parameters.keys() == default_model.parameters.keys()
    assert gui.model.neurongroups.keys() == default_model.neurongroups.keys()
    assert gui.model.synapses.keys() == default_model.synapses.keys()
    assert gui.model.monitors.keys() == default_model.monitors.keys()
    # File -> New also loads the default model:
    gui.btn_new_clicked()
    assert gui.model.parameters.keys() == default_model.parameters.keys()
    assert gui.model.neurongroups.keys() == default_model.neurongroups.keys()
    assert gui.model.synapses.keys() == default_model.synapses.keys()
    assert gui.model.monitors.keys() == default_model.monitors.keys()


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
    filename_without_extension = tmp_path / "test"
    # the gui will add the .nseq extension automatically
    filename = filename_without_extension.with_suffix(".nseq")
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    assert filename.exists()
    loaded_model = B2Model.load_model(filename)
    assert mfx_ornstein.neurongroups.keys() == loaded_model.neurongroups.keys()
    assert mfx_ornstein.synapses.keys() == loaded_model.synapses.keys()
    assert mfx_ornstein.monitors.keys() == loaded_model.monitors.keys()


def test_gui_export_with_simulation_data(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: B2Model
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    # generate some simulation data
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.1", True))
    gui.btn_continue_simulation_clicked()
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
    assert mfx_ornstein.parameters == {}
    mfx_ornstein.parameters = {"stim": b2.TimedArray(stim, dt=dt)}
    gui.update_model(mfx_ornstein)
    stim_editor = gui.stim_editor
    stim_console = gui.stim_console
    assert stim_editor.stims_dict is not None
    assert_allclose(stim_editor.stims_dict["stim"].timed_array.values, stim)
    # set all stimuli to 1: StimWidget and model stimuli updated
    stim_console.input.setText("stims['stim'][:,:] = 1")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    stim[:, :] = 1
    assert_allclose(stim_editor.stims_dict["stim"].timed_array.values, stim)
    assert gui.model.parameters is not None
    assert_allclose(gui.model.parameters["stim"].values, stim)
    assert stim_editor.stim_widget.n_neurons == n_neurons
    # set stims to an invalid value: StimWidget and model not updated
    stim_console.input.setText("stims['stim'] = 5")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_editor.stim_widget.n_neurons == n_neurons
    assert_allclose(stim_editor.stims_dict["stim"].timed_array.values, stim)
    assert_allclose(gui.model.parameters["stim"].values, stim)
    # assign random valid stim array with different shape, StimWidget and model updated
    stim_console.input.setText("stims['stim'] = np.random.rand(127, 83)")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_editor.stims_dict["stim"].timed_array.values.shape == (127, 83)
    assert gui.model.parameters["stim"].values.shape == (127, 83)
    assert stim_editor.stim_widget.n_neurons == 83


def test_gui_synapse_ram_limit(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    # get current config settings to restore after the test, since choosing the ram limit now overwrites the config file
    orig_setting = gui.settings.max_ram_synapse_plot_matrix
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
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 334
    # 2 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("2", True))
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 500
    # 0 MB RAM limit == unlimited
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0", True))
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 1000
    # invalid string displays error message
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("imnotanumber!", True)
    )
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.tVals.size
    assert n_timesteps == 1000
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "invalid input" in qmessagebox_calls[0]["title"].lower()

    # restore original setting
    gui.settings.max_ram_synapse_plot_matrix = orig_setting
    gui.save_config_settings()


def test_gui_brian2_backend(monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    # retrieve current config settings to restore after the test, since choosing the backend now overwrites the config file
    orig_setting = gui.settings.brian2_backend
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert b2.prefs.codegen.target == orig_setting
    # mock getItem to return the desired backend choice
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("cython", True)
    )
    gui.choose_brian2_backend()
    assert b2.prefs.codegen.target == "cython"
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("numpy", True)
    )
    gui.choose_brian2_backend()
    assert b2.prefs.codegen.target == "numpy"
    # check it also works if the initial value is somehow invalid
    gui.settings.brian2_backend = "invalid"
    gui.choose_brian2_backend()
    assert b2.prefs.codegen.target == "numpy"

    # restore original setting
    gui.settings.brian2_backend = orig_setting
    gui.save_config_settings()


def test_gui_stim_padding(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    # get current config settings to restore after the test, since choosing the stim padding now overwrites the config file
    orig_setting = gui.settings.stim_padding_secs
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.currentIndex == 0
    # set 10secs of stim padding
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("10", True))
    gui.choose_stim_padding_secs()
    qtbot.wait(delay_after_update_model_call_in_ms)
    # simulate 0.1s
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.1", True))
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    # should have ~0.1 secs of simulation data, and ~10.1 secs of stim array data with dt=0.1ms
    assert gui.heatmap_widget.tVals[-1] == approx(100, rel=2)
    assert gui.stim_editor.stim_widget.n_timestamps == 10100

    # restore original setting
    gui.settings.stim_padding_secs = orig_setting
    gui.save_config_settings()


def test_gui_restore_settings_on_restart(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    gui = Gui()
    # retrieve current config settings to restore after the test, since choosing the backend now overwrites the config file
    orig_backend = gui.settings.brian2_backend
    orig_ram_limit = gui.settings.max_ram_synapse_plot_matrix
    orig_stim_padding = gui.settings.stim_padding_secs
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    # change all settings
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("cython", True)
    )
    gui.choose_brian2_backend()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("512", True))
    gui.choose_max_ram_synapse_plot_matrix()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("6", True))
    gui.choose_stim_padding_secs()
    assert b2.prefs.codegen.target == "cython"
    assert gui.settings.max_ram_synapse_plot_matrix == 512
    assert gui.settings.stim_padding_secs == 6

    # close gui and reopen to check if settings were saved and reloaded correctly
    gui.close()
    gui2 = Gui()
    gui2.show()
    qtbot.addWidget(gui2)
    qtbot.waitUntil(lambda: gui2.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert b2.prefs.codegen.target == "cython"
    assert gui2.settings.max_ram_synapse_plot_matrix == 512
    assert gui2.settings.stim_padding_secs == 6

    # restore original settings
    gui2.settings.brian2_backend = orig_backend
    gui2.settings.max_ram_synapse_plot_matrix = orig_ram_limit
    gui2.settings.stim_padding_secs = orig_stim_padding


def test_gui_save_config_settings(qtbot: QtBot) -> None:
    gui = Gui()
    # retrieve current config settings to restore after the test, since choosing the backend now overwrites the config file
    orig_backend = gui.settings.brian2_backend
    orig_ram_limit = gui.settings.max_ram_synapse_plot_matrix
    orig_stim_padding = gui.settings.stim_padding_secs
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)

    # change all settings
    gui.settings.brian2_backend = "numpy" if orig_backend == "cython" else "cython"
    gui.settings.max_ram_synapse_plot_matrix = (
        orig_ram_limit + 1 if orig_ram_limit < 1024 else orig_ram_limit - 1
    )
    gui.settings.stim_padding_secs = (
        orig_stim_padding + 1 if orig_stim_padding < 20 else orig_stim_padding - 1
    )

    # check if settings file exists and contains the correct values
    config_dir = Path(user_config_dir("neuroseq", "ssciwr"))
    config_file = config_dir / "config.json"
    assert config_file.exists()

    # check if settings were saved correctly
    with open(config_file) as f:
        config = json.load(f)
    assert config["brian2_backend"] == gui.settings.brian2_backend
    assert (
        config["max_ram_synapse_plot_matrix"]
        == gui.settings.max_ram_synapse_plot_matrix
    )
    assert config["stim_padding_secs"] == gui.settings.stim_padding_secs

    # restore original settings
    gui.settings.brian2_backend = orig_backend
    gui.settings.max_ram_synapse_plot_matrix = orig_ram_limit
    gui.settings.stim_padding_secs = orig_stim_padding


def test_gui_load_config_settings(qtbot: QtBot) -> None:
    gui = Gui()
    # retrieve current config settings to restore after the test, since choosing the backend now overwrites the config file
    orig_backend = gui.settings.brian2_backend
    orig_ram_limit = gui.settings.max_ram_synapse_plot_matrix
    orig_stim_padding = gui.settings.stim_padding_secs
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)

    # change all settings
    new_backend = "numpy" if orig_backend == "cython" else "cython"
    new_ram_limit = orig_ram_limit + 1 if orig_ram_limit < 1024 else orig_ram_limit - 1
    new_stim_padding = (
        orig_stim_padding + 1 if orig_stim_padding < 20 else orig_stim_padding - 1
    )
    gui.settings.brian2_backend = new_backend
    gui.settings.max_ram_synapse_plot_matrix = new_ram_limit
    gui.settings.stim_padding_secs = new_stim_padding

    # set settings to original to check if loading works
    gui.settings.__dict__["brian2_backend"] = orig_backend
    gui.settings.__dict__["max_ram_synapse_plot_matrix"] = orig_ram_limit
    gui.settings.__dict__["stim_padding_secs"] = orig_stim_padding

    # load settings from file
    gui.settings.load_config_settings()

    # check if settings were loaded correctly
    assert gui.settings.brian2_backend == new_backend
    assert gui.settings.max_ram_synapse_plot_matrix == new_ram_limit
    assert gui.settings.stim_padding_secs == new_stim_padding

    # try loading invalid values
    gui.settings.brian2_backend = "invalid backend"
    gui.settings.max_ram_synapse_plot_matrix = -5
    gui.settings.stim_padding_secs = -3

    # try loading invalid values
    # brian2 backend - should reset to default 'numpy'
    gui.settings.load_config_settings()
    assert gui.settings.brian2_backend == "numpy"
    # max_ram_synapse_plot_matrix - should be set to 0
    assert gui.settings.max_ram_synapse_plot_matrix == 0
    # stim_padding_secs - should be set to 0
    assert gui.settings.stim_padding_secs == 0

    # try setting RAM to string
    gui.settings.brian2_backend = orig_backend  # valid again
    gui.settings.max_ram_synapse_plot_matrix = "this is not even a number!"  # type: ignore[assignment]
    gui.settings.stim_padding_secs = orig_stim_padding  # valid again
    # should log a ValueError and default to 256
    gui.settings.load_config_settings()
    assert gui.settings.max_ram_synapse_plot_matrix == 256

    # try setting stim_padding_secs to string
    gui.settings.max_ram_synapse_plot_matrix = orig_ram_limit  # valid again
    gui.settings.stim_padding_secs = "this is not even a number!"  # type: ignore[assignment]
    # should log a ValueError and default to 5s
    gui.settings.load_config_settings()
    assert gui.settings.stim_padding_secs == 5

    # restore original settings
    gui.settings.brian2_backend = orig_backend
    gui.settings.max_ram_synapse_plot_matrix = orig_ram_limit
    gui.settings.stim_padding_secs = orig_stim_padding


def test_gui_playback_speed(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.relative_playback_speed == pytest.approx(1.0)
    # mock getItem to return the desired playback speed
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("2.6743", True)
    )
    gui.btn_playback_speed_clicked()
    assert gui.heatmap_widget.relative_playback_speed == pytest.approx(2.6743)
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("0.1234", True)
    )
    gui.btn_playback_speed_clicked()
    assert gui.heatmap_widget.relative_playback_speed == pytest.approx(0.1234)
    # pressing cancel should not change the playback speed
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1.0", False))
    gui.btn_playback_speed_clicked()
    assert gui.heatmap_widget.relative_playback_speed == pytest.approx(0.1234)
    # invalid input should show an error message and not change the playback speed
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.0", True))
    gui.btn_playback_speed_clicked()
    assert gui.heatmap_widget.relative_playback_speed == pytest.approx(0.1234)
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "invalid input" in qmessagebox_calls[0]["title"].lower()


def test_gui_simulate_external_model(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    # load external simulation function
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: ("external_simulation_example/simulate.py", True),
    )
    assert (
        gui.stim_editor.stim_widget.n_timestamps * gui.stim_editor.stim_widget.dt
        == approx(2000, rel=2)
    )
    # simulate for 10 seconds
    simulation_time_secs = 10
    gui.connect_external_simulation_function()
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args: (f"{simulation_time_secs}", True),
    )
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    simulation_time_ms = simulation_time_secs * 1000
    # stim array should be padded to simulation time + default padding of 5 secs
    assert (
        gui.stim_editor.stim_widget.n_timestamps * gui.stim_editor.stim_widget.dt
        == approx(simulation_time_ms + 5000, rel=2)
    )
    assert gui.trace_widget.n_neurons == 32
    assert gui.trace_widget.n_timestamps == 2000
    assert gui.spike_widget.n_neurons == 32
    assert gui.spike_widget.dt == approx(simulation_time_ms / 2000, rel=2)
    assert gui.heatmap_widget.tVals[-1] == approx(simulation_time_ms, rel=2)


def test_integration_modeleditor_delete_restore(
    qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(gui.isVisible)
    qtbot.wait(delay_after_update_model_call_in_ms)

    gui.btn_edit_clicked()

    modeleditor = gui.findChild(ModelEditorDialog)

    qtbot.addWidget(modeleditor)
    qtbot.waitUntil(modeleditor.isVisible)

    # find synapse tree
    synapsetree = modeleditor.findChild(EditableSynapseTree)

    assert "Ce" in modeleditor.b2model.synapses

    # delete Ce synapse by triggering the remove signal
    ce_synapse_element = synapsetree.findItems(
        "Ce",
        QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
        0,
    )[0]

    with qtbot.waitSignal(
        ce_synapse_element.param.sigRemoved, timeout=1000, raising=True
    ):
        ce_synapse_element.param.remove()

    # assert Ce is gone from the definition
    assert "Ce" not in modeleditor.b2model.synapses
    assert "Ce" in gui.model.synapses

    # try to accept, error
    qtbot.mouseClick(
        modeleditor.button_box.button(QtWidgets.QDialogButtonBox.Ok),
        QtCore.Qt.LeftButton,
    )

    # check error message
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "model rebuilding error" in qmessagebox_calls[0]["title"].lower()
    assert (
        "error when rebuilding the model after editing"
        in qmessagebox_calls[0]["text"].lower()
    )

    # reject-> only option available, restores old model
    qtbot.mouseClick(
        modeleditor.button_box.button(QtWidgets.QDialogButtonBox.Cancel),
        QtCore.Qt.LeftButton,
    )
    qtbot.waitUntil(lambda: not modeleditor.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)

    # assert that Ce is in the definition and in the network -> restoring old state works
    assert "Ce" in gui.model.synapses
    assert gui.model.network["Ce"] is not None

    # run the model and make sure it works as it should
    gui.model.run_simulation(0.2 * b2.second)

    assert gui.model.network["Traces_v"].v.shape == (30, 2000)
