from __future__ import annotations

from pathlib import Path

import brian2 as b2
import pytest
from brian2 import numpy_ as np
from brian2.tests.utils import assert_allclose
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

import neuroseq.save_options
from neuroseq.example_models import default_model
from neuroseq.example_models import example_models
from neuroseq.gui import Gui
from neuroseq.model import Model
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.save_options import SaveOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

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
    gui.time_slider.toggle_pause()
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


def test_gui_simulate_and_playback(monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.get_timestamp_index() == 0
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    simulation_time_s = 1.0
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time_s}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    simulation_time_ms = simulation_time_s * 1000
    gui.btn_continue_simulation_clicked()
    assert gui.cmb_synapse.count() == 1
    assert gui.cmb_synapse.itemText(0) == "Simulation :: Ce / we"
    assert gui.cmb_trace.count() == 1
    assert gui.cmb_trace.itemText(0) == "Simulation :: P / v"
    assert gui.cmb_spike.count() == 1
    assert gui.cmb_spike.itemText(0) == "Simulation :: P"
    assert gui.heatmap_widget.data is not None
    assert gui.heatmap_widget.timestamps is not None
    n_timesteps = gui.heatmap_widget.timestamps.size
    gui.time_slider.toggle_pause()
    qtbot.waitUntil(lambda: gui.heatmap_widget.get_timestamp_index() > 0)
    # pause and then continue playback
    gui.time_slider.toggle_pause()
    gui.time_slider.toggle_pause()
    qtbot.waitUntil(
        lambda: gui.heatmap_widget.get_timestamp_index() == n_timesteps - 1,
        timeout=100000,
    )
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.trace_widget.timeline is not None
    assert gui.stim_editor.stim_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.trace_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.spike_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.heatmap_widget.get_timestamp_index() == n_timesteps - 1
    gui.time_slider.skip_to_start()
    assert gui.stim_editor.stim_widget.timeline.t == approx(0, rel=2)
    assert gui.trace_widget.timeline.t == approx(0, rel=2)
    assert gui.spike_widget.timeline.t == approx(0, rel=2)
    assert gui.heatmap_widget.get_timestamp_index() == 0
    gui.time_slider.skip_to_end()
    assert gui.stim_editor.stim_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.trace_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.spike_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.heatmap_widget.get_timestamp_index() == n_timesteps - 1
    gui.time_slider.stop()
    assert gui.stim_editor.stim_widget.timeline.t == approx(0, rel=2)
    assert gui.trace_widget.timeline.t == approx(0, rel=2)
    assert gui.spike_widget.timeline.t == approx(0, rel=2)
    assert gui.heatmap_widget.get_timestamp_index() == 0


def test_gui_new(tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    model = default_model()
    # GUI loads default model if no model file is provided:
    assert gui.model.b2_model.parameters.keys() == model.b2_model.parameters.keys()
    assert gui.model.b2_model.neurongroups.keys() == model.b2_model.neurongroups.keys()
    assert gui.model.b2_model.synapses.keys() == model.b2_model.synapses.keys()
    assert gui.model.b2_model.monitors.keys() == model.b2_model.monitors.keys()
    # File -> New also loads the default model:
    gui.btn_new_clicked()
    assert gui.model.b2_model.parameters.keys() == model.b2_model.parameters.keys()
    assert gui.model.b2_model.neurongroups.keys() == model.b2_model.neurongroups.keys()
    assert gui.model.b2_model.synapses.keys() == model.b2_model.synapses.keys()
    assert gui.model.b2_model.monitors.keys() == model.b2_model.monitors.keys()


def test_gui_open(monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_path: Path) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    model = example_models[list(sorted(example_models.keys()))[-1]]()
    model_path = tmp_path / "model.nseq"
    model.save(
        model_path,
        SaveOptions(
            include_stim_arrays=True,
            include_model=True,
            include_simulation_data=False,
            include_experimental_data=False,
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: (f"{model_path}", True),
    )
    gui.btn_open_clicked()

    assert gui.model.b2_model.parameters is not None
    assert list(gui.model.b2_model.parameters.keys()) == list(
        model.b2_model.parameters.keys()
    )
    assert list(gui.model.b2_model.neurongroups.keys()) == list(
        model.b2_model.neurongroups.keys()
    )
    assert list(gui.model.b2_model.synapses.keys()) == list(
        model.b2_model.synapses.keys()
    )
    assert list(gui.model.b2_model.monitors.keys()) == list(
        model.b2_model.monitors.keys()
    )


class MockSaveOptionsDialog:
    save_options = SaveOptions(True, True, True, True)

    def exec(self) -> bool:
        return True


def test_gui_save(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: Model
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
    monkeypatch.setattr(
        neuroseq.gui,
        "SaveOptionsDialog",
        MockSaveOptionsDialog,
    )
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    assert filename.exists()
    loaded_model = Model(filename)
    assert (
        mfx_ornstein.b2_model.neurongroups.keys()
        == loaded_model.b2_model.neurongroups.keys()
    )
    assert (
        mfx_ornstein.b2_model.synapses.keys() == loaded_model.b2_model.synapses.keys()
    )
    assert (
        mfx_ornstein.b2_model.monitors.keys() == loaded_model.b2_model.monitors.keys()
    )


def test_gui_export_with_simulation_data(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, mfx_ornstein: Model
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
    gui.update_model(Model())
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
    mfx_ornstein: Model,
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
    assert "test" in gui.model.b2_model.checkpoints
    # simulate
    simulation_time = 0.1
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.model.b2_model.network["M"].t[-1] == approx(simulation_time, rel=1e-2)
    # restore checkpoint
    gui.btn_restore_checkpoint_clicked()
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    assert len(gui.model.b2_model.network["M"].t) == 0


def test_gui_edit_stim_with_console(qtbot: QtBot, mfx_ornstein: Model) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 10 * b2.ms
    stim = np.zeros((n_timestamps, n_neurons))
    assert mfx_ornstein.stim_arrays == {}
    mfx_ornstein.stim_arrays = {
        "stim": StimArray("stim", StimType.Excitation, stim, "mV", dt)
    }
    gui.update_model(mfx_ornstein)
    stim_editor = gui.stim_editor
    stim_console = gui.stim_console
    assert stim_editor.stims_dict is not None
    assert_allclose(stim_editor.stims_dict["stim"].stim_array.values, stim)
    # set all stimuli to 1: StimWidget and model stimuli updated
    stim_console.input.setText("stims['stim'][:,:] = 1")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    stim[:, :] = 1
    assert_allclose(stim_editor.stims_dict["stim"].stim_array.values, stim)
    assert_allclose(gui.model.stim_arrays["stim"].values, stim)
    assert stim_editor.stim_widget.n_neurons == n_neurons
    # set stims to an invalid value: StimWidget and model not updated
    stim_console.input.setText("stims['stim'] = 5")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_editor.stim_widget.n_neurons == n_neurons
    assert_allclose(stim_editor.stims_dict["stim"].stim_array.values, stim)
    assert_allclose(gui.model.stim_arrays["stim"].values, stim)
    # assign random valid stim array with different shape, StimWidget and model updated
    stim_console.input.setText("stims['stim'] = np.random.rand(127, 83)")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert stim_editor.stims_dict["stim"].stim_array.values.shape == (127, 83)
    assert gui.model.stim_arrays["stim"].values.shape == (127, 83)
    assert stim_editor.stim_widget.n_neurons == 83


def test_gui_synapse_ram_limit(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    tmp_config_dir,
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.get_timestamp_index() == 0
    simulation_time = 0.1
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.data is not None
    assert gui.heatmap_widget.timestamps is not None
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 1000
    # 1 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1", True))
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 334
    # 2 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("2", True))
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 500
    # 0 MB RAM limit == unlimited
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0", True))
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 1000
    # invalid string displays error message
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("imnotanumber!", True)
    )
    gui.choose_max_ram_synapse_plot_matrix()
    qtbot.wait(delay_after_update_model_call_in_ms)
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 1000
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "invalid input" in qmessagebox_calls[0]["title"].lower()


def test_gui_brian2_backend(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_config_dir
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert b2.prefs.codegen.target == "numpy"
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


def test_gui_stim_padding(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    tmp_config_dir,
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.get_timestamp_index() == 0
    # set 10secs of stim padding
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("10", True))
    gui.choose_stim_padding_secs()
    qtbot.wait(delay_after_update_model_call_in_ms)
    # simulate 0.1s
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.1", True))
    gui.btn_continue_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    # should have ~0.1 secs of simulation data, and ~10.1 secs of stim array data with dt=0.1ms
    assert gui.heatmap_widget.data is not None
    assert gui.heatmap_widget.timestamps is not None
    assert gui.heatmap_widget.timestamps[-1] == approx(100, rel=2)
    assert gui.stim_editor.stim_widget.n_timestamps == 10100


def test_gui_restore_settings_on_restart(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_config_dir
) -> None:
    gui = Gui()
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
    qtbot.wait(delay_after_update_model_call_in_ms)

    gui2 = Gui()
    gui2.show()
    qtbot.addWidget(gui2)
    qtbot.waitUntil(lambda: gui2.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert b2.prefs.codegen.target == "cython"
    assert gui2.settings.max_ram_synapse_plot_matrix == 512
    assert gui2.settings.stim_padding_secs == 6


def test_gui_playback_speed(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.time_slider.relative_playback_speed == pytest.approx(1.0)
    # mock getItem to return the desired playback speed
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("2.6743", True)
    )
    gui.btn_playback_speed_clicked()
    assert gui.time_slider.relative_playback_speed == pytest.approx(2.6743)
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("0.1234", True)
    )
    gui.btn_playback_speed_clicked()
    assert gui.time_slider.relative_playback_speed == pytest.approx(0.1234)
    # pressing cancel should not change the playback speed
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1.0", False))
    gui.btn_playback_speed_clicked()
    assert gui.time_slider.relative_playback_speed == pytest.approx(0.1234)
    # invalid input should show an error message and not change the playback speed
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.0", True))
    gui.btn_playback_speed_clicked()
    assert gui.time_slider.relative_playback_speed == pytest.approx(0.1234)
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
        gui.stim_editor.stim_widget.n_timestamps * gui.stim_editor.stim_widget.dt_ms
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
    gui.btn_do_external_simulation_clicked()
    qtbot.wait(delay_after_update_model_call_in_ms)
    simulation_time_ms = simulation_time_secs * 1000
    # stim array should be padded to simulation time + default padding of 5 secs
    assert (
        gui.stim_editor.stim_widget.n_timestamps * gui.stim_editor.stim_widget.dt_ms
        == approx(simulation_time_ms + 5000, rel=2)
    )
    # select the newly added simulation data in the dropdowns
    gui.set_neuron(gui.cmb_trace.itemText(gui.cmb_trace.count() - 1))
    gui.set_spike(gui.cmb_spike.itemText(gui.cmb_spike.count() - 1))
    gui.set_synapse(gui.cmb_synapse.itemText(gui.cmb_synapse.count() - 1))
    assert gui.trace_widget.n_neurons == 32
    assert gui.trace_widget.n_timestamps == 2000
    assert gui.spike_widget.n_neurons == 32
    assert gui.spike_widget.dt_ms == approx(simulation_time_ms / 2000, rel=2)
    assert gui.heatmap_widget.data is not None
    assert gui.heatmap_widget.timestamps is not None
    assert gui.heatmap_widget.timestamps[-1] == approx(simulation_time_ms, rel=2)


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

    assert gui.model.b2_model is not modeleditor.b2model

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

    # assert Ce is gone from the model in the model editor, but still in the GUI model
    assert "Ce" not in modeleditor.b2model.synapses
    assert "Ce" in gui.model.b2_model.synapses

    # try to accept, error
    qtbot.mouseClick(
        modeleditor.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok),
        QtCore.Qt.MouseButton.LeftButton,
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
        modeleditor.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel),
        QtCore.Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: not modeleditor.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)

    # assert that Ce is in the definition and in the network -> restoring old state works
    assert "Ce" in gui.model.b2_model.synapses

    # run the model and make sure it works as it should
    gui.model.run_simulation(0.1)
    assert gui.model.b2_model.network["Ce"] is not None

    assert gui.model.b2_model.network["trace"].v.shape == (30, 1000)
