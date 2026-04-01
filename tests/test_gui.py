from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import brian2 as b2
import pytest
from brian2 import numpy_ as np
from brian2.tests.utils import assert_allclose
from pytest import MonkeyPatch
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

import neuroseq
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui import Gui
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

# no obvious signal to wait for after the model is updated to know when the GUI is ready,
# so for now we just wait for a fixed time
delay_after_update_model_call_in_ms = 50


def test_gui_click_exit(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.btn_exit_clicked()
    qtbot.waitUntil(lambda: not gui.isVisible())
    gui.close()


def test_gui_open_example_project_menu_has_subsections(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    assert gui.open_example_project_menu is not None
    assert list(gui.open_example_project_section_menus) == list(example_projects)
    assert len(gui.open_example_project_actions) == sum(
        len(section_projects) for section_projects in example_projects.values()
    )
    assert {"Getting Started Project", "non_linear"}.issubset(
        example_projects["Basics"]
    )

    gui.close()


def test_gui_simulate_and_playback(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project
) -> None:
    gui = Gui()
    gui.load_project(test_project)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.idx == 0
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
    qtbot.waitUntil(lambda: gui.heatmap_widget.idx > 0)
    # pause and then continue playback
    gui.time_slider.toggle_pause()
    gui.time_slider.toggle_pause()
    qtbot.waitUntil(
        lambda: gui.heatmap_widget.idx == n_timesteps - 1,
        timeout=100000,
    )
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.trace_widget.timeline is not None
    assert gui.stim_editor.stim_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.trace_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.spike_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.heatmap_widget.idx == n_timesteps - 1
    gui.time_slider.skip_to_start()
    assert gui.stim_editor.stim_widget.timeline.t == approx(0, rel=2)
    assert gui.trace_widget.timeline.t == approx(0, rel=2)
    assert gui.spike_widget.timeline.t == approx(0, rel=2)
    assert gui.heatmap_widget.idx == 0
    gui.time_slider.skip_to_end()
    assert gui.stim_editor.stim_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.trace_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.spike_widget.timeline.t == approx(simulation_time_ms, rel=2)
    assert gui.heatmap_widget.idx == n_timesteps - 1
    gui.time_slider.stop()
    assert gui.stim_editor.stim_widget.timeline.t == approx(0, rel=2)
    assert gui.trace_widget.timeline.t == approx(0, rel=2)
    assert gui.spike_widget.timeline.t == approx(0, rel=2)
    assert gui.heatmap_widget.idx == 0
    gui.close()


def test_gui_new(tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    project = default_project()
    # GUI loads default model if no model file is provided:
    assert gui.project.b2_model.parameters.keys() == project.b2_model.parameters.keys()
    assert (
        gui.project.b2_model.neurongroups.keys() == project.b2_model.neurongroups.keys()
    )
    assert gui.project.b2_model.synapses.keys() == project.b2_model.synapses.keys()
    assert gui.project.b2_model.monitors.keys() == project.b2_model.monitors.keys()
    # File -> New also loads the default model:
    gui.btn_new_clicked()
    assert gui.project.b2_model.parameters.keys() == project.b2_model.parameters.keys()
    assert (
        gui.project.b2_model.neurongroups.keys() == project.b2_model.neurongroups.keys()
    )
    assert gui.project.b2_model.synapses.keys() == project.b2_model.synapses.keys()
    assert gui.project.b2_model.monitors.keys() == project.b2_model.monitors.keys()
    gui.close()


def test_gui_open(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_path: Path, test_project: Project
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    project = test_project
    project_path = tmp_path / "model.nseq"
    project.save(
        project_path,
        SaveLoadOptions(
            include_stim_arrays=True,
            include_model=True,
            include_simulation_data=False,
            include_experimental_data=False,
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: (f"{project_path}", True),
    )
    gui.btn_open_clicked()

    assert gui.project.b2_model.parameters is not None
    assert list(gui.project.b2_model.parameters.keys()) == list(
        project.b2_model.parameters.keys()
    )
    assert list(gui.project.b2_model.neurongroups.keys()) == list(
        project.b2_model.neurongroups.keys()
    )
    assert list(gui.project.b2_model.synapses.keys()) == list(
        project.b2_model.synapses.keys()
    )
    assert list(gui.project.b2_model.monitors.keys()) == list(
        project.b2_model.monitors.keys()
    )
    gui.close()


def test_gui_toggle_grid(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    widgets = [
        gui.stim_editor.stim_widget,
        gui.trace_widget,
        gui.spike_widget,
        gui.data_widget.trace_widget,
        gui.data_widget.spike_widget,
    ]
    assert all(widget.show_grid for widget in widgets)
    gui.toggle_grid(False)
    assert all(not widget.show_grid for widget in widgets)
    gui.toggle_grid(True)
    assert all(widget.show_grid for widget in widgets)
    gui.close()


class MockSaveLoadOptionsDialog:
    options = SaveLoadOptions(
        include_stim_arrays=True,
        include_model=True,
        include_simulation_data=True,
        include_experimental_data=True,
    )

    def __init__(self, *args, **kwargs) -> None:
        pass

    def exec(self) -> bool:
        return True


def test_gui_save(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.update_project(project_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    filename_without_extension = tmp_path / "test"
    # the gui will add the .nseq extension automatically
    filename = filename_without_extension.with_suffix(".nseq")
    assert not filename.exists()
    MockSaveLoadOptionsDialog.options.include_stim_arrays = True
    MockSaveLoadOptionsDialog.options.include_model = True
    MockSaveLoadOptionsDialog.options.include_simulation_data = True
    MockSaveLoadOptionsDialog.options.include_experimental_data = True
    monkeypatch.setattr(
        neuroseq.gui,
        "SaveLoadOptionsDialog",
        MockSaveLoadOptionsDialog,
    )
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    assert filename.exists()
    assert gui.project.filepath == filename
    assert str(filename) in gui.windowTitle()
    loaded_model = Project(filename)
    assert loaded_model.filepath == gui.project.filepath
    assert (
        project_ornstein.b2_model.neurongroups.keys()
        == loaded_model.b2_model.neurongroups.keys()
    )
    assert (
        project_ornstein.b2_model.synapses.keys()
        == loaded_model.b2_model.synapses.keys()
    )
    assert (
        project_ornstein.b2_model.monitors.keys()
        == loaded_model.b2_model.monitors.keys()
    )
    gui.close()


def test_gui_import(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    project_ornstein: Project,
    test_project: Project,
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    # create a nseq file to import from
    project = test_project
    model_path = tmp_path / "model.nseq"
    project.save(
        model_path,
        SaveLoadOptions(
            include_stim_arrays=True,
            include_model=True,
            include_simulation_data=False,
            include_experimental_data=False,
        ),
    )
    gui.update_project(project_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    # mock getOpenFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: (f"{model_path}", True),
    )
    MockSaveLoadOptionsDialog.options.include_stim_arrays = True
    MockSaveLoadOptionsDialog.options.include_model = False
    MockSaveLoadOptionsDialog.options.include_simulation_data = False
    MockSaveLoadOptionsDialog.options.include_experimental_data = False
    monkeypatch.setattr(
        neuroseq.gui,
        "SaveLoadOptionsDialog",
        MockSaveLoadOptionsDialog,
    )
    gui.btn_import_clicked()
    # b2model is not modified
    assert (
        project_ornstein.b2_model.neurongroups.keys()
        == gui.project.b2_model.neurongroups.keys()
    )
    assert (
        project_ornstein.b2_model.synapses.keys()
        == gui.project.b2_model.synapses.keys()
    )
    assert (
        project_ornstein.b2_model.monitors.keys()
        == gui.project.b2_model.monitors.keys()
    )
    # but stim arrays are imported
    assert gui.project.stim_arrays.keys() == project.stim_arrays.keys()
    gui.close()


def test_gui_export_with_simulation_data(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
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
    gui.close()


def test_gui_export_empty_project(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.update_project(Project())
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
    gui.close()


def test_gui_export_igor2_patchclamp_stim(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    project = Project()
    stim_values = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [0.0, 30.0],
        ],
        dtype=np.float32,
    )
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, stim_values, "pA", 0.001)
    )

    class MockIgor2PatchClampExportDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            assert stim_name == "stim"
            assert stim_array is project.stim_arrays["stim"]
            self.experiment_name = "one_neuron_stim_example"
            self.selected_neuron_index = 1

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "igor_export.csv"
    monkeypatch.setattr(
        neuroseq.gui,
        "Igor2PatchClampExportDialog",
        MockIgor2PatchClampExportDialog,
    )
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (f"{output_path}", "CSV file (*.csv)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_igor2_patchclamp_stim"),
    )

    Gui.btn_export_igor2_patchclamp_stim_clicked(fake_gui)

    assert output_path.exists()
    assert (
        output_path.read_text(encoding="utf-8")
        == "one_neuron_stim_example\n10.0\n20.0\n30.0\n"
    )
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "export successful" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_igor2_patchclamp_stim_requires_stim_data(
    qmessagebox_calls: list,
) -> None:
    fake_gui: Any = SimpleNamespace(
        project=Project(),
        cmb_stim=SimpleNamespace(currentText=lambda: ""),
        logger=logging.getLogger(
            "test_gui_export_igor2_patchclamp_stim_requires_stim_data"
        ),
    )

    Gui.btn_export_igor2_patchclamp_stim_clicked(fake_gui)

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "no stimulation entities" in qmessagebox_calls[0]["title"].lower()


def test_gui_import_igor2_wave_updates_selected_stim(
    monkeypatch: MonkeyPatch,
) -> None:
    project = Project()

    class MockIgorImportDialog:
        def __init__(self, dialog_project: Project, parent=None) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_stim_name = "igor_wave"

        def exec(self) -> int:
            self.project.add_stim_array(
                StimArray(
                    "igor_wave",
                    StimType.External,
                    np.array([[1.0], [2.0]], dtype=np.float32),
                    "pA",
                    0.001,
                )
            )
            return QtWidgets.QDialog.DialogCode.Accepted

    stim_combo = QtWidgets.QComboBox()

    def _update_project(updated_project: Project) -> None:
        stim_combo.clear()
        stim_combo.addItems(list(updated_project.stim_arrays.keys()))

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=stim_combo,
        update_project=_update_project,
    )

    monkeypatch.setattr(neuroseq.gui, "IgorImportDialog", MockIgorImportDialog)
    Gui.btn_import_igor2_wave_clicked(fake_gui)

    assert fake_gui.cmb_stim.currentText() == "igor_wave"
    assert "igor_wave" in project.stim_arrays


def test_gui_checkpoints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    project_ornstein: Project,
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
    gui.update_project(project_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.btn_store_checkpoint_clicked()
    assert "test" in gui.project.b2_model.checkpoints
    # simulate
    simulation_time = 0.1
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time}", True)
    )
    gui.btn_continue_simulation_clicked()
    assert "run" not in gui.continue_simulation_action.text().lower()
    assert "continue" in gui.continue_simulation_action.text().lower()
    assert gui.project.b2_model.network["M"].t[-1] == approx(simulation_time, rel=1e-2)
    # restore checkpoint
    gui.btn_restore_checkpoint_clicked()
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    assert len(gui.project.b2_model.network["M"].t) == 0
    gui.close()


def test_gui_edit_stim_with_console(qtbot: QtBot, project_ornstein: Project) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 10 * b2.ms
    stim = np.zeros((n_timestamps, n_neurons))
    assert project_ornstein.stim_arrays == {}
    project_ornstein.add_stim_array(
        StimArray("stim", StimType.Excitation, stim, "mV", dt)
    )
    gui.update_project(project_ornstein)
    stim_editor = gui.stim_editor
    stim_console = gui.console_widget
    assert stim_editor.stims_dict is not None
    # set all stimuli to 1: StimWidget and model stimuli updated
    stim_console.input.setText("project.stim_arrays['stim'].values[:,:] = 1")
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    stim[:, :] = 1
    assert_allclose(gui.project.stim_arrays["stim"].values, stim)
    assert stim_editor.stim_widget.n_neurons == n_neurons
    # assign random valid stim array with different shape, StimWidget and model updated
    stim_console.input.setText(
        "project.stim_arrays['stim'].values = np.random.rand(127, 83)"
    )
    qtbot.keyPress(stim_console.input, QtCore.Qt.Key.Key_Enter)
    assert gui.project.stim_arrays["stim"].values.shape == (127, 83)
    assert stim_editor.stim_widget.n_neurons == 83
    gui.close()


def test_gui_synapse_ram_limit(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    tmp_config_dir,
    test_project: Project,
) -> None:
    gui = Gui()
    gui.load_project(test_project)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    assert gui.heatmap_widget.idx == 0
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
    gui.close()


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
    gui.close()


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
    assert gui.heatmap_widget.idx == 0
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
    gui.close()


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
    gui2.close()


def test_gui_playback_speed(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    test_project: Project,
) -> None:
    gui = Gui()
    gui.load_project(test_project)
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
    gui.close()


class MockImportExperimentDialog:
    append_to_experiment = False

    def __init__(self, *args, **kwargs) -> None:
        pass

    def exec(self) -> bool:
        return True


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
    monkeypatch.setattr(
        neuroseq.gui,
        "ImportExperimentDialog",
        MockImportExperimentDialog,
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
    gui.close()


def test_integration_modeleditor_delete_restore(
    qtbot: QtBot, qmessagebox_calls: list, test_project: Project
) -> None:
    gui = Gui()
    gui.load_project(test_project)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(gui.isVisible)
    qtbot.wait(delay_after_update_model_call_in_ms)

    gui.btn_edit_clicked()

    modeleditor = gui.findChild(ModelEditorDialog)

    qtbot.addWidget(modeleditor)
    qtbot.waitUntil(modeleditor.isVisible)

    assert gui.project.b2_model is not modeleditor.b2model

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
    assert "Ce" in gui.project.b2_model.synapses

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
    assert "Ce" in gui.project.b2_model.synapses

    # run the model and make sure it works as it should
    gui.project.run_simulation(0.1)
    assert gui.project.b2_model.network["Ce"] is not None

    assert gui.project.b2_model.network["trace"].v.shape == (30, 1000)
    gui.close()
