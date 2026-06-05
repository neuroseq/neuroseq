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
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui import Gui
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_editor import DrawMode
from neuroseq.stim_editor import StimEditor
from neuroseq.utils import InteractMode

# no obvious signal to wait for after the model is updated to know when the GUI is ready,
# so for now we just wait for a fixed time
delay_after_update_model_call_in_ms = 50


def _wait_for_brian2_simulation(qtbot: QtBot, gui: Gui, timeout: int = 90000) -> None:
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=timeout)


class _MockAxes:
    def __init__(self) -> None:
        self.plot_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.scatter_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def plot(self, *args: Any, **kwargs: Any) -> None:
        self.plot_calls.append((args, kwargs))

    def scatter(self, *args: Any, **kwargs: Any) -> None:
        self.scatter_calls.append((args, kwargs))

    def set_xlabel(self, *args: Any, **kwargs: Any) -> None:
        return

    def set_ylabel(self, *args: Any, **kwargs: Any) -> None:
        return

    def set_title(self, *args: Any, **kwargs: Any) -> None:
        return

    def tick_params(self, *args: Any, **kwargs: Any) -> None:
        return


class _MockFigure:
    def __init__(self, save_error: Exception | None = None) -> None:
        self.save_error = save_error
        self.saved: list[tuple[str, dict[str, Any]]] = []
        self.tight_layout_called = False

    def tight_layout(self) -> None:
        self.tight_layout_called = True

    def savefig(self, path: str, **kwargs: Any) -> None:
        if self.save_error is not None:
            raise self.save_error
        self.saved.append((path, kwargs))


def _make_trace_export_gui(
    trace_name: str,
    trace_data: TraceData | None,
) -> Any:
    return SimpleNamespace(
        cmb_trace=SimpleNamespace(currentText=lambda: trace_name),
        project=SimpleNamespace(
            get_data=lambda name, data_type: (
                trace_data if data_type is TraceData else None
            )
        ),
        logger=logging.getLogger("test_trace_export"),
    )


def _make_spike_export_gui(
    spike_name: str,
    spike_data: SpikeData | None,
) -> Any:
    return SimpleNamespace(
        cmb_spike=SimpleNamespace(currentText=lambda: spike_name),
        project=SimpleNamespace(
            get_data=lambda name, data_type: (
                spike_data if data_type is SpikeData else None
            )
        ),
        logger=logging.getLogger("test_spike_export"),
    )


@pytest.mark.parametrize(
    ("method_name", "software", "inner_name"),
    [
        (
            "btn_import_stimulus_light_clicked",
            "SysCon2",
            "btn_import_syscon2_seq_clicked",
        ),
        (
            "btn_import_stimulus_patchclamp_clicked",
            "Igor2",
            "btn_import_igor2_wave_clicked",
        ),
        (
            "btn_import_trace_patchclamp_clicked",
            "Igor2",
            "btn_import_igor2_trace_clicked",
        ),
        (
            "btn_import_trace_patchclamp_clicked",
            "NumPy (.npy)",
            "btn_import_patchclamp_data_clicked",
        ),
        (
            "btn_export_patch_clamp_stim_clicked",
            "Igor2 - single array (.ibw / .csv)",
            "btn_export_igor2_single_wave_clicked",
        ),
        (
            "btn_export_patch_clamp_stim_clicked",
            "Igor2 - all arrays (.pxp)",
            "btn_export_igor2_pxp_stim_clicked",
        ),
        (
            "btn_export_light_stimulus_clicked",
            "SysCon2",
            "open_stimulation_generator_dialog",
        ),
    ],
)
def test_gui_pre_dialog_wrapper_calls_inner_handler_on_accept(
    monkeypatch: MonkeyPatch,
    method_name: str,
    software: str,
    inner_name: str,
) -> None:
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: (software, True),
    )
    called: list[bool] = []
    fake_gui: Any = SimpleNamespace(**{inner_name: lambda: called.append(True)})

    getattr(Gui, method_name)(fake_gui)

    assert called == [True]


@pytest.mark.parametrize(
    ("method_name", "software", "inner_name"),
    [
        (
            "btn_import_stimulus_light_clicked",
            "SysCon2",
            "btn_import_syscon2_seq_clicked",
        ),
        (
            "btn_import_stimulus_patchclamp_clicked",
            "Igor2",
            "btn_import_igor2_wave_clicked",
        ),
        (
            "btn_import_trace_patchclamp_clicked",
            "Igor2",
            "btn_import_igor2_trace_clicked",
        ),
        (
            "btn_import_trace_patchclamp_clicked",
            "NumPy (.npy)",
            "btn_import_patchclamp_data_clicked",
        ),
        (
            "btn_export_patch_clamp_stim_clicked",
            "Igor2 - single array (.ibw / .csv)",
            "btn_export_igor2_single_wave_clicked",
        ),
        (
            "btn_export_patch_clamp_stim_clicked",
            "Igor2 - all arrays (.pxp)",
            "btn_export_igor2_pxp_stim_clicked",
        ),
        (
            "btn_export_light_stimulus_clicked",
            "SysCon2",
            "open_stimulation_generator_dialog",
        ),
    ],
)
def test_gui_pre_dialog_wrapper_does_nothing_on_cancel(
    monkeypatch: MonkeyPatch,
    method_name: str,
    software: str,
    inner_name: str,
) -> None:
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: (software, False),
    )
    called: list[bool] = []
    fake_gui: Any = SimpleNamespace(**{inner_name: lambda: called.append(True)})

    getattr(Gui, method_name)(fake_gui)

    assert called == []


def test_patchclamp_pre_dialogs_include_cc_vc_convention(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_labels: list[str] = []

    def fake_get_item(*args: Any, **kwargs: Any) -> tuple[str, bool]:
        del kwargs
        captured_labels.append(args[2])
        return ("Igor2", False)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", fake_get_item)

    fake_gui: Any = SimpleNamespace(
        btn_import_igor2_wave_clicked=lambda: None,
        btn_import_igor2_trace_clicked=lambda: None,
        btn_import_patchclamp_data_clicked=lambda: None,
    )

    Gui.btn_import_stimulus_patchclamp_clicked(fake_gui)
    Gui.btn_import_trace_patchclamp_clicked(fake_gui)

    assert len(captured_labels) == 2
    stim_label, trace_label = captured_labels
    assert "Current Clamp (CC)" in stim_label
    assert "Voltage Clamp (VC)" in stim_label
    assert "CURRENT is the stimulus" in stim_label
    assert "VOLTAGE is the stimulus" in stim_label
    assert "Current Clamp (CC)" in trace_label
    assert "Voltage Clamp (VC)" in trace_label
    assert "VOLTAGE is the trace" in trace_label
    assert "CURRENT is the trace" in trace_label


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

    # locate the "Open example project" submenu in the File menu
    example_menu = next(
        (
            action.menu()
            for action in gui.menu_file.actions()
            if action.menu() is not None
            and "Open example project" in action.text().replace("&", "")
        ),
        None,
    )
    assert example_menu is not None

    # its child submenus are the sections, each containing one action per project
    section_menus = {
        action.text().replace("&", ""): action.menu()
        for action in example_menu.actions()
        if action.menu() is not None
    }
    assert list(section_menus) == list(example_projects)
    assert sum(len(menu.actions()) for menu in section_menus.values()) == sum(
        len(section_projects) for section_projects in example_projects.values()
    )
    assert {"Getting Started Project", "non_linear"}.issubset(
        example_projects["Basics"]
    )

    gui.close()


def test_gui_cellpose_import_action_moved_to_file_menu(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    file_action_texts = [action.text() for action in gui.menu_file.actions()]
    assert "Import cell spatial &footprints" in file_action_texts

    import_data_action = next(
        (
            action
            for action in gui.menu_file.actions()
            if action.menu() is not None and action.text() == "&Import data"
        ),
        None,
    )
    assert import_data_action is not None
    import_data_menu = import_data_action.menu()
    assert import_data_menu is not None
    import_data_action_texts = [action.text() for action in import_data_menu.actions()]
    assert "Import cell &segmentation" not in import_data_action_texts
    assert "Import cell spatial &footprints" not in import_data_action_texts

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
    simulation_time_s = 0.2
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time_s}", True)
    )
    qtbot.wait(delay_after_update_model_call_in_ms)
    simulation_time_ms = simulation_time_s * 1000
    gui.btn_continue_simulation_clicked()
    _wait_for_brian2_simulation(qtbot, gui)
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


def test_gui_simulate_and_cancel(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project
) -> None:
    gui = Gui()
    gui.load_project(test_project)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1.0", True))

    gui.btn_continue_simulation_clicked()
    assert gui._io_worker is not None
    # cancel simulation on first progress update
    gui._io_worker.progress.connect(lambda _: gui.progress_dialog.canceled.emit())

    _wait_for_brian2_simulation(qtbot, gui)

    assert gui.project.b2_model.cancel_requested is True
    assert gui._io_worker is None
    assert not gui.progress_dialog.isVisible()
    assert QtWidgets.QApplication.overrideCursor() is None
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


def test_btn_new_clicked_calls_gc_collect(monkeypatch: MonkeyPatch) -> None:
    collect_calls = [0]
    sentinel_project = Project()
    loaded_project: list[Project] = []

    monkeypatch.setattr(
        "neuroseq.gui.gc.collect",
        lambda: collect_calls.__setitem__(0, collect_calls[0] + 1),
    )
    monkeypatch.setattr("neuroseq.gui.default_project", lambda: sentinel_project)

    fake_gui: Any = SimpleNamespace(
        load_project=lambda project: loaded_project.append(project)
    )
    Gui.btn_new_clicked(fake_gui)

    assert collect_calls[0] == 1
    assert loaded_project == [sentinel_project]


def test_load_example_project_calls_gc_collect(monkeypatch: MonkeyPatch) -> None:
    collect_calls = [0]
    generator_calls = [0]
    sentinel_project = Project()
    loaded_project: list[Project] = []

    monkeypatch.setattr(
        "neuroseq.gui.gc.collect",
        lambda: collect_calls.__setitem__(0, collect_calls[0] + 1),
    )

    def generator() -> Project:
        generator_calls[0] += 1
        return sentinel_project

    fake_gui: Any = SimpleNamespace(
        load_project=lambda project: loaded_project.append(project)
    )
    Gui._load_example_project(fake_gui, generator)

    assert collect_calls[0] == 1
    assert generator_calls[0] == 1
    assert loaded_project == [sentinel_project]


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
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)

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
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)
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
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)
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
    _wait_for_brian2_simulation(qtbot, gui)
    filename = tmp_path / "test.h5"
    assert not filename.exists()
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (f"{filename}", True)
    )
    gui.btn_export_simulation_results_clicked()
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
    gui.btn_export_simulation_results_clicked()
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "export error" in qmessagebox_calls[0]["title"].lower()
    assert not filename.exists()
    gui.close()


def test_gui_export_igor2_single_wave_csv(
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

    class MockIgor2SingleWaveExportDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            assert stim_name == "stim"
            assert stim_array is project.stim_arrays["stim"]
            self.experiment_name = "one_neuron_stim_example"
            self.selected_neuron_index = 1
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0
            self.as_sweeps: bool = False

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "igor_export.csv"
    monkeypatch.setattr(
        neuroseq.gui,
        "Igor2SingleWaveExportDialog",
        MockIgor2SingleWaveExportDialog,
    )
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (f"{output_path}", "CSV file (*.csv)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_igor2_single_wave_csv"),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    assert output_path.exists()
    assert (
        output_path.read_text(encoding="utf-8")
        == "one_neuron_stim_example\n10.0\n20.0\n30.0\n"
    )
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "export successful" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_igor2_single_wave_ibw(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    import importlib

    binarywave = importlib.import_module("igor2.binarywave")
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

    class MockDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            assert stim_name == "stim"
            assert stim_array is project.stim_arrays["stim"]
            self.experiment_name = "stim_neuron_1"
            self.selected_neuron_index = 1
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0
            self.as_sweeps: bool = False

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "stim_neuron_1.ibw"
    monkeypatch.setattr(neuroseq.gui, "Igor2SingleWaveExportDialog", MockDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw"),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    assert output_path.exists()
    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        decimal=5,
    )
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "export successful" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_igor2_single_wave_ibw_with_as_sweeps_succeeds(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    import importlib

    binarywave = importlib.import_module("igor2.binarywave")
    project = Project()
    stim_values = np.tile(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        (4, 1),
    )
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, stim_values, "pA", 0.001)
    )

    class MockDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            self.experiment_name = "stim_sweeps"
            self.selected_neuron_index = 0
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0
            self.as_sweeps = True

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "stim_sweeps.ibw"
    monkeypatch.setattr(neuroseq.gui, "Igor2SingleWaveExportDialog", MockDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw_sweeps"),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.shape == (3, 4)
    assert "Split into 4 sweeps × 3 samples." in qmessagebox_calls[0]["text"]


def test_gui_export_igor2_single_wave_ibw_as_sweeps_falls_back_for_non_periodic(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    import importlib

    binarywave = importlib.import_module("igor2.binarywave")
    project = Project()
    stim_values = np.arange(7, dtype=np.float32).reshape(7, 1)
    project.add_stim_array(
        StimArray("stim", StimType.Excitation, stim_values, "pA", 0.001)
    )

    class MockDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            self.experiment_name = "stim_aperiodic"
            self.selected_neuron_index = 0
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0
            self.as_sweeps = True

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "stim_aperiodic.ibw"
    monkeypatch.setattr(neuroseq.gui, "Igor2SingleWaveExportDialog", MockDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    info_messages: list[str] = []
    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=SimpleNamespace(
            info=lambda message, *args: info_messages.append(message % args)
        ),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    assert wave_data.ndim == 1
    assert any("no period was detected" in message for message in info_messages)


def test_gui_export_igor2_single_wave_requires_stim_data(
    qmessagebox_calls: list,
) -> None:
    fake_gui: Any = SimpleNamespace(
        project=Project(),
        cmb_stim=SimpleNamespace(currentText=lambda: ""),
        logger=logging.getLogger(
            "test_gui_export_igor2_single_wave_requires_stim_data"
        ),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "no stimulation entities" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_igor2_pxp_stim(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "exc",
            StimType.Excitation,
            np.ones((3, 2), dtype=np.float32),
            "pA",
            0.001,
        )
    )
    project.add_stim_array(
        StimArray(
            "inh",
            StimType.Inhibition,
            np.ones((3, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )

    class MockPxpDialog:
        def __init__(self, current_project: Project, parent=None) -> None:
            assert current_project is project
            self.selected_array_names = ["exc", "inh"]
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "stim_arrays.pxp"
    monkeypatch.setattr(neuroseq.gui, "Igor2PxpStimExportDialog", MockPxpDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Packed Experiment (*.pxp)"),
    )

    fake_gui: Any = SimpleNamespace(project=project)

    Gui.btn_export_igor2_pxp_stim_clicked(fake_gui)

    assert output_path.exists()
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "export complete" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_igor2_pxp_stim_requires_selection(
    monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.ones((3, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )

    class MockPxpDialog:
        def __init__(self, current_project: Project, parent=None) -> None:
            assert current_project is project
            self.selected_array_names: list[str] = []
            self.unit_override: str | None = None
            self.scaling_factor: float = 1.0

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(neuroseq.gui, "Igor2PxpStimExportDialog", MockPxpDialog)
    fake_gui: Any = SimpleNamespace(project=project)

    Gui.btn_export_igor2_pxp_stim_clicked(fake_gui)

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "nothing selected" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_light_stimulus_igor2_pxp_routes_to_pxp_export(
    monkeypatch: MonkeyPatch,
) -> None:
    """Selecting 'Igor2 - all arrays (.pxp)' must dispatch to the pxp handler."""
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Igor2 - all arrays (.pxp)", True),
    )

    called: dict[str, int] = {"pxp": 0, "ibw": 0, "syscon": 0}

    fake_gui: Any = SimpleNamespace(
        btn_export_igor2_pxp_stim_clicked=lambda: called.__setitem__(
            "pxp", called["pxp"] + 1
        ),
        btn_export_igor2_single_wave_clicked=lambda: called.__setitem__(
            "ibw", called["ibw"] + 1
        ),
        open_stimulation_generator_dialog=lambda: called.__setitem__(
            "syscon", called["syscon"] + 1
        ),
    )

    Gui.btn_export_light_stimulus_clicked(fake_gui)

    assert called == {"pxp": 1, "ibw": 0, "syscon": 0}


def test_gui_export_light_stimulus_syscon2_routes_to_generator(
    monkeypatch: MonkeyPatch,
) -> None:
    """Selecting SysCon2 still opens the stimulation generator dialog."""
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("SysCon2", True),
    )

    called: dict[str, int] = {"pxp": 0, "ibw": 0, "syscon": 0}

    fake_gui: Any = SimpleNamespace(
        btn_export_igor2_pxp_stim_clicked=lambda: called.__setitem__(
            "pxp", called["pxp"] + 1
        ),
        btn_export_igor2_single_wave_clicked=lambda: called.__setitem__(
            "ibw", called["ibw"] + 1
        ),
        open_stimulation_generator_dialog=lambda: called.__setitem__(
            "syscon", called["syscon"] + 1
        ),
    )

    Gui.btn_export_light_stimulus_clicked(fake_gui)

    assert called == {"pxp": 0, "ibw": 0, "syscon": 1}


def test_gui_export_light_stimulus_igor2_ibw_routes_to_single_wave_export(
    monkeypatch: MonkeyPatch,
) -> None:
    """Selecting 'Igor2 - single neuron' routes to the single-wave export flow."""
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("Igor2 - single neuron (.ibw / .csv)", True),
    )

    called: dict[str, int] = {"pxp": 0, "ibw": 0, "syscon": 0}

    fake_gui: Any = SimpleNamespace(
        btn_export_igor2_pxp_stim_clicked=lambda: called.__setitem__(
            "pxp", called["pxp"] + 1
        ),
        btn_export_igor2_single_wave_clicked=lambda: called.__setitem__(
            "ibw", called["ibw"] + 1
        ),
        open_stimulation_generator_dialog=lambda: called.__setitem__(
            "syscon", called["syscon"] + 1
        ),
    )

    Gui.btn_export_light_stimulus_clicked(fake_gui)

    assert called == {"pxp": 0, "ibw": 1, "syscon": 0}


def test_gui_export_igor2_single_wave_ibw_passes_unit_and_scaling(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    """Single-wave dialog's unit_override and scaling_factor reach the .ibw exporter."""
    import importlib

    binarywave = importlib.import_module("igor2.binarywave")
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "pA",
            0.001,
        )
    )

    class MockDialog:
        def __init__(self, stim_name: str, stim_array: StimArray, parent=None) -> None:
            self.experiment_name = "scaled_wave"
            self.selected_neuron_index = 0
            self.unit_override = "mW/mm^2"
            self.scaling_factor = 10.0
            self.as_sweeps = False

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "scaled_wave.ibw"
    monkeypatch.setattr(neuroseq.gui, "Igor2SingleWaveExportDialog", MockDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw_extras"),
    )

    Gui.btn_export_igor2_single_wave_clicked(fake_gui)

    loaded = binarywave.load(output_path)
    wave_data = loaded["wave"]["wData"].astype(np.float32)
    np.testing.assert_array_almost_equal(
        wave_data,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        decimal=5,
    )
    extended = bytes(loaded["wave"]["data_units"]).decode("ascii")
    assert extended == "mW/mm^2"


def test_gui_export_igor2_pxp_stim_passes_unit_and_scaling(
    tmp_path: Path, monkeypatch: MonkeyPatch, qmessagebox_calls: list
) -> None:
    """Dialog's unit_override and scaling_factor must flow into export_igor2_pxp."""
    project = Project()
    project.add_stim_array(
        StimArray(
            "exc",
            StimType.Excitation,
            np.ones((3, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )

    class MockPxpDialog:
        def __init__(self, current_project: Project, parent=None) -> None:
            assert current_project is project
            self.selected_array_names = ["exc"]
            self.unit_override = "mW/mm^2"
            self.scaling_factor = 3.0

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

    output_path = tmp_path / "scaled.pxp"
    monkeypatch.setattr(neuroseq.gui, "Igor2PxpStimExportDialog", MockPxpDialog)
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Packed Experiment (*.pxp)"),
    )

    captured: dict[str, Any] = {}

    def _fake_export(out, stims, **kwargs):
        captured["out"] = out
        captured["array_names"] = kwargs.get("array_names")
        captured["unit_override"] = kwargs.get("unit_override")
        captured["scaling_factor"] = kwargs.get("scaling_factor")
        out.write_bytes(b"")
        from neuroseq.exporter_igor2 import Igor2PxpExport

        return Igor2PxpExport(path=out, wave_names=["exc_n0"])

    monkeypatch.setattr(neuroseq.gui, "export_igor2_pxp", _fake_export)

    fake_gui: Any = SimpleNamespace(project=project)
    Gui.btn_export_igor2_pxp_stim_clicked(fake_gui)

    assert captured["array_names"] == ["exc"]
    assert captured["unit_override"] == "mW/mm^2"
    assert captured["scaling_factor"] == 3.0


def test_gui_export_traces_no_selection(qmessagebox_calls: list) -> None:
    fake_gui = _make_trace_export_gui("", None)

    Gui.btn_export_traces_clicked(fake_gui)

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "no trace data" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_traces_returns_when_file_dialog_cancelled(
    monkeypatch: MonkeyPatch,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    trace_data = TraceData(
        name="voltage",
        traces=np.zeros((3, 100), dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    fake_gui = _make_trace_export_gui("exp :: voltage", trace_data)

    def _unexpected_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("subplots should not be called when no file is selected")

    monkeypatch.setattr(neuroseq.gui, "getsavefilename", lambda *args: ("", ""))
    monkeypatch.setattr(plt, "subplots", _unexpected_subplots)

    Gui.btn_export_traces_clicked(fake_gui)

    assert qmessagebox_calls == []


def test_gui_export_traces_saves_figure(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    trace_data = TraceData(
        name="voltage",
        traces=np.zeros((3, 100), dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    output_path = tmp_path / "traces.png"
    fig = _MockFigure()
    ax = _MockAxes()
    closed_figs: list[_MockFigure] = []
    fake_gui = _make_trace_export_gui("exp :: voltage", trace_data)

    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plt, "close", lambda figure: closed_figs.append(figure))
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "PNG image (*.png)"),
    )

    Gui.btn_export_traces_clicked(fake_gui)

    assert fig.saved == [
        (str(output_path), {"dpi": 300, "bbox_inches": "tight"}),
    ]
    assert fig.tight_layout_called is True
    assert len(ax.plot_calls) == 3
    assert closed_figs == [fig]
    assert any(call["name"] == "information" for call in qmessagebox_calls)


def test_gui_export_traces_closes_figure_on_save_error(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    trace_data = TraceData(
        name="voltage",
        traces=np.zeros((2, 50), dtype=np.float32),
        indices=np.array([0, 1], dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    output_path = tmp_path / "traces.png"
    fig = _MockFigure(save_error=OSError("disk full"))
    ax = _MockAxes()
    closed_figs: list[_MockFigure] = []
    fake_gui = _make_trace_export_gui("exp :: voltage", trace_data)

    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plt, "close", lambda figure: closed_figs.append(figure))
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "PNG image (*.png)"),
    )

    Gui.btn_export_traces_clicked(fake_gui)

    assert closed_figs == [fig]
    assert any(call["name"] == "critical" for call in qmessagebox_calls)


def test_gui_export_spikes_no_selection(qmessagebox_calls: list) -> None:
    fake_gui = _make_spike_export_gui("", None)

    Gui.btn_export_spikes_clicked(fake_gui)

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "no spike data" in qmessagebox_calls[0]["title"].lower()


def test_gui_export_spikes_returns_when_file_dialog_cancelled(
    monkeypatch: MonkeyPatch,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1], dtype=np.int32),
        spike_times=np.array([0.1, 0.2], dtype=np.float32),
    )
    fake_gui = _make_spike_export_gui("exp :: spikes", spike_data)

    def _unexpected_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("subplots should not be called when no file is selected")

    monkeypatch.setattr(neuroseq.gui, "getsavefilename", lambda *args: ("", ""))
    monkeypatch.setattr(plt, "subplots", _unexpected_subplots)

    Gui.btn_export_spikes_clicked(fake_gui)

    assert qmessagebox_calls == []


def test_gui_export_spikes_saves_figure(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1, 0], dtype=np.int32),
        spike_times=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    output_path = tmp_path / "spikes.png"
    fig = _MockFigure()
    ax = _MockAxes()
    closed_figs: list[_MockFigure] = []
    fake_gui = _make_spike_export_gui("exp :: spikes", spike_data)

    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plt, "close", lambda figure: closed_figs.append(figure))
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "PNG image (*.png)"),
    )

    Gui.btn_export_spikes_clicked(fake_gui)

    assert fig.saved == [
        (str(output_path), {"dpi": 300, "bbox_inches": "tight"}),
    ]
    assert fig.tight_layout_called is True
    assert len(ax.scatter_calls) == 1
    assert closed_figs == [fig]
    assert any(call["name"] == "information" for call in qmessagebox_calls)


def test_gui_export_spikes_closes_figure_on_save_error(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    qmessagebox_calls: list,
) -> None:
    import matplotlib.pyplot as plt

    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1], dtype=np.int32),
        spike_times=np.array([0.1, 0.2], dtype=np.float32),
    )
    output_path = tmp_path / "spikes.png"
    fig = _MockFigure(save_error=OSError("disk full"))
    ax = _MockAxes()
    closed_figs: list[_MockFigure] = []
    fake_gui = _make_spike_export_gui("exp :: spikes", spike_data)

    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plt, "close", lambda figure: closed_figs.append(figure))
    monkeypatch.setattr(
        neuroseq.gui,
        "getsavefilename",
        lambda *args: (str(output_path), "PNG image (*.png)"),
    )

    Gui.btn_export_spikes_clicked(fake_gui)

    assert closed_figs == [fig]
    assert any(call["name"] == "critical" for call in qmessagebox_calls)


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

    class FakeComboBox:
        def __init__(self) -> None:
            self._items: list[str] = []
            self._current_text = ""

        def clear(self) -> None:
            self._items = []
            self._current_text = ""

        def addItems(self, items: list[str]) -> None:
            self._items.extend(items)

        def setCurrentText(self, text: str) -> None:
            self._current_text = text

        def currentText(self) -> str:
            return self._current_text

    stim_combo = FakeComboBox()

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


def test_gui_import_igor2_trace_updates_selected_trace(
    monkeypatch: MonkeyPatch,
) -> None:
    import neuroseq.igor_trace_import_dialog

    project = Project()
    captured_data: list[TraceData | None] = []
    trace_widget = SimpleNamespace(n_neurons=0)

    def _set_trace_data(data: TraceData | None) -> None:
        captured_data.append(data)
        trace_widget.n_neurons = 0 if data is None else int(np.max(data.indices)) + 1

    trace_widget.set_data = _set_trace_data

    class MockIgorTraceImportDialog:
        def __init__(self, dialog_project: Project, parent=None) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_experiment_name = "Experiment A"
            self.selected_trace_name = "Igor Trace"

        def exec(self) -> int:
            self.project.add_experiment(
                Experiment(
                    name="Experiment A",
                    duration=0.002,
                    timepoints=[],
                    data=[
                        TraceData(
                            name="Igor Trace",
                            traces=np.array([[1.0, 2.0]], dtype=np.float32),
                            indices=np.array([0], dtype=np.uint32),
                            units="mV",
                            dt=0.001,
                        )
                    ],
                )
            )
            return QtWidgets.QDialog.DialogCode.Accepted

    class FakeComboBox:
        def __init__(self) -> None:
            self._items: list[str] = []
            self._current_text = ""

        def clear(self) -> None:
            self._items = []
            self._current_text = ""

        def addItems(self, items: list[str]) -> None:
            self._items.extend(items)

        def setCurrentText(self, text: str) -> None:
            self._current_text = text

        def currentText(self) -> str:
            return self._current_text

    trace_combo = FakeComboBox()

    def _update_project(updated_project: Project) -> None:
        trace_combo.clear()
        trace_combo.addItems(updated_project.get_data_names(TraceData))

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_trace=trace_combo,
        trace_widget=trace_widget,
        update_project=_update_project,
        _focus_neuron_range=lambda n_neurons: None,
    )
    fake_gui._apply_trace_selection = lambda cmb_name: Gui._apply_trace_selection(  # type: ignore[attr-defined]
        fake_gui, cmb_name
    )

    monkeypatch.setattr(
        neuroseq.igor_trace_import_dialog,
        "IgorTraceImportDialog",
        MockIgorTraceImportDialog,
    )
    Gui.btn_import_igor2_trace_clicked(fake_gui)

    assert fake_gui.cmb_trace.currentText() == "Experiment A :: Igor Trace"
    assert "Experiment A" in project.experiments
    assert project.stim_arrays == {}
    assert captured_data[-1] is not None
    assert captured_data[-1].name == "Igor Trace"


def test_gui_import_igor2_trace_refreshes_existing_selected_trace_plot(
    monkeypatch: MonkeyPatch,
) -> None:
    import neuroseq.igor_trace_import_dialog

    project = Project()
    project.add_experiment(
        Experiment(
            name="Experiment A",
            duration=0.002,
            timepoints=[],
            data=[
                TraceData(
                    name="Igor Trace",
                    traces=np.array([[1.0, 2.0]], dtype=np.float32),
                    indices=np.array([0], dtype=np.uint32),
                    units="mV",
                    dt=0.001,
                )
            ],
        )
    )

    class MockIgorTraceImportDialog:
        def __init__(self, dialog_project: Project, parent=None) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_experiment_name = "Experiment A"
            self.selected_trace_name = "Igor Trace"

        def exec(self) -> int:
            trace_data = self.project.get_data("Experiment A :: Igor Trace", TraceData)
            assert trace_data is not None
            trace_data.traces = np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
            trace_data.indices = np.array([0, 5], dtype=np.uint32)
            return QtWidgets.QDialog.DialogCode.Accepted

    class FakeComboBox:
        def __init__(self, initial_text: str) -> None:
            self._items: list[str] = []
            self._current_text = initial_text

        def clear(self) -> None:
            self._items = []

        def addItems(self, items: list[str]) -> None:
            self._items.extend(items)

        def setCurrentText(self, text: str) -> None:
            self._current_text = text

        def currentText(self) -> str:
            return self._current_text

    captured_data: list[TraceData | None] = []
    focus_calls: list[int] = []
    trace_combo = FakeComboBox("Experiment A :: Igor Trace")
    trace_widget = SimpleNamespace(n_neurons=0)

    def _set_trace_data_2(data: TraceData | None) -> None:
        captured_data.append(data)
        trace_widget.n_neurons = 0 if data is None else int(np.max(data.indices)) + 1

    trace_widget.set_data = _set_trace_data_2

    def _update_project(updated_project: Project) -> None:
        trace_combo.clear()
        trace_combo.addItems(updated_project.get_data_names(TraceData))

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_trace=trace_combo,
        trace_widget=trace_widget,
        update_project=_update_project,
        _focus_neuron_range=lambda n_neurons: focus_calls.append(n_neurons),
    )
    fake_gui._apply_trace_selection = lambda cmb_name: Gui._apply_trace_selection(  # type: ignore[attr-defined]
        fake_gui, cmb_name
    )

    monkeypatch.setattr(
        neuroseq.igor_trace_import_dialog,
        "IgorTraceImportDialog",
        MockIgorTraceImportDialog,
    )

    Gui.btn_import_igor2_trace_clicked(fake_gui)

    assert fake_gui.cmb_trace.currentText() == "Experiment A :: Igor Trace"
    assert captured_data[-1] is not None
    assert captured_data[-1].traces.shape == (2, 2)
    np.testing.assert_array_equal(
        captured_data[-1].indices,
        np.array([0, 5], dtype=np.uint32),
    )
    assert focus_calls[-1] == 6


def test_gui_imported_igor_stim_can_be_edited_by_click(
    monkeypatch: MonkeyPatch,
) -> None:
    class MockIgorImportDialog:
        def __init__(self, dialog_project: Project, parent=None) -> None:
            self.project = dialog_project
            self.selected_stim_name = "igor_wave"

        def exec(self) -> int:
            self.project.add_stim_array(
                StimArray(
                    "igor_wave",
                    StimType.External,
                    np.zeros((4, 1), dtype=np.float32),
                    "pA",
                    0.001,
                )
            )
            return QtWidgets.QDialog.DialogCode.Accepted

    project = Project()
    draw_value = 3.0

    fake_editor: Any = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        current_stim_modifier=SimpleNamespace(
            stim_array=StimArray(
                "igor_wave",
                StimType.External,
                np.zeros((4, 1), dtype=np.float32),
                "pA",
                0.001,
            )
        ),
        mode=InteractMode.DRAW,
        draw_mode=DrawMode.DRAW,
        draw_stim_length=1,
        draw_stim=SimpleNamespace(draw_stim_type=neuroseq.stim.DrawStimType.Constant),
        _is_valid_slice=lambda neuron_slice, time_slice: True,
        draw_stim_region=lambda neuron_slice, time_slice, erase=False: (
            project.stim_arrays["igor_wave"].values.__setitem__(
                (time_slice, neuron_slice), 0.0 if erase else draw_value
            )
        ),
        _draw_drag_origin_time_index=None,
    )

    class FakeComboBox:
        def __init__(self) -> None:
            self._items: list[str] = []
            self._current_text = ""

        def clear(self) -> None:
            self._items = []
            self._current_text = ""

        def addItems(self, items: list[str]) -> None:
            self._items.extend(items)

        def setCurrentText(self, text: str) -> None:
            self._current_text = text
            if text in project.stim_arrays:
                fake_editor.current_stim_modifier = SimpleNamespace(
                    stim_array=project.stim_arrays[text]
                )

        def currentText(self) -> str:
            return self._current_text

    stim_combo = FakeComboBox()

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

    assert fake_editor.current_stim_modifier is not None
    assert fake_editor.current_stim_modifier.stim_array.name == "igor_wave"
    assert fake_gui.cmb_stim.currentText() == "igor_wave"

    StimEditor.stim_widget_mouse_clicked(fake_editor, 0, 0, 1)  # type: ignore[arg-type]

    assert project.stim_arrays["igor_wave"].values[0, 0] == approx(draw_value)


def test_gui_set_neuron_refocuses_to_selected_trace_range() -> None:
    trace_data = TraceData(
        name="cellpose traces",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.1,
    )
    focus_calls: list[int] = []
    captured_data: list[TraceData | None] = []
    trace_widget = SimpleNamespace(n_neurons=0)

    def _set_trace_data(data: TraceData | None) -> None:
        captured_data.append(data)
        trace_widget.n_neurons = 0 if data is None else int(np.max(data.indices)) + 1

    trace_widget.set_data = _set_trace_data
    fake_gui: Any = SimpleNamespace(
        project=SimpleNamespace(
            get_data=lambda name, data_type: (
                trace_data if data_type is TraceData else None
            )
        ),
        trace_widget=trace_widget,
        _focus_neuron_range=lambda n_neurons: focus_calls.append(n_neurons),
    )

    Gui._apply_trace_selection(fake_gui, "Cellpose :: traces")

    assert captured_data == [trace_data]
    assert focus_calls == [1]


def test_gui_set_spike_refocuses_to_selected_spike_range() -> None:
    spike_data = SpikeData(
        name="cellpose spikes",
        spike_indices=np.array([0, 0, 0], dtype=np.int32),
        spike_times=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    focus_calls: list[int] = []
    captured_data: list[SpikeData | None] = []
    spike_widget = SimpleNamespace(n_neurons=0)

    def _set_spike_data(data: SpikeData | None) -> None:
        captured_data.append(data)
        spike_widget.n_neurons = (
            0 if data is None else int(np.max(data.spike_indices)) + 1
        )

    spike_widget.set_data = _set_spike_data
    fake_gui: Any = SimpleNamespace(
        project=SimpleNamespace(
            get_data=lambda name, data_type: (
                spike_data if data_type is SpikeData else None
            )
        ),
        spike_widget=spike_widget,
        _focus_neuron_range=lambda n_neurons: focus_calls.append(n_neurons),
    )

    Gui._apply_spike_selection(fake_gui, "Cellpose :: spikes")

    assert captured_data == [spike_data]
    assert focus_calls == [1]


def test_gui_set_data_refocuses_combined_trace_view() -> None:
    trace_data = TraceData(
        name="cellpose traces",
        traces=np.array([[1.0, 2.0]], dtype=np.float32),
        indices=np.array([0], dtype=np.uint32),
        units="mV",
        dt=0.1,
    )
    focus_calls: list[int] = []
    captured_data: list[TraceData | SpikeData | None] = []
    trace_widget = SimpleNamespace(n_neurons=0)
    spike_widget = SimpleNamespace(n_neurons=0)
    data_widget = SimpleNamespace(
        isVisible=lambda: True,
        trace_widget=trace_widget,
        spike_widget=spike_widget,
    )

    def _set_combined_data(data: TraceData | SpikeData | None) -> None:
        captured_data.append(data)
        trace_widget.n_neurons = 0
        spike_widget.n_neurons = 0
        if isinstance(data, TraceData):
            trace_widget.n_neurons = int(np.max(data.indices)) + 1
        elif isinstance(data, SpikeData):
            spike_widget.n_neurons = int(np.max(data.spike_indices)) + 1

    data_widget.set_data = _set_combined_data
    fake_gui: Any = SimpleNamespace(
        project=SimpleNamespace(
            get_data=lambda name, data_type: (
                trace_data if data_type is TraceData else None
            )
        ),
        data_widget=data_widget,
        _focus_neuron_range=lambda n_neurons: focus_calls.append(n_neurons),
    )

    Gui._apply_data_selection(fake_gui, "Cellpose :: traces [trace]")

    assert captured_data == [trace_data]
    assert focus_calls == [1]


def test_gui_apply_cellpose_import_result_expands_stim_arrays(
    monkeypatch: MonkeyPatch,
) -> None:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((10, 40), dtype=np.float32),
            "nS",
            0.001,
        )
    )

    class _FakeGui:
        def __init__(self, project: Project) -> None:
            self.project = project
            self.last_cellpose_import: CellposeImportResult | None = None
            self.updated_projects: list[Project] = []
            self.update_limits_calls = 0

        def update_project(self, project: Project) -> None:
            self.updated_projects.append(project)

        def update_limits(self) -> None:
            self.update_limits_calls += 1

        def _cellpose_neuron_count(
            self,
            imported: CellposeImportResult | None,
        ) -> int:
            return Gui._cellpose_neuron_count(self, imported)  # type: ignore[arg-type]

        def _expand_stim_arrays_to_neuron_count(
            self,
            target_neuron_count: int,
        ) -> list[str]:
            return Gui._expand_stim_arrays_to_neuron_count(
                self,  # type: ignore[arg-type]
                target_neuron_count,
            )

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    fake_gui = _FakeGui(project)
    imported = CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.arange(100, dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
    )

    resized_names = Gui._apply_cellpose_import_result(fake_gui, imported)  # type: ignore[arg-type]

    assert resized_names == ["stim"]
    assert fake_gui.last_cellpose_import is not None
    assert project.stim_arrays["stim"].values.shape[1] == 100
    assert fake_gui.updated_projects == [project]
    assert fake_gui.update_limits_calls == 0


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
    _wait_for_brian2_simulation(qtbot, gui)
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
    _wait_for_brian2_simulation(qtbot, gui)
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
    _wait_for_brian2_simulation(qtbot, gui)
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
