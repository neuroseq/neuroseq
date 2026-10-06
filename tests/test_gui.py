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
from neuroseq.custom_analysis_widget import CustomAnalysisType
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.detached_panel_window import DetachedPanelWindow
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui import Gui
from neuroseq.gui import ToolbarProgressWidget
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
        self.lines: list[_MockLine] = []
        self.scatter_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def plot(self, *args: Any, **kwargs: Any) -> tuple[_MockLine]:
        self.plot_calls.append((args, kwargs))
        line = _MockLine()
        self.lines.append(line)
        return (line,)

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


class _MockLine:
    def __init__(self) -> None:
        self.gid: str | None = None

    def set_gid(self, gid: str) -> None:
        self.gid = gid


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
    assert "CURRENT is the input" in stim_label
    assert "VOLTAGE is the input" in stim_label
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


def _find_action(menu: QtWidgets.QMenu, text: str) -> QtWidgets.QAction:
    return next(a for a in menu.actions() if a.text().replace("&", "") == text)


def test_syscon2_panel_targets_drive_spatial_footprint_crosses(qtbot: QtBot) -> None:
    """Feature (#582): whichever sequence SysCon2Panel last sent to SysCon2
    should mark its targeted neurons with a cross in the Spatial Footprints
    panel - wired via the shared SpatialFootprintState, not a direct
    reference between the two widgets."""
    gui = Gui()
    qtbot.addWidget(gui)

    gui.syscon2_panel.targets_changed.emit(frozenset({2, 5}))

    assert gui._spatial_footprint_state.targeted_stim_ids == frozenset({2, 5})


def test_gui_open_fiji_and_cellpose_buttons(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    import neuroseq.external_tools as external_tools

    gui = Gui()
    qtbot.addWidget(gui)

    fiji_action = _find_action(gui.menu_tools, "Open Fiji")
    cellpose_action = _find_action(gui.menu_tools, "Open Cellpose")

    fiji_calls: list[object] = []
    cellpose_calls: list[object] = []
    monkeypatch.setattr(
        external_tools, "launch_fiji", lambda parent: fiji_calls.append(parent)
    )
    monkeypatch.setattr(
        external_tools, "launch_cellpose", lambda parent: cellpose_calls.append(parent)
    )

    fiji_action.trigger()
    cellpose_action.trigger()
    assert fiji_calls == [gui]
    assert cellpose_calls == [gui]
    gui.close()


def test_tools_menu_holds_fiji_cellpose_and_change_fiji_path(
    qtbot: QtBot,
) -> None:
    """Open Fiji / Open Cellpose / Change Fiji Path moved out of the Data

    toolbar and Settings menu into their own Tools menu, between Settings
    and Help (issue #535).
    """
    gui = Gui()
    qtbot.addWidget(gui)

    menu_bar = gui.menuBar()
    assert menu_bar is not None
    top_level_menus = [a.menu() for a in menu_bar.actions()]
    settings_index = top_level_menus.index(gui.menu_settings)
    tools_index = top_level_menus.index(gui.menu_tools)
    help_index = top_level_menus.index(gui.menu_help)
    assert settings_index < tools_index < help_index

    texts = [a.text().replace("&", "") for a in gui.menu_tools.actions()]
    assert texts[0] == "Open Fiji"
    assert texts[1] == "Open Cellpose"
    assert gui.menu_tools.actions()[2].isSeparator()
    assert texts[3] == "Change Fiji Path…"

    settings_texts = [a.text().replace("&", "") for a in gui.menu_settings.actions()]
    assert "Change Fiji Path…" not in settings_texts
    gui.close()


def test_stim_array_ops_button_sits_next_to_stim_data_combobox_with_full_menu(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    actions = gui.toolbar_data.actions()
    stim_combo_index = next(
        i
        for i, action in enumerate(actions)
        if gui.toolbar_data.widgetForAction(action) is gui.cmb_stim
    )
    ops_button_index = next(
        i
        for i, action in enumerate(actions)
        if gui.toolbar_data.widgetForAction(action) is gui.btn_stim_array_ops
    )
    assert ops_button_index == stim_combo_index + 1

    menu_actions = gui.stim_array_ops_menu.actions()
    assert menu_actions[0] is gui.stim_editor.action_create_stim_array
    assert menu_actions[1] is gui.stim_editor.action_edit_stim_array
    assert menu_actions[2] is gui.stim_editor.action_delete_stim_array
    assert menu_actions[3] is gui.stim_editor.action_clear_stim_array
    assert menu_actions[4].isSeparator()
    assert menu_actions[5] is gui.stim_editor.action_trim_to_mapped_neurons
    assert menu_actions[6].isSeparator()
    assert menu_actions[7] is gui.stim_editor.action_pad_stim_array
    assert menu_actions[8] is gui.stim_editor.action_trim_stim_array
    assert menu_actions[9] is gui.stim_editor.action_loop_stim_array
    gui.close()


def test_view_toolbar_has_separator_before_toggle_selection_text(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    actions = gui.toolbar_view.actions()
    index = next(
        i
        for i, action in enumerate(actions)
        if action.text() == "Toggle Selection Text"
    )
    assert actions[index - 1].isSeparator()
    gui.close()


def test_show_all_stim_arrays_action_is_in_the_view_toolbar(qtbot: QtBot) -> None:
    """Moved out of the stim editor's left toolbar (issue #535); reused by

    reference rather than duplicated, so it stays the same QAction that
    toggle_show_all_stim_arrays() checks/toggles.
    """
    gui = Gui()
    qtbot.addWidget(gui)

    action = gui.stim_editor.action_show_all_stim_arrays
    assert action in gui.toolbar_view.actions()
    assert action in gui.menu_view.actions()
    assert action.isChecked()
    gui.close()


def test_design_static_and_dynamic_buttons_are_in_the_view_toolbar(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    """Moved out of the stim editor's left toolbar (issue #535); unlike their

    icon-only neighbors, these carry a text label beside their icon.
    """
    static_calls: list[int] = []
    dynamic_calls: list[int] = []
    # Patched on the class before the button's clicked signal connects to a
    # bound method of it, so the connection picks up the patched version.
    monkeypatch.setattr(
        StimEditor, "open_pattern_designer", lambda self: static_calls.append(1)
    )
    monkeypatch.setattr(
        StimEditor, "open_dynamic_input_designer", lambda self: dynamic_calls.append(1)
    )
    gui = Gui()
    qtbot.addWidget(gui)

    for button in (gui.btn_design_static_input, gui.btn_design_dynamic_input):
        assert button.parentWidget() is gui.toolbar_view
        assert not button.icon().isNull()
        assert button.text()

    assert gui.btn_design_static_input.text() == "Design static input"
    assert gui.btn_design_dynamic_input.text() == "Design dynamic input"

    gui.btn_design_static_input.click()
    gui.btn_design_dynamic_input.click()
    assert static_calls == [1]
    assert dynamic_calls == [1]
    gui.close()


def test_file_toolbar_groups_actions_with_separators(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    actions = gui.toolbar_file.actions()
    texts = [a.text().replace("&", "") for a in actions]

    def index_of(text: str) -> int:
        return texts.index(text)

    # {New, Open Project, Import from Project} | {Save Project} |
    # {Edit Model} | {Generate Synthetic Data, Manage Experiments}
    assert actions[index_of("Save Project") - 1].isSeparator()
    assert actions[index_of("Edit Model") - 1].isSeparator()
    assert actions[index_of("Generate Synthetic Data...") - 1].isSeparator()
    assert texts[index_of("Generate Synthetic Data...") + 1] == "Manage Experiments"
    gui.close()


def test_playback_toolbar_separates_speed_from_transport_controls(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    actions = gui.toolbar_playback.actions()
    index = next(
        i for i, a in enumerate(actions) if a.text().replace("&", "") == "Skip to Start"
    )
    assert actions[index - 1].isSeparator()
    gui.close()


def test_data_toolbar_groups_dataset_pairs_with_separators(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    toolbar = gui.toolbar_data
    actions = toolbar.actions()

    def label_indices(text: str) -> list[int]:
        return [
            i
            for i, a in enumerate(actions)
            if isinstance(toolbar.widgetForAction(a), QtWidgets.QLabel)
            and toolbar.widgetForAction(a).text().strip() == text
        ]

    spike_index = label_indices("Spike data:")[0]
    all_data_index = label_indices("All data:")[0]
    assert actions[spike_index - 1].isSeparator()
    assert actions[all_data_index - 1].isSeparator()
    gui.close()


def test_manage_experiments_action_sits_next_to_generate_synthetic_data(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    actions = gui.menu_file.actions()
    generate_index = next(
        i
        for i, action in enumerate(actions)
        if action.text().replace("&", "") == "Generate Synthetic Data..."
    )
    assert actions[generate_index + 1].text().replace("&", "") == "Manage Experiments"
    gui.close()


def _make_gui_cellpose_result(
    mapping: dict[int, int] | None = None,
) -> CellposeImportResult:
    segmentation = np.array(
        [
            [0, 0, 1],
            [0, 2, 2],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    )
    return CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((3, 3, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1, 2], dtype=np.int32),
        stimulus_to_cellpose_mapping=mapping,
    )


def test_trim_to_mapped_neurons_action_lives_in_the_stim_array_ops_menu(
    qtbot: QtBot,
) -> None:
    """The action moved out of the Spatial Footprints panel and into the

    stim array ops dropdown next to the stim data combobox (issue #535),
    disabled until there's an actual mapping to trim to."""
    gui = Gui()
    qtbot.addWidget(gui)
    action = gui.stim_editor.action_trim_to_mapped_neurons
    assert action in gui.stim_array_ops_menu.actions()
    assert not action.isEnabled()

    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui.update_project(project)
    assert not action.isEnabled()

    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0})
    gui.spatial_footprint_widget.set_data(project.cellpose_data)
    assert action.isEnabled()

    gui.spatial_footprint_widget.set_data(None)
    assert not action.isEnabled()
    gui.close()


def test_trim_to_mapped_neurons_action_triggers_the_trim_handler(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0})
    gui.update_project(project)
    assert gui.stim_editor.action_trim_to_mapped_neurons.isEnabled()
    calls: list[int] = []
    gui.stim_editor.trim_to_mapped_neurons_requested.connect(lambda: calls.append(1))

    gui.stim_editor.action_trim_to_mapped_neurons.trigger()

    assert calls == [1]
    gui.close()


def test_trim_to_mapped_neurons_reports_nothing_to_trim_without_mapping(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui.update_project(project)

    gui.btn_trim_to_mapped_neurons_clicked()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "Nothing to trim" in qmessagebox_calls[0]["title"]
    gui.close()


def test_trim_to_mapped_neurons_reports_nothing_to_trim_when_fully_mapped(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0, 1: 1, 2: 2})
    gui.update_project(project)

    gui.btn_trim_to_mapped_neurons_clicked()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "Nothing to trim" in qmessagebox_calls[0]["title"]
    gui.close()


def test_trim_to_mapped_neurons_cancelled_confirm_makes_no_changes(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((5, 3), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0, 1: 1})
    gui.update_project(project)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    gui.btn_trim_to_mapped_neurons_clicked()

    assert gui.project.stim_arrays["stim"].values.shape == (5, 3)
    gui.close()


def test_trim_to_mapped_neurons_confirmed_trims_and_forces_save_as(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((5, 3), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0, 1: 1})
    gui.update_project(project)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    new_filename_without_extension = tmp_path / "trimmed"
    new_filename = new_filename_without_extension.with_suffix(".nseq")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{new_filename_without_extension}", True),
    )

    gui.btn_trim_to_mapped_neurons_clicked()
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)

    assert gui.project.stim_arrays["stim"].values.shape == (5, 2)
    assert new_filename.exists()
    assert gui.project.filepath == new_filename
    gui.close()


def _add_detach_panel_test_data(gui: Gui) -> tuple[TraceData, SpikeData, SynapseData]:
    trace_data = TraceData(
        name="trace",
        traces=np.arange(12, dtype=np.float32).reshape(3, 4),
        indices=np.arange(3, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1, 2], dtype=np.uint32),
        spike_times=np.array([0.001, 0.002, 0.003], dtype=np.float32),
    )
    synapse_data = SynapseData(
        name="synapses",
        values=np.arange(18, dtype=np.float32).reshape(3, 3, 2),
        timestamps=np.array([0.0, 0.001], dtype=np.float32),
    )
    project = Project()
    project.project_section = "Section"
    project.project_name = "Project"
    project.add_experiment(
        Experiment(
            name="Experiment",
            duration=0.004,
            timepoints=[],
            data=[trace_data, spike_data, synapse_data],
        )
    )
    project.cellpose_data = _make_gui_cellpose_result()
    gui.update_project(project)
    gui.cmb_trace.setCurrentText("Experiment :: trace")
    gui._apply_trace_selection("Experiment :: trace")
    gui.cmb_spike.setCurrentText("Experiment :: spikes")
    gui._apply_spike_selection("Experiment :: spikes")
    gui.cmb_synapse.setCurrentText("Experiment :: synapses")
    gui.set_synapse("Experiment :: synapses")
    gui.cmb_data.setCurrentText("Experiment :: trace [trace]")
    gui.console_widget.input.history = ["project"]
    gui.console_widget.output.setPlainText(">>> project\n<Project>")
    return trace_data, spike_data, synapse_data


def test_spatial_footprint_panel_toggle_shows_and_hides(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    action = gui._view_actions[gui.spatial_footprint_widget]

    assert action.text() == "Show &Spatial Footprints"
    assert action.isChecked()
    assert gui.spatial_footprint_widget.isVisible()

    action.trigger()
    assert not action.isChecked()
    assert not gui.spatial_footprint_widget.isVisible()

    action.trigger()
    assert action.isChecked()
    assert gui.spatial_footprint_widget.isVisible()
    gui.close()


def test_detach_panel_hides_original_and_shows_fresh_window(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    trace_data, _, synapse_data = _add_detach_panel_test_data(gui)
    panels: list[QtWidgets.QWidget] = [
        gui.trace_widget,
        gui.data_widget,
        gui.heatmap_widget,
        gui.spatial_footprint_widget,
        gui.console_widget,
    ]

    for panel in panels:
        gui._auto_show_widget(panel)
        gui.detach_panel(panel, keep_original=False)
        window = gui._detached_panel_windows[-1]

        assert window.windowTitle().endswith(" - Section / Project")
        assert not gui._view_actions[panel].isChecked()
        assert not panel.isVisible()
        if panel is gui.trace_widget:
            assert window._widget.data is trace_data
        elif panel is gui.data_widget:
            assert window._widget.trace_widget.data is trace_data
        elif panel is gui.heatmap_widget:
            np.testing.assert_array_equal(window._widget.data, synapse_data.values.T)
        elif panel is gui.spatial_footprint_widget:
            assert window._widget.overlay_widget.segmentation is not None
        elif panel is gui.console_widget:
            assert window._widget.input.history == ["project"]
            assert window._widget.output.toPlainText() == ">>> project\n<Project>"

        window.close()

    gui.close()


def test_reshowing_a_plainly_detached_panel_closes_its_detached_window(
    qtbot: QtBot,
) -> None:
    """A plain "Detach panel" (not "Detach copy") hides the docked panel;
    checking "Show X" again should bring it back into the dock and close
    the now-redundant floating window, not leave both visible."""
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel(gui.trace_widget, keep_original=False)
    window = gui._detached_panel_windows[-1]
    assert not gui._view_actions[gui.trace_widget].isChecked()
    assert window.isVisible()

    gui._view_actions[gui.trace_widget].trigger()  # re-check "Show Traces"

    assert gui.trace_widget.isVisible()
    assert not window.isVisible()
    assert window not in gui._detached_panel_windows
    gui.close()


def test_reshowing_a_copy_detached_panel_leaves_its_window_open(
    qtbot: QtBot,
) -> None:
    """ "Detach copy of panel" never unchecks "Show X" in the first place,
    but toggling it off and back on should not touch the independent
    copy window either."""
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel_copy(gui.trace_widget)
    window = gui._detached_panel_windows[-1]
    action = gui._view_actions[gui.trace_widget]
    assert action.isChecked()

    action.trigger()  # uncheck
    action.trigger()  # re-check

    assert window.isVisible()
    assert window in gui._detached_panel_windows
    window.close()
    gui.close()


def test_detach_copy_keeps_original_visible(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel_copy(gui.trace_widget)
    window = gui._detached_panel_windows[-1]

    assert window.windowTitle() == "Traces - Section / Project"
    assert gui._view_actions[gui.trace_widget].isChecked()
    assert gui.trace_widget.isVisible()
    assert window.isVisible()
    window.close()
    gui.close()


def test_detached_trace_copy_shares_state_with_original(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel_copy(gui.trace_widget)
    window = gui._detached_panel_windows[-1]
    detached_widget = window._widget

    # both views are bound to the same shared state, so data set before the
    # detach is already visible in the detached view too
    assert detached_widget.data is gui.trace_widget.data

    # greying a neuron via the detached view is reflected in the docked one
    detached_widget.set_neuron_greyed(1, True)
    assert gui.trace_widget.is_neuron_greyed(1)
    assert gui.trace_widget.n_neurons == 2

    # and greying via the docked view is reflected back in the detached one
    gui.trace_widget.set_neuron_greyed(1, False)
    assert not detached_widget.is_neuron_greyed(1)
    assert detached_widget.n_neurons == 3

    window.close()
    gui.close()


def test_hovering_a_neuron_in_trace_or_stim_highlights_it_in_spatial_footprints(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)

    assert gui._spatial_footprint_state.hovered_stim_id is None

    gui.trace_widget.neuron_hovered.emit(1)
    assert gui._spatial_footprint_state.hovered_stim_id == 1

    gui.stim_editor.stim_widget.neuron_hovered.emit(2)
    assert gui._spatial_footprint_state.hovered_stim_id == 2

    gui.trace_widget.neuron_hovered.emit(None)
    assert gui._spatial_footprint_state.hovered_stim_id is None
    gui.close()


def test_hovering_in_a_detached_trace_copy_also_highlights_the_neuron(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel_copy(gui.trace_widget)
    window = gui._detached_panel_windows[-1]
    detached_widget = window._widget

    detached_widget.neuron_hovered.emit(0)

    assert gui._spatial_footprint_state.hovered_stim_id == 0
    window.close()
    gui.close()


def test_detached_spatial_footprint_copy_shares_state_with_original(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.spatial_footprint_widget)

    gui.detach_panel_copy(gui.spatial_footprint_widget)
    window = gui._detached_panel_windows[-1]
    detached_widget = window._widget

    # clicking a neuron in the detached view selects it in the docked one
    # too, including the downstream trace/stim rows the docked panel drives
    detached_widget.overlay_widget.neuron_clicked.emit(1)
    assert gui.spatial_footprint_widget._greyed_cellpose_ids == {0, 2}
    assert not gui.trace_widget.is_neuron_greyed(1)
    assert gui.trace_widget.is_neuron_greyed(0)
    assert gui.trace_widget.is_neuron_greyed(2)

    # clicking it again from the docked view deselects it, un-greying both
    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    assert detached_widget._greyed_cellpose_ids == set()
    assert not gui.trace_widget.is_neuron_greyed(0)
    assert not gui.trace_widget.is_neuron_greyed(2)

    window.close()
    gui.close()


def test_neuron_click_rebuilds_stim_and_trace_layout_only_once(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)

    # Selecting one cell greys out every other cell in the segmentation at
    # once; the stim/trace panels must recompute their layout exactly once
    # for the whole change, not once per greyed-out row (see the batch
    # `neurons_greyed_out_changed` signal added alongside the old
    # per-neuron `neuron_greyed_out_changed`).
    stim_layout_calls = 0
    original_stim_layout = gui.stim_editor.stim_widget._apply_visible_neuron_layout

    def _counting_stim_layout(*args: object, **kwargs: object) -> None:
        nonlocal stim_layout_calls
        stim_layout_calls += 1
        original_stim_layout(*args, **kwargs)

    gui.stim_editor.stim_widget._apply_visible_neuron_layout = (  # type: ignore[method-assign]
        _counting_stim_layout
    )

    trace_layout_calls = 0
    original_trace_layout = gui.trace_widget._apply_visible_neuron_layout

    def _counting_trace_layout(*args: object, **kwargs: object) -> None:
        nonlocal trace_layout_calls
        trace_layout_calls += 1
        original_trace_layout(*args, **kwargs)

    gui.trace_widget._apply_visible_neuron_layout = (  # type: ignore[method-assign]
        _counting_trace_layout
    )

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(0)

    assert stim_layout_calls == 1
    assert trace_layout_calls == 1
    assert gui.trace_widget.is_neuron_greyed(1)
    assert gui.trace_widget.is_neuron_greyed(2)
    assert not gui.trace_widget.is_neuron_greyed(0)
    gui.close()


def test_detached_console_copy_close_button_closes_its_own_window_only(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.console_widget)

    gui.detach_panel_copy(gui.console_widget)
    window = gui._detached_panel_windows[-1]

    window._widget.close_requested.emit()

    assert not window.isVisible()
    assert gui.console_widget.isVisible()
    assert gui._view_actions[gui.console_widget].isChecked()
    gui.close()


def test_detach_context_menu_installed_on_panel_canvases(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    for widget in (
        gui.data_widget,
        gui.data_widget.heatmap_widget.widget,
        gui.heatmap_widget.widget,
        gui.custom_analysis_widget.canvas,
        gui.console_widget,
        gui.chat_widget.history_view,
    ):
        assert (
            widget.contextMenuPolicy() == QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )

    # The Spatial Footprints panel's own rendering canvas is excluded - a
    # right click there is its lasso-select gesture, not a detach request
    # (see test_overlay_view_left_click_selects_right_click_does_not in
    # tests/test_spatial_footprint_widget.py).
    assert (
        gui.spatial_footprint_widget.overlay_widget.overlay_view.widget.contextMenuPolicy()
        != QtCore.Qt.ContextMenuPolicy.CustomContextMenu
    )
    assert (
        gui.spatial_footprint_widget.overlay_widget.contextMenuPolicy()
        == QtCore.Qt.ContextMenuPolicy.CustomContextMenu
    )

    assert (
        gui.stim_editor.stim_widget.contextMenuPolicy()
        != QtCore.Qt.ContextMenuPolicy.CustomContextMenu
    )
    gui.close()


def test_detach_custom_analysis_before_anything_analyzed_shows_message(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)

    gui.detach_panel(gui.custom_analysis_widget, keep_original=False)

    assert qmessagebox_calls[-1]["name"] == "information"
    assert "nothing to detach" in qmessagebox_calls[-1]["title"].lower()
    assert gui.findChildren(DetachedPanelWindow) == []
    gui._close_confirmed = True
    gui.close()


def test_detach_custom_analysis_after_analysis_seeds_from_last_shown(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    trace_data, _, _ = _add_detach_panel_test_data(gui)

    gui._custom_analysis(
        trace_data,
        CustomAnalysisType.Histogram,
        False,
        None,
        None,
    )
    gui.detach_panel(gui.custom_analysis_widget, keep_original=True)
    window = gui._detached_panel_windows[-1]

    assert isinstance(window._widget, CustomAnalysisWidget)
    assert window._widget.traces is not None
    assert window._widget.traces.shape == trace_data.traces.shape
    assert window.windowTitle() == "Custom Analysis - Section / Project"
    window.close()
    gui.close()


def test_custom_analysis_popup_title_includes_originating_tab(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    trace_data, _, _ = _add_detach_panel_test_data(gui)

    gui._custom_analysis(
        trace_data,
        CustomAnalysisType.Histogram,
        True,
        None,
        None,
    )
    window = gui.findChildren(CustomAnalysisWindow)[-1]

    assert window.windowTitle().endswith(" - Section / Project")
    window.close()
    gui.close()


def test_closing_the_main_window_closes_dynamic_input_routine_dialogs(
    qtbot: QtBot,
) -> None:
    """The routine dialogs (and their own plot/heatmap popups) are created
    without a Qt parent - see StimEditor._show_routine_dialog's docstring -
    so closing the main window would otherwise leave them open as orphaned
    windows: findChildren() never finds them, since they were never made
    Qt children of the main window in the first place."""
    gui = Gui()
    qtbot.addWidget(gui)
    gui.update_project(default_project())

    gui.stim_editor.open_dynamic_input_designer()
    selector = gui.stim_editor._routine_selector_dialog
    assert selector is not None
    selector.cmb_routine.setCurrentText(StimEditor.RHEOBASE_ROUTINE)
    selector._on_continue()
    dialog = gui.stim_editor._rheobase_dialog
    assert dialog is not None
    assert dialog.isVisible()

    gui._close_confirmed = True
    gui.close()

    assert not dialog.isVisible()
    assert gui.stim_editor._rheobase_dialog is None


def _make_timed_project(n_timepoints: int, dt: float = 0.01) -> Project:
    project = Project()
    traces = TraceData(
        name="trace",
        traces=np.zeros((2, n_timepoints), dtype=np.float32),
        indices=np.arange(2, dtype=np.uint32),
        units="mV",
        dt=dt,
    )
    project.add_experiment(
        Experiment(
            name="exp",
            duration=n_timepoints * dt,
            timepoints=[],
            data=[traces],
        )
    )
    return project


def test_switching_project_tabs_restores_each_tabs_own_time_slider_position(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    tab0_project = gui.project
    tab1_project = _make_timed_project(500, dt=0.01)  # 5.0s

    gui._open_project_in_new_tab(tab1_project, reset_controller=True)
    assert gui.time_slider.t_ms == 0  # a freshly opened tab has no history

    gui.time_slider.set_current_time_ms(3000)
    gui._activate_tab_index(0, reset_controller=True)
    assert gui.time_slider.t_ms == 0  # tab0 was never scrubbed
    assert tab1_project.last_time_ms == 3000  # remembered on the way out

    gui.time_slider.set_current_time_ms(200)
    gui._activate_tab_index(1, reset_controller=True)
    assert gui.time_slider.t_ms == 3000  # restored from tab1_project.last_time_ms
    assert tab0_project.last_time_ms == 200

    gui._activate_tab_index(0, reset_controller=True)
    assert gui.time_slider.t_ms == 200  # restored from tab0_project.last_time_ms
    gui.close()


def test_switching_to_tab_with_no_stim_arrays_still_updates_time_slider_range(
    qtbot: QtBot,
) -> None:
    # StimEditor.set_project() only emits limits_modified (which historically
    # drove the time-slider range) when the project has stim arrays. A tab
    # with none previously left the slider's range stale from whichever
    # project was active before it.
    gui = Gui()
    qtbot.addWidget(gui)
    tab1_project = _make_timed_project(500, dt=0.01)  # 5.0s
    assert not tab1_project.stim_arrays

    gui._open_project_in_new_tab(tab1_project, reset_controller=True)

    assert gui.time_slider.t_max_ms == 5000
    gui.close()


def test_closing_the_only_dirty_tab_prompts_to_save_then_opens_example_project(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    assert len(gui.tabs) == 1
    gui.project.mark_dirty()

    warn_calls: list[tuple] = []

    def _fake_warning(*args, **kwargs):
        warn_calls.append(args)
        return QtWidgets.QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)

    gui._close_tab(0)

    assert len(warn_calls) == 1  # asked about saving before replacing the tab
    assert len(gui.tabs) == 1
    assert gui.project.project_name == "Getting Started Project"
    gui.close()


def test_closing_the_only_clean_tab_opens_example_project_without_prompting(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    assert len(gui.tabs) == 1
    assert not gui.project.dirty

    warn_calls: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warn_calls.append(args),
    )

    gui._close_tab(0)

    assert warn_calls == []  # nothing unsaved, so no prompt was needed
    assert len(gui.tabs) == 1
    assert gui.project.project_name == "Getting Started Project"
    gui.close()


def test_switching_project_tabs_restores_each_tabs_own_spatial_footprint_selection(
    qtbot: QtBot,
) -> None:
    """Regression test for #555: the Spatial Footprints panel's selection
    lives on one shared SpatialFootprintState (see SpatialFootprintWidget's
    ``state`` parameter), and set_data() - called on every tab switch -
    always clears it. Switching tabs used to silently deselect whatever the
    user had selected, even when switching straight back."""
    gui = Gui()
    qtbot.addWidget(gui)
    tab0_project = gui.project
    tab0_project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui.update_project(tab0_project)

    tab1_project = Project()
    tab1_project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui._open_project_in_new_tab(tab1_project, reset_controller=True)
    assert gui._spatial_footprint_state.selected_cellpose_ids == set()

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(2)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {2}

    gui._activate_tab_index(0, reset_controller=True)
    assert gui._spatial_footprint_state.selected_cellpose_ids == set()
    assert tab1_project.selected_cellpose_ids == {2}  # remembered on the way out

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {1}

    gui._activate_tab_index(1, reset_controller=True)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {2}  # restored

    gui._activate_tab_index(0, reset_controller=True)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {1}  # restored
    gui.close()


def test_closing_gui_with_open_detached_windows_does_not_crash(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)

    gui.detach_panel(gui.console_widget, keep_original=True)
    gui.detach_panel(gui.spatial_footprint_widget, keep_original=True)

    assert len(gui._detached_panel_windows) == 2
    gui.close()
    assert gui._detached_panel_windows == []


def test_clicking_neuron_in_spatial_footprint_greys_trace_and_stim_rows(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((10, 3), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    trace_data = TraceData(
        name="traces",
        traces=np.zeros((3, 10), dtype=np.float32),
        indices=np.arange(3, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    spike_data = SpikeData(
        name="spikes",
        spike_indices=np.array([0, 1, 2], dtype=np.uint32),
        spike_times=np.array([0.001, 0.002, 0.003], dtype=np.float32),
    )
    project.add_experiment(
        Experiment(
            name="Experiment",
            duration=0.01,
            timepoints=[0.01],
            data=[trace_data, spike_data],
        )
    )
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 2, 1: 1, 2: 0})

    gui.update_project(project)
    gui._apply_trace_selection("Experiment :: traces")
    gui._apply_spike_selection("Experiment :: spikes")
    original_traces = trace_data.traces.copy()
    original_stim_values = project.stim_arrays["stim"].values.copy()

    # clicking selects cellpose cell 2 (-> stim 0), hiding everything else
    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(2)

    assert not gui.trace_widget.is_neuron_greyed(0)
    assert gui.trace_widget.is_neuron_greyed(1)
    assert gui.trace_widget.is_neuron_greyed(2)
    assert gui.stim_editor.stim_widget.is_neuron_greyed(1)
    assert gui.stim_editor.stim_widget.is_neuron_greyed(2)
    assert gui.spike_widget.is_neuron_greyed(1)
    assert gui.spike_widget.is_neuron_greyed(2)
    # the greyed rows are actually hidden, and the remaining row compacts -
    # the same way, and on the same rows, in every panel that stacks neurons
    # on its y-axis, so a given row still means the same neuron everywhere
    assert gui.trace_widget.n_neurons == 1
    assert gui.stim_editor.stim_widget.n_neurons == 1
    assert gui.spike_widget.n_neurons == 1
    assert gui.trace_widget.graphic is not None
    assert gui.trace_widget.graphic.graphics[0].visible
    # underlying project data is never touched by hiding
    reloaded_trace_data = project.experiments["Experiment"].data[0]
    assert isinstance(reloaded_trace_data, TraceData)
    assert np.allclose(reloaded_trace_data.traces, original_traces)
    assert np.allclose(project.stim_arrays["stim"].values, original_stim_values)

    # deselecting the only selected cell clears the selection - show all again
    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(2)

    assert not gui.trace_widget.is_neuron_greyed(1)
    assert not gui.trace_widget.is_neuron_greyed(2)
    assert not gui.stim_editor.stim_widget.is_neuron_greyed(1)
    assert not gui.stim_editor.stim_widget.is_neuron_greyed(2)
    assert not gui.spike_widget.is_neuron_greyed(1)
    assert not gui.spike_widget.is_neuron_greyed(2)
    assert gui.trace_widget.n_neurons == 3
    assert gui.stim_editor.stim_widget.n_neurons == 3
    assert gui.spike_widget.n_neurons == 3
    gui.close()


def _stim_trace_overlay_project() -> tuple[Project, TraceData]:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stim",
            StimType.Excitation,
            np.zeros((10, 3), dtype=np.float32),
            "mV",
            0.001,
        )
    )
    trace_data = TraceData(
        name="traces",
        traces=np.zeros((3, 10), dtype=np.float32),
        indices=np.arange(3, dtype=np.uint32),
        units="mV",
        dt=0.001,
    )
    project.add_experiment(
        Experiment(
            name="Experiment",
            duration=0.01,
            timepoints=[0.01],
            data=[trace_data],
        )
    )
    return project, trace_data


def test_toggle_stim_trace_overlay_shows_and_hides_stim_panel(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    gui.update_project(project)
    gui._apply_trace_selection("Experiment :: traces")

    assert gui.trace_widget._stim_overlay_graphic is None
    assert gui.stim_editor.isVisible()

    gui.action_overlay_stim_on_traces.trigger()

    assert gui.trace_widget._stim_overlay_enabled
    assert gui.trace_widget._stim_overlay_graphic is not None
    assert not gui.stim_editor.isVisible()

    gui.action_overlay_stim_on_traces.trigger()

    assert not gui.trace_widget._stim_overlay_enabled
    assert gui.trace_widget._stim_overlay_graphic is None
    assert gui.stim_editor.isVisible()
    gui.close()


def test_toggle_stim_trace_overlay_respects_manually_hidden_stim_panel(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    gui.update_project(project)
    gui._apply_trace_selection("Experiment :: traces")
    gui._view_actions[gui.stim_editor].trigger()
    assert not gui.stim_editor.isVisible()

    gui.action_overlay_stim_on_traces.trigger()
    assert not gui.stim_editor.isVisible()  # already hidden, unaffected

    gui.action_overlay_stim_on_traces.trigger()
    assert not gui.stim_editor.isVisible()  # stays hidden, as it was before overlay
    gui.close()


def test_stim_trace_overlay_updates_when_selected_stim_changes(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    other_values = np.full((10, 3), 5.0, dtype=np.float32)
    project.add_stim_array(
        StimArray("stim2", StimType.Excitation, other_values, "mV", 0.001)
    )
    gui.update_project(project)
    gui._apply_trace_selection("Experiment :: traces")
    gui.action_overlay_stim_on_traces.trigger()
    assert gui.trace_widget._stim_overlay_array is project.stim_arrays["stim"]

    gui.cmb_stim.setCurrentText("stim2")

    assert gui.trace_widget._stim_overlay_array is project.stim_arrays["stim2"]
    gui.close()


def test_time_slider_highlights_stimulated_neurons_in_spatial_footprints(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    project.stim_arrays["stim"].values[5, 1] = 3.0  # neuron 1 stimulated at t=5ms
    project.cellpose_data = _make_gui_cellpose_result()
    gui.update_project(project)

    gui.time_slider.set_current_time_ms(5)
    assert gui._spatial_footprint_state.active_stim_ids == frozenset({1})

    gui.time_slider.set_current_time_ms(0)
    assert gui._spatial_footprint_state.active_stim_ids == frozenset()
    gui.close()


def test_switching_current_stim_array_refreshes_active_stim_highlight(
    qtbot: QtBot,
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    other_values = np.zeros((10, 3), dtype=np.float32)
    other_values[0, 2] = 1.0
    project.add_stim_array(
        StimArray("stim2", StimType.Excitation, other_values, "mV", 0.001)
    )
    project.cellpose_data = _make_gui_cellpose_result()
    gui.update_project(project)
    assert gui._spatial_footprint_state.active_stim_ids == frozenset()

    gui.cmb_stim.setCurrentText("stim2")

    assert gui._spatial_footprint_state.active_stim_ids == frozenset({2})
    gui.close()


def test_import_cellpose_data_populates_and_shows_spatial_footprint_panel(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    imported = _make_gui_cellpose_result()

    class _FakeDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

        def get_imported_data(self) -> CellposeImportResult:
            return imported

    monkeypatch.setattr("neuroseq.gui.CellposeImportDialog", _FakeDialog)
    gui = Gui()
    qtbot.addWidget(gui)

    gui.btn_import_cellpose_data_clicked()

    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()
    assert gui.spatial_footprint_widget.isVisible()
    gui.close()


def test_import_tif_with_new_segmentation_populates_spatial_footprint_panel(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    imported = _make_gui_cellpose_result()

    class _FakeDialog:
        experiment_name = "fake"
        experiment_duration = 1.0
        imported_data: list[Any] = []
        new_cellpose_result = imported
        imported_tif_paths: list[Any] = []
        imported_fps = 30.0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr("neuroseq.tif_import_dialog.TifImportDialog", _FakeDialog)
    gui = Gui()
    qtbot.addWidget(gui)

    gui.btn_import_tif_clicked()

    assert gui.project.cellpose_data is not None
    assert gui.project.cellpose_data.image_path == imported.image_path
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()
    assert gui.spatial_footprint_widget.isVisible()
    gui.close()


def test_import_tif_reusing_loaded_segmentation_does_not_reset_spatial_footprint_panel(
    monkeypatch: MonkeyPatch,
) -> None:
    existing = _make_gui_cellpose_result(mapping={0: 2, 1: 1, 2: 0})
    appended: list[Experiment] = []

    class _FakeDialog:
        experiment_name = "fake"
        experiment_duration = 1.0
        imported_data: list[Any] = []
        new_cellpose_result = None
        imported_tif_paths: list[Any] = []
        imported_fps = 30.0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr("neuroseq.tif_import_dialog.TifImportDialog", _FakeDialog)

    class _FakeGui:
        def __init__(self) -> None:
            self.project = Project()
            self.project.cellpose_data = existing
            self.last_cellpose_import = existing
            self.spatial_footprint_widget = SimpleNamespace()

        def _append_experimental_data(self, experiment: Experiment) -> None:
            appended.append(experiment)

        def _apply_cellpose_import_result(
            self,
            imported: CellposeImportResult,
        ) -> list[str]:
            raise AssertionError(f"unexpected Cellpose propagation: {imported}")

    fake_gui = _FakeGui()

    Gui.btn_import_tif_clicked(fake_gui)  # type: ignore[arg-type]

    assert len(appended) == 1
    assert fake_gui.project.cellpose_data is existing
    assert fake_gui.last_cellpose_import is existing
    assert fake_gui.project.cellpose_data.stimulus_to_cellpose_mapping == {
        0: 2,
        1: 1,
        2: 0,
    }


def test_spatial_footprint_empty_button_opens_cellpose_import(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    imported = _make_gui_cellpose_result()

    class _FakeDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

        def get_imported_data(self) -> CellposeImportResult:
            return imported

    monkeypatch.setattr("neuroseq.gui.CellposeImportDialog", _FakeDialog)
    gui = Gui()
    qtbot.addWidget(gui)

    gui.spatial_footprint_widget.import_button.click()

    assert gui.project.cellpose_data is not None
    assert gui.project.cellpose_data.image_path == imported.image_path
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()
    gui.close()


def test_deleting_cellpose_data_clears_spatial_footprint_panel(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    gui.project.cellpose_data = _make_gui_cellpose_result()
    gui.spatial_footprint_widget.set_data(gui.project.cellpose_data)

    gui._delete_cellpose_data()

    assert gui.project.cellpose_data is None
    assert gui.spatial_footprint_widget.overlay_widget.overlay_image is None
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is None
    assert not gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert gui.spatial_footprint_widget.overlay_widget.isHidden()
    gui.close()


def test_gui_choose_fiji_path_forces_prompt(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    import neuroseq.fiji as fiji_module

    gui = Gui()
    qtbot.addWidget(gui)

    calls: list[dict] = []
    monkeypatch.setattr(
        fiji_module,
        "resolve_fiji_path",
        lambda parent, *, force_prompt=False: calls.append(
            {"parent": parent, "force_prompt": force_prompt}
        ),
    )
    gui.choose_fiji_path()
    assert calls == [{"parent": gui, "force_prompt": True}]
    gui.close()


def test_gui_theme_menu_toggles_theme_and_persists(
    qtbot: QtBot, tmp_config_dir
) -> None:
    from neuroseq import theme as theme_module

    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    # locate the Theme submenu in the Settings menu
    theme_menu = next(
        (
            action.menu()
            for action in gui.menu_settings.actions()
            if action.menu() is not None and "Theme" in action.text().replace("&", "")
        ),
        None,
    )
    assert theme_menu is not None
    labels = {action.text().replace("&", ""): action for action in theme_menu.actions()}
    assert set(labels) == {"Light", "Dark", "Retro light theme (beta)"}

    labels["Dark"].trigger()
    assert gui.settings.theme == theme_module.DARK
    assert gui.trace_widget._theme == theme_module.DARK
    assert gui.data_widget.spike_widget._theme == theme_module.DARK
    assert labels["Dark"].isChecked()

    app = QtWidgets.QApplication.instance()
    labels["Retro light theme (beta)"].trigger()
    assert gui.settings.theme == theme_module.RETRO
    assert gui.trace_widget._theme == theme_module.RETRO
    assert labels["Retro light theme (beta)"].isChecked()
    # the retro skin is driven by an application-wide stylesheet
    assert app is not None
    assert app.styleSheet() == theme_module.app_stylesheet(theme_module.RETRO)

    labels["Light"].trigger()
    # ... which is cleared again when leaving the retro skin
    assert app.styleSheet() == ""
    assert gui.settings.theme == theme_module.LIGHT
    assert gui.trace_widget._theme == theme_module.LIGHT
    assert labels["Light"].isChecked()
    gui.close()


def test_gui_applies_persisted_theme_on_startup(qtbot: QtBot, tmp_config_dir) -> None:
    from neuroseq import theme as theme_module
    from neuroseq.gui_settings import GuiSettings

    # Persist a dark theme to the (temporary) config before launching.
    GuiSettings().theme = theme_module.DARK

    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    assert gui.settings.theme == theme_module.DARK
    assert gui.trace_widget._theme == theme_module.DARK
    assert gui._theme_actions[theme_module.DARK].isChecked()
    gui.close()


def test_gui_starts_in_retro_theme_without_config(qtbot: QtBot, tmp_config_dir) -> None:
    from neuroseq import theme as theme_module

    # No config has been written, so the GUI must come up in the default theme.
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    assert theme_module.DEFAULT == theme_module.RETRO
    assert gui.settings.theme == theme_module.RETRO
    assert gui.trace_widget._theme == theme_module.RETRO
    assert gui._theme_actions[theme_module.RETRO].isChecked()
    app = QtWidgets.QApplication.instance()
    assert app is not None
    assert app.styleSheet() == theme_module.app_stylesheet(theme_module.RETRO)
    gui.close()
    # leave the shared QApplication unskinned for the rest of the suite
    theme_module.apply_app_theme(app, theme_module.LIGHT)


def test_timeseries_panels_and_slider_share_one_width(qtbot: QtBot) -> None:
    """Stim editor, traces, spikes and the time slider must line up exactly.

    They share a time axis, so a sample at a given x in one panel has to sit
    at the same x in the others and under the slider handle. They live in one
    vertical splitter for that reason; the synapse / cell-segmentation column
    sits beside the whole stack rather than beside the stim editor alone.
    """
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.trace_widget.setVisible(True)
    gui.spike_widget.setVisible(True)
    gui.data_widget.setVisible(True)
    qtbot.waitUntil(lambda: gui.spike_widget.isVisible())

    panels: list[QtWidgets.QWidget] = [
        gui.stim_editor,
        gui.trace_widget,
        gui.spike_widget,
        gui.data_widget,
        gui.time_slider,
    ]
    widths = {panel.width() for panel in panels}
    lefts = {panel.mapTo(gui, QtCore.QPoint(0, 0)).x() for panel in panels}
    assert len(widths) == 1, f"panels have differing widths: {widths}"
    assert len(lefts) == 1, f"panels have differing left edges: {lefts}"
    # ...and they are genuinely narrower than the window, i.e. the right-hand
    # column is beside them rather than below.
    assert next(iter(widths)) < gui.width()
    gui.close()


def test_retro_time_slider_elapsed_side_is_green(qtbot: QtBot) -> None:
    from neuroseq import theme as theme_module

    stylesheet = theme_module.app_stylesheet(theme_module.RETRO)
    sub_page = stylesheet.split("QSlider::sub-page:horizontal")[1].split("}")[0]
    # The elapsed side of the groove is lit LCD green, not the dim groove color.
    assert theme_module._RETRO_LCD_FG in sub_page


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
    assert "Cell &segmentation" in file_action_texts

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
    assert "Cell &segmentation" not in import_data_action_texts

    gui.close()


def test_gui_has_no_export_graphics_menu(qtbot: QtBot) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())

    submenus = {
        action.text().replace("&", ""): action.menu()
        for action in gui.menu_file.actions()
        if action.menu() is not None
    }
    assert "Export graphics" not in submenus

    assert "Export" in submenus
    assert submenus["Export"] is not None
    export_actions = {
        action.text().replace("&", "") for action in submenus["Export"].actions()
    }
    assert "Traces" not in export_actions
    assert "Spikes" not in export_actions

    gui.close()


def test_gui_simulate_and_playback(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project, tmp_config_dir
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


class _MainThreadCancelTrigger(QtCore.QObject):
    """Relay ``BackgroundIOWorker.progress`` (emitted on the worker thread)
    into a ``canceled`` emission that runs on the main thread, the way a
    real Cancel-button click would.

    Connecting the worker's cross-thread signal straight to a plain lambda
    (as this test used to) runs that lambda - and therefore its
    ``progress_dialog.canceled.emit()`` call, and everything ``_on_cancel``
    does in response, including direct QWidget mutations - on whichever
    thread emits ``progress``. Qt's AutoConnection can only auto-queue a
    cross-thread call onto a *QObject* receiver's own thread; a bare
    function/lambda has no thread affinity for it to detect, so PyQt just
    runs it synchronously, in-thread, instead. A bound method of a QObject
    living on the main thread (this class) gives AutoConnection something
    to detect, so it correctly queues the call onto the main thread instead
    - avoiding a background-thread QWidget mutation that is undefined
    behaviour in Qt and has been observed to hang/crash intermittently on
    macOS CI specifically (never reproduced on Linux, which tends to
    tolerate the violation silently)."""

    def __init__(self, progress_dialog: ToolbarProgressWidget) -> None:
        super().__init__()
        self._progress_dialog = progress_dialog

    @QtCore.Slot(int)
    def trigger(self, _value: int) -> None:
        self._progress_dialog.canceled.emit()


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
    cancel_trigger = _MainThreadCancelTrigger(gui.progress_dialog)
    gui._io_worker.progress.connect(cancel_trigger.trigger)

    _wait_for_brian2_simulation(qtbot, gui)

    assert gui.project.b2_model.cancel_requested is True
    assert gui._io_worker is None
    # isHidden() (this widget's own state) is the reliable signal that
    # cancel()/hide() worked - see the comment in
    # test_simulation_progress_is_embedded_in_view_toolbar for why
    # isVisible() isn't asserted here.
    assert gui.progress_dialog.isHidden()
    assert QtWidgets.QApplication.overrideCursor() is None
    gui.close()


def test_file_menu_shortcuts_are_disabled_during_a_running_io_task(
    qtbot: QtBot, test_project: Project
) -> None:
    """Regression test for #556: ``_set_io_actions_enabled`` used to only
    disable the File menu/toolbar/tab bar *containers*.  Disabling a QMenu
    or QToolBar greys it out and blocks mouse clicks, but each QAction it
    holds keeps its own ``enabled`` flag - so a Qt.WindowShortcut such as
    Ctrl+N/Ctrl+O/Ctrl+E kept firing even while the File menu looked
    disabled, letting a user open/switch tabs while a Brian2 simulation
    (or any other IO task) was still running for the previously active
    tab."""
    gui = Gui()
    gui.load_project(test_project)
    qtbot.addWidget(gui)

    actions = gui._iter_menu_actions(gui.menu_file)
    assert any(a.shortcut().toString() == "Ctrl+N" for a in actions)
    assert all(a.isEnabled() for a in actions)

    gui._set_io_actions_enabled(False)
    assert gui.menu_file.isEnabled() is False
    assert all(not a.isEnabled() for a in gui._iter_menu_actions(gui.menu_file))

    gui._set_io_actions_enabled(True)
    assert gui.menu_file.isEnabled() is True
    assert all(a.isEnabled() for a in gui._iter_menu_actions(gui.menu_file))
    gui.close()


def test_brian2_simulation_finish_does_not_touch_a_different_tab(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    test_project: Project,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Regression test for #556: if the active tab changes while a Brian2
    simulation is running (e.g. through the shortcut bypass fixed above),
    its completion callback used to unconditionally refresh the shared
    widgets from ``self.project`` and tell stim_editor's Design dynamic
    input dialogs the simulation finished - both of which, by then, belong
    to whichever *other* tab is active, not the one that was simulated."""
    gui = Gui()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    simulated_project = gui.project
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.2", True))

    gui.btn_continue_simulation_clicked()
    assert gui._io_worker is not None

    # Switch tabs while the simulation for `simulated_project` is still
    # running on the worker thread - this is exactly what the (now fixed)
    # shortcut bypass let a user do despite the "disabled" File menu/tab bar.
    other_project = default_project()
    gui._open_project_in_new_tab(other_project, reset_controller=True)
    assert gui.project is other_project

    _wait_for_brian2_simulation(qtbot, gui)

    # The simulation still ran to completion and its results live on the
    # project that was actually simulated...
    assert float(simulated_project.b2_model.network.t) > 0

    # ...but the other (now active) tab was left alone: no combo-box entries
    # borrowed from the simulated project, and the user was told where the
    # results actually went instead of the change silently landing elsewhere.
    assert gui.project is other_project
    assert gui.cmb_trace.count() == 0
    assert any(
        call["name"] == "information" and "project switch" in call["text"]
        for call in qmessagebox_calls
    )
    gui._close_confirmed = True
    gui.close()


def test_simulation_progress_is_embedded_in_view_toolbar(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible)

    assert gui.progress_dialog.parentWidget() is gui.toolbar_view
    assert isinstance(gui.progress_dialog.progress_bar, QtWidgets.QProgressBar)
    assert not isinstance(gui.progress_dialog, QtWidgets.QDialog)
    assert gui.progress_dialog.progress_bar.width() == 100
    assert gui.progress_dialog.cancel_button.width() == 58
    layout = gui.progress_dialog.layout()
    assert layout is not None
    assert layout.itemAt(0).widget() is gui.progress_dialog.progress_bar
    assert layout.itemAt(1).widget() is gui.progress_dialog.cancel_button
    assert layout.itemAt(2).widget() is gui.progress_dialog.label

    gui.progress_dialog.setLabelText("Simulating Brian2...")
    gui.progress_dialog.setCancelButtonText("Cancel")
    gui.progress_dialog.setValue(42)
    gui.progress_dialog.show()
    # show() is deferred under the offscreen QPA plugin used in tests, so
    # wait for it to actually land before asserting on it.
    qtbot.waitUntil(gui.progress_dialog.isVisible, timeout=2000)

    assert gui.progress_dialog.isVisible()
    assert gui.progress_dialog.value() == 42
    assert gui.progress_dialog.cancel_button.isVisible()

    gui.progress_dialog.reset()
    # isHidden() (this widget's own state) is the reliable signal that
    # reset()/hide() worked. isVisible() also asks whether every ancestor is
    # visible, and for a widget embedded via QWidgetAction in a QToolBar that
    # composite state can lag isHidden() under the offscreen QPA plugin used
    # in tests - not something reset() controls, so it isn't asserted here.
    assert gui.progress_dialog.isHidden()
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


def test_create_new_project_calls_gc_collect(monkeypatch: MonkeyPatch) -> None:
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
    Gui._create_new_project(fake_gui)

    assert collect_calls[0] == 1
    assert loaded_project == [sentinel_project]


def test_prompt_save_before_action_branches(monkeypatch: MonkeyPatch) -> None:
    proceeded = [0]
    save_events: list[str] = []
    answer_box = {"answer": QtWidgets.QMessageBox.StandardButton.Cancel}

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *a, **k: answer_box["answer"]
    )

    def make_fake_gui() -> Any:
        def fake_save(on_success: Any = None) -> bool:
            save_events.append("save")
            if on_success is not None:
                on_success(None)
            return True

        project = Project()
        project.dirty = True
        return SimpleNamespace(_save_project=fake_save, project=project)

    def proceed() -> None:
        proceeded[0] += 1

    # Cancel: neither save nor proceed.
    Gui._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert proceeded[0] == 0
    assert save_events == []

    # Discard: proceed immediately, without saving.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Discard
    Gui._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert proceeded[0] == 1
    assert save_events == []

    # Save: save first, then proceed via the on_success chain.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Save
    Gui._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert save_events == ["save"]
    assert proceeded[0] == 2


def test_prompt_save_before_action_skips_prompt_when_not_dirty(
    monkeypatch: MonkeyPatch,
) -> None:
    """No unsaved changes -> proceed right away, no Save/Discard/Cancel

    prompt at all (issue #535)."""
    warning_calls: list[object] = []

    def _fake_warning(*a: object, **k: object) -> QtWidgets.QMessageBox.StandardButton:
        warning_calls.append(a)
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)
    fake_gui: Any = SimpleNamespace(project=Project())  # dirty=False by default
    proceeded = [0]

    Gui._prompt_save_before_action(
        fake_gui, "T", "M", lambda: proceeded.__setitem__(0, 1)
    )

    assert warning_calls == []
    assert proceeded == [1]


def test_prompt_save_before_close_branches(monkeypatch: MonkeyPatch) -> None:
    save_callbacks: list[Any] = []
    closed = [0]
    answer_box = {"answer": QtWidgets.QMessageBox.StandardButton.Cancel}

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *a, **k: answer_box["answer"]
    )

    def fake_save(on_success: Any = None) -> bool:
        save_callbacks.append(on_success)
        return True

    dirty_project = Project()
    dirty_project.dirty = True
    fake_gui: Any = SimpleNamespace(
        _close_confirmed=False,
        _save_project=fake_save,
        close=lambda: closed.__setitem__(0, closed[0] + 1),
        tabs=[dirty_project],
    )

    # Cancel: keep the app open.
    assert not Gui._prompt_save_before_close(fake_gui)
    assert not fake_gui._close_confirmed
    assert save_callbacks == []
    assert closed == [0]

    # Discard: close immediately.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Discard
    assert Gui._prompt_save_before_close(fake_gui)
    assert fake_gui._close_confirmed
    assert save_callbacks == []
    assert closed == [0]

    # Save: block the current close, then close from the save success callback.
    fake_gui._close_confirmed = False
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Save
    assert not Gui._prompt_save_before_close(fake_gui)
    assert not fake_gui._close_confirmed
    assert len(save_callbacks) == 1
    assert closed == [0]

    save_callbacks[0](None)
    assert fake_gui._close_confirmed
    assert closed == [1]


def test_prompt_save_before_close_skips_prompt_when_no_tab_is_dirty(
    monkeypatch: MonkeyPatch,
) -> None:
    """No open tab has unsaved changes -> close proceeds right away, no

    Save/Discard/Cancel prompt at all (issue #535)."""
    warning_calls: list[object] = []

    def _fake_warning(*a: object, **k: object) -> QtWidgets.QMessageBox.StandardButton:
        warning_calls.append(a)
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)
    # Two clean tabs; neither is dirty.
    fake_gui: Any = SimpleNamespace(
        _close_confirmed=False,
        tabs=[Project(), Project()],
    )

    assert Gui._prompt_save_before_close(fake_gui)

    assert warning_calls == []
    assert fake_gui._close_confirmed


def test_qmessagebox_calls_fixture_accepts_the_buttons_and_default_button_args(
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Regression test: _prompt_save_before_close/_prompt_save_before_action
    call QMessageBox.warning(self, title, text, buttons, default_button) -
    5 positional args. The qmessagebox_calls fixture's mock used to accept
    only 3, so a dirty tab's close/save prompt raised a TypeError from
    inside a Qt event handler (e.g. closeEvent, or qtbot's end-of-test
    auto-close of a leftover widget) - a real crash risk, since an exception
    escaping a Qt virtual method override there is undefined behaviour
    rather than a normal test failure."""
    QtWidgets.QMessageBox.warning(
        None,
        "Close tab",
        "Closing this tab will discard any unsaved changes.",
        QtWidgets.QMessageBox.StandardButton.Save
        | QtWidgets.QMessageBox.StandardButton.Discard
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Save,
    )
    assert qmessagebox_calls == [
        {
            "name": "warning",
            "title": "Close tab",
            "text": "Closing this tab will discard any unsaved changes.",
        }
    ]


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


def test_gui_open_adds_to_recent_projects(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_path: Path, test_project: Project
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    project_path = tmp_path / "model.nseq"
    test_project.save(
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

    assert gui.settings.recent_projects[0] == str(project_path.resolve())
    gui.close()


def test_gui_save_adds_to_recent_projects(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    gui.update_project(project_ornstein)
    filename_without_extension = tmp_path / "saved"
    filename = filename_without_extension.with_suffix(".nseq")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)

    assert gui.settings.recent_projects[0] == str(filename.resolve())
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


def test_gui_toggle_minimum_stimulus_size(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    stim_widget = gui.stim_editor.stim_widget
    assert stim_widget.keep_short_stimuli_visible
    gui.toggle_minimum_stimulus_size(False)
    assert not stim_widget.keep_short_stimuli_visible
    gui.toggle_minimum_stimulus_size(True)
    assert stim_widget.keep_short_stimuli_visible
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

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("save options dialog should not be shown")

    monkeypatch.setattr(neuroseq.gui, "SaveLoadOptionsDialog", _fail_if_called)
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


def test_gui_save_quick_save_skips_dialogs_when_filepath_set(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    """Once a project has a file path, plain Save (Ctrl+S) writes straight
    back to it without showing the options or file-picker dialogs."""
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.update_project(project_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)

    filename = tmp_path / "existing.nseq"
    gui.project.save(filename, SaveLoadOptions(True, True, True, True))
    gui.project.filepath = filename
    assert filename.exists()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dialog should not be shown for a quick save")

    monkeypatch.setattr(neuroseq.gui, "SaveLoadOptionsDialog", _fail_if_called)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", _fail_if_called)

    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)
    assert gui.project.filepath == filename
    gui.close()


def test_gui_save_as_uses_all_options_without_options_dialog(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    """Save As prompts for a destination and includes every project component."""
    gui = Gui()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)
    gui.update_project(project_ornstein)
    qtbot.wait(delay_after_update_model_call_in_ms)

    original_filename = tmp_path / "existing.nseq"
    gui.project.save(original_filename, SaveLoadOptions(True, True, True, True))
    gui.project.filepath = original_filename

    new_filename_without_extension = tmp_path / "saved_as"
    new_filename = new_filename_without_extension.with_suffix(".nseq")
    assert not new_filename.exists()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("save options dialog should not be shown")

    monkeypatch.setattr(neuroseq.gui, "SaveLoadOptionsDialog", _fail_if_called)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{new_filename_without_extension}", True),
    )

    gui.btn_save_as_clicked()
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=15000)
    assert new_filename.exists()
    assert gui.project.filepath == new_filename
    # the original file must be untouched by the Save As
    assert original_filename.exists()
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
    assert "no input entities" in qmessagebox_calls[0]["title"].lower()


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
        (
            str(output_path),
            {"dpi": 300, "bbox_inches": "tight", "transparent": False},
        ),
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


def test_apply_spike_selection_does_not_auto_show_for_a_never_simulated_monitor() -> (
    None
):
    # A model that defines a spike monitor but was never simulated yields a
    # real (non-None) SpikeData with zero spikes - SpikeWidget.set_data()
    # itself treats that as empty and leaves .data unset. Auto-show must key
    # off that, not off get_data()'s non-None return, or every such project
    # pops the panel open with nothing in it.
    empty_spike_data = SpikeData(
        name="P",
        spike_indices=np.array([], dtype=np.int32),
        spike_times=np.array([], dtype=np.float32),
    )
    auto_show_calls: list[Any] = []
    spike_widget = SimpleNamespace(data=None, n_neurons=0)

    def _set_spike_data(data: SpikeData | None) -> None:
        spike_widget.data = (
            None if data is None or data.spike_times.shape[0] == 0 else data
        )

    spike_widget.set_data = _set_spike_data
    fake_gui: Any = SimpleNamespace(
        project=SimpleNamespace(get_data=lambda name, data_type: empty_spike_data),
        spike_widget=spike_widget,
        _focus_neuron_range=lambda n_neurons: None,
        _auto_show_widget=lambda widget: auto_show_calls.append(widget),
    )

    Gui._apply_spike_selection(fake_gui, "Simulation :: P")

    assert spike_widget.data is None
    assert auto_show_calls == []


def test_set_synapse_does_not_auto_show_for_a_never_simulated_monitor() -> None:
    empty_synapse_data = SynapseData(
        name="Ce / we",
        values=np.zeros((0, 0, 0), dtype=np.float32),
        timestamps=np.array([], dtype=np.float32),
    )
    auto_show_calls: list[Any] = []
    heatmap_widget = SimpleNamespace(data=None)

    def _set_heatmap_data(data: SynapseData | None) -> None:
        heatmap_widget.data = (
            None if data is None or data.values.shape[0] == 0 else data
        )

    heatmap_widget.set_data = _set_heatmap_data
    fake_gui: Any = SimpleNamespace(
        settings=SimpleNamespace(max_ram_synapse_plot_matrix=0),
        project=SimpleNamespace(
            max_ram_synapse_plot_matrix=0,
            get_data=lambda name, data_type: empty_synapse_data,
        ),
        heatmap_widget=heatmap_widget,
        _auto_show_widget=lambda widget: auto_show_calls.append(widget),
    )

    # set_synapse is @with_wait_cursor-wrapped, which touches the live
    # QApplication cursor - call the unwrapped function so this test doesn't
    # need a real Qt application.
    Gui.set_synapse.__wrapped__(fake_gui, "Simulation :: Ce / we")  # type: ignore[attr-defined]

    assert heatmap_widget.data is None
    assert auto_show_calls == []


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


class _RemapFakeGui:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.last_cellpose_import: CellposeImportResult | None = None

    def update_limits(self) -> None:
        pass

    def _cellpose_neuron_count(
        self,
        imported: CellposeImportResult | None,
    ) -> int:
        return Gui._cellpose_neuron_count(self, imported)  # type: ignore[arg-type]


def _cellpose_result_with_mapping(
    mapping: dict[int, int] | None,
) -> CellposeImportResult:
    return CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        stimulus_to_cellpose_mapping=mapping,
    )


def test_gui_apply_cellpose_import_result_remaps_existing_trace_data() -> None:
    """Reproduces the #518 scenario: import Suite2p trace data under one Cellpose
    mapping, then re-run the Cellpose import with a different mapping - the
    already-imported trace's displayed indices must follow the new mapping."""
    project = Project()
    project.cellpose_data = _cellpose_result_with_mapping({4: 0, 7: 1})
    trace_data = TraceData(
        "traces",
        np.zeros((2, 10), dtype=np.float32),
        np.array([4, 7], dtype=np.uint32),
        "mV",
        0.1,
        np.array([0, 1], dtype=np.uint32),
    )
    project.add_experiment(Experiment("exp", 1.0, [0.0, 1.0], [trace_data]))
    fake_gui = _RemapFakeGui(project)

    Gui._apply_cellpose_import_result(
        fake_gui,  # type: ignore[arg-type]
        _cellpose_result_with_mapping({9: 0, 2: 1}),
    )

    np.testing.assert_array_equal(trace_data.indices, np.array([9, 2], dtype=np.uint32))


def test_gui_apply_cellpose_import_result_warns_on_remap_conflict(
    qmessagebox_calls: list,
) -> None:
    project = Project()
    project.cellpose_data = _cellpose_result_with_mapping({4: 0, 7: 1})
    trace_data = TraceData(
        "traces",
        np.zeros((2, 10), dtype=np.float32),
        np.array([4, 7], dtype=np.uint32),
        "mV",
        0.1,
        np.array([0, 1], dtype=np.uint32),
    )
    project.add_experiment(Experiment("exp", 1.0, [0.0, 1.0], [trace_data]))
    fake_gui = _RemapFakeGui(project)

    # Cellpose id 0 maps explicitly to stim 1, cellpose id 1 falls back to
    # identity (stim 1 too) - a collision that cannot be resolved.
    Gui._apply_cellpose_import_result(
        fake_gui,  # type: ignore[arg-type]
        _cellpose_result_with_mapping({1: 0}),
    )

    np.testing.assert_array_equal(trace_data.indices, np.array([4, 7], dtype=np.uint32))
    assert any(call["name"] == "warning" for call in qmessagebox_calls)
    warning_call = next(call for call in qmessagebox_calls if call["name"] == "warning")
    assert "exp" in warning_call["text"]


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
    assert gui.time_slider.relative_playback_speed == pytest.approx(0.25)
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

    # accept: the rebuild now succeeds by pruning the monitor / initial
    # condition that depended on the deleted Ce synapse (this previously raised
    # a "model rebuilding error" and forced a cancel).
    qtbot.mouseClick(
        modeleditor.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok),
        QtCore.Qt.MouseButton.LeftButton,
    )
    qtbot.waitUntil(lambda: not modeleditor.isVisible())
    qtbot.wait(delay_after_update_model_call_in_ms)

    # no error dialog was shown
    assert len(qmessagebox_calls) == 0

    # the edited model rebuilt without Ce (and its orphaned dependents), and the
    # network assembled cleanly
    edited = modeleditor.b2model
    assert "Ce" not in edited.synapses
    assert "Ce" not in edited.initials
    assert not any(m.get("source") == "Ce" for m in edited.monitors.values())
    assert "Ce" not in [obj.name for obj in edited.network.objects]
    gui.close()


def test_warn_if_cellpose_model_mismatch(qmessagebox_calls: list) -> None:
    project = default_project()  # model has 10 neurons

    class _FakeGui:
        def __init__(self, project: Project) -> None:
            self.project = project

        def _cellpose_neuron_count(self, imported: CellposeImportResult | None) -> int:
            return Gui._cellpose_neuron_count(self, imported)  # type: ignore[arg-type]

    fake = _FakeGui(project)

    def _cellpose_with(n_cells: int) -> CellposeImportResult:
        return CellposeImportResult(
            image_path=Path("dummy.png"),
            segmentation_array=np.zeros((2, 2), dtype=np.int32),
            overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
            trace_data=None,
            spike_data=None,
            original_cellpose_ids=np.arange(n_cells, dtype=np.int32),
            stimulus_to_cellpose_mapping=None,
        )

    # no cellpose data -> no warning
    project.cellpose_data = None
    Gui._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 0

    # matching counts (10 cells, 10 neurons) -> no warning
    project.cellpose_data = _cellpose_with(10)
    Gui._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 0

    # mismatch (25 cells vs 10 neurons) -> warning
    project.cellpose_data = _cellpose_with(25)
    Gui._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "mismatch" in qmessagebox_calls[0]["title"].lower()
    assert "10" in qmessagebox_calls[0]["text"]
    assert "25" in qmessagebox_calls[0]["text"]


def test_brian2_busy_notifies_dynamic_input(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible)
    gui._io_worker = object()  # type: ignore[assignment]  # in-flight IO task
    dropped: list[str] = []
    monkeypatch.setattr(
        gui.stim_editor,
        "dynamic_input_simulation_failed",
        lambda message: dropped.append(message),
    )

    gui._do_brian2_simulation_threaded(0.1)

    # The dropped request is reported to the designer instead of stalling it.
    assert dropped
    gui._io_worker = None
    gui.close()


def test_toolbar_progress_does_not_auto_hide_at_maximum(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible)
    pd = gui.progress_dialog
    pd.setRange(0, 100)
    pd.setValue(50)
    pd.show()
    qtbot.wait(10)  # let the deferred show settle (offscreen QPA)

    # Reaching the maximum must not hide the bar mid-operation.
    pd.setValue(100)
    assert pd.isVisible()
    assert pd.value() == 100

    # Teardown still hides it explicitly.
    pd.reset()
    assert pd.isHidden()
    gui.close()


def test_gui_export_dialog_opens_on_the_current_input_array(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project
) -> None:
    """#588: the stimulation export dialog preselects the input array open
    in the editor, not the first one created."""
    import neuroseq.gui as gui_module

    test_project.add_stim_array(
        StimArray(
            "second_stim",
            StimType.Excitation,
            np.zeros((100, 3), dtype=np.float32),
            "nS",
            0.001,
        )
    )
    gui = Gui()
    qtbot.addWidget(gui)
    gui.load_project(test_project)
    gui.stim_editor.cmb_stim.setCurrentText("second_stim")
    opened: list[str] = []

    class RecordingDialog(gui_module.StimulationGeneratorDialog):
        def exec(self) -> int:
            opened.append(self.stim_combo.currentText())
            return 0

    monkeypatch.setattr(gui_module, "StimulationGeneratorDialog", RecordingDialog)

    gui.open_stimulation_generator_dialog()

    assert opened == ["second_stim"]
    gui.close()
