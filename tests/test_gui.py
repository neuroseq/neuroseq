from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import MethodType
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
import neuroseq.device_control.syscon2_panel as syscon2_panel_module
import neuroseq.export.stimulus_export_dialog as stimulus_export_dialog_module
import neuroseq.stimulation.stim
import neuroseq.stimulation.stim_editor
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.data_manager_dialog import DataManagerDialog
from neuroseq.example_projects import default_project
from neuroseq.example_projects import example_projects
from neuroseq.gui import Gui
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.model_editor import EditableSynapseTree
from neuroseq.model_editor import ModelEditorDialog
from neuroseq.project import Project
from neuroseq.project_workspace import DetachedPanelWindow
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.project_workspace import ToolbarProgressWidget
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.dynamic_input.dynamic_input_controller import (
    DynamicInputController,
)
from neuroseq.stimulation.static_input.static_input_controller import (
    StaticInputController,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_editor import DrawMode
from neuroseq.stimulation.stim_editor import StimEditor
from neuroseq.utils import InteractMode
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisType
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.visualization.data_widget import DataWidget
from neuroseq.visualization.trace_widget import TraceWidget
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _wait_for_brian2_simulation(qtbot: QtBot, gui: ProjectWorkspace) -> None:
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)


@pytest.mark.parametrize(
    ("handler", "setting", "label"),
    [
        (
            "choose_max_ram_synapse_plot_matrix",
            "max_ram_synapse_plot_matrix",
            "synapse matrix plot",
        ),
        ("choose_max_ram_trace_plot_matrix", "max_ram_trace_plot_matrix", "trace plot"),
    ],
)
@pytest.mark.parametrize(
    ("text", "accepted", "expected", "error"),
    [
        ("512", True, 512, False),
        ("0", True, 0, False),
        ("-1", True, 0, False),
        ("invalid", True, 256, True),
        ("512", False, 256, False),
    ],
)
def test_plot_ram_limit_dialog(
    monkeypatch, handler, setting, label, text, accepted, expected, error
):
    from unittest.mock import Mock

    settings = SimpleNamespace(
        max_ram_synapse_plot_matrix=256, max_ram_trace_plot_matrix=256
    )
    refresh = Mock()
    window = SimpleNamespace(settings=settings, _update_data=refresh)
    window._choose_plot_ram_limit = lambda *args: (
        ProjectWorkspace._choose_plot_ram_limit(window, *args)
    )
    chooser = Mock(return_value=(text, accepted))
    critical = Mock()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", chooser)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)

    getattr(ProjectWorkspace, handler)(window)

    chooser.assert_called_once_with(
        window,
        "RAM limit",
        f"RAM limit in MB for {label} (0 for no limit)",
        QtWidgets.QLineEdit.EchoMode.Normal,
        "256",
    )
    assert getattr(settings, setting) == expected
    other_setting = (
        "max_ram_trace_plot_matrix"
        if setting == "max_ram_synapse_plot_matrix"
        else "max_ram_synapse_plot_matrix"
    )
    assert getattr(settings, other_setting) == 256
    if accepted and not error:
        refresh.assert_called_once_with(keep_undo_history=True)
    else:
        refresh.assert_not_called()
    assert critical.called == error


def test_gui_click_exit(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
    gui.btn_exit_clicked()
    qtbot.waitUntil(lambda: not gui.isVisible(), timeout=WAIT_TIMEOUT_MS)


def _find_action(menu: QtWidgets.QMenu, text: str) -> QtWidgets.QAction:
    return next(a for a in menu.actions() if a.text().replace("&", "") == text)


def test_syscon2_panel_targets_drive_spatial_footprint_crosses(qtbot: QtBot) -> None:
    """Feature (#582): whichever sequence SysCon2Panel last sent to SysCon2
    should mark its targeted neurons with a cross in the Spatial Footprints
    panel - wired via the shared SpatialFootprintState, not a direct
    reference between the two widgets."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    gui.syscon2_panel.targets_changed.emit(frozenset({2, 5}))

    assert gui._spatial_footprint_state.targeted_stim_ids == frozenset({2, 5})


def test_gui_open_fiji_and_cellpose_buttons(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    import neuroseq.external_tools as external_tools

    gui = ProjectWorkspace()
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


def test_tools_menu_holds_fiji_cellpose_and_change_fiji_path(
    qtbot: QtBot,
) -> None:
    """Open Fiji / Open Cellpose / Change Fiji Path moved out of the Data

    toolbar and Settings menu into their own Tools menu, between Settings
    and Help (issue #535).
    """
    gui = ProjectWorkspace()
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


def test_stim_array_ops_button_sits_next_to_stim_data_combobox_with_full_menu(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
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


def test_view_toolbar_has_separator_before_toggle_selection_text(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    actions = gui.toolbar_view.actions()
    index = next(
        i
        for i, action in enumerate(actions)
        if action.text() == "Toggle Selection Text"
    )
    assert actions[index - 1].isSeparator()


def test_show_all_stim_arrays_action_is_in_the_view_toolbar(qtbot: QtBot) -> None:
    """Moved out of the stim editor's left toolbar (issue #535); reused by

    reference rather than duplicated, so it stays the same QAction that
    toggle_show_all_stim_arrays() checks/toggles.
    """
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    action = gui.stim_editor.action_show_all_stim_arrays
    assert action in gui.toolbar_view.actions()
    assert action in gui.menu_view.actions()
    assert action.isChecked()


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
        StaticInputController,
        "open_pattern_designer",
        lambda self: static_calls.append(1),
    )
    monkeypatch.setattr(
        DynamicInputController,
        "open_dynamic_input_designer",
        lambda self: dynamic_calls.append(1),
    )
    gui = ProjectWorkspace()
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
    design_actions = [
        action
        for action in gui.stim_editor.context_menu.actions()
        if action.text() in {"Design static input", "Design dynamic input"}
    ]
    assert [action.text() for action in design_actions] == [
        "Design static input",
        "Design dynamic input",
    ]
    for action in design_actions:
        action.trigger()
    assert static_calls == [1, 1]
    assert dynamic_calls == [1, 1]


def test_file_toolbar_groups_actions_with_separators(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    actions = gui.toolbar_file.actions()
    texts = [a.text().replace("&", "") for a in actions]

    def index_of(text: str) -> int:
        return texts.index(text)

    # {New, Open Project, Import from Project} | {Save Project} |
    # {Edit Model} | {Generate Synthetic Data, Manage Data}
    assert actions[index_of("Save Project") - 1].isSeparator()
    assert actions[index_of("Edit Model") - 1].isSeparator()
    assert actions[index_of("Generate Synthetic Data...") - 1].isSeparator()
    assert texts[index_of("Generate Synthetic Data...") + 1] == "Manage Data"


def test_playback_toolbar_separates_speed_from_transport_controls(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    actions = gui.toolbar_playback.actions()
    index = next(
        i for i, a in enumerate(actions) if a.text().replace("&", "") == "Skip to Start"
    )
    assert actions[index - 1].isSeparator()


def test_data_toolbar_groups_dataset_pairs_with_separators(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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


def test_manage_data_action_sits_next_to_generate_synthetic_data(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    actions = gui.menu_file.actions()
    generate_index = next(
        i
        for i, action in enumerate(actions)
        if action.text().replace("&", "") == "Generate Synthetic Data..."
    )
    assert actions[generate_index + 1].text().replace("&", "") == "Manage Data"
    actions[generate_index + 1].trigger()
    dialog = gui.findChild(DataManagerDialog)
    assert dialog is not None
    assert dialog.windowTitle() == "Manage Data"


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
    gui = ProjectWorkspace()
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


def test_trim_to_mapped_neurons_action_triggers_the_trim_handler(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0})
    gui.update_project(project)
    assert gui.stim_editor.action_trim_to_mapped_neurons.isEnabled()
    calls: list[int] = []
    gui.stim_editor.trim_to_mapped_neurons_requested.connect(lambda: calls.append(1))

    gui.stim_editor.action_trim_to_mapped_neurons.trigger()

    assert calls == [1]


def test_trim_to_mapped_neurons_reports_nothing_to_trim_without_mapping(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui.update_project(project)

    gui.btn_trim_to_mapped_neurons_clicked()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "Nothing to trim" in qmessagebox_calls[0]["title"]


def test_trim_to_mapped_neurons_reports_nothing_to_trim_when_fully_mapped(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = Project()
    project.cellpose_data = _make_gui_cellpose_result(mapping={0: 0, 1: 1, 2: 2})
    gui.update_project(project)

    gui.btn_trim_to_mapped_neurons_clicked()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "Nothing to trim" in qmessagebox_calls[0]["title"]


def test_trim_to_mapped_neurons_cancelled_confirm_makes_no_changes(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    gui = ProjectWorkspace()
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


def test_trim_to_mapped_neurons_confirmed_trims_and_forces_save_as(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    gui = ProjectWorkspace()
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
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)

    assert gui.project.stim_arrays["stim"].values.shape == (5, 2)
    assert new_filename.exists()
    assert gui.project.filepath == new_filename


def _add_detach_panel_test_data(
    gui: ProjectWorkspace,
) -> tuple[TraceData, SpikeData, SynapseData]:
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
    gui = ProjectWorkspace()
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


def test_detach_panel_hides_original_and_shows_fresh_window(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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


def test_reshowing_a_plainly_detached_panel_closes_its_detached_window(
    qtbot: QtBot,
) -> None:
    """A plain "Detach panel" (not "Detach copy") hides the docked panel;
    checking "Show X" again should bring it back into the dock and close
    the now-redundant floating window, not leave both visible."""
    gui = ProjectWorkspace()
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


def test_reshowing_a_copy_detached_panel_leaves_its_window_open(
    qtbot: QtBot,
) -> None:
    """ "Detach copy of panel" never unchecks "Show X" in the first place,
    but toggling it off and back on should not touch the independent
    copy window either."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel(gui.trace_widget, keep_original=True)
    window = gui._detached_panel_windows[-1]
    action = gui._view_actions[gui.trace_widget]
    assert action.isChecked()

    action.trigger()  # uncheck
    action.trigger()  # re-check

    assert window.isVisible()
    assert window in gui._detached_panel_windows


def test_detach_copy_keeps_original_visible(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel(gui.trace_widget, keep_original=True)
    window = gui._detached_panel_windows[-1]

    assert window.windowTitle() == "Traces - Section / Project"
    assert gui._view_actions[gui.trace_widget].isChecked()
    assert gui.trace_widget.isVisible()
    assert window.isVisible()


def test_detached_trace_copy_shares_state_with_original(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel(gui.trace_widget, keep_original=True)
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


def test_hovering_a_neuron_in_trace_or_stim_highlights_it_in_spatial_footprints(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)

    assert gui._spatial_footprint_state.hovered_stim_id is None

    gui.trace_widget.neuron_hovered.emit(1)
    assert gui._spatial_footprint_state.hovered_stim_id == 1

    gui.stim_editor.stim_widget.neuron_hovered.emit(2)
    assert gui._spatial_footprint_state.hovered_stim_id == 2

    gui.trace_widget.neuron_hovered.emit(None)
    assert gui._spatial_footprint_state.hovered_stim_id is None


def test_hovering_in_a_detached_trace_copy_also_highlights_the_neuron(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.trace_widget)

    gui.detach_panel(gui.trace_widget, keep_original=True)
    window = gui._detached_panel_windows[-1]
    detached_widget = window._widget

    detached_widget.neuron_hovered.emit(0)

    assert gui._spatial_footprint_state.hovered_stim_id == 0


def test_detached_spatial_footprint_copy_shares_state_with_original(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.spatial_footprint_widget)

    gui.detach_panel(gui.spatial_footprint_widget, keep_original=True)
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


def test_neuron_click_rebuilds_stim_and_trace_layout_only_once(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
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


def test_deselecting_a_cell_restores_the_vertical_zoom(qtbot: QtBot) -> None:
    """Issue #604: selecting a cell fits the plots to its rows; deselecting
    it must go back to the earlier vertical zoom, not zoom all the way out."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    controllers = [gui.controller, gui.trace_controller, gui.spike_controller]
    for controller in controllers:
        controller.set_vertical_view((1.5, 1.25))
    zoomed_in = [controller.vertical_view() for controller in controllers]
    assert zoomed_in[1] == pytest.approx((1.5, 1.25))

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    assert gui.trace_controller.vertical_view() != pytest.approx(zoomed_in[1])

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    for controller, view in zip(controllers, zoomed_in, strict=True):
        assert controller.vertical_view() == pytest.approx(view)


def test_right_click_on_a_cell_centres_its_trace(qtbot: QtBot) -> None:
    """Issue #603: right-clicking a cell scrolls the traces to its neuron
    and keeps the vertical zoom, without hiding any trace."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui.trace_controller.set_vertical_view((1.0, 0.0))

    gui.spatial_footprint_widget.overlay_widget.neuron_right_clicked.emit(2)

    assert gui.trace_controller.vertical_view() == pytest.approx((1.0, 2.0))
    assert not any(gui.trace_widget.is_neuron_greyed(n) for n in range(3))


def test_switching_project_ignores_the_previous_projects_data_names(
    qtbot: QtBot,
) -> None:
    """Issue #595: closing a tab or opening a project re-windows the shown
    data before the data selectors are refilled, so they can still name the
    previous project's simulation monitors, which this project lacks."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    # As while update_project() runs: the selectors are not refilled yet,
    # and their own change handlers have not run.
    for combo in (gui.cmb_trace, gui.cmb_spike, gui.cmb_synapse):
        combo.blockSignals(True)
        combo.addItem("Simulation :: P")
        combo.setCurrentText("Simulation :: P")
        combo.blockSignals(False)
    gui.cmb_data.blockSignals(True)
    gui.cmb_data.addItem("Simulation :: P [spike]")
    gui.cmb_data.setCurrentText("Simulation :: P [spike]")
    gui.cmb_data.blockSignals(False)
    gui._last_sim_timeline_secs = None

    gui._rewindow_simulation_views()  # raised KeyError: 'Simulation :: P'


def test_cell_selection_applies_to_every_trace_view(qtbot: QtBot) -> None:
    """Issue #596: a selected cell also filters the universal data widget,
    and choosing data again keeps the selection in every view."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.data_widget)
    gui._apply_data_selection("Experiment :: trace [trace]")

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    views = [gui.trace_widget, gui.spike_widget, gui.data_widget.trace_widget]
    for view in views:
        assert [n for n in range(3) if view.is_neuron_greyed(n)] == [0, 2]

    gui.set_neuron(gui.cmb_trace.currentText())
    gui.set_spike(gui.cmb_spike.currentText())
    gui._apply_data_selection("Experiment :: spikes [spike]")
    views[2] = gui.data_widget.spike_widget
    for view in views:
        assert [n for n in range(3) if view.is_neuron_greyed(n)] == [0, 2]

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)  # deselect
    for view in views:
        assert not any(view.is_neuron_greyed(n) for n in range(3))


def test_detached_console_copy_close_button_closes_its_own_window_only(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui)
    gui._auto_show_widget(gui.console_widget)

    gui.detach_panel(gui.console_widget, keep_original=True)
    window = gui._detached_panel_windows[-1]

    window._widget.close_requested.emit()

    assert not window.isVisible()
    assert gui.console_widget.isVisible()
    assert gui._view_actions[gui.console_widget].isChecked()


def test_detach_context_menu_installed_on_panel_canvases(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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


def test_detached_chat_continues_the_docked_conversation(qtbot: QtBot) -> None:
    from neuroseq.llm.backend import Message
    from neuroseq.llm.backend import Role

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.chat_widget._history = [Message(Role.USER, "hi"), Message(Role.ASSISTANT, "yo")]

    detached = gui._build_fresh_chat_widget()
    qtbot.addWidget(detached)

    assert detached._history == gui.chat_widget._history
    assert detached._project is gui.project
    assert "yo" in detached.history_view.toPlainText()


def test_detach_custom_analysis_before_anything_analyzed_shows_message(
    qtbot: QtBot,
    qmessagebox_calls: list,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    gui.detach_panel(gui.custom_analysis_widget, keep_original=False)

    assert qmessagebox_calls[-1]["name"] == "information"
    assert "nothing to detach" in qmessagebox_calls[-1]["title"].lower()
    assert gui.findChildren(DetachedPanelWindow) == []
    gui._close_confirmed = True


def test_detach_custom_analysis_after_analysis_seeds_from_last_shown(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
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


def test_custom_analysis_popup_title_includes_originating_tab(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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


def test_closing_the_main_window_closes_static_input_designer(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.update_project(default_project())
    gui.static_input_controller.open_pattern_designer()
    dialog = gui.static_input_controller._pattern_dialog
    assert dialog is not None
    assert dialog.isVisible()

    gui._close_confirmed = True
    gui.close()

    assert not dialog.isVisible()
    assert gui.static_input_controller._pattern_dialog is None


def test_replacing_the_project_discards_static_input_designer(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.update_project(default_project())
    gui.static_input_controller.open_pattern_designer()
    dialog = gui.static_input_controller._pattern_dialog
    assert dialog is not None

    gui.update_project(default_project())

    assert not dialog.isVisible()
    assert gui.static_input_controller._pattern_dialog is None
    gui._close_confirmed = True


def test_closing_the_main_window_closes_dynamic_input_routine_dialogs(
    qtbot: QtBot,
) -> None:
    """The routine dialogs (and their own plot/heatmap popups) are created
    without a Qt parent - see DynamicInputController._show_routine_dialog's docstring -
    so closing the main window would otherwise leave them open as orphaned
    windows: findChildren() never finds them, since they were never made
    Qt children of the main window in the first place."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.update_project(default_project())

    gui.dynamic_input_controller.open_dynamic_input_designer()
    selector = gui.dynamic_input_controller._routine_selector_dialog
    assert selector is not None
    selector.cmb_routine.setCurrentText(DynamicInputController.RHEOBASE_ROUTINE)
    selector._on_continue()
    dialog = gui.dynamic_input_controller._rheobase_dialog
    assert dialog is not None
    assert dialog.isVisible()

    gui._close_confirmed = True
    gui.close()

    assert not dialog.isVisible()
    assert gui.dynamic_input_controller._rheobase_dialog is None


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
    tab1_project = _make_timed_project(500, dt=0.01)  # 5.0s

    gui._add_workspace(tab1_project)
    assert gui.time_slider.t_ms == 0  # a freshly opened tab has no history

    gui.time_slider.set_current_time_ms(3000)
    gui.project_tabs.setCurrentIndex(0)
    assert gui.time_slider.t_ms == 0  # tab0 was never scrubbed

    gui.time_slider.set_current_time_ms(200)
    gui.project_tabs.setCurrentIndex(1)
    assert gui.time_slider.t_ms == 3000

    gui.project_tabs.setCurrentIndex(0)
    assert gui.time_slider.t_ms == 200


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

    gui._add_workspace(tab1_project)

    assert gui.time_slider.t_max_ms == 5000


def test_closing_the_only_dirty_tab_prompts_to_save_then_opens_example_project(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    assert len(gui.workspaces) == 1
    gui.project.mark_dirty()

    warn_calls: list[tuple] = []

    def _fake_warning(*args, **kwargs):
        warn_calls.append(args)
        return QtWidgets.QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)

    gui._close_tab(0)

    assert len(warn_calls) == 1  # asked about saving before replacing the tab
    assert len(gui.workspaces) == 1
    assert gui.project.project_name == "Getting Started Project"


def test_closing_the_only_clean_tab_opens_example_project_without_prompting(
    monkeypatch: MonkeyPatch, qtbot: QtBot
) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    assert len(gui.workspaces) == 1
    assert not gui.project.dirty

    warn_calls: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warn_calls.append(args),
    )

    gui._close_tab(0)

    assert warn_calls == []  # nothing unsaved, so no prompt was needed
    assert len(gui.workspaces) == 1
    assert gui.project.project_name == "Getting Started Project"


def test_closing_a_tab_shows_the_next_projects_own_data(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Issues #612/#621: after closing a tab, the window showed the closed
    project's imaging, cells, traces and input arrays on top of the project
    that took its place."""
    import tifffile

    gui = Gui()
    qtbot.addWidget(gui)
    _add_detach_panel_test_data(gui.active_workspace)
    movie = tmp_path / "movie.tif"
    tifffile.imwrite(movie, np.zeros((3, 4, 4), dtype=np.uint16))
    gui.project.add_calcium_video([movie], 10.0)
    gui._refresh_calcium_video_widgets()
    closed = gui.project
    gui._add_workspace(Project())
    blank = gui.project
    gui._add_workspace(Project())

    gui.project_tabs.setCurrentIndex(0)
    gui._remove_workspace(gui.workspaces[0])

    assert gui.active_workspace.project is blank
    assert gui.project_tabs.currentIndex() == 0
    assert gui.calcium_video_widget._paths == []
    assert gui.spatial_footprint_widget._cellpose_data is None
    assert gui.trace_widget.data is None
    assert gui.cmb_trace.count() == 0
    assert blank.calcium_videos == [] and blank.cellpose_data is None
    assert all(w.project is not closed for w in gui.workspaces)


def test_switching_project_tabs_restores_each_tabs_own_spatial_footprint_selection(
    qtbot: QtBot,
) -> None:
    """Each workspace keeps its own selection when other tabs become active."""
    gui = Gui()
    qtbot.addWidget(gui)
    tab0_project = gui.project
    tab0_project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui.update_project(tab0_project)

    tab1_project = Project()
    tab1_project.cellpose_data = _make_gui_cellpose_result(mapping=None)
    gui._add_workspace(tab1_project)
    assert gui._spatial_footprint_state.selected_cellpose_ids == set()

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(2)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {2}

    gui.project_tabs.setCurrentIndex(0)
    assert gui._spatial_footprint_state.selected_cellpose_ids == set()

    gui.spatial_footprint_widget.overlay_widget.neuron_clicked.emit(1)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {1}

    gui.project_tabs.setCurrentIndex(1)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {2}  # restored

    gui.project_tabs.setCurrentIndex(0)
    assert gui._spatial_footprint_state.selected_cellpose_ids == {1}  # restored


def test_closing_gui_with_open_detached_windows_does_not_crash(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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
    gui = ProjectWorkspace()
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
    gui = ProjectWorkspace()
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


def test_toggle_stim_trace_overlay_respects_manually_hidden_stim_panel(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
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


def test_stim_trace_overlay_updates_when_selected_stim_changes(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
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


def _overlay_values(widget) -> np.ndarray:
    assert widget._stim_overlay_graphic is not None
    return np.asarray(widget._stim_overlay_graphic.data.value)


def _all_data_trace_overlay_gui(qtbot: QtBot) -> tuple[ProjectWorkspace, Project]:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    project.stim_arrays["stim"].values[2:4, 1] = 3.0
    gui.update_project(project)
    gui._apply_trace_selection("Experiment :: traces")
    gui.cmb_data.setCurrentText("Experiment :: traces [trace]")
    gui._view_actions[gui.data_widget].trigger()  # show the All data panel
    assert gui.data_widget.trace_widget.n_neurons == 3
    return gui, project


def test_all_data_trace_overlay_toggles_independently(qtbot: QtBot) -> None:
    """Issue #663: the All data panel's traces get their own input overlay."""
    gui, project = _all_data_trace_overlay_gui(qtbot)
    data_traces = gui.data_widget.trace_widget
    assert data_traces._stim_overlay_graphic is None

    gui.action_overlay_stim_on_data_traces.trigger()

    assert data_traces._stim_overlay_array is project.stim_arrays["stim"]
    rendered = _overlay_values(data_traces)
    assert rendered.shape == (3, 10)
    assert np.all(rendered[1, 2:4] == 3.0)
    assert gui.trace_widget._stim_overlay_graphic is None
    assert not gui.action_overlay_stim_on_traces.isChecked()
    # Unlike the Neuron data toggle, the Stimuli panel stays put.
    assert gui.stim_editor.isVisible()

    gui.action_overlay_stim_on_traces.trigger()
    gui.action_overlay_stim_on_data_traces.trigger()

    assert gui.trace_widget._stim_overlay_graphic is not None
    assert data_traces._stim_overlay_graphic is None


def test_trace_view_context_menu_toggles_overlay_and_syncs_toolbar(
    qtbot: QtBot,
) -> None:
    gui, _ = _all_data_trace_overlay_gui(qtbot)
    data_traces = gui.data_widget.trace_widget

    data_traces._rebuild_custom_analysis_menu()
    overlay_action = next(
        a
        for a in data_traces.context_menu.actions()
        if a.text() == "Overlay input on traces"
    )
    assert not overlay_action.isChecked()
    overlay_action.trigger()

    assert data_traces.stim_overlay_enabled
    assert data_traces._stim_overlay_graphic is not None
    assert gui.action_overlay_stim_on_data_traces.isChecked()
    assert not gui.action_overlay_stim_on_traces.isChecked()

    gui.trace_widget._rebuild_custom_analysis_menu()
    main_action = next(
        a
        for a in gui.trace_widget.context_menu.actions()
        if a.text() == "Overlay input on traces"
    )
    main_action.trigger()
    assert gui.action_overlay_stim_on_traces.isChecked()
    assert gui.trace_widget._stim_overlay_graphic is not None
    assert not gui.stim_editor.isVisible()


def test_all_data_trace_overlay_follows_stim_edits_and_undo(qtbot: QtBot) -> None:
    gui, project = _all_data_trace_overlay_gui(qtbot)
    gui.action_overlay_stim_on_data_traces.trigger()
    data_traces = gui.data_widget.trace_widget

    new_values = np.zeros((10, 3), dtype=np.float32)
    new_values[5:7, 0] = 2.0
    project.replace_stim_values("stim", new_values)

    rendered = _overlay_values(data_traces)
    assert np.all(rendered[0, 5:7] == 2.0)
    assert np.all(rendered[1] == 0.0)

    project.history.undo()

    rendered = _overlay_values(data_traces)
    assert np.all(rendered[0] == 0.0)
    assert np.all(rendered[1, 2:4] == 3.0)


def test_all_data_trace_overlay_follows_selected_stim(qtbot: QtBot) -> None:
    gui, project = _all_data_trace_overlay_gui(qtbot)
    project.add_stim_array(
        StimArray(
            "stim2",
            StimType.Excitation,
            np.full((10, 3), 5.0, dtype=np.float32),
            "mV",
            0.001,
        )
    )
    gui.update_project(project)
    gui.cmb_data.setCurrentText("Experiment :: traces [trace]")
    gui.action_overlay_stim_on_data_traces.trigger()

    gui.cmb_stim.setCurrentText("stim2")

    assert (
        gui.data_widget.trace_widget._stim_overlay_array is project.stim_arrays["stim2"]
    )
    assert np.all(_overlay_values(gui.data_widget.trace_widget) == 5.0)


def test_detached_trace_views_keep_their_own_input_overlay(qtbot: QtBot) -> None:
    gui, project = _all_data_trace_overlay_gui(qtbot)
    gui.action_overlay_stim_on_data_traces.trigger()

    gui.detach_panel(gui.data_widget, keep_original=True)
    data_copy = gui._detached_panel_windows[-1].panel_widget
    assert isinstance(data_copy, DataWidget)
    assert data_copy.trace_widget.stim_overlay_enabled
    assert data_copy.trace_widget._stim_overlay_graphic is not None

    gui.detach_panel(gui.trace_widget, keep_original=True)
    trace_copy = gui._detached_panel_windows[-1].panel_widget
    assert isinstance(trace_copy, TraceWidget)
    assert not trace_copy.stim_overlay_enabled

    trace_copy.set_stim_overlay_enabled(True)
    data_copy.trace_widget.set_stim_overlay_enabled(False)
    assert trace_copy._stim_overlay_graphic is not None
    assert data_copy.trace_widget._stim_overlay_graphic is None
    assert gui.data_widget.trace_widget._stim_overlay_graphic is not None

    new_values = np.zeros((10, 3), dtype=np.float32)
    new_values[0:2, 2] = 4.0
    project.replace_stim_values("stim", new_values)

    assert np.all(_overlay_values(trace_copy)[2, 0:2] == 4.0)


def test_time_slider_highlights_stimulated_neurons_in_spatial_footprints(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project, _ = _stim_trace_overlay_project()
    project.stim_arrays["stim"].values[5, 1] = 3.0  # neuron 1 stimulated at t=5ms
    project.cellpose_data = _make_gui_cellpose_result()
    gui.update_project(project)

    gui.time_slider.set_current_time_ms(5)
    assert gui._spatial_footprint_state.active_stim_ids == frozenset({1})

    gui.time_slider.set_current_time_ms(0)
    assert gui._spatial_footprint_state.active_stim_ids == frozenset()


def test_switching_current_stim_array_refreshes_active_stim_highlight(
    qtbot: QtBot,
) -> None:
    gui = ProjectWorkspace()
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

    monkeypatch.setattr("neuroseq.project_workspace.CellposeImportDialog", _FakeDialog)
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    gui.btn_import_cellpose_data_clicked()

    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()
    assert gui.spatial_footprint_widget.isVisible()


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
        batch_imports: list[Any] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "neuroseq.importers.tif_import_dialog.TifImportDialog", _FakeDialog
    )
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    gui.btn_import_tif_clicked()

    assert gui.project.cellpose_data is not None
    assert gui.project.cellpose_data.image_path == imported.image_path
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()
    assert gui.spatial_footprint_widget.isVisible()


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
        batch_imports: list[Any] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "neuroseq.importers.tif_import_dialog.TifImportDialog", _FakeDialog
    )

    class _FakeGui:
        def __init__(self) -> None:
            self.project = Project()
            self.project.cellpose_data = existing
            self.last_cellpose_import = existing
            self.spatial_footprint_widget = SimpleNamespace()
            self.import_folders = None

        def _append_experimental_data(self, experiment: Experiment) -> None:
            appended.append(experiment)

        def _apply_cellpose_import_result(
            self,
            imported: CellposeImportResult,
        ) -> list[str]:
            raise AssertionError(f"unexpected Cellpose propagation: {imported}")

    fake_gui = _FakeGui()

    ProjectWorkspace.btn_import_tif_clicked(fake_gui)  # type: ignore[arg-type]

    assert len(appended) == 1
    assert fake_gui.project.cellpose_data is existing
    assert fake_gui.last_cellpose_import is existing
    assert fake_gui.project.cellpose_data.stimulus_to_cellpose_mapping == {
        0: 2,
        1: 1,
        2: 0,
    }


def test_batch_tif_import_adds_one_data_group_per_file(
    qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """#643: each imported file becomes its own data group (named after the
    file, made unique) with its own movie; failed files add nothing."""
    from neuroseq.importers.tif_import_dialog import TifBatchImport

    def _trace() -> TraceData:
        return TraceData(
            "Simply averaged ROI traces",
            np.zeros((2, 30), dtype=np.float32),
            np.arange(2, dtype=np.uint32),
            "fluorescence",
            1 / 30.0,
        )

    class _FakeDialog:
        experiment_name = "unused"
        experiment_duration = 0.0
        imported_data: list[Any] = []
        new_cellpose_result = None
        imported_tif_paths: list[Any] = []
        imported_fps = 30.0
        imported_channels = 1
        imported_functional_channel = 1
        batch_imports = [
            TifBatchImport(tmp_path / "a.tif", "imported", [_trace()], 1.0),
            TifBatchImport(tmp_path / "b.tif", "failed", message="bad"),
            TifBatchImport(tmp_path / "sub" / "a.tif", "imported", [_trace()], 1.0),
        ]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "neuroseq.importers.tif_import_dialog.TifImportDialog", _FakeDialog
    )
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    before = set(gui.project.experiments)

    gui.btn_import_tif_clicked()

    added = set(gui.project.experiments) - before
    assert added == {"a", "a (2)"}
    videos = {video["name"]: video["paths"] for video in gui.project.calcium_videos}
    assert videos["a"] == [str(tmp_path / "a.tif")]
    assert videos["a (2)"] == [str(tmp_path / "sub" / "a.tif")]
    assert gui.cmb_trace.currentText().startswith("a (2) ::")
    gui.project.history.undo()
    assert set(gui.project.experiments) - before == {"a"}


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

    monkeypatch.setattr("neuroseq.project_workspace.CellposeImportDialog", _FakeDialog)
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)

    gui.spatial_footprint_widget.import_button.click()

    assert gui.project.cellpose_data is not None
    assert gui.project.cellpose_data.image_path == imported.image_path
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is not None
    assert gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert not gui.spatial_footprint_widget.overlay_widget.isHidden()


def test_deleting_cellpose_data_clears_spatial_footprint_panel(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.project.cellpose_data = _make_gui_cellpose_result()
    gui.spatial_footprint_widget.set_data(gui.project.cellpose_data)

    gui.project.update_limits()
    gui.project.dirty = False

    gui._delete_cellpose_data()

    assert gui.project.cellpose_data is None
    # Removing the data is an unsaved project change.
    assert gui.project.dirty
    assert gui.spatial_footprint_widget.overlay_widget.overlay_image is None
    assert gui.spatial_footprint_widget.overlay_widget.segmentation is None
    assert not gui.spatial_footprint_widget.empty_state_widget.isHidden()
    assert gui.spatial_footprint_widget.overlay_widget.isHidden()


def test_gui_choose_fiji_path_forces_prompt(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    import neuroseq.fiji as fiji_module

    gui = ProjectWorkspace()
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


def test_gui_theme_menu_toggles_theme_and_persists(
    qtbot: QtBot, tmp_config_dir
) -> None:
    from neuroseq import theme as theme_module

    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

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


def test_gui_applies_persisted_theme_on_startup(qtbot: QtBot, tmp_config_dir) -> None:
    from neuroseq import theme as theme_module
    from neuroseq.gui_settings import GuiSettings

    # Persist a dark theme to the (temporary) config before launching.
    GuiSettings().theme = theme_module.DARK

    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

    assert gui.settings.theme == theme_module.DARK
    assert gui.trace_widget._theme == theme_module.DARK
    assert gui._theme_actions[theme_module.DARK].isChecked()


def test_gui_starts_in_retro_theme_without_config(qtbot: QtBot, tmp_config_dir) -> None:
    from neuroseq import theme as theme_module

    # No config has been written, so the GUI must come up in the default theme.
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

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
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
    gui.trace_widget.setVisible(True)
    gui.spike_widget.setVisible(True)
    gui.data_widget.setVisible(True)
    qtbot.waitUntil(lambda: gui.spike_widget.isVisible(), timeout=WAIT_TIMEOUT_MS)

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
    assert next(iter(widths)) < gui._main_scroll.widget().width()


def test_retro_time_slider_elapsed_side_is_green(qtbot: QtBot) -> None:
    from neuroseq import theme as theme_module

    stylesheet = theme_module.app_stylesheet(theme_module.RETRO)
    sub_page = stylesheet.split("QSlider::sub-page:horizontal")[1].split("}")[0]
    # The elapsed side of the groove is lit LCD green, not the dim groove color.
    assert theme_module._RETRO_LCD_FG in sub_page


def test_gui_open_example_project_menu_has_subsections(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

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


def test_gui_cellpose_import_action_moved_to_file_menu(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

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
    assert "&Output" in import_data_action_texts
    assert "&Trace" not in import_data_action_texts
    assert "Import cell &segmentation" not in import_data_action_texts
    assert "Cell &segmentation" not in import_data_action_texts


def test_gui_has_no_export_graphics_menu(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)

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


def test_gui_simulate_and_playback(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project, tmp_config_dir
) -> None:
    gui = ProjectWorkspace()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    assert gui.heatmap_widget.idx == 0
    assert "run" in gui.continue_simulation_action.text().lower()
    assert "continue" not in gui.continue_simulation_action.text().lower()
    simulation_time_s = 0.2
    # mock getText to return the desired simulation time
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: (f"{simulation_time_s}", True)
    )
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
    qtbot.waitUntil(lambda: gui.heatmap_widget.idx > 0, timeout=WAIT_TIMEOUT_MS)
    # pause and then continue playback
    gui.time_slider.toggle_pause()
    gui.time_slider.toggle_pause()
    qtbot.waitUntil(
        lambda: gui.heatmap_widget.idx == n_timesteps - 1,
        timeout=WAIT_TIMEOUT_MS,
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
    gui = ProjectWorkspace()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("1.0", True))

    gui.btn_continue_simulation_clicked()
    assert gui._background_job.worker is not None
    # cancel simulation on first progress update
    cancel_trigger = _MainThreadCancelTrigger(gui.progress_dialog)
    gui._background_job.worker.progress.connect(cancel_trigger.trigger)

    _wait_for_brian2_simulation(qtbot, gui)

    assert gui.project.b2_model.cancel_requested is True
    assert gui._background_job.worker is None
    # isHidden() (this widget's own state) is the reliable signal that
    # cancel()/hide() worked - see the comment in
    # test_simulation_progress_is_embedded_in_view_toolbar for why
    # isVisible() isn't asserted here.
    assert gui.progress_dialog.isHidden()
    assert QtWidgets.QApplication.overrideCursor() is None


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
    gui = ProjectWorkspace()
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


def test_brian2_simulation_finish_updates_only_its_workspace(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    test_project: Project,
) -> None:
    gui = Gui(test_project)
    qtbot.addWidget(gui)
    workspace = gui.active_workspace
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.2", True))
    workspace.btn_continue_simulation_clicked()
    other = gui._add_workspace(default_project())
    _wait_for_brian2_simulation(qtbot, workspace)
    assert float(test_project.b2_model.network.t) > 0
    assert workspace.cmb_trace.count() > 0
    assert gui.active_workspace is other
    assert other.cmb_trace.count() == 0
    test_project.dirty = False


def test_simulation_progress_is_embedded_in_view_toolbar(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible, timeout=WAIT_TIMEOUT_MS)

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
    qtbot.waitUntil(gui.progress_dialog.isVisible, timeout=WAIT_TIMEOUT_MS)

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


def test_gui_new(tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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


def test_create_new_project_calls_gc_collect(monkeypatch: MonkeyPatch) -> None:
    collect_calls = [0]
    sentinel_project = Project()
    loaded_project: list[Project] = []

    monkeypatch.setattr(
        "neuroseq.project_workspace.gc.collect",
        lambda: collect_calls.__setitem__(0, collect_calls[0] + 1),
    )
    monkeypatch.setattr(
        "neuroseq.project_workspace.default_project", lambda: sentinel_project
    )

    fake_gui: Any = SimpleNamespace(
        load_project=lambda project: loaded_project.append(project)
    )
    ProjectWorkspace._create_new_project(fake_gui)

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
        project = Project()
        project.dirty = True

        def fake_save(on_success: Any = None) -> bool:
            save_events.append("save")
            project.dirty = False
            if on_success is not None:
                on_success(None)
            return True

        fake: Any = SimpleNamespace(_save_project=fake_save, project=project)
        fake._prompt_save_before_action = (
            lambda title, message, proceed: ProjectWorkspace._prompt_save_before_action(
                fake, title, message, proceed
            )
        )
        return fake

    def proceed() -> None:
        proceeded[0] += 1

    # Cancel: neither save nor proceed.
    ProjectWorkspace._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert proceeded[0] == 0
    assert save_events == []

    # Discard: proceed immediately, without saving.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Discard
    ProjectWorkspace._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert proceeded[0] == 1
    assert save_events == []

    # Save: save first, then proceed via the on_success chain.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Save
    ProjectWorkspace._prompt_save_before_action(make_fake_gui(), "T", "M", proceed)
    assert save_events == ["save"]
    assert proceeded[0] == 2


def test_prompt_save_before_action_asks_again_after_edits_during_save(
    monkeypatch: MonkeyPatch,
) -> None:
    """An edit made while the save ran keeps the project dirty: ask again
    instead of proceeding and dropping it."""
    answers = [
        QtWidgets.QMessageBox.StandardButton.Save,
        QtWidgets.QMessageBox.StandardButton.Cancel,
    ]
    prompts: list[object] = []

    def _fake_warning(*a: object, **k: object) -> QtWidgets.QMessageBox.StandardButton:
        prompts.append(a)
        return answers.pop(0)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)
    project = Project()
    project.mark_dirty()

    def save_while_editing(on_success: Any = None) -> bool:
        change_count = project._change_count
        project.mark_dirty()  # edit lands while the save is writing
        if project._change_count == change_count:
            project.dirty = False
        on_success(None)
        return True

    fake: Any = SimpleNamespace(project=project, _save_project=save_while_editing)
    fake._prompt_save_before_action = (
        lambda title, message, proceed: ProjectWorkspace._prompt_save_before_action(
            fake, title, message, proceed
        )
    )
    proceeded = [0]

    ProjectWorkspace._prompt_save_before_action(
        fake, "T", "M", lambda: proceeded.__setitem__(0, 1)
    )

    assert len(prompts) == 2
    assert proceeded == [0]
    assert project.dirty


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

    ProjectWorkspace._prompt_save_before_action(
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
        project=dirty_project,
    )

    # Cancel: keep the app open.
    assert not ProjectWorkspace._prompt_save_before_close(fake_gui)
    assert not fake_gui._close_confirmed
    assert save_callbacks == []
    assert closed == [0]

    # Discard: close immediately.
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Discard
    assert ProjectWorkspace._prompt_save_before_close(fake_gui)
    assert fake_gui._close_confirmed
    assert save_callbacks == []
    assert closed == [0]

    # Save: block the current close, then close from the save success callback.
    fake_gui._close_confirmed = False
    answer_box["answer"] = QtWidgets.QMessageBox.StandardButton.Save
    assert not ProjectWorkspace._prompt_save_before_close(fake_gui)
    assert not fake_gui._close_confirmed
    assert len(save_callbacks) == 1
    assert closed == [0]

    # The callback closes again through the prompt instead of confirming
    # outright, so edits made during the save are asked about.
    save_callbacks[0](None)
    assert not fake_gui._close_confirmed
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
    # A standalone workspace owns one clean project.
    fake_gui: Any = SimpleNamespace(
        _close_confirmed=False,
        project=Project(),
    )

    assert ProjectWorkspace._prompt_save_before_close(fake_gui)

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
        "neuroseq.project_workspace.gc.collect",
        lambda: collect_calls.__setitem__(0, collect_calls[0] + 1),
    )

    def generator() -> Project:
        generator_calls[0] += 1
        return sentinel_project

    fake_gui: Any = SimpleNamespace(
        load_project=lambda project: loaded_project.append(project)
    )
    ProjectWorkspace._load_example_project(fake_gui, generator)

    assert collect_calls[0] == 1
    assert generator_calls[0] == 1
    assert loaded_project == [sentinel_project]


def test_gui_open(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_path: Path, test_project: Project
) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)

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


def test_gui_open_adds_to_recent_projects(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_path: Path, test_project: Project
) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)

    assert gui.settings.recent_projects[0] == str(project_path.resolve())


def test_gui_save_adds_to_recent_projects(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
    gui.update_project(project_ornstein)
    filename_without_extension = tmp_path / "saved"
    filename = filename_without_extension.with_suffix(".nseq")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)

    assert gui.settings.recent_projects[0] == str(filename.resolve())


def test_gui_toggle_grid(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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


def test_gui_toggle_minimum_stimulus_size(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    stim_widget = gui.stim_editor.stim_widget
    assert stim_widget.keep_short_stimuli_visible
    gui.toggle_minimum_stimulus_size(False)
    assert not stim_widget.keep_short_stimuli_visible
    gui.toggle_minimum_stimulus_size(True)
    assert stim_widget.keep_short_stimuli_visible


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
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    gui.update_project(project_ornstein)
    filename_without_extension = tmp_path / "test"
    # the gui will add the .nseq extension automatically
    filename = filename_without_extension.with_suffix(".nseq")
    assert not filename.exists()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("save options dialog should not be shown")

    monkeypatch.setattr(
        neuroseq.project_workspace, "SaveLoadOptionsDialog", _fail_if_called
    )
    # mock getSaveFileName to return the desired filename
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{filename_without_extension}", True),
    )
    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)
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


def test_gui_save_quick_save_skips_dialogs_when_filepath_set(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    """Once a project has a file path, plain Save (Ctrl+S) writes straight
    back to it without showing the options or file-picker dialogs."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    gui.update_project(project_ornstein)

    filename = tmp_path / "existing.nseq"
    gui.project.save(filename, SaveLoadOptions(True, True, True, True))
    gui.project.filepath = filename
    assert filename.exists()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dialog should not be shown for a quick save")

    monkeypatch.setattr(
        neuroseq.project_workspace, "SaveLoadOptionsDialog", _fail_if_called
    )
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", _fail_if_called)

    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)
    assert gui.project.filepath == filename


def test_gui_save_as_uses_all_options_without_options_dialog(
    tmp_path: Path, monkeypatch: MonkeyPatch, qtbot: QtBot, project_ornstein: Project
) -> None:
    """Save As prompts for a destination and includes every project component."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    gui.update_project(project_ornstein)

    original_filename = tmp_path / "existing.nseq"
    gui.project.save(original_filename, SaveLoadOptions(True, True, True, True))
    gui.project.filepath = original_filename

    new_filename_without_extension = tmp_path / "saved_as"
    new_filename = new_filename_without_extension.with_suffix(".nseq")
    assert not new_filename.exists()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("save options dialog should not be shown")

    monkeypatch.setattr(
        neuroseq.project_workspace, "SaveLoadOptionsDialog", _fail_if_called
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (f"{new_filename_without_extension}", True),
    )

    gui.btn_save_as_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)
    assert new_filename.exists()
    assert gui.project.filepath == new_filename
    # the original file must be untouched by the Save As
    assert original_filename.exists()


def test_gui_import(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    project_ornstein: Project,
    test_project: Project,
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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
        neuroseq.project_workspace,
        "SaveLoadOptionsDialog",
        MockSaveLoadOptionsDialog,
    )
    gui.btn_import_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=WAIT_TIMEOUT_MS)
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
        stimulus_export_dialog_module,
        "Igor2SingleWaveExportDialog",
        MockIgor2SingleWaveExportDialog,
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (f"{output_path}", "CSV file (*.csv)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_igor2_single_wave_csv"),
    )

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2SingleWaveExportDialog", MockDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw"),
    )

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2SingleWaveExportDialog", MockDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw_sweeps"),
    )

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2SingleWaveExportDialog", MockDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    info_messages: list[str] = []
    monkeypatch.setattr(
        stimulus_export_dialog_module.logger,
        "info",
        lambda message, *args: info_messages.append(message % args),
    )
    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=SimpleNamespace(
            info=lambda message, *args: info_messages.append(message % args)
        ),
    )

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2PxpStimExportDialog", MockPxpDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Packed Experiment (*.pxp)"),
    )

    fake_gui: Any = SimpleNamespace(project=project)

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project, "igor-pxp", parent=fake_gui
    )

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

    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2PxpStimExportDialog", MockPxpDialog
    )
    fake_gui: Any = SimpleNamespace(project=project)

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project, "igor-pxp", parent=fake_gui
    )

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "nothing selected" in qmessagebox_calls[0]["title"].lower()


@pytest.mark.parametrize(
    ("handler", "mode"),
    [
        ("btn_export_patch_clamp_stim_clicked", "patch-clamp"),
        ("btn_export_light_stimulus_clicked", "light"),
    ],
)
def test_gui_stimulus_export_delegates_format_selection(monkeypatch, handler, mode):
    from unittest.mock import Mock

    entry_point = Mock()
    monkeypatch.setattr(
        neuroseq.project_workspace, "show_stimulus_export_dialog", entry_point
    )
    project = Project()
    panel = SimpleNamespace(open_generator_dialog=Mock())
    window = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "selected"),
        syscon2_panel=panel,
    )

    getattr(ProjectWorkspace, handler)(window)

    args, kwargs = entry_point.call_args
    assert args == (project, mode)
    assert kwargs.pop("parent") is window
    assert kwargs.pop("current_stim_name") == "selected"
    if mode == "light":
        # The menu export writes files only; it never uploads to SysCon2.
        kwargs.pop("open_syscon2_generator")()
        panel.open_generator_dialog.assert_called_once_with(send_to_syscon=False)
    assert kwargs == {}


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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2SingleWaveExportDialog", MockDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
        "getsavefilename",
        lambda *args: (str(output_path), "Igor Binary Wave (*.ibw)"),
    )

    fake_gui: Any = SimpleNamespace(
        project=project,
        cmb_stim=SimpleNamespace(currentText=lambda: "stim"),
        logger=logging.getLogger("test_gui_export_ibw_extras"),
    )

    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project,
        "igor-single",
        parent=fake_gui,
        current_stim_name=fake_gui.cmb_stim.currentText(),
    )

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
    monkeypatch.setattr(
        stimulus_export_dialog_module, "Igor2PxpStimExportDialog", MockPxpDialog
    )
    monkeypatch.setattr(
        stimulus_export_dialog_module,
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
        from neuroseq.export.exporter_igor2 import Igor2PxpExport

        return Igor2PxpExport(path=out, wave_names=["exc_n0"])

    monkeypatch.setattr(stimulus_export_dialog_module, "export_igor2_pxp", _fake_export)

    fake_gui: Any = SimpleNamespace(project=project)
    stimulus_export_dialog_module.show_stimulus_export_dialog(
        fake_gui.project, "igor-pxp", parent=fake_gui
    )

    assert captured["array_names"] == ["exc"]
    assert captured["unit_override"] == "mW/mm^2"
    assert captured["scaling_factor"] == 3.0


def _bind_import_helpers(fake_gui: Any) -> Any:
    """Give a SimpleNamespace GUI double the real import-completion helpers."""
    if not hasattr(fake_gui, "import_folders"):
        fake_gui.import_folders = None
    fake_gui._complete_import = MethodType(ProjectWorkspace._complete_import, fake_gui)
    fake_gui._complete_stimulus_import = MethodType(
        ProjectWorkspace._complete_stimulus_import, fake_gui
    )
    return fake_gui


@pytest.mark.parametrize(
    ("handler", "dialog_name"),
    [
        ("btn_import_syscon2_seq_clicked", "Syscon2ImportDialog"),
        ("btn_import_patch_clamp_input_clicked", "PatchClampDataInputImportDialog"),
    ],
)
@pytest.mark.parametrize("with_path", [False, True])
@pytest.mark.parametrize(
    ("accepted", "selection"), [(True, "imported"), (True, ""), (False, "imported")]
)
def test_stimulus_import_completion(
    monkeypatch, tmp_path, handler, dialog_name, with_path, accepted, selection
):
    from unittest.mock import Mock
    from unittest.mock import call

    events = Mock()
    project = SimpleNamespace(mark_dirty=events.mark_dirty)
    window = _bind_import_helpers(
        SimpleNamespace(
            project=project,
            update_project=events.refresh,
            cmb_stim=SimpleNamespace(setCurrentText=events.select),
        )
    )
    dialog = Mock(selected_stim_name=selection)
    dialog.load_file = events.load
    dialog.exec = events.execute
    dialog.exec.return_value = (
        QtWidgets.QDialog.DialogCode.Accepted
        if accepted
        else QtWidgets.QDialog.DialogCode.Rejected
    )
    factory = Mock(return_value=dialog)
    monkeypatch.setattr(neuroseq.project_workspace, dialog_name, factory)
    path = tmp_path / "stimulus.csv" if with_path else None

    getattr(ProjectWorkspace, handler)(window, path=path)

    factory.assert_called_once_with(project, parent=window, import_folders=None)
    dialog.deleteLater.assert_called_once_with()
    expected = [call.load(str(path))] if with_path else []
    expected.append(call.execute())
    if accepted:
        expected.extend([call.mark_dirty(), call.refresh(project)])
        if selection:
            expected.append(call.select(selection))
    assert events.mock_calls == expected


def test_gui_import_igor2_wave_updates_selected_stim(
    monkeypatch: MonkeyPatch,
) -> None:
    project = Project()

    class MockPatchClampDataInputImportDialog:
        def __init__(
            self, dialog_project: Project, parent=None, import_folders=None
        ) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_stim_name = "igor_wave"

        def deleteLater(self) -> None:
            pass

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

    monkeypatch.setattr(
        neuroseq.project_workspace,
        "PatchClampDataInputImportDialog",
        MockPatchClampDataInputImportDialog,
    )
    _bind_import_helpers(fake_gui)
    ProjectWorkspace.btn_import_patch_clamp_input_clicked(fake_gui)

    assert fake_gui.cmb_stim.currentText() == "igor_wave"
    assert "igor_wave" in project.stim_arrays


def test_gui_import_igor2_trace_updates_selected_trace(
    monkeypatch: MonkeyPatch,
) -> None:
    import neuroseq.importers.patch_clamp_data_output_import_dialog

    project = Project()
    captured_data: list[TraceData | None] = []
    trace_widget = SimpleNamespace(n_neurons=0)

    def _set_trace_data(data: TraceData | None) -> None:
        captured_data.append(data)
        trace_widget.n_neurons = 0 if data is None else int(np.max(data.indices)) + 1

    trace_widget.set_data = _set_trace_data

    class MockPatchClampDataOutputImportDialog:
        def __init__(
            self, dialog_project: Project, parent=None, import_folders=None
        ) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_experiment_name = "Experiment A"
            self.selected_trace_name = "Igor Trace"

        def deleteLater(self) -> None:
            pass

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
        _keep_cell_selection=lambda widget: None,
    )
    fake_gui._apply_trace_selection = lambda cmb_name: (
        ProjectWorkspace._apply_trace_selection(  # type: ignore[attr-defined]
            fake_gui, cmb_name
        )
    )

    monkeypatch.setattr(
        neuroseq.importers.patch_clamp_data_output_import_dialog,
        "PatchClampDataOutputImportDialog",
        MockPatchClampDataOutputImportDialog,
    )
    _bind_import_helpers(fake_gui)
    ProjectWorkspace.btn_import_patch_clamp_output_clicked(fake_gui)

    assert fake_gui.cmb_trace.currentText() == "Experiment A :: Igor Trace"
    assert "Experiment A" in project.experiments
    assert project.stim_arrays == {}
    assert captured_data[-1] is not None
    assert captured_data[-1].name == "Igor Trace"


def test_gui_import_igor2_trace_refreshes_existing_selected_trace_plot(
    monkeypatch: MonkeyPatch,
) -> None:
    import neuroseq.importers.patch_clamp_data_output_import_dialog

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

    class MockPatchClampDataOutputImportDialog:
        def __init__(
            self, dialog_project: Project, parent=None, import_folders=None
        ) -> None:
            assert dialog_project is project
            self.project = dialog_project
            self.selected_experiment_name = "Experiment A"
            self.selected_trace_name = "Igor Trace"

        def deleteLater(self) -> None:
            pass

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
        _keep_cell_selection=lambda widget: None,
    )
    fake_gui._apply_trace_selection = lambda cmb_name: (
        ProjectWorkspace._apply_trace_selection(  # type: ignore[attr-defined]
            fake_gui, cmb_name
        )
    )

    monkeypatch.setattr(
        neuroseq.importers.patch_clamp_data_output_import_dialog,
        "PatchClampDataOutputImportDialog",
        MockPatchClampDataOutputImportDialog,
    )

    _bind_import_helpers(fake_gui)
    ProjectWorkspace.btn_import_patch_clamp_output_clicked(fake_gui)

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
    class MockPatchClampDataInputImportDialog:
        def __init__(
            self, dialog_project: Project, parent=None, import_folders=None
        ) -> None:
            self.project = dialog_project
            self.selected_stim_name = "igor_wave"

        def deleteLater(self) -> None:
            pass

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
        draw_stim=SimpleNamespace(
            draw_stim_type=neuroseq.stimulation.stim.DrawStimType.Constant
        ),
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
    monkeypatch.setattr(
        neuroseq.project_workspace,
        "PatchClampDataInputImportDialog",
        MockPatchClampDataInputImportDialog,
    )

    _bind_import_helpers(fake_gui)
    ProjectWorkspace.btn_import_patch_clamp_input_clicked(fake_gui)

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
        _keep_cell_selection=lambda widget: None,
    )

    ProjectWorkspace._apply_trace_selection(fake_gui, "Cellpose :: traces")

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
        _keep_cell_selection=lambda widget: None,
    )

    ProjectWorkspace._apply_spike_selection(fake_gui, "Cellpose :: spikes")

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
        _keep_cell_selection=lambda widget: None,
        _auto_show_widget=lambda widget: auto_show_calls.append(widget),
    )

    ProjectWorkspace._apply_spike_selection(fake_gui, "Simulation :: P")

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
    ProjectWorkspace.set_synapse.__wrapped__(fake_gui, "Simulation :: Ce / we")  # type: ignore[attr-defined]

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
        _keep_cell_selection=lambda widget: None,
    )

    ProjectWorkspace._apply_data_selection(fake_gui, "Cellpose :: traces [trace]")

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

        def _expand_stim_arrays_to_neuron_count(
            self,
            target_neuron_count: int,
        ) -> list[str]:
            return ProjectWorkspace._expand_stim_arrays_to_neuron_count(
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

    resized_names = ProjectWorkspace._apply_cellpose_import_result(fake_gui, imported)  # type: ignore[arg-type]

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

    ProjectWorkspace._apply_cellpose_import_result(
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

    # Two stimulus IDs target Cellpose id 0 - a mapping that cannot be resolved.
    ProjectWorkspace._apply_cellpose_import_result(
        fake_gui,  # type: ignore[arg-type]
        _cellpose_result_with_mapping({0: 0, 1: 0}),
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
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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


def test_gui_edit_stim_with_console(qtbot: QtBot, project_ornstein: Project) -> None:
    gui = ProjectWorkspace()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitUntil(lambda: gui.isVisible(), timeout=WAIT_TIMEOUT_MS)
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


def test_gui_synapse_ram_limit(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    tmp_config_dir,
    test_project: Project,
) -> None:
    gui = ProjectWorkspace()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 334
    # 2 MB RAM limit
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("2", True))
    gui.choose_max_ram_synapse_plot_matrix()
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 500
    # 0 MB RAM limit == unlimited
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0", True))
    gui.choose_max_ram_synapse_plot_matrix()
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 1000
    # invalid string displays error message
    assert len(qmessagebox_calls) == 0
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args: ("imnotanumber!", True)
    )
    gui.choose_max_ram_synapse_plot_matrix()
    n_timesteps = gui.heatmap_widget.timestamps.size
    assert n_timesteps == 1000
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "invalid input" in qmessagebox_calls[0]["title"].lower()


def test_gui_brian2_backend(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_config_dir
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    assert gui.heatmap_widget.idx == 0
    # set 10secs of stim padding
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("10", True))
    gui.choose_stim_padding_secs()
    # simulate 0.1s
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args: ("0.1", True))
    gui.btn_continue_simulation_clicked()
    _wait_for_brian2_simulation(qtbot, gui)
    # should have ~0.1 secs of simulation data, and ~10.1 secs of stim array data with dt=0.1ms
    assert gui.heatmap_widget.data is not None
    assert gui.heatmap_widget.timestamps is not None
    assert gui.heatmap_widget.timestamps[-1] == approx(100, rel=2)
    assert gui.stim_editor.stim_widget.n_timestamps == 10100


def test_gui_restore_settings_on_restart(
    monkeypatch: MonkeyPatch, qtbot: QtBot, tmp_config_dir
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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

    gui2 = ProjectWorkspace()
    qtbot.addWidget(gui2)
    with qtbot.waitExposed(gui2, timeout=WAIT_TIMEOUT_MS):
        gui2.show()
    assert b2.prefs.codegen.target == "cython"
    assert gui2.settings.max_ram_synapse_plot_matrix == 512
    assert gui2.settings.stim_padding_secs == 6


def test_gui_playback_speed(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
    qmessagebox_calls: list,
    test_project: Project,
) -> None:
    gui = ProjectWorkspace()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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


class MockAddExperimentDataDialog:
    append_to_experiment = False

    def __init__(self, *args, **kwargs) -> None:
        pass

    def exec(self) -> bool:
        return True


def test_gui_simulate_external_model(
    monkeypatch: MonkeyPatch, qtbot: QtBot, qmessagebox_calls: list
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
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
        neuroseq.project_workspace,
        "AddExperimentDataDialog",
        MockAddExperimentDataDialog,
    )
    gui.btn_do_external_simulation_clicked()
    qtbot.waitUntil(lambda: not gui._background_job.busy, timeout=WAIT_TIMEOUT_MS)
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
    qtbot: QtBot, qmessagebox_calls: list, test_project: Project
) -> None:
    gui = ProjectWorkspace()
    gui.load_project(test_project)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()

    gui.btn_edit_clicked()

    modeleditor = gui.findChild(ModelEditorDialog)

    qtbot.addWidget(modeleditor)
    qtbot.waitUntil(modeleditor.isVisible, timeout=WAIT_TIMEOUT_MS)

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
        ce_synapse_element.param.sigRemoved, raising=True, timeout=WAIT_TIMEOUT_MS
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
    qtbot.waitUntil(lambda: not modeleditor.isVisible(), timeout=WAIT_TIMEOUT_MS)

    # no error dialog was shown
    assert len(qmessagebox_calls) == 0

    # the edited model rebuilt without Ce (and its orphaned dependents), and the
    # network assembled cleanly
    edited = modeleditor.b2model
    assert "Ce" not in edited.synapses
    assert "Ce" not in edited.initials
    assert not any(m.get("source") == "Ce" for m in edited.monitors.values())
    assert "Ce" not in [obj.name for obj in edited.network.objects]


def test_warn_if_cellpose_model_mismatch(qmessagebox_calls: list) -> None:
    project = default_project()  # model has 10 neurons

    class _FakeGui:
        def __init__(self, project: Project) -> None:
            self.project = project

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
    ProjectWorkspace._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 0

    # matching counts (10 cells, 10 neurons) -> no warning
    project.cellpose_data = _cellpose_with(10)
    ProjectWorkspace._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 0

    # mismatch (25 cells vs 10 neurons) -> warning
    project.cellpose_data = _cellpose_with(25)
    ProjectWorkspace._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "mismatch" in qmessagebox_calls[0]["title"].lower()
    assert "10" in qmessagebox_calls[0]["text"]
    assert "25" in qmessagebox_calls[0]["text"]

    # 25 segmented ROIs but a mapping that picks 10 of them as neurons -> the
    # neuron count is the mapping's size, so no warning
    mapped = _cellpose_with(25)
    mapped.stimulus_to_cellpose_mapping = {i: i for i in range(10)}
    project.cellpose_data = mapped
    ProjectWorkspace._warn_if_cellpose_model_mismatch(fake)  # type: ignore[arg-type]
    assert len(qmessagebox_calls) == 1


def test_brian2_busy_notifies_dynamic_input(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible, timeout=WAIT_TIMEOUT_MS)
    release = threading.Event()
    gui._dispatch_io_task(
        lambda report: release.wait(WAIT_TIMEOUT_MS / 1000), "Busy", lambda _: None
    )
    dropped: list[str] = []
    monkeypatch.setattr(
        gui.dynamic_input_controller,
        "dynamic_input_simulation_failed",
        lambda message: dropped.append(message),
    )

    gui._do_brian2_simulation_threaded(0.1)

    # The dropped request is reported to the designer instead of stalling it.
    assert dropped
    release.set()
    qtbot.waitUntil(lambda: not gui._background_job.busy, timeout=WAIT_TIMEOUT_MS)


def test_toolbar_progress_does_not_auto_hide_at_maximum(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitUntil(gui.isVisible, timeout=WAIT_TIMEOUT_MS)
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


def test_gui_export_dialog_opens_on_the_current_input_array(
    monkeypatch: MonkeyPatch, qtbot: QtBot, test_project: Project
) -> None:
    """#588: the stimulation export dialog preselects the input array open
    in the editor, not the first one created."""
    test_project.add_stim_array(
        StimArray(
            "second_stim",
            StimType.Excitation,
            np.zeros((100, 3), dtype=np.float32),
            "nS",
            0.001,
        )
    )
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.load_project(test_project)
    gui.stim_editor.cmb_stim.setCurrentText("second_stim")
    opened: list[str] = []

    class RecordingDialog(syscon2_panel_module.StimulationGeneratorDialog):
        def exec(self) -> int:
            opened.append(self.stim_combo.currentText())
            return 0

    monkeypatch.setattr(
        syscon2_panel_module, "StimulationGeneratorDialog", RecordingDialog
    )

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("SysCon2", True)
    )
    gui.btn_export_light_stimulus_clicked()

    assert opened == ["second_stim"]


def test_import_keeps_syscon_routine_settings_and_selected_stim(qtbot, monkeypatch):
    from neuroseq.export.syscon2_export_service import SysconExportSettings
    from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = default_project()
    gui.update_project(project)
    # Use a second array to detect an accidental reset to the first one on import.
    original = next(iter(project.stim_arrays.values()))
    selected = StimArray(
        "probe_array",
        original.stim_type,
        np.zeros_like(original.values),
        original.units,
        original.dt,
    )
    project.add_stim_array(selected)
    gui.stim_editor.set_project(project, stim_to_select=selected.name)
    dialog = RheobaseDialog(gui.stim_editor)
    gui.dynamic_input_controller._rheobase_dialog = dialog
    dialog.cmb_target.setCurrentText(dialog.SYSCON2_TARGET)
    dialog._running = True
    dialog._awaiting_simulation = True
    host = gui.syscon2_panel.stimulation_host
    host.phase = "awaiting_acquisition"
    settings = SysconExportSettings(config_text="config")
    gui.syscon2_panel._export_settings = settings
    monkeypatch.setattr(
        neuroseq.project_workspace,
        "AddExperimentDataDialog",
        MockAddExperimentDataDialog,
    )
    recording = Experiment("new_recording", 1.0, [], [SpikeData("spikes", [0], [0.2])])

    gui._append_experimental_data(recording)

    assert "new_recording" in project.experiments
    assert gui.dynamic_input_controller._rheobase_dialog is dialog
    assert dialog._running and dialog._awaiting_simulation
    assert host.phase == "awaiting_acquisition"
    assert gui.syscon2_panel.export_settings is settings
    assert gui.stim_editor.current_stim_modifier.stim_array is selected
    assert dialog.cmb_spikes.findText("new_recording :: spikes") >= 0
    gui.update_project(default_project())
    assert gui.dynamic_input_controller._rheobase_dialog is None
    assert gui.syscon2_panel.export_settings is None
    gui._close_confirmed = True


def test_main_menu_export_updates_setup_preview_without_import(
    qtbot, monkeypatch, tmp_path
):
    from qtpy import QtGui

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    path = tmp_path / "overview.png"
    image = QtGui.QPixmap(100, 50)
    image.fill()
    image.save(str(path))

    class ExportDialog(syscon2_panel_module.StimulationGeneratorDialog):
        def exec(self):
            self.sequence_generated.emit("offline.seq", str(path), frozenset({1}))
            return 0

    monkeypatch.setattr(
        syscon2_panel_module, "StimulationGeneratorDialog", ExportDialog
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("SysCon2", True)
    )
    gui.btn_export_light_stimulus_clicked()
    assert not gui.syscon2_panel.preview_label.pixmap().isNull()
    assert "not imported" in gui.syscon2_panel.preview_status_label.text()
    assert gui.syscon2_panel.export_settings is not None


def test_main_menu_export_uses_stored_folder_and_settings(qtbot, monkeypatch, tmp_path):
    """The menu SysCon2 export opens the same generator as the panel's "Export with settings…":
    last-used folder, remembered settings and target power, and it stores
    the chosen folder for next time."""
    from neuroseq.export.syscon2_export_service import SysconExportSettings

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.load_project(default_project())
    panel = gui.syscon2_panel
    stored = tmp_path / "stored"
    chosen = tmp_path / "chosen"
    monkeypatch.setattr(panel._settings, "syscon_export_directory", str(stored))
    settings = SysconExportSettings(config_text="config", target_power_mw=12.0)
    panel._show_export_settings(settings)
    seen: dict[str, object] = {}

    class RecordingDialog(syscon2_panel_module.StimulationGeneratorDialog):
        def apply_settings(self, applied):
            seen["settings"] = applied

        def exec(self):
            seen["folder"] = self.output_folder
            seen["power"] = self.target_power_spin.value()
            self.output_folder = chosen
            return 0

    monkeypatch.setattr(
        syscon2_panel_module, "StimulationGeneratorDialog", RecordingDialog
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("SysCon2", True)
    )

    gui.btn_export_light_stimulus_clicked()

    assert seen == {"settings": settings, "folder": stored, "power": 12.0}
    assert panel._settings.syscon_export_directory == str(chosen)
    gui._close_confirmed = True


def test_hardware_rheobase_keeps_its_delivered_pulses_in_the_stim_array(qtbot):
    """The probes delivered to BrainScaleS-2 stay in the input array when the
    sweep ends; only the padding after the last delivered probe goes."""
    from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
    from tests.fake_stimulation_host import FakeBss2Host

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = default_project()
    gui.update_project(project)
    stim = gui.stim_editor.current_stim_modifier.stim_array
    gui.stim_editor.zero_current_stim_array()
    host = FakeBss2Host()
    gui.stim_editor.bss2_host = host  # type: ignore[assignment]
    dialog = RheobaseDialog(gui.stim_editor)
    qtbot.addWidget(dialog)
    dialog.refresh()
    dialog.cmb_target.setCurrentText(dialog.BSS2_TARGET)
    dialog.cmb_sweeping_strategy.setCurrentText(
        dialog.FIXED_DURATION_VARYING_AMPLITUDE_STRATEGY
    )
    dialog.spn_repeats.setValue(1)
    dialog.spn_tolerance.setValue(0.3)
    dialog.start()
    for _ in range(20):
        if not dialog._running:
            break
        onset = dialog._pulse_onset_seconds - dialog._block_range[0] * float(stim.dt)
        times = [onset + 0.002] if dialog._pending_amplitude >= 0.5 else []
        host.complete(
            {"block_spikes": SpikeData("s", np.zeros(len(times)), np.array(times))}
        )
    assert not dialog._running
    stim = gui.stim_editor.current_stim_modifier.stim_array
    last_block_end = dialog._delivered_end_sample
    assert last_block_end > 0
    # Every delivered probe is still there, and nothing after the last one.
    assert stim.values.shape[0] == last_block_end
    pulses = np.flatnonzero(np.diff((stim.values[:, 0] > 0).astype(int)) == 1)
    assert len(pulses) == len(dialog._samples)
    project.dirty = False


def test_hardware_rheobase_drops_a_probe_that_was_never_delivered(qtbot):
    from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
    from tests.fake_stimulation_host import FakeBss2Host

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = default_project()
    gui.update_project(project)
    gui.stim_editor.zero_current_stim_array()
    host = FakeBss2Host()
    gui.stim_editor.bss2_host = host  # type: ignore[assignment]
    dialog = RheobaseDialog(gui.stim_editor)
    qtbot.addWidget(dialog)
    dialog.refresh()
    dialog.cmb_target.setCurrentText(dialog.BSS2_TARGET)
    dialog.start()
    dialog.stop()  # before the chip ran the first probe
    stim = gui.stim_editor.current_stim_modifier.stim_array
    assert not np.any(stim.values)
    project.dirty = False


def test_refused_hardware_rheobase_keeps_the_existing_input(qtbot, monkeypatch):
    from neuroseq.stimulation.dynamic_input import rheobase_dialog
    from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
    from tests.fake_stimulation_host import FakeBss2Host

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = default_project()
    gui.update_project(project)
    before = gui.stim_editor.current_stim_modifier.stim_array.values.copy()
    assert np.any(before)
    gui.stim_editor.bss2_host = FakeBss2Host()  # type: ignore[assignment]
    # The start prompt is answered "continue after the last stimulus".
    monkeypatch.setattr(
        rheobase_dialog, "confirm_start_routine", lambda *_args, **_kwargs: "continue"
    )
    dialog = RheobaseDialog(gui.stim_editor)
    qtbot.addWidget(dialog)
    dialog.refresh()
    dialog.cmb_target.setCurrentText(dialog.BSS2_TARGET)
    dialog.spn_max_amplitude.setValue(5.0)  # refused: above 1
    dialog.start()
    assert not dialog._running
    after = gui.stim_editor.current_stim_modifier.stim_array.values
    assert np.any(after)
    np.testing.assert_array_equal(after[: before.shape[0]], before[: after.shape[0]])
    last = int(np.flatnonzero(np.any(before != 0, axis=1))[-1])
    assert after.shape[0] > last
    project.dirty = False


def test_file_export_offers_traces_and_spikes_as_numpy_arrays(
    qtbot: QtBot, mocker
) -> None:
    from neuroseq.export import numpy_export_dialog

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    submenus = {
        action.text().replace("&", ""): action.menu()
        for action in gui.menu_file.actions()
        if action.menu() is not None
    }
    actions = {
        action.text().replace("&", ""): action
        for action in submenus["Export"].actions()
    }
    opened = mocker.patch.object(numpy_export_dialog.NumpyExportDialog, "exec")
    for label in ("Traces as NumPy Array…", "Spikes as NumPy Array…"):
        actions[label].trigger()
    assert opened.call_count == 2


def test_renaming_a_data_group_in_manage_data_updates_the_data_selectors(
    qtbot: QtBot, test_experiments: list[Experiment]
) -> None:
    """#654: a rename in Manage Data shows up in the Data toolbar selectors."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.project.add_experiment(test_experiments[0])
    gui.update_project(gui.project)
    trace_name = test_experiments[0].data[0].name
    gui.btn_manage_data_clicked()
    dialog = gui.findChild(DataManagerDialog)
    assert dialog is not None

    def items(combo: QtWidgets.QComboBox) -> list[str]:
        return [combo.itemText(i) for i in range(combo.count())]

    dialog.group_item("e1").setText(0, "Calcium")
    qtbot.waitUntil(
        lambda: f"Calcium :: {trace_name}" in items(gui.cmb_trace),
        timeout=WAIT_TIMEOUT_MS,
    )
    assert all(not item.startswith("e1 ::") for item in items(gui.cmb_trace))
    gui.project.history.undo()
    qtbot.waitUntil(
        lambda: f"e1 :: {trace_name}" in items(gui.cmb_trace),
        timeout=WAIT_TIMEOUT_MS,
    )
    dialog.close()
    gui._close_confirmed = True
