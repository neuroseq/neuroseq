from types import SimpleNamespace
from typing import cast

import numpy as np
from qtpy import QtWidgets

from neuroseq.example_projects import default_project
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.stimulation.dynamic_input.dynamic_input_controller import (
    DynamicInputController,
)
from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_editor import StimEditor
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)


def test_controller_forwards_lifecycle_to_active_routines(qtbot) -> None:
    controller = DynamicInputController(cast(StimEditor, SimpleNamespace()))
    events = []
    dialog = SimpleNamespace(
        simulation_finished=lambda: events.append("finished"),
        simulation_failed=lambda message: events.append(message),
        refresh=lambda: events.append("refresh"),
        apply_theme=lambda theme: events.append(theme),
        close=lambda: events.append("close"),
        deleteLater=lambda: events.append("delete"),
    )
    controller._rheobase_dialog = cast(RheobaseDialog, dialog)

    controller.dynamic_input_simulation_finished()
    controller.dynamic_input_simulation_failed("failed")
    controller.refresh()
    controller.apply_dynamic_input_theme("dark")
    controller.discard_dynamic_input_dialogs()
    controller.discard_dynamic_input_dialogs()

    assert events == ["finished", "failed", "refresh", "dark", "close", "delete"]
    assert controller._rheobase_dialog is None
    assert controller._dynamic_input_theme == "dark"


def test_controller_requires_an_editable_stimulation_array(qtbot) -> None:
    editor = SimpleNamespace(project=None, current_stim_modifier=None)
    controller = DynamicInputController(cast(StimEditor, editor))

    controller.open_dynamic_input_designer()

    assert controller._routine_selector_dialog is None


def _two_array_project() -> Project:
    project = Project()
    for name in ("a", "b"):
        project.add_stim_array(
            StimArray(name, StimType.Excitation, np.zeros((50, 2)), "nS", 0.001)
        )
    return project


def _editor(qtbot, project: Project) -> StimEditor:
    editor = StimEditor(
        ConstrainedPanZoomController(), QtWidgets.QComboBox(), QtWidgets.QToolBar()
    )
    qtbot.addWidget(editor)
    editor.set_project(project, stim_to_select="b")
    return editor


def test_running_routine_locks_array_selection_and_undo(qtbot) -> None:
    """A routine writes each block into the array it started with: switching
    arrays would redirect the next block, and undo would remove a delivered one."""
    project = _two_array_project()
    editor = _editor(qtbot, project)
    assert editor.current_stim_modifier is not None
    editor.current_stim_modifier.do(slice(0, 1), slice(0, 2), 1.0)
    controller = DynamicInputController(editor)
    controller._show_routine_dialog("_rheobase_dialog", RheobaseDialog)
    dialog = controller._rheobase_dialog
    assert dialog is not None
    qtbot.addWidget(dialog)

    dialog._running = True

    assert not editor.cmb_stim.isEnabled()
    assert not editor.action_undo.isEnabled()
    editor.undo()
    assert project.stim_arrays["b"].values[0, 0] == 1.0
    # Rebuilding the views of the same project keeps the routine's array.
    controller.update_project(project)
    editor.set_project(project)
    assert controller._rheobase_dialog is dialog
    assert editor.cmb_stim.currentText() == "b"
    assert not editor.cmb_stim.isEnabled()

    dialog._running = False

    assert editor.cmb_stim.isEnabled()
    assert editor.action_undo.isEnabled()
    editor.undo()
    assert project.stim_arrays["b"].values[0, 0] == 0.0


def test_controller_discards_routines_only_for_another_project(qtbot) -> None:
    project = _two_array_project()
    editor = _editor(qtbot, project)
    controller = DynamicInputController(editor)
    controller._show_routine_dialog("_rheobase_dialog", RheobaseDialog)
    dialog = controller._rheobase_dialog
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._running = True

    controller.update_project(project)
    assert controller._rheobase_dialog is dialog

    controller.update_project(_two_array_project())
    assert controller._rheobase_dialog is None
    assert not dialog._running
    assert editor.cmb_stim.isEnabled()


def test_workspace_refresh_keeps_running_routine_and_spares_new_project(
    qtbot,
) -> None:
    """Data Manager deletes, imports and model edits refresh the same project
    and must not end a running routine; replacing the project ends it before
    the editor switches, so its trim never reaches the new project's array."""
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.update_project(default_project())
    controller = gui.dynamic_input_controller
    controller._show_routine_dialog("_rheobase_dialog", RheobaseDialog)
    dialog = controller._rheobase_dialog
    assert dialog is not None
    dialog._running = True

    gui.update_project(gui.project)

    assert controller._rheobase_dialog is dialog
    assert dialog._running

    new_project = default_project()
    lengths = {
        name: stim.values.shape[0] for name, stim in new_project.stim_arrays.items()
    }
    gui.update_project(new_project)

    assert controller._rheobase_dialog is None
    assert {
        name: stim.values.shape[0] for name, stim in new_project.stim_arrays.items()
    } == lengths
    assert gui.stim_editor.cmb_stim.isEnabled()
    gui._close_confirmed = True
    gui.close()
