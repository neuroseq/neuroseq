from types import SimpleNamespace
from typing import cast

from neuroseq.dynamic_input.dynamic_input_controller import DynamicInputController
from neuroseq.dynamic_input.rheobase_dialog import RheobaseDialog
from neuroseq.stimulation.stim_editor import StimEditor


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
