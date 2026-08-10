from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.routine_overview_dialog import ROUTINE_OVERVIEWS
from neuroseq.routine_overview_dialog import RoutineSelectorDialog
from neuroseq.routine_overview_dialog import routine_overview
from neuroseq.stim_editor import StimEditor


def test_every_dynamic_routine_has_an_overview() -> None:
    # Each routine offered by the selector must have a registered overview so
    # the always-on preview never falls back to "no preview available".
    for routine in StimEditor.DYNAMIC_INPUT_ROUTINES:
        overview = routine_overview(routine)
        assert overview is not None
        assert overview.goal
        assert overview.parameters
        assert overview.plot_title


def test_routine_overview_unknown_routine_returns_none() -> None:
    assert routine_overview("not a routine") is None


def test_selector_shows_preview_for_initial_selection(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    first = StimEditor.DYNAMIC_INPUT_ROUTINES[0]
    overview = routine_overview(first)
    assert overview is not None
    # Preview is populated immediately (always on), in the same window.
    assert dialog.lbl_title.text() == overview.title
    assert dialog.lbl_goal.text() == overview.goal
    assert dialog._axes.get_title() == overview.plot_title
    assert dialog._axes.lines
    dialog.close()


def test_selector_preview_updates_when_selection_changes(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    for routine in StimEditor.DYNAMIC_INPUT_ROUTINES:
        dialog.cmb_routine.setCurrentText(routine)
        overview = routine_overview(routine)
        assert overview is not None
        assert dialog.lbl_title.text() == overview.title
        assert dialog._axes.get_title() == overview.plot_title
        assert dialog._axes.lines
    dialog.close()


def test_selector_continue_emits_selected_routine(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    chosen: list[str] = []
    dialog.routine_chosen.connect(chosen.append)
    dialog.cmb_routine.setCurrentText(StimEditor.STDP_WINDOW_ROUTINE)
    dialog._on_continue()
    assert chosen == [StimEditor.STDP_WINDOW_ROUTINE]
    dialog.close()


def test_selector_apply_theme_runs(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(ROUTINE_OVERVIEWS))
    qtbot.addWidget(dialog)
    # Recoloring for either theme must not raise.
    dialog.apply_theme("dark")
    dialog.apply_theme("light")
    dialog.close()
