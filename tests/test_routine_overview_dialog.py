from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.routine_overview_dialog import ROUTINE_OVERVIEWS
from neuroseq.routine_overview_dialog import RoutineOverviewDialog
from neuroseq.routine_overview_dialog import routine_overview
from neuroseq.stim_editor import StimEditor


def test_every_dynamic_routine_has_an_overview() -> None:
    # Each routine offered by the picker must have a registered overview so the
    # new pre-dialog step never silently falls back to opening the dialog.
    for routine in StimEditor.DYNAMIC_INPUT_ROUTINES:
        overview = routine_overview(routine)
        assert overview is not None
        assert overview.goal
        assert overview.parameters
        assert overview.plot_title


def test_routine_overview_unknown_routine_returns_none() -> None:
    assert routine_overview("not a routine") is None


@pytest.mark.parametrize("routine", list(ROUTINE_OVERVIEWS))
def test_routine_overview_dialog_renders_goal_and_sample_plot(
    routine: str, qtbot: QtBot
) -> None:
    overview = routine_overview(routine)
    assert overview is not None
    dialog = RoutineOverviewDialog(overview)
    qtbot.addWidget(dialog)

    # The sample plot is drawn with the routine's plot title, and drawing it
    # produced at least one line (the illustrative curve).
    assert dialog._axes.get_title() == overview.plot_title
    assert dialog._axes.lines

    # Continue emits continue_requested (the caller opens the real dialog).
    fired: list[bool] = []
    dialog.continue_requested.connect(lambda: fired.append(True))
    dialog._on_continue()
    assert fired == [True]
    dialog.close()


def test_routine_overview_dialog_apply_theme_runs(qtbot: QtBot) -> None:
    overview = next(iter(ROUTINE_OVERVIEWS.values()))
    dialog = RoutineOverviewDialog(overview)
    qtbot.addWidget(dialog)
    # Recoloring for either theme must not raise.
    dialog.apply_theme("dark")
    dialog.apply_theme("light")
    dialog.close()
